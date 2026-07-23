"""sr_bf16_optimizer_test — verify the stochastic-rounding bf16 numerics (offline, pure numpy).

Checks the properties that make SR worth it:
  1. SR is UNBIASED: mean of many SR casts of a sub-ULP value ≈ the true value (RTNE is biased to a grid point).
  2. sub-ULP ACCUMULATION: repeatedly adding a tiny delta to a bf16 accumulator STALLS under RTNE but
     tracks fp32 under SR.
  3. bf16-master AdamW: SR reaches lower final RMSE than RTNE and near the fp32 baseline.
  4. fp31 decompose/reconstruct round-trips.
  5. agent contract returns done with the memory-savings report.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import sr_bf16_optimizer as S


def _run():
    print("=== SR-BF16-OPTIMIZER VERIFIER ===")
    checks = {}
    rng = np.random.default_rng(0)

    # 1. unbiasedness of SR vs bias of RTNE on a value between two bf16 grid points
    x = np.full(200000, 1.234567, np.float32)
    sr_mean = float(np.mean(S.fp32_to_bf16_sr(x, rng)))
    rtne_val = float(S.fp32_to_bf16_rtne(np.float32([1.234567]))[0])
    checks["sr_unbiased"] = abs(sr_mean - 1.234567) < 1e-3
    checks["rtne_is_biased"] = abs(rtne_val - 1.234567) > abs(sr_mean - 1.234567)
    print(f"  -> SR mean {sr_mean:.6f} vs true 1.234567 ; RTNE grid {rtne_val:.6f}")

    # 2. sub-ULP accumulation: add a delta ~1/20 of the local ULP many times (region with a constant ULP).
    base = np.float32(4.0)                                                       # bf16 ULP near 4 ≈ 2^-5
    ulp = float(S.fp32_to_bf16_rtne(np.float32([4.0 + 0.03125]))[0] - 4.0) or 0.03125
    N = 1000; delta = np.float32(ulp / 20.0); expected = base + N * float(delta)
    acc_rtne = S.fp32_to_bf16_rtne(np.array([base], np.float32))
    for _ in range(N):                                                          # RTNE stalls: sub-ULP → no change
        acc_rtne = S.fp32_to_bf16_rtne(acc_rtne + delta)
    # SR is UNBIASED → average of many independent runs tracks the fp32 sum (single runs have real variance)
    finals = []
    for tr in range(25):
        a = S.fp32_to_bf16_sr(np.array([base], np.float32), np.random.default_rng(100 + tr))
        for _ in range(N):
            a = S.fp32_to_bf16_sr(a + delta, np.random.default_rng(9000 * tr + _))
        finals.append(float(a[0]))
    sr_mean_final = float(np.mean(finals))
    checks["rtne_stalls"] = abs(float(acc_rtne[0]) - base) < 0.5 * ulp
    checks["sr_accumulates"] = abs(sr_mean_final - expected) < 4 * ulp        # unbiased accumulation
    print(f"  -> {N} sub-ULP adds: RTNE={float(acc_rtne[0]):.4f} (stalled@{base}) "
          f"SR mean={sr_mean_final:.4f} exp={expected:.4f} (ulp={ulp:.4f})")

    # 3. bf16-master AdamW convergence
    dim = 64
    c = rng.standard_normal(dim).astype(np.float32) * 0.3 + 1.234567
    w0 = c + rng.standard_normal(dim).astype(np.float32) * 0.5
    grad = lambda w: w.astype(np.float32) - c
    rmse = {m: float(np.sqrt(np.mean((S.adam_bf16_master(grad, w0, steps=600, lr=5e-3, rounding=m, seed=1) - c) ** 2)))
            for m in ("fp32", "rtne", "sr")}
    checks["sr_beats_rtne_adam"] = rmse["sr"] < 0.3 * rmse["rtne"]            # SR escapes the RTNE stall
    checks["sr_near_bf16_floor"] = rmse["sr"] < 0.02                          # reaches the bf16 resolution floor
    print(f"  -> AdamW RMSE fp32={rmse['fp32']:.2e} rtne={rmse['rtne']:.2e} sr={rmse['sr']:.2e}")

    # 4. fp31 round-trip
    v = rng.standard_normal(1000).astype(np.float32) * 5
    xb, err = S.fp31_decompose(v)
    rec = S.fp31_reconstruct(xb, err)
    checks["fp31_roundtrip"] = float(np.max(np.abs(rec - v))) < 1e-3
    checks["fp31_better_than_bf16"] = float(np.mean(np.abs(rec - v))) < float(np.mean(np.abs(xb - v)))
    print(f"  -> fp31 max|err|={np.max(np.abs(rec - v)):.2e} vs bf16 {np.max(np.abs(xb - v)):.2e}")

    # 5. agent contract
    st, d, to, msg = S.run({"spec": {"dim": 48, "steps": 400}}, "t")
    checks["agent_done"] = st == "done" and "rmse" in d and d["bytes_per_param_sr"] == 6
    print(f"  -> agent: {st} | {msg[:90]}")

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== sr-bf16-optimizer: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
