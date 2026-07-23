"""finance_pack_test — verifier for the portfolio/forecasting agents (offline, synthetic)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import finance_pack as Fp


def _run():
    print("=== FINANCE PACK VERIFIER ===")
    checks = {}

    # position sizer: stays in [0,2]; higher vol → allocation closer to neutral (1.0)
    sig = np.array([0.5, 0.5, 0.5])
    a_lowvol = Fp.size_positions(sig, vol=np.array([0.05, 0.05, 0.05]), target_vol=0.1)
    a_hivol = Fp.size_positions(sig, vol=np.array([0.5, 0.5, 0.5]), target_vol=0.1)
    checks["sizer_bounds"] = a_lowvol.min() >= 0 and a_lowvol.max() <= 2
    checks["sizer_vol_scaling"] = abs(a_hivol[0] - 1.0) < abs(a_lowvol[0] - 1.0)  # high vol → nearer neutral

    # market odds: no-vig sums to 1; even odds → 0.5/0.5
    pa, pb = Fp.no_vig((-110, -110))
    checks["novig_sums_one"] = abs((pa + pb) - 1.0) < 1e-9 and abs(pa - 0.5) < 1e-9
    fav_pa, fav_pb = Fp.no_vig((-200, +170))  # favorite
    checks["novig_favorite"] = fav_pa > fav_pb
    bl = Fp.blend_market(np.array([0.6]), np.array([0.8]), weight=0.5)
    checks["blend_interp"] = abs(bl[0] - 0.7) < 1e-9

    # forecast-drivers-then-derive: linear formula recovered from drivers
    D = np.random.RandomState(0).rand(50, 3); coef = np.array([2.0, -1.0, 0.5])
    out = Fp.derive_from_drivers(D, lambda r: r @ coef + 1.0)
    checks["derive_formula"] = np.allclose(out, D @ coef + 1.0)

    # label-lag-anchor: blends toward recent-label mean
    an = Fp.anchor_blend(np.array([1.0, 1.0]), recent_labels=[0.0, 0.0, 0.0], w=0.5)
    checks["anchor_blend"] = np.allclose(an, [0.5, 0.5])

    # distributional recalibrator: per-group affine removes a group-specific offset
    rng = np.random.RandomState(1)
    preds = rng.rand(200); groups = np.array([0] * 100 + [1] * 100)
    y = preds.copy(); y[groups == 1] += 0.5   # group 1 has a +0.5 bias vs preds
    out, params = Fp.recalibrate_by_group(preds, y, groups)
    bias_after = np.mean(out[groups == 1] - y[groups == 1])
    checks["recal_removes_bias"] = abs(bias_after) < 1e-6

    # agent contracts
    st, d, to, msg = Fp.run_sizer({"spec": {"signal": sig.tolist(), "vol": [0.1, 0.1, 0.1]}}, "t")
    checks["sizer_agent"] = st == "done" and "_alloc" in d
    st, d, to, msg = Fp.run_recal({"spec": {"preds": preds.tolist(), "y": y.tolist(), "groups": groups.tolist()}}, "t")
    checks["recal_agent"] = st == "done" and len(d["params"]) == 2

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== finance-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
