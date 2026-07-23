"""gpu_patterns_test — data-wise verifier for the GPU-Puzzles parallel primitives + roofline model.

Core properties:
  1. prefix_scan == np.cumsum; segment_reduce reduces per segment; tiled_matmul == A@B for several tiles.
  2. roofline: a big matmul is compute-bound, an elementwise add is memory-bound; ridge point correct;
     attainable never exceeds peak.
  3. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import gpu_patterns as GP


def _run():
    print("=== GPU-PATTERNS VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}

    # 1. primitives
    x = rng.randn(37)
    checks["scan_eq_cumsum"] = np.allclose(GP.prefix_scan(x), np.cumsum(x))
    seg = np.array([0, 0, 1, 1, 1, 2]); v = np.array([1., 2., 3., 4., 5., 6.])
    sr = GP.segment_reduce(v, seg, "sum")
    checks["segment_reduce"] = sr == {0: 3.0, 1: 12.0, 2: 6.0}
    A = rng.randn(48, 40); B = rng.randn(40, 32)
    checks["tiled_matmul_16"] = np.allclose(GP.tiled_matmul(A, B, 16), A @ B)
    checks["tiled_matmul_7"] = np.allclose(GP.tiled_matmul(A, B, 7), A @ B)   # non-divisor tile
    print(f"  -> scan/reduce/tiled-matmul reference primitives verified")

    # 2. roofline
    peak_flops, peak_bw = 200e12, 2e12          # ridge AI = 100 FLOP/byte
    M = 4096
    rmm = GP.roofline(2 * M**3, 3 * M * M * 2, peak_flops, peak_bw)
    radd = GP.roofline(M * M, 3 * M * M * 2, peak_flops, peak_bw)
    print(f"  -> matmul AI={rmm['arithmetic_intensity']:.0f} {rmm['bound']} | add AI={radd['arithmetic_intensity']:.2f} {radd['bound']} | ridge={rmm['ridge_point']:.0f}")
    checks["ridge_correct"] = abs(rmm["ridge_point"] - 100.0) < 1e-6
    checks["matmul_compute_bound"] = rmm["bound"] == "compute"
    checks["add_memory_bound"] = radd["bound"] == "memory"
    checks["attainable_le_peak"] = rmm["attainable_flops"] <= peak_flops + 1 and radd["attainable_flops"] <= peak_flops
    checks["ai_monotone"] = rmm["arithmetic_intensity"] > radd["arithmetic_intensity"]

    # 3. agent
    st, dta, to, msg = GP.run_gpupatterns({"spec": {"m": 4096}}, "t")
    checks["agent_done"] = st == "done" and dta["matmul_bound"] == "compute" and dta["add_bound"] == "memory"

    for k, val in checks.items():
        print(f"  {'OK' if val else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== gpu-patterns: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
