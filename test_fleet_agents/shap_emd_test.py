"""shap_emd_test — data-wise verifier for SHAP-guided Earth-Mover's-Distance (thiagorr162/shap-emd).

Core properties:
  1. cost_matrix_from_curves: symmetric, non-negative, zero diagonal; identical curves → zero cost.
  2. emd2 is a proper metric on the simplex under C: d(a,a)=0, symmetric, and matches scipy's 1-D
     Wasserstein when C is the |i-j| line metric.
  3. shap_emd_distance normalizes and orders (closer compositions → smaller distance).
  4. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import shap_emd as SE


def _run():
    print("=== SHAP-EMD VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}

    # 1. cost matrix properties
    x = np.linspace(0, 1, 50)
    curves = np.array([np.sin((k + 1) * np.pi * x) for k in range(6)])
    C = SE.cost_matrix_from_curves(curves, x)
    checks["cost_symmetric"] = np.allclose(C, C.T)
    checks["cost_nonneg"] = bool((C >= 0).all())
    checks["cost_zero_diag"] = np.allclose(np.diag(C), 0)
    checks["cost_identical_zero"] = abs(SE.cost_matrix_from_curves(np.array([x, x]), x)[0, 1]) < 1e-9

    # 2. emd2 metric + agreement with 1-D Wasserstein on the line metric
    F = 6
    Cline = np.abs(np.subtract.outer(np.arange(F), np.arange(F))).astype(float)
    a = rng.dirichlet(np.ones(F)); b = rng.dirichlet(np.ones(F))
    d_aa = SE.emd2(a, a, Cline); d_ab = SE.emd2(a, b, Cline); d_ba = SE.emd2(b, a, Cline)
    from scipy.stats import wasserstein_distance
    w_ref = wasserstein_distance(np.arange(F), np.arange(F), a, b)
    print(f"  -> emd2(a,b)={d_ab:.4f}  scipy-1D-Wasserstein={w_ref:.4f}  d(a,a)={d_aa:.2e}")
    checks["emd_self_zero"] = d_aa < 1e-6
    checks["emd_symmetric"] = abs(d_ab - d_ba) < 1e-6
    checks["emd_matches_wasserstein_1d"] = abs(d_ab - w_ref) < 1e-4

    # 3. ordering: a nearer neighbor has smaller distance than a far one
    near = a.copy(); near[0] += 0.01; near[1] -= 0.01; near = np.clip(near, 0, None); near /= near.sum()
    far = rng.dirichlet(np.ones(F) * 0.3)     # very different, spiky
    norm = SE.normalization_p99(rng.dirichlet(np.ones(F), size=10), C)
    d_near = SE.shap_emd_distance(a, near, C, norm); d_far = SE.shap_emd_distance(a, far, C, norm)
    print(f"  -> normalized d(a,near)={d_near:.4f} < d(a,far)={d_far:.4f}  (norm={norm:.4f})")
    checks["ordering"] = d_near < d_far
    checks["norm_positive"] = norm > 0

    # 4. agent contract
    st, dta, to, msg = SE.run_shapemd({"spec": {"n_features": 8, "n_ref": 10}}, "t")
    checks["agent_done"] = st == "done" and dta["d_self"] < 1e-6

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== shap-emd: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
