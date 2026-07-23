"""final_pure_pack_test — verifier for the last pure tools (offline)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import final_pure_pack as P


def _run():
    print("=== FINAL PURE PACK VERIFIER ===")
    checks = {}

    # spatial augmentor: rotation preserves pairwise distances; flipx mirrors about center
    pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    rot = P.augment_coords(pts, "rotate", center=[0, 0], angle=0.9)
    def pd(a): return np.linalg.norm(a[0] - a[1]) + np.linalg.norm(a[1] - a[2]) + np.linalg.norm(a[0] - a[2])
    checks["rotate_preserves_dist"] = abs(pd(rot) - pd(pts)) < 1e-9
    fl = P.augment_coords(pts, "flipx", center=[0.5, 0.5])
    checks["flipx_mirrors"] = np.allclose(fl[:, 0], [1.0, 0.0, 1.0])
    dr = P.augment_coords(pts, "dropout", drop_idx=[1])
    checks["dropout"] = np.allclose(dr[1], pts.mean(0))

    # infer-cascade: high-conf at stage0 stays; low-conf escalates to last stage
    conf = np.array([[0.9, 0.5, 0.5], [0.1, 0.2, 0.99], [0.1, 0.2, 0.1]])
    stage = P.cascade_stage(conf, thresholds=[0.8, 0.8, 0.8])
    checks["cascade_stay"] = stage[0] == 0            # confident at stage 0
    checks["cascade_escalate"] = stage[1] == 2         # only stage 2 clears
    checks["cascade_fallback"] = stage[2] == 2         # none clear → last stage

    # drill generator: fills templates
    drills = P.generate_drills([("What is {a}+{b}?", "{c}")], {"a": ["1"], "b": ["2"], "c": ["3"]}, n=5)
    checks["drills_count"] = len(drills) == 5 and drills[0]["prompt"] == "What is 1+2?" and drills[0]["answer"] == "3"

    # heteroscedastic: sigma high where models disagree, ~0 where they agree
    ens = np.array([[1.0, 5.0], [1.0, 5.0], [1.0, 5.0]])   # agree on both cols
    mu, sigma = P.ensemble_uncertainty(ens)
    checks["hetero_agree_zero_sigma"] = np.allclose(sigma, 0.0)
    ens2 = np.array([[1.0, 0.0], [2.0, 10.0], [3.0, 20.0]])  # disagree, col1 more
    mu2, sigma2 = P.ensemble_uncertainty(ens2)
    checks["hetero_disagree_sigma"] = sigma2[1] > sigma2[0] > 0
    # GaussianNLL lower when sigma matches error
    nll_good = P.gaussian_nll_loss([1.0], [1.0], [0.1]); nll_bad = P.gaussian_nll_loss([1.0], [3.0], [0.1])
    checks["gnll_penalizes_error"] = nll_bad > nll_good

    # agent contracts
    st, d, to, msg = P.run_augment({"spec": {"points": pts.tolist(), "op": "rotate", "center": [0, 0], "angle": 0.5}}, "t")
    checks["augment_agent"] = st == "done" and "_points" in d
    st, d, to, msg = P.run_hetero({"spec": {"ensemble_preds": ens2.tolist(), "y": [2.0, 10.0]}}, "t")
    checks["hetero_agent"] = st == "done" and d["gaussian_nll"] is not None

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== final-pure-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
