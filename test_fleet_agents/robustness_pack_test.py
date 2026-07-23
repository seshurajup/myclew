"""robustness_pack_test — verifier for shift/decode/constraint/routing agents (offline, synthetic)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import robustness_pack as R


def _run():
    print("=== ROBUSTNESS PACK VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}

    # shift-adapt: test shifted +2 on feat0; train rows near +2 get higher weight
    Xtr = rng.normal(0, 1, (400, 4)); Xte = rng.normal(0, 1, (400, 4)); Xte[:, 0] += 2.0
    w = R.shift_weights(Xtr, Xte)
    hi = Xtr[:, 0] > 1.0; lo = Xtr[:, 0] < -1.0
    checks["shift_weights_direction"] = w[hi].mean() > w[lo].mean()
    hold = R.shift_aligned_holdout(Xtr, Xte, 0.1)
    checks["shift_holdout_testlike"] = Xtr[hold, 0].mean() > Xtr[:, 0].mean()

    # geospatial-fe: target depends on grid cell → cell feature correlates with target
    coords = rng.rand(500, 2); y = (coords[:, 0] > 0.5).astype(float) + rng.normal(0, 0.1, 500)
    cte = rng.rand(100, 2)
    tr, te = R.geo_features(coords, y, cte, n_bins=8, k=10)
    checks["geo_signal"] = abs(np.corrcoef(tr[:, 0], y)[0, 1]) > 0.5
    checks["geo_test_cols"] = te.shape[1] == 2

    # linear-constraint-projector: Total = a+b+c enforced
    preds = np.array([[1.0, 2.0, 3.0, 5.0], [0.0, 1.0, 1.0, 3.0]])  # cols: a,b,c,total
    A = np.array([[1.0, 1.0, 1.0, -1.0]]); b = np.array([0.0])       # a+b+c-total=0
    proj = R.project_constraints(preds, A, b)
    checks["constraint_satisfied"] = np.allclose(proj[:, :3].sum(1), proj[:, 3], atol=1e-9)
    checks["constraint_minimal"] = np.linalg.norm(proj - preds) < np.linalg.norm(preds)

    # runtime-budget-router: budget picks highest gain/cost within limit
    costs = np.array([1.0, 1.0, 1.0, 1.0]); gain = np.array([10.0, 1.0, 1.0, 1.0])
    mask, spent = R.budget_route(costs, budget=2.0, quality_gain=gain)
    checks["router_budget"] = spent <= 2.0 and mask[0] and mask.sum() == 2

    # mbr: 3 near-identical + 1 outlier → pick from the consensus cluster
    cands = [np.array([0, 0.0]), np.array([0.1, 0]), np.array([0, 0.1]), np.array([9, 9.0])]
    best, scores = R.mbr_select(cands, lambda a, b: -float(np.linalg.norm(a - b)))
    checks["mbr_consensus"] = best in (0, 1, 2)

    # noisy-label-cleaner: conflicting labels for same key → majority
    keys = ["a", "a", "a", "b"]; labels = [1, 1, 0, 0]
    hard, soft = R.clean_labels(keys, labels)
    checks["cleaner_majority"] = hard["a"] == 1 and abs(soft["a"] - 2 / 3) < 1e-9

    # knn-label-transfer: separable → recovers labels
    Xt = np.vstack([rng.normal(-3, 0.5, (100, 2)), rng.normal(3, 0.5, (100, 2))])
    yt = np.array([0] * 100 + [1] * 100)
    Xq = np.vstack([rng.normal(-3, 0.5, (30, 2)), rng.normal(3, 0.5, (30, 2))])
    pred = R.knn_transfer(Xt, yt, Xq, k=5)
    checks["transfer_accuracy"] = (pred == np.array([0] * 30 + [1] * 30)).mean() > 0.95

    # agent contracts
    st, d, to, msg = R.run_shift({"spec": {"X_train": Xtr.tolist(), "X_test": Xte.tolist()}}, "t")
    checks["shift_agent"] = st == "done" and "holdout_idx" in d
    st, d, to, msg = R.run_project({"spec": {"preds": preds.tolist(), "A": A.tolist(), "b": b.tolist()}}, "t")
    checks["project_agent"] = st == "done"

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== robustness-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
