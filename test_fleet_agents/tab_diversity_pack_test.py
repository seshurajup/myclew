"""tab_diversity_pack_test — verifier for the recurring pure-tabular diversity agents (offline, synthetic)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import tab_diversity_pack as D


def _run():
    print("=== TAB DIVERSITY PACK VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}

    # synth-artifact-fe: snap-to-original recovers exact matches + digit features
    orig = np.array([1.0, 2.5, 3.0, 4.75])
    x = np.array([2.5, 2.6, 3.0, 10.0])
    X, names = D.synth_artifact_features(x, original=orig)
    checks["synth_has_snap"] = "snap_diff" in names and "n_decimals" in names
    si = names.index("snap_is_exact")
    checks["synth_snap_exact"] = X[0, si] == 1.0 and X[3, si] == 0.0  # 2.5 in orig, 10.0 not
    checks["synth_finite"] = np.all(np.isfinite(X))

    # oof-diversity-prune: 2 near-identical + 1 decorrelated → keep 2
    base = rng.normal(0, 1, 500)
    oof = {"a": base + rng.normal(0, 1e-4, 500), "b": base + rng.normal(0, 1e-4, 500), "c": rng.normal(0, 1, 500)}
    kept, C = D.diversity_prune(oof, corr_threshold=0.999)
    checks["prune_drops_twin"] = len(kept) == 2 and "c" in kept

    # feature-select: 5 informative + 15 noise → top-K mostly informative
    n = 800; Xi = rng.normal(0, 1, (n, 20))
    y = Xi[:, :5].sum(1) + rng.normal(0, 0.3, n)  # only first 5 matter
    idx, imp = D.consensus_select(Xi, y, top_k=5, task="regression")
    checks["fselect_finds_informative"] = len(set(idx) & set(range(5))) >= 4

    # residual-boost: linear base underfits a nonlinear target; booster recovers it
    Xr = rng.normal(0, 1, (600, 3))
    yr = Xr[:, 0] ** 2 + np.sin(3 * Xr[:, 1]) + 0.1 * rng.normal(0, 1, 600)  # nonlinear
    from sklearn.linear_model import Ridge
    base_only = Ridge(1.0).fit(Xr, yr).predict(Xr)
    oof_rb, _ = D.residual_boost(Ridge(1.0), __import__("sklearn.ensemble", fromlist=["HistGradientBoostingRegressor"]).HistGradientBoostingRegressor(max_iter=300), Xr, yr)
    mse_base = np.mean((base_only - yr) ** 2); mse_rb = np.mean((oof_rb - yr) ** 2)
    checks["residual_boost_improves"] = mse_rb < mse_base * 0.5
    print(f"  -> residual-boost MSE {mse_rb:.4f} < base {mse_base:.4f}")

    # knn-feature: OOF kNN target-mean correlates with target
    from sklearn.model_selection import KFold
    folds = list(KFold(5, shuffle=True, random_state=0).split(Xr))
    kf, _ = D.knn_target_feature(Xr, yr, folds, k=15)
    checks["knn_feature_signal"] = abs(np.corrcoef(kf[:, 0], yr)[0, 1]) > 0.3

    # full-retrain-calibrator: iteration formula
    checks["retrain_iters"] = D.retrain_iterations(100, 5) == 125   # 100*(1+1/4)=125
    # seed-average reduces variance
    sa = D.seed_average(lambda s: np.array([s, s + 1.0]), [1, 2, 3])
    checks["seed_average"] = np.allclose(sa, [2.0, 3.0])

    # agent contracts
    st, d, to, msg = D.run_prune({"spec": {"oof": {k: v.tolist() for k, v in oof.items()}}}, "t")
    checks["prune_agent"] = st == "done" and len(d["kept"]) == 2

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== tab-diversity-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
