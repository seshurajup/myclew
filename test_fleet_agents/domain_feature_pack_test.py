"""domain_feature_pack_test — verifier for fin-ta / imu-fe / online-walk-forward + math-master soft_spearman
+ purged-embargo CV. Real behavioral assertions on synthetic data (offline)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import domain_feature_pack as D
from fleet_agents import math_master as MM
from fleet_agents import tab_common as TC
from fleet_agents import comp_config as CC


def _run():
    print("=== DOMAIN FEATURE PACK VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}

    # fin-ta: a trending+vol price series → features finite, includes rsi/vol/hurst
    price = 100 + np.cumsum(rng.normal(0.05, 1.0, 400))
    X, names = D.fin_ta_features(price)
    checks["finta_shape"] = X.shape[0] == 400 and X.shape[1] == len(names)
    checks["finta_has_rsi_vol_hurst"] = "rsi14" in names and any("vol" in n for n in names) and "hurst" in names
    checks["finta_finite"] = np.all(np.isfinite(X))

    # imu: accel stream → gravity-removed magnitude differs from raw magnitude
    accel = rng.normal(0, 1, (300, 3)); accel[:, 2] += 9.8   # gravity on z
    Xi, ni = D.imu_features(accel)
    checks["imu_features"] = Xi.shape == (300, 6) and "lin_mag" in ni
    checks["imu_gravity_removed"] = Xi[:, ni.index("lin_mag")].mean() < Xi[:, ni.index("mag")].mean()

    # online walk-forward beats a STATIC model on drifting data
    n = 600; t = np.arange(n); Xd = rng.normal(0, 1, (n, 3))
    drift_w = np.stack([np.sin(t / 100), np.cos(t / 100), t / n], 1)   # coefficients drift over time
    y = np.sum(Xd * drift_w, axis=1) + rng.normal(0, 0.1, n)
    from sklearn.linear_model import Ridge
    oof = D.walk_forward(Xd, y, lambda: Ridge(1.0), retrain_every=40, warmup=100)
    m = ~np.isnan(oof)
    static = Ridge(1.0).fit(Xd[:100], y[:100]); pred_static = static.predict(Xd[m])
    mse_online = np.mean((oof[m] - y[m]) ** 2); mse_static = np.mean((pred_static - y[m]) ** 2)
    checks["online_beats_static"] = mse_online < mse_static
    print(f"  -> online MSE {mse_online:.4f} < static MSE {mse_static:.4f}")

    # math-master soft_spearman + spearman_sharpe
    checks["soft_spearman"] = abs(MM.soft_spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    days_t = [[1, 2, 3], [3, 2, 1], [1, 2, 3]]; days_p = [[1, 2, 3], [3, 2, 1], [1, 2, 3]]
    checks["spearman_sharpe"] = MM.spearman_sharpe(days_t, days_p) > 0

    # purged-embargo CV: train excludes val block + embargo, folds disjoint on val
    y2 = np.zeros(200)
    cfg = CC.CompConfig(slug="ts", cv_scheme="purged-embargo", n_folds=5)
    folds = TC.make_cv(cfg, y2)
    checks["purged_folds"] = len(folds) == 5
    tr0, va0 = folds[1]
    checks["purged_embargo_gap"] = len(set(tr0) & set(va0)) == 0 and (min(va0) - 1 not in set(tr0) or True)
    # embargo actually removes neighbors: the index right before val block should NOT be in train
    just_before = min(va0) - 1
    checks["embargo_removes_neighbor"] = just_before not in set(tr0)

    # agent contracts
    st, d, to, msg = D.run_fin({"spec": {"prices": price.tolist()}}, "t")
    checks["fin_agent"] = st == "done" and d["n_features"] == len(names)

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== domain-feature-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
