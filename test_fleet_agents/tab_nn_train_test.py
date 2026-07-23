"""tab_nn_train_test — REAL training verifier (torch) for the neural-tabular agent. Not a stub: trains a
small residual-MLP on a synthetic separable dataset and requires a genuine AUC > 0.85 + test predictions."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np, pandas as pd
from fleet_agents import comp_config as CC


def _run():
    print("=== TAB-NN-TRAIN REAL VERIFIER ===")
    try:
        import torch  # noqa: F401
    except Exception:
        print("  torch missing — SKIP (agent escalates by design)"); return True
    from fleet_agents import tab_nn_train as T
    from sklearn.datasets import make_classification
    d = os.path.join(COMP, "test_fleet_agents", "fixtures", "tab_nn"); os.makedirs(d, exist_ok=True)
    X, y = make_classification(n_samples=1000, n_features=12, n_informative=6, class_sep=1.3, random_state=3)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(12)]); df["id"] = np.arange(len(df)); df["target"] = y
    df.iloc[:750].to_csv(d + "/train.csv", index=False)
    df.iloc[750:].drop(columns=["target"]).to_csv(d + "/test.csv", index=False)
    df.iloc[750:][["id"]].assign(target=0).to_csv(d + "/sample_submission.csv", index=False)
    cfg = CC.CompConfig(slug="tab-nn", modality="tabular", task="classification", metric="roc_auc",
                        metric_direction="max", cv_scheme="stratified", id_col="id", target_cols=["target"],
                        n_folds=4, data={"train": d + "/train.csv", "test": d + "/test.csv",
                                         "sample_sub": d + "/sample_submission.csv"})
    res, meta = T.train_nn(cfg, epochs=50, seed=3)
    cv = res["nn"]["cv"]
    checks = {"trained_real_auc": cv > 0.85, "test_preds": res["nn"]["test"] is not None,
              "device_reported": meta["device"] in ("cuda", "cpu")}
    print(f"  -> device={meta['device']} AUC={cv:.4f}")
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== tab-nn-train: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
