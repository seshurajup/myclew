"""tab_pack_test — data-wise verifier for the TABULAR pack on a synthetic mini-comp (self-contained, offline).

Builds a real toy tabular competition (make_classification → train.csv / test.csv / sample_submission.csv),
then runs the WHOLE turnkey spine and asserts:
  • comp-onboard fingerprints the fixture from its manifest → tabular/roc_auc/pack=tab,
  • tab-profile reports shape + target balance + no false leakage,
  • tab-train CV-trains the installed backends with a real (>0.8) AUC on separable data,
  • tab-stack blend ≥ best single backend,
  • tab-autobaseline writes a submission.csv with the EXACT sample-submission schema and the right row count.
This is the fixture gate from critique C2.4: the pack is "done" only when profile→…→submission runs GREEN.
"""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
import pandas as pd
from fleet_agents import comp_config as CC
from fleet_agents import comp_onboard as ON
from fleet_agents import tab_profile as TP
from fleet_agents import tab_train as TT
from fleet_agents import tab_stack as TS
from fleet_agents import tab_autobaseline as TA

FIX = os.path.join(COMP, "test_fleet_agents", "fixtures", "tab_toy")


def _build_fixture():
    from sklearn.datasets import make_classification
    os.makedirs(FIX, exist_ok=True)
    X, y = make_classification(n_samples=1200, n_features=12, n_informative=6, n_redundant=2,
                               n_classes=2, class_sep=1.2, random_state=7)
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["id"] = np.arange(len(df)); df["target"] = y
    tr = df.iloc[:900].copy(); te = df.iloc[900:].copy()
    tr.to_csv(os.path.join(FIX, "train.csv"), index=False)
    te.drop(columns=["target"]).to_csv(os.path.join(FIX, "test.csv"), index=False)
    te[["id"]].assign(target=0).to_csv(os.path.join(FIX, "sample_submission.csv"), index=False)


def _build_cat_fixture():
    """A comp where signal lives in a HIGH-CARDINALITY categorical → target-encoding should unlock it,
    naive label-encoding (arbitrary integer ids) should not. Proves the GM tab-fe lever."""
    rng = np.random.RandomState(11)
    n = 2000; n_cat = 60
    cat_effect = rng.RandomState if False else rng.uniform(-3, 3, n_cat)  # per-category log-odds
    cat = rng.randint(0, n_cat, n)
    noise = rng.normal(0, 1.0, n)
    logit = cat_effect[cat] + noise
    y = (logit > 0).astype(int)
    df = pd.DataFrame({"id": np.arange(n), "cat_id": [f"C{c}" for c in cat],
                       "noise1": rng.normal(0, 1, n), "noise2": rng.normal(0, 1, n)})
    df["target"] = y
    d = os.path.join(COMP, "test_fleet_agents", "fixtures", "tab_cat"); os.makedirs(d, exist_ok=True)
    tr = df.iloc[:1500]; te = df.iloc[1500:]
    tr.to_csv(os.path.join(d, "train.csv"), index=False)
    te.drop(columns=["target"]).to_csv(os.path.join(d, "test.csv"), index=False)
    te[["id"]].assign(target=0).to_csv(os.path.join(d, "sample_submission.csv"), index=False)
    return CC.CompConfig(slug="tab-cat", modality="tabular", task="classification", metric="roc_auc",
                         metric_direction="max", cv_scheme="stratified", id_col="id", target_cols=["target"],
                         n_folds=4, data={"train": os.path.join(d, "train.csv"),
                                          "test": os.path.join(d, "test.csv"),
                                          "sample_sub": os.path.join(d, "sample_submission.csv")})


def _cfg():
    return CC.CompConfig(
        slug="tab-toy", modality="tabular", paradigm="predictive", task="classification",
        metric="roc_auc", metric_direction="max", cv_scheme="stratified", id_col="id",
        target_cols=["target"], n_folds=4,
        data={"train": os.path.join(FIX, "train.csv"), "test": os.path.join(FIX, "test.csv"),
              "sample_sub": os.path.join(FIX, "sample_submission.csv")})


def _run():
    print("=== TABULAR PACK DATA-WISE VERIFIER (synthetic fixture) ===")
    _build_fixture()
    cfg = _cfg()
    checks = {}

    # onboard the fixture from its manifest alone (cold-start proof)
    onb = ON.infer_config("tab-toy", files=["train.csv", "test.csv", "sample_submission.csv"],
                          eval_text="area under the ROC curve", sample_header=["id", "target"])
    checks["onboard_modality"] = onb.modality == "tabular"
    checks["onboard_pack"] = onb.pack() == "tab"

    # profile
    rep = TP.profile(cfg)
    checks["profile_shape"] = rep["n_train"] == 900 and rep["n_features"] == 12
    checks["profile_target_balance"] = "target_balance" in rep
    checks["profile_no_false_leak"] = rep.get("leakage_suspects") == []

    # train
    preds, meta = TT.train_backends(cfg, seed=7)
    checks["train_backends_ran"] = len(preds) >= 1
    checks["train_auc_real"] = all(preds[b]["cv"] > 0.8 for b in preds)
    checks["train_test_preds"] = all(preds[b]["test"] is not None for b in preds)

    # stack ≥ best single
    w, oof, test, blend_cv = TS.optimize_blend(cfg, preds, meta["y"])
    best_single = max(preds[b]["cv"] for b in preds)
    checks["stack_ge_best_single"] = blend_cv >= best_single - 1e-6

    # autobaseline writes a valid submission
    out = tempfile.mktemp(suffix=".csv")
    res = TA.run_pipeline(cfg, out_path=out, seed=7)
    checks["submission_written"] = os.path.exists(res.get("submission", ""))
    if checks["submission_written"]:
        sub = pd.read_csv(res["submission"])
        samp = pd.read_csv(cfg.data["sample_sub"])
        checks["submission_schema"] = list(sub.columns) == list(samp.columns)
        checks["submission_rows"] = len(sub) == len(samp)
    checks["autobaseline_cv"] = res["blend_cv"] > 0.8

    # GM tab-fe uplift: on a comp with a predictive HIGH-CARDINALITY categorical, target-encoding (fe=True)
    # must beat naive label-encoding (fe=False). This is the grandmaster-quality proof.
    cfg2 = _build_cat_fixture()
    base_preds, _ = TT.train_backends(cfg2, backends=["histgbm"], seed=7, fe=False)
    fe_preds, _ = TT.train_backends(cfg2, backends=["histgbm"], seed=7, fe=True)
    base_auc = base_preds["histgbm"]["cv"]; fe_auc = fe_preds["histgbm"]["cv"]
    checks["tab_fe_lifts_cv"] = fe_auc > base_auc + 0.01
    print(f"  -> tab-fe uplift: label-enc AUC={base_auc:.4f} → target-enc AUC={fe_auc:.4f} (+{fe_auc-base_auc:.4f})")

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    print(f"  -> backends={list(preds)} single_CV={ {b: round(preds[b]['cv'],4) for b in preds} } blend={blend_cv:.4f}")
    ok = all(checks.values())
    print(f"=== tab-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
