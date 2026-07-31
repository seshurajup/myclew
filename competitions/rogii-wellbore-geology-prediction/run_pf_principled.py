"""Comp-level runner: principled per-well noise PF (system-identification, NO sweep) → blend → field-CV.
Invokes the shared agents (geology_trackB.trackB_oof with noise_id=True → pf_noise_id.identify per well,
geology_honest.build_features). Agent-based execution; committed to the competition folder."""
import os, sys, types, importlib.util, time
import numpy as np, pandas as pd

FA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet_agents")
if "fleet_agents" not in sys.modules:
    pkg = types.ModuleType("fleet_agents"); pkg.__path__ = [FA]; sys.modules["fleet_agents"] = pkg
def _load(n):
    sp = importlib.util.spec_from_file_location(f"fleet_agents.{n}", os.path.join(FA, f"{n}.py"))
    m = importlib.util.module_from_spec(sp); sys.modules[f"fleet_agents.{n}"] = m; sp.loader.exec_module(m); return m

tb = _load("geology_trackB"); _load("pf_noise_id"); gh = _load("geology_honest")
t0 = time.time()
out = "results/pf_principled.csv"
tb.trackB_oof("input/train", out, training=True, n_particles=500, n_seeds=128, scale=5.0, gpu=True, noise_id=True)
print("principled PF done", round(time.time()-t0), flush=True)
pf = pd.read_csv(out); pf_oof = pf.rename(columns={"trackB_dtvt": "dtvt_pred"}); pf_oof["dtvt_true"] = pf["true_dtvt"]
feats = gh.build_features("input/train", pf_oof[["id","well","dtvt_pred","trackB_conf","dtvt_true"]], training=True)
feats.to_parquet("results/honest_feat_principled.parquet")
folds = pd.read_csv("config/well_field_folds.csv")[["well","field_fold"]]
hf = feats.merge(folds, on="well", how="left"); hf = hf[hf.field_fold.notna()].copy()
import xgboost as xgb
y = hf.dtvt_true.to_numpy(float); pf_d = hf.pf_dtvt.to_numpy(float); fold = hf.field_fold.to_numpy(int)
META = [c for c in ["pf_dtvt","conf","md_since","z_since","z_rate","rowidx_since","beta","icpt","zsig","n_known","gr","gr_res","a","tvt_ps","twspan","proj_dtvt"] if c in hf.columns]
X = hf[META].to_numpy(float); gbm = np.full_like(y, np.nan)
rmse = lambda p: float(np.sqrt(np.mean((p-y)**2)))
for vf in range(5):
    tr = fold != vf; va = fold == vf
    m = xgb.train(dict(device="cuda", tree_method="hist", max_depth=6, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10),
                  xgb.DMatrix(X[tr], label=y[tr]), num_boost_round=400)
    gbm[va] = m.predict(xgb.DMatrix(X[va]))
print("principled PF field-CV:", rmse(pf_d))
best = None
for w in np.linspace(0, 1, 21):
    r = rmse(w*pf_d + (1-w)*gbm)
    if best is None or r < best[0]: best = (r, round(float(w),2))
print("principled blend field-CV:", best, " (gs130 blend 10.56, blend_AB 10.75)")
