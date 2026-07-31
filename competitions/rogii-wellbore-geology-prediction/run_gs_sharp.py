"""Test a SHARPER PF (gs<1) — motivated by the LB trend (gs1.0->8.77, gs1.3->9.11 => decreasing gs beats 8.77).
Runs PF at a given gs_scale, builds features, blends, reports BOTH field-disjoint CV and well-random CV
(the interspersed LB-proxy). Agent-invoking, committed runner."""
import os, sys, types, importlib.util, time
import numpy as np, pandas as pd
GS = float(sys.argv[1]) if len(sys.argv) > 1 else 0.8
FA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fleet_agents")
if "fleet_agents" not in sys.modules:
    pkg = types.ModuleType("fleet_agents"); pkg.__path__ = [FA]; sys.modules["fleet_agents"] = pkg
def _load(n):
    sp = importlib.util.spec_from_file_location(f"fleet_agents.{n}", os.path.join(FA, f"{n}.py"))
    m = importlib.util.module_from_spec(sp); sys.modules[f"fleet_agents.{n}"] = m; sp.loader.exec_module(m); return m
tb = _load("geology_trackB"); gh = _load("geology_honest")
t0 = time.time()
out = f"results/pf_gs{int(GS*100)}.csv"
tb.trackB_oof("input/train", out, training=True, n_particles=500, n_seeds=128, scale=5.0, gpu=True, gs_scale=GS)
print(f"PF gs{GS} done", round(time.time()-t0), flush=True)
pf = pd.read_csv(out); pf_oof = pf.rename(columns={"trackB_dtvt": "dtvt_pred"}); pf_oof["dtvt_true"] = pf["true_dtvt"]
feats = gh.build_features("input/train", pf_oof[["id","well","dtvt_pred","trackB_conf","dtvt_true"]], training=True)
feats.to_parquet(f"results/honest_feat_gs{int(GS*100)}.parquet")
import xgboost as xgb
META = [c for c in ["pf_dtvt","conf","md_since","z_since","z_rate","rowidx_since","beta","icpt","zsig","n_known","gr","gr_res","a","tvt_ps","twspan","proj_dtvt"] if c in feats.columns]
y = feats.dtvt_true.to_numpy(float); pf_d = feats.pf_dtvt.to_numpy(float); X = feats[META].to_numpy(float)
rmse = lambda p, yy: float(np.sqrt(np.mean((p-yy)**2)))
def cv_blend(fold_col_csv, fold_key):
    fol = pd.read_csv(fold_col_csv)[["well", fold_key]]
    hf = feats.merge(fol, on="well", how="left"); m = hf[fold_key].notna()
    yv = hf.dtvt_true.to_numpy(float); pv = hf.pf_dtvt.to_numpy(float); fold = hf[fold_key].fillna(-1).to_numpy(int)
    Xv = hf[META].to_numpy(float); gbm = np.full_like(yv, np.nan)
    for vf in sorted(set(fold[fold>=0])):
        tr = fold != vf; va = fold == vf
        mdl = xgb.train(dict(device="cuda", tree_method="hist", max_depth=6, eta=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10),
                        xgb.DMatrix(Xv[tr], label=yv[tr]), num_boost_round=400)
        gbm[va] = mdl.predict(xgb.DMatrix(Xv[va]))
    best = min(((rmse(w*pv+(1-w)*gbm, yv), round(float(w),2)) for w in np.linspace(0,1,21)))
    return rmse(pv, yv), best
pf_f, bl_f = cv_blend("config/well_field_folds.csv", "field_fold")
pf_w, bl_w = cv_blend("config/well_random_folds.csv", "well_fold")
print(f"gs{GS}  FIELD-CV: PF {pf_f:.3f} blend {bl_f}")
print(f"gs{GS}  WELL-RANDOM-CV (LB proxy): PF {pf_w:.3f} blend {bl_w}   [blend_AB well-CV ~10.54, LB 8.773]")
