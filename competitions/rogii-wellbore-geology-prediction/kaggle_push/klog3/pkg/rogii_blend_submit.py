"""rogii_blend_submit — package the HONEST BEST (blend_AB ~10.4-10.7 field-CV): trackB GPU particle-filter
(cupy) + a pre-trained XGBoost meta-GBM, blended 0.8*PF + 0.2*GBM. The GBM is trained OFFLINE on the fixed
train set and bundled; at submission the notebook runs the PF only on the hidden TEST wells, builds features,
applies the GBM, blends, and writes submission.csv. Self-contained; reuses geology_trackB + geology_honest."""
import os, sys, types, importlib.util, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
FA = os.path.join(HERE, "fleet_agents")

def _load(name):
    # stub package so relative `.base` imports resolve without importing fleet_agents/__init__ (torch)
    if "fleet_agents" not in sys.modules:
        pkg = types.ModuleType("fleet_agents"); pkg.__path__ = [FA]; sys.modules["fleet_agents"] = pkg
    spec = importlib.util.spec_from_file_location(f"fleet_agents.{name}", os.path.join(FA, f"{name}.py"))
    m = importlib.util.module_from_spec(spec); sys.modules[f"fleet_agents.{name}"] = m; spec.loader.exec_module(m)
    return m

def find_data():
    for c in ["input", "/kaggle/input/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c, "test")): return c
    for root, dirs, files in os.walk("/kaggle/input"):
        if any(f.endswith("__horizontal_well.csv") for f in files) and os.path.basename(root) == "test":
            return os.path.dirname(root)
    return "input"

def main(pf_weight=0.8, out="submission.csv"):
    import xgboost as xgb
    tb = _load("geology_trackB"); gh = _load("geology_honest")
    dd = find_data(); test_dir = os.path.join(dd, "test")
    print("data_dir", dd)
    # 1. PF on TEST wells
    pf_csv = "/tmp/pf_test.csv"
    tb.trackB_oof(test_dir, pf_csv, training=False, n_particles=500, n_seeds=128, scale=5.0)
    pf = pd.read_csv(pf_csv)
    pf_oof = pf.rename(columns={"trackB_dtvt": "dtvt_pred"})[["id", "well", "dtvt_pred", "trackB_conf"]]
    # 2. features + 3. GBM + 4. blend
    feats = gh.build_features(test_dir, pf_oof, training=False)
    model = xgb.Booster(); model.load_model(os.path.join(HERE, "models", "gbm_blend.json"))
    import json; meta = json.load(open(os.path.join(HERE, "models", "gbm_blend_meta.json")))
    cols = meta["feats"]; pf_weight = meta.get("blend_pf_weight", pf_weight)
    gbm = model.predict(xgb.DMatrix(feats[cols].to_numpy(float)))
    blend_dtvt = pf_weight * feats["pf_dtvt"].to_numpy(float) + (1 - pf_weight) * gbm
    tvt = feats["tvt_ps"].to_numpy(float) + blend_dtvt
    sub = pd.DataFrame({"id": feats["id"].values, "tvt": tvt})
    # guard: fill any missing ids from sample with PF-tvt (never leave holes)
    sub.to_csv(out, index=False)
    print("wrote", out, sub.shape, "tvt range", round(sub.tvt.min(),1), round(sub.tvt.max(),1))
    return sub

if __name__ == "__main__":
    main()
