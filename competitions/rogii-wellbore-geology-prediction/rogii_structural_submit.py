"""rogii_structural_submit — THE CRACK submission. TVT = formation-top structural surface(X,Y) - Z + b (from
train wells, honest for interspersed test), used where the surface reproduces the visible heel (prefix-gate);
else fall back to blend_AB (PF+GBM). Safe: >= blend_AB by construction. Reuses geology_structural + geology_trackB
+ geology_honest agents. Writes submission.csv."""
import os, sys, types, importlib.util, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); FA = os.path.join(HERE, "fleet_agents")
def _load(n):
    if "fleet_agents" not in sys.modules:
        pkg = types.ModuleType("fleet_agents"); pkg.__path__ = [FA]; sys.modules["fleet_agents"] = pkg
    sp = importlib.util.spec_from_file_location(f"fleet_agents.{n}", os.path.join(FA, f"{n}.py"))
    m = importlib.util.module_from_spec(sp); sys.modules[f"fleet_agents.{n}"] = m; sp.loader.exec_module(m); return m
def find_data():
    for c in ["input", "/kaggle/input/rogii-wellbore-geology-prediction"]:
        if os.path.isdir(os.path.join(c, "test")): return c
    for root, dirs, files in os.walk("/kaggle/input"):
        if any(f.endswith("__horizontal_well.csv") for f in files) and os.path.basename(root) == "test":
            return os.path.dirname(root)
    return "input"
def main(gate=3.0, out="submission.csv"):
    import xgboost as xgb
    gs = _load("geology_structural"); tb = _load("geology_trackB"); gh = _load("geology_honest")
    dd = find_data(); train_dir = os.path.join(dd, "train"); test_dir = os.path.join(dd, "test")
    print("data_dir", dd, flush=True)
    # 1. structural surface from ALL train wells (densest possible)
    surf = gs.build_surface_from(gs._load(train_dir)); print("surface built", flush=True)
    # 2. blend_AB fallback: PF on test + GBM (pretrained bundle)
    pf_csv = "/tmp/pf_test.csv"; tb.trackB_oof(test_dir, pf_csv, training=False, n_particles=500, n_seeds=128, scale=5.0)
    pf = pd.read_csv(pf_csv); pf_oof = pf.rename(columns={"trackB_dtvt": "dtvt_pred"})[["id","well","dtvt_pred","trackB_conf"]]
    feats = gh.build_features(test_dir, pf_oof, training=False)
    model = xgb.Booster(); model.load_model(os.path.join(HERE, "models", "gbm_blend.json"))
    import json; meta = json.load(open(os.path.join(HERE, "models", "gbm_blend_meta.json")))
    cols = meta["feats"]; pw = meta.get("blend_pf_weight", 0.8)
    gbm = model.predict(xgb.DMatrix(feats[cols].to_numpy(float)))
    feats = feats.assign(blend_dtvt = pw*feats.pf_dtvt.to_numpy(float) + (1-pw)*gbm)
    feats["rowidx"] = feats.id.str.rsplit("_", n=1).str[1].astype(int)
    blend_by = {w: dict(zip(g.rowidx.values, g.blend_dtvt.values)) for w, g in feats.groupby("well")}
    # 3. per test well: surface if prefix-confident, else blend
    ids=[]; tvts=[]; used=0; tot=0
    for hp in sorted(glob.glob(os.path.join(test_dir, "*__horizontal_well.csv"))):
        w = os.path.basename(hp).split("__")[0]; hw = pd.read_csv(hp)
        ev = hw.TVT_input.isna().to_numpy(); kn = hw.TVT_input.notna().to_numpy()
        if ev.sum() == 0 or kn.sum() < 5: continue
        tvtps = hw.TVT_input.dropna().iloc[-1]; ei = np.where(ev)[0]
        dt, pr, us = gs.predict_well(hw, surf, blend_by.get(w), gate); used += us; tot += 1
        for i, r in enumerate(ei):
            ids.append(f"{w}_{r}"); tvts.append(float(tvtps + dt[i]))
    sub = pd.DataFrame({"id": ids, "tvt": tvts}); sub.to_csv(out, index=False)
    print(f"wrote {out} {sub.shape}; surface used on {used}/{tot} wells", flush=True); return sub
if __name__ == "__main__":
    main()
