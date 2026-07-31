"""rogii_soft_submit — OUR honest best: multi-seed PF (2xT4) + GBM blend_AB, with the DENSE formation-surface
SOFT-ANCHOR (SPW=60, K=20, tau=4) pulling the confident wells toward the exact TVT=surface(X,Y)-Z+b. Seed count
via env NBASE (2/4/6...). All our own code (no koolbox). Writes submission.csv."""
import os, sys, types, importlib.util, glob, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); FA = os.path.join(HERE, "fleet_agents")
NBASE = int(os.environ.get("NBASE", "4")); TAU = 4.0; SPW = 60; KNN = 20
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
def main(out="submission.csv"):
    import xgboost as xgb
    tb = _load("geology_trackB"); gh = _load("geology_honest"); gs = _load("geology_structural")
    dd = find_data(); train_dir = os.path.join(dd, "train"); test_dir = os.path.join(dd, "test")
    print(f"data_dir {dd} | NBASE={NBASE} seeds | tau={TAU} SPW={SPW}", flush=True)
    # 1. dense formation surface from ALL train wells
    dsurf = gs.build_dense_surface(train_dir, spw=SPW); print("dense surface built", flush=True)
    # 2. multi-seed PF on test (2xT4) + GBM -> blend_AB base
    pf_csv = "/tmp/pf_test.csv"
    tb.trackB_oof(test_dir, pf_csv, training=False, n_particles=500, n_seeds=128, scale=5.0, gpu=True, n_base_seeds=NBASE)
    pf = pd.read_csv(pf_csv); pf_oof = pf.rename(columns={"trackB_dtvt": "dtvt_pred"})[["id","well","dtvt_pred","trackB_conf"]]
    feats = gh.build_features(test_dir, pf_oof, training=False)
    model = xgb.Booster(); model.load_model(os.path.join(HERE, "models", "gbm_blend.json"))
    meta = json.load(open(os.path.join(HERE, "models", "gbm_blend_meta.json")))
    cols = meta["feats"]; pw = meta.get("blend_pf_weight", 0.8)
    gbm = model.predict(xgb.DMatrix(feats[cols].to_numpy(float)))
    feats = feats.assign(blend = pw*feats.pf_dtvt.to_numpy(float) + (1-pw)*gbm)
    feats["rowidx"] = feats.id.str.rsplit("_", n=1).str[1].astype(int)
    base_by = {w: dict(zip(g.rowidx.values, g.blend.values)) for w, g in feats.groupby("well")}
    # 3. soft-anchor the surface onto the blend
    ids=[]; tvts=[]; sw=0; tot=0
    for hp in sorted(glob.glob(os.path.join(test_dir, "*__horizontal_well.csv"))):
        w = os.path.basename(hp).split("__")[0]; hw = pd.read_csv(hp)
        ev = hw.TVT_input.isna().to_numpy(); kn = hw.TVT_input.notna().to_numpy()
        if ev.sum() == 0 or kn.sum() < 5: continue
        tvtps = hw.TVT_input.dropna().iloc[-1]
        try:
            dt, pr, a = gs.soft_anchor_dtvt(hw, dsurf, base_by.get(w, {}), tau=TAU, k=KNN)
        except Exception:
            dt = {int(r): base_by.get(w, {}).get(int(r), 0.0) for r in np.where(ev)[0]}; a = 0.0
        sw += (a > 0.5); tot += 1
        for r in np.where(ev)[0]:
            ids.append(f"{w}_{r}"); tvts.append(float(tvtps + dt.get(int(r), 0.0)))
    sub = pd.DataFrame({"id": ids, "tvt": tvts}); sub.to_csv(out, index=False)
    print(f"wrote {out} {sub.shape}; surface-dominant on {sw}/{tot} wells", flush=True); return sub
if __name__ == "__main__":
    main()
