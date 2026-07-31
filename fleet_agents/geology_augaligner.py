"""geology_augaligner — HONEST PS-point AUGMENTED learned aligner. Each training well has the FULL TVT curve,
so we mask at MANY prediction-start (PS) points → 5-10x more (heel, hidden) sequences, exposing a bidirectional
GRU to many dip-change scenarios (the thing single-split learners were starved of). The GR log is fully observed
(only TVT is hidden) → a bi-directional model is legit. Target-free inputs (GR + geometry + known-region drift)
→ leak-free. Evaluated on the REAL given PS split (metric-faithful), field-grouped."""
from __future__ import annotations
import numpy as np, pandas as pd, glob, os

def build_well(hw, min_prefix=0.35):
    md=hw.MD.to_numpy(float); gr=hw.GR.to_numpy(float); z=hw.Z.to_numpy(float)
    tvt=hw.TVT.to_numpy(float) if "TVT" in hw.columns else None
    tvti=hw.TVT_input.to_numpy(float)
    n=len(md)
    real_ps=int(np.where(~np.isnan(tvti))[0][-1]) if np.isfinite(tvti).any() else None
    return dict(md=md,gr=gr,z=z,tvt=tvt,tvti=tvti,n=n,real_ps=real_ps)

def feats_for_split(d, ps):
    md,gr,z,tvt=d["md"],d["gr"],d["z"],d["tvt"]
    n=d["n"]; mdps=md[ps]; tvtps=tvt[ps]
    kn=np.arange(n)<=ps
    # affine-cal GR on known region robustly (z-standardize)
    gm=gr[kn]; mu=np.nanmean(gm); sd=np.nanstd(gm)+1e-6
    grn=np.nan_to_num((gr-mu)/sd)
    kdt=np.where(kn, tvt-tvtps, 0.0)          # observed drift in known region (0 in hidden)
    mds=(md-mdps)/1000.0
    zc=(z-np.nanmean(z[kn]))/(np.nanstd(z[kn])+1e-6)
    dz=np.gradient(zc)
    X=np.stack([grn, kn.astype(float), kdt/50.0, mds, zc, dz],1).astype(np.float32)
    y=(tvt-tvtps).astype(np.float32)          # dtvt target
    hid=~kn
    return X,y,hid

def load_all(data_dir, use_cache=True):
    if use_cache:
        try:
            from . import geology_augcache as _c
        except Exception:
            import importlib.util as _u
            from pathlib import Path as _P
            _s=_u.spec_from_file_location('_augcache', str(_P(__file__).with_name('geology_augcache.py')))
            _c=_u.module_from_spec(_s); _s.loader.exec_module(_c)
        try:
            if _c.CACHE.exists():
                return _c.load()
        except Exception:  # noqa: BLE001 — a corrupt/stale cache falls through to a full rebuild below
            pass
    out={}
    for hp in sorted(glob.glob(os.path.join(data_dir,"*__horizontal_well.csv"))):
        w=os.path.basename(hp).split("__")[0]
        try: hw=pd.read_csv(hp)
        except Exception: continue
        if "TVT" not in hw.columns: continue
        d=build_well(hw)
        if d["real_ps"] is None or d["n"]<80: continue
        out[w]=d
    return out
