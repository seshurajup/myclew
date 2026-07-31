"""geology_structural — THE CRACK. TVT is EXACTLY the formation-top TVD minus Z plus a per-well constant
(TVT_i = ANCC_i - Z_i + b_well, std 0.006 ft — parallel formations). The formation top is a smooth STRUCTURAL
SURFACE over (X,Y). So: reconstruct that surface from train wells' known (X,Y,TVT+Z), interpolate to a test
well's (X,Y), then TVT = surface(X,Y) - Z, with b fixed on the visible prefix. Honest for INTERSPERSED test
(the public/private LB pool). For ISOLATED wells (surface unreliable) fall back to the PF, gated by how well
the surface reproduces the KNOWN heel (leak-free prefix validation). Fast: global low-order (X,Y) polynomial
(regional dip) + local kNN residual. Reusable via run_oof / predict_dir."""
from __future__ import annotations
import os, glob, numpy as np, pandas as pd
from scipy.spatial import cKDTree

def _poly_features(X, Y, deg=3):
    f = [np.ones_like(X)]
    for i in range(1, deg + 1):
        for j in range(i + 1):
            f.append((X ** (i - j)) * (Y ** j))
    return np.stack(f, 1)

class Surface:
    """Global degree-3 (X,Y) polynomial for the datum s=TVT+Z (regional structure) + local kNN residual."""
    def __init__(self, X, Y, S, kres=8, sub=400000):
        if len(X) > sub:
            idx = np.random.default_rng(0).choice(len(X), sub, replace=False); X, Y, S = X[idx], Y[idx], S[idx]
        self.mx, self.my = X.mean(), Y.mean(); self.sx, self.sy = X.std() + 1e-6, Y.std() + 1e-6
        Xn, Yn = (X - self.mx) / self.sx, (Y - self.my) / self.sy
        F = _poly_features(Xn, Yn); self.coef, *_ = np.linalg.lstsq(F, S, rcond=None)
        resid = S - F @ self.coef
        self.tree = cKDTree(np.c_[Xn, Yn]); self.res = resid; self.kres = kres
    def predict(self, X, Y):
        Xn, Yn = (X - self.mx) / self.sx, (Y - self.my) / self.sy
        base = _poly_features(Xn, Yn) @ self.coef
        dist, ii = self.tree.query(np.c_[Xn, Yn], k=self.kres)
        w = 1.0 / (dist + 1e-6); corr = (w * self.res[ii]).sum(1) / w.sum(1)
        return base + corr, dist[:, 0]

def _load(data_dir, need_true=True):
    W = {}
    for hp in sorted(glob.glob(os.path.join(data_dir, "*__horizontal_well.csv"))):
        w = os.path.basename(hp).split("__")[0]; hw = pd.read_csv(hp)
        if need_true and "TVT" not in hw.columns: continue
        W[w] = hw
    return W

def build_surface_from(wells: dict):
    """Datum surface s=TVT+Z from wells that HAVE full TVT (train)."""
    Xs, Ys, Ss = [], [], []
    for w, hw in wells.items():
        if "TVT" not in hw.columns: continue
        Xs.append(hw.X.to_numpy(float)); Ys.append(hw.Y.to_numpy(float)); Ss.append((hw.TVT + hw.Z).to_numpy(float))
    return Surface(np.concatenate(Xs), np.concatenate(Ys), np.concatenate(Ss))

def predict_well(hw, surf: Surface, pf_dtvt=None, gate=3.0):
    """Return (dtvt over hidden rows, prefix_rmse, used_surface). Falls back to pf_dtvt (dict rowidx->dtvt) if
    the surface fails the prefix check. TVT = surface(X,Y) - Z + b, b from the visible prefix."""
    X = hw.X.to_numpy(float); Y = hw.Y.to_numpy(float); Z = hw.Z.to_numpy(float)
    kn = hw.TVT_input.notna().to_numpy(); ev = hw.TVT_input.isna().to_numpy()
    s_pred, d0 = surf.predict(X, Y)                         # datum s=TVT+Z at (X,Y)
    tin = hw.TVT_input.to_numpy(float)
    b = np.median(tin[kn] - (s_pred[kn] - Z[kn]))
    tvt = s_pred - Z + b                                    # absolute TVT
    prefix_rmse = float(np.sqrt(np.mean((tvt[kn] - tin[kn]) ** 2)))
    tvtps = tin[kn][-1]; ei = np.where(ev)[0]
    surf_dt = tvt[ei] - tvtps
    if prefix_rmse < gate or pf_dtvt is None:
        return surf_dt, prefix_rmse, True
    pf = np.array([pf_dtvt.get(r, np.nan) for r in ei])
    return np.where(np.isnan(pf), surf_dt, pf), prefix_rmse, False

def run_oof(data_dir, folds_csv, fold_key, pf_parquet="results/honest_feat.parquet", gate=3.0):
    W = _load(data_dir); fol = pd.read_csv(folds_csv).set_index("well")[fold_key].to_dict()
    pfmap = {}
    try:
        hf = pd.read_parquet(pf_parquet); hf["rowidx"] = hf.id.str.rsplit("_", n=1).str[1].astype(int)
        pfmap = {w: dict(zip(g.rowidx.values, g.pf_dtvt.values)) for w, g in hf.groupby("well")}
    except Exception:  # noqa: BLE001 — no PF parquet = run without pf_dtvt; visible via the used/tot counters
        pass
    P, T = [], []; used = 0; tot = 0
    for vf in sorted(set(v for v in fol.values() if pd.notna(v))):
        train = {w: hw for w, hw in W.items() if fol.get(w) is not None and fol.get(w) != vf}
        surf = build_surface_from(train)
        for w, hw in W.items():
            if fol.get(w) != vf: continue
            ev = hw.TVT_input.isna().to_numpy() & hw.TVT.notna().to_numpy(); kn = hw.TVT_input.notna().to_numpy()
            if ev.sum() < 10 or kn.sum() < 10: continue
            dt, pr, us = predict_well(hw, surf, pfmap.get(w), gate); used += us; tot += 1
            tvtps = hw.TVT_input.dropna().iloc[-1]
            P.append(dt); T.append(hw.TVT.to_numpy(float)[ev] - tvtps)
    P = np.concatenate(P); T = np.concatenate(T); m = np.isfinite(P) & np.isfinite(T)
    return float(np.sqrt(np.mean((P[m] - T[m]) ** 2))), used, tot


# ============================ dense soft-anchor (the honest best: SPW=60, K=20, tau=4) ============================
FORMATIONS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]

class DenseSurface:
    """Formation-top surface from DENSE per-well samples (SPW centers/well) — captures along-well structural
    variation (median-per-well loses it; too many centers over-weight one well = worse, SPW=60 is optimal).
    Distance-weighted kNN over the sampled (X,Y)->ANCC cloud."""
    def __init__(self, wells: dict, spw=60):
        Xs, As = [], []
        for w, hw in wells.items():
            if "ANCC" not in hw.columns: continue
            df = hw[["X", "Y", "ANCC"]].dropna()
            if len(df) == 0: continue
            n = len(df); idx = np.linspace(0, n - 1, min(spw, n)).astype(int)
            Xs.append(df[["X", "Y"]].to_numpy()[idx]); As.append(df["ANCC"].to_numpy()[idx])
        self.X = np.vstack(Xs); self.A = np.concatenate(As); self.tree = cKDTree(self.X)
    def anc(self, X, Y, k=20):
        dist, ii = self.tree.query(np.c_[X, Y], k=k, workers=-1)
        w = 1.0 / (dist + 1e-3); return (w * self.A[ii]).sum(1) / w.sum(1)

def build_dense_surface(train_dir, spw=60):
    return DenseSurface(_load(train_dir), spw=spw)

def soft_anchor_dtvt(hw, dsurf: DenseSurface, base_dtvt: dict, tau=4.0, k=20):
    """Soft-blend the dense formation surface with a base predictor (blend_AB/PF). weight a=exp(-prefix_rmse/tau):
    where the surface reproduces the known heel, trust it more; else fall to base. Returns {rowidx: dtvt}."""
    X = hw.X.to_numpy(float); Y = hw.Y.to_numpy(float); Z = hw.Z.to_numpy(float)
    kn = hw.TVT_input.notna().to_numpy(); ev = hw.TVT_input.isna().to_numpy()
    tin = hw.TVT_input.to_numpy(float)
    anc = dsurf.anc(X, Y, k=k)
    b = np.median(tin[kn] - (anc[kn] - Z[kn])); tvt = anc - Z + b
    prefix_rmse = float(np.sqrt(np.mean((tvt[kn] - tin[kn]) ** 2)))
    a = float(np.exp(-prefix_rmse / tau)); tvtps = tin[kn][-1]
    out = {}
    for r in np.where(ev)[0]:
        surf_dt = tvt[r] - tvtps; bs = base_dtvt.get(int(r))
        out[int(r)] = surf_dt if bs is None else (a * surf_dt + (1 - a) * bs)
    return out, prefix_rmse, a
