"""geology_contact — FORMATION-CONSTRAINED alignment (the honest non-PF lever from the public frontier).
The typewell Geology column labels 6 parallel formations (ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA) at their TVT
boundaries. Pointwise GR matching is multimodal (eda proof 8: ~21 minima/well) because facies repeat ACROSS
formations. Fix: constrain each hidden-row GR match to the formation band consistent with the local
dip-continuation prior — collapsing the global multimodal search to a local, unimodal one. Leak-free: uses
only the typewell Geology (available at test) + the well's own GR/heel. Returns per-well hidden-row TVT."""
from __future__ import annotations
import numpy as np, pandas as pd

def _formation_bands(tw):
    g = tw["Geology"].fillna("").astype(str).str.strip().to_numpy()
    T = tw["TVT"].to_numpy(float)
    bands = {}
    for lab in [x for x in np.unique(g) if x]:
        tt = T[g == lab]
        if len(tt): bands[lab] = (tt.min(), tt.max())
    return bands

def predict_contact(hw, tw, band_pad=3.0, win=25):
    tws = tw.sort_values("TVT"); T = tws["TVT"].to_numpy(float); G = tws["GR"].to_numpy(float)
    m = np.isfinite(T) & np.isfinite(G); T, G = T[m], G[m]
    out = hw["TVT_input"].to_numpy(float).copy()
    kn = hw["TVT_input"].notna().to_numpy(); ev = ~kn
    if ev.sum() == 0 or kn.sum() < 30 or len(T) < 20: return out
    MD = hw["MD"].to_numpy(float); GR = hw["GR"].to_numpy(float); Z = hw["Z"].to_numpy(float)
    ki = np.where(kn)[0]; mdps = MD[ki[-1]]; tvtps = out[ki[-1]]
    # affine-calibrate HW GR to typewell scale on the heel
    tw_at_k = np.interp(out[ki], T, G); a, b = np.polyfit(GR[ki][np.isfinite(GR[ki])], tw_at_k[np.isfinite(GR[ki])], 1) \
        if np.isfinite(GR[ki]).sum() > 10 else (1.0, 0.0)
    Gc = a * GR + b
    # dip-continuation prior
    kk = ki[-min(400, len(ki)):]; dip0 = np.polyfit(MD[kk], out[kk], 1)[0]
    # HOME FORMATION from the prefix (the well stays in ~1 formation 87% of the time): its typewell TVT band
    # is a FIXED anchor that caps far-toe drift, unlike the drifting dip prior.
    bands = _formation_bands(tw)
    home = None
    for lab, (lo, hi) in bands.items():
        if lo - band_pad <= tvtps <= hi + band_pad: home = (lo, hi); break
    if home is None:                                            # fall back to a band around the anchor
        home = (tvtps - 40, tvtps + 40)
    hlo, hhi = home[0] - band_pad, home[1] + band_pad
    sel = (T >= hlo) & (T <= hhi)
    if sel.sum() < 5:
        out[np.where(ev)[0]] = np.clip(tvtps, T[0], T[-1]); return out
    Ts, Gs = T[sel], G[sel]
    ei = np.where(ev)[0]
    for r in ei:
        prior = np.clip(tvtps + dip0 * (MD[r] - mdps), hlo, hhi)   # dip prior CLAMPED to the home band
        d = np.abs(Gs - Gc[r]) + 0.25 * np.abs(Ts - prior)        # GR match within the home formation + prior reg
        out[r] = Ts[np.argmin(d)]
    return out

def run_oof(data_dir, folds_csv, fold_key, limit=None):
    import glob, os
    fol = pd.read_csv(folds_csv).set_index("well")[fold_key].to_dict()
    rows = []
    for hp in sorted(glob.glob(os.path.join(data_dir, "*__horizontal_well.csv")))[:limit]:
        w = os.path.basename(hp).split("__")[0]
        try: hw = pd.read_csv(hp); tw = pd.read_csv(os.path.join(data_dir, f"{w}__typewell.csv"))
        except Exception: continue
        if "TVT" not in hw.columns or w not in fol: continue
        pred = predict_contact(hw, tw); ev = hw["TVT_input"].isna().to_numpy() & hw["TVT"].notna().to_numpy()
        if ev.sum() == 0: continue
        yt = hw["TVT"].to_numpy(float)[ev]; yp = pred[ev]
        for a, b in zip(yp, yt): rows.append((w, a, b))
    d = pd.DataFrame(rows, columns=["well", "pred", "true"])
    pooled = float(np.sqrt(np.mean((d.pred - d.true) ** 2)))
    return pooled, d
