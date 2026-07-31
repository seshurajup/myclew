"""geology_linealign — HONEST constrained per-well LINE aligner (Pilkwang-style), the base aligner that
targets the ~6.6 per-well line-oracle. For each well: affine-calibrate the horizontal GR to the typewell
scale on the visible heel, then fit a 2-param trajectory dtvt(MD)=off+slope*(MD-MD_ps) by a robust CLIPPED
GR-misfit against the typewell GR, verified against the visible prefix, with a bimodal hedge. No target used
(GR + typewell + known TVT only) → leak-free. Replaces the flexible particle filter that over-commits to
spurious multimodal GR matches on the hard (divergent) wells."""
from __future__ import annotations
import numpy as np, pandas as pd

def _interp_sorted(x, xp, fp):
    # xp sorted ascending; vectorized linear interp with edge clamp
    idx = np.clip(np.searchsorted(xp, x) - 1, 0, len(xp) - 2)
    x0 = xp[idx]; x1 = xp[idx + 1]; f0 = fp[idx]; f1 = fp[idx + 1]
    t = np.where(x1 > x0, (x - x0) / (x1 - x0 + 1e-9), 0.0)
    return f0 + t * (f1 - f0)

def align_well(hw: pd.DataFrame, tw: pd.DataFrame,
               slope_rng=0.05, n_slope=201, off_rng=15.0, n_off=61,
               clip=4.0, sub=300, prefix_tol=2.0):
    tws = tw.sort_values("TVT")
    twT = tws["TVT"].to_numpy(float); twG = tws["GR"].to_numpy(float)
    m = np.isfinite(twT) & np.isfinite(twG); twT, twG = twT[m], twG[m]
    out = hw["TVT_input"].to_numpy(float).copy()
    kn = hw["TVT_input"].notna().to_numpy(); ev = ~kn
    if ev.sum() == 0 or kn.sum() < 30 or len(twT) < 10:
        return out
    md = hw["MD"].to_numpy(float); gr = hw["GR"].to_numpy(float)
    ki = np.where(kn)[0]; mdps = md[ki[-1]]; tvtps = out[ki[-1]]
    # affine-calibrate HW GR to typewell scale on the visible heel
    tw_at_k = _interp_sorted(out[ki], twT, twG)
    A = np.polyfit(gr[ki][np.isfinite(gr[ki])], tw_at_k[np.isfinite(gr[ki])], 1) \
        if np.isfinite(gr[ki]).sum() > 10 else np.array([1.0, 0.0])
    def cal(g): return A[0] * g + A[1]
    # known-region dip prior (slope of TVT vs MD over last part of heel)
    kk = ki[-min(400, len(ki)):]
    dip0 = np.polyfit(md[kk], out[kk], 1)[0]
    ei = np.where(ev)[0]
    # subsample hidden rows for the fit
    fit = ei if len(ei) <= sub else ei[np.linspace(0, len(ei) - 1, sub).astype(int)]
    gm = cal(gr[fit]); good = np.isfinite(gm)
    fit = fit[good]; gm = gm[good]
    if len(fit) < 10:
        return out
    mdf = md[fit]
    slopes = dip0 + np.linspace(-slope_rng, slope_rng, n_slope)
    offs = np.linspace(-off_rng, off_rng, n_off)
    S, O = np.meshgrid(slopes, offs, indexing="ij")     # (nS,nO)
    S = S.ravel(); O = O.ravel()
    lo, hi = twT[0], twT[-1]
    # predicted TVT per (combo, row): tvtps + off + slope*(md-mdps)
    pred = tvtps + O[:, None] + S[:, None] * (mdf[None, :] - mdps)
    pred = np.clip(pred, lo, hi)
    egr = _interp_sorted(pred.ravel(), twT, twG).reshape(pred.shape)
    s = np.std(gm) + 1e-6
    r = np.clip((gm[None, :] - egr) / s, -clip, clip)
    J = np.mean(r ** 2, axis=1)                          # robust misfit per combo
    order = np.argsort(J)
    best = order[0]
    # bimodal hedge: 2nd distinct-TVT minimum, prefix-verify both, pick prefix-consistent
    def prefix_rmse(slope, off):
        p = tvtps + off + slope * (md[kk] - mdps)
        return np.sqrt(np.mean((p - out[kk]) ** 2))
    cand = [best]
    for j in order[1:200]:
        if abs(O[j] - O[best]) > 5.0:
            cand.append(j); break
    scored = sorted(cand, key=lambda j: (J[j] + 0.3 * prefix_rmse(S[j], O[j])))
    bj = scored[0]
    slope, off = S[bj], O[bj]
    out[ei] = np.clip(tvtps + off + slope * (md[ei] - mdps), lo, hi)
    return out

def run_oof(data_dir, wells=None, limit=None):
    import glob, os
    hs = sorted(glob.glob(os.path.join(data_dir, "*__horizontal_well.csv")))
    rows = []
    for hp in hs:
        w = os.path.basename(hp).split("__")[0]
        if wells is not None and w not in wells:
            continue
        try:
            hw = pd.read_csv(hp); tw = pd.read_csv(os.path.join(data_dir, f"{w}__typewell.csv"))
        except Exception:
            continue
        pred = align_well(hw, tw)
        ev = hw["TVT_input"].isna().to_numpy()
        if ev.sum() == 0 or "TVT" not in hw.columns:
            continue
        yt = hw["TVT"].to_numpy(float)[ev]; yp = pred[ev]
        tvtps = hw["TVT_input"].dropna().iloc[-1]
        rows.append((w, np.sqrt(np.mean((yp - yt) ** 2)),
                     np.sqrt(np.mean(((yp - tvtps) - (yt - tvtps)) ** 2)), ev.sum(), yp, yt))
        if limit and len(rows) >= limit:
            break
    return rows
