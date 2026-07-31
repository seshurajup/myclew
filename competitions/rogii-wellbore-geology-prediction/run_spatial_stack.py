"""run_spatial_stack — the MISSING CV-6 ingredient: PF + spatial-kNN neighbour-dip prior + GBM stack.
Neighbour prior is OOF-SAFE (only training-fold wells are neighbours). Field-disjoint CV vs blend_AB 10.43.

Idea (public decode): formation dip is regionally coherent, so a well's dtvt(md_since) profile is well
predicted by the SAME-md_since dtvt of its spatially nearest wells. We build that neighbour prior with a
cKDTree over each well's PS-point (X,Y), then let a GBM combine [pf_dtvt, neighbour_prior, geometry]."""
import glob, os, time, numpy as np, pandas as pd
from scipy.spatial import cKDTree

FEAT = "results/honest_feat_gs80.parquet"
FOLDS = "config/well_field_folds.csv"
K = 15            # neighbours
TAU = None        # None => inverse-distance; else gaussian bandwidth

def ps_xy(data_dir="input/train"):
    """PS-point (X,Y) per well = the last known (TVT_input non-nan) row — where prediction starts."""
    rows = {}
    for hp in glob.glob(os.path.join(data_dir, "*__horizontal_well.csv")):
        w = os.path.basename(hp).split("__")[0]
        d = pd.read_csv(hp, usecols=["X", "Y", "TVT_input"])
        kn = d.TVT_input.notna().to_numpy()
        if kn.sum() == 0:
            continue
        i = np.where(kn)[0][-1]
        rows[w] = (float(d.X.iloc[i]), float(d.Y.iloc[i]))
    return rows

def build_profiles(feat):
    """Per well: sorted (md_since, dtvt_true) profile for interpolation."""
    prof = {}
    for w, g in feat.groupby("well"):
        g = g.sort_values("md_since")
        prof[w] = (g.md_since.to_numpy(float), g.dtvt_true.to_numpy(float))
    return prof

def neighbour_prior(feat, xy, prof, folds):
    """For every row, distance-weighted avg of the K nearest TRAIN-FOLD wells' dtvt at that md_since."""
    out = np.full(len(feat), np.nan)
    well_arr = feat.well.to_numpy(); md_arr = feat.md_since.to_numpy(float)
    idx_by_well = {w: np.where(well_arr == w)[0] for w in feat.well.unique()}
    for vf in sorted(set(v for v in folds.values() if pd.notna(v))):
        tr = [w for w in xy if folds.get(w) is not None and folds.get(w) != vf and w in prof]
        va = [w for w in xy if folds.get(w) == vf and w in idx_by_well]
        if not tr or not va:
            continue
        pts = np.array([xy[w] for w in tr]); tree = cKDTree(pts)
        for w in va:
            qi = idx_by_well[w]
            dist, nn = tree.query(xy[w], k=min(K, len(tr)))
            dist = np.atleast_1d(dist); nn = np.atleast_1d(nn)
            wgt = np.exp(-(dist / TAU) ** 2) if TAU else 1.0 / (dist + 1e-6)
            wgt = wgt / wgt.sum()
            m = md_arr[qi]
            acc = np.zeros(len(qi))
            for wi, ni in zip(wgt, nn):
                nx, ny = prof[tr[ni]]
                acc += wi * np.interp(m, nx, ny)   # neighbour dtvt at this md_since
            out[qi] = acc
    return out

def pooled(p, t):
    m = np.isfinite(p) & np.isfinite(t)
    return float(np.sqrt(np.mean((p[m] - t[m]) ** 2)))

def main():
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.time()
    feat = pd.read_parquet(FEAT)
    folds = pd.read_csv(FOLDS).set_index("well").field_fold.to_dict()
    feat = feat[feat.well.map(folds).notna()].reset_index(drop=True)
    xy = ps_xy(); prof = build_profiles(feat)
    print(f"loaded {len(feat):,} rows, {feat.well.nunique()} wells ({(time.time()-t0):.0f}s)", flush=True)
    feat["nbr"] = neighbour_prior(feat, xy, prof, folds)
    print(f"neighbour prior built ({(time.time()-t0):.0f}s); coverage={feat.nbr.notna().mean():.3f}", flush=True)

    tru = feat.dtvt_true.to_numpy(float)
    print(f"[baseline] PF pf_dtvt            field-CV = {pooled(feat.pf_dtvt.to_numpy(float), tru):.4f}")
    print(f"[baseline] blend_AB reference    field-CV = 10.4353 (submitted -> LB 8.773)")
    print(f"[NEW] neighbour prior ALONE      field-CV = {pooled(feat.nbr.to_numpy(float), tru):.4f}")

    # GBM stack, field-disjoint OOF
    cols = ["pf_dtvt", "nbr", "md_since", "z_since", "z_rate", "conf", "tvt_ps", "twspan"]
    cols = [c for c in cols if c in feat.columns]
    X = feat[cols].to_numpy(np.float32); y = tru
    fold_id = feat.well.map(folds).to_numpy()
    oof = np.full(len(feat), np.nan)
    for vf in sorted(set(v for v in folds.values() if pd.notna(v))):
        tr = fold_id != vf; va = fold_id == vf
        gb = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=6,
                                           l2_regularization=1.0, max_bins=255)
        gb.fit(X[tr], y[tr])
        oof[va] = gb.predict(X[va])
        print(f"  fold{int(vf)} stack cum = {pooled(oof, y):.4f} ({(time.time()-t0):.0f}s)", flush=True)
    print(f"[NEW] PF+nbr+GBM STACK           field-CV = {pooled(oof, y):.4f}")
    print(f"=== gate: beats blend_AB 10.4353 ? {'YES' if pooled(oof,y)<10.4353 else 'no'} ===")

if __name__ == "__main__":
    main()
