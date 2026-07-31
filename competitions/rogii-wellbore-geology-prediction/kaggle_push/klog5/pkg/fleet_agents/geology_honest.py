"""Honest target-free geosteering engine for rogii-wellbore-geology-prediction.

Per-well, self-contained (no cross-well training) alignment of the horizontal GR curve
against the well's own dense typewell GR(TVT) reference to recover TVT in the eval zone.
All levers are leak-free: only the visible heel (TVT_input notna) + typewell are used.

Core levers (in priority order):
  1. affine_cal      : GR gain/offset calibration on the visible heel (removes tool scale).
  2. multi_scale_ncc : windowed normalized cross-correlation shape match -> TVT per eval row.
  3. selector        : per-well hold-out-tail scoring picks the best variant (ncc/pf/hold).
  4. continuity+warmup+smoothing on the recovered TVT track.

Prediction target is dtvt = TVT - tvt_ps (== competition RMSE).
"""
import glob
import os
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ data
def list_pairs(data_dir):
    pairs = []
    for hp in sorted(glob.glob(os.path.join(data_dir, "*__horizontal_well.csv"))):
        wid = os.path.basename(hp).split("__")[0]
        tp = os.path.join(data_dir, f"{wid}__typewell.csv")
        if os.path.exists(tp):
            pairs.append((wid, hp, tp))
    return pairs


def _smooth(x, w):
    if w <= 1:
        return np.asarray(x, float)
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


# ------------------------------------------------------------------ affine calibration
def affine_cal(kgr, tw_at_k, min_pts=20):
    """Fit kgr ~ a*tw_at_k + b on the heel; return (a, b). Maps horizontal GR to typewell scale
    via  tw_scale = (G - b) / a."""
    v = np.isfinite(kgr) & np.isfinite(tw_at_k)
    if v.sum() < min_pts or np.std(tw_at_k[v]) < 1e-6 or np.std(kgr[v]) < 1e-6:
        return 1.0, float(np.nanmean(kgr) - np.nanmean(tw_at_k)) if v.any() else 0.0
    a, b = np.polyfit(tw_at_k[v], kgr[v], 1)
    if not np.isfinite(a) or abs(a) < 1e-6:
        return 1.0, float(np.nanmean(kgr) - np.nanmean(tw_at_k))
    return float(a), float(b)


# ------------------------------------------------------------------ multi-scale NCC
def multi_scale_ncc(ref_gr, ref_tvt, query_gr, hws=(8, 15, 25), stride=2,
                    band=None, band_center=None):
    """Match each query (horizontal) GR window against the reference (typewell) GR windows;
    return (tvt_pred[n_query], score[n_query]) using the score-weighted multi-scale mean.

    band: if given, for query row i restrict candidate typewell centers to
          band_center[i] +/- band (in TVT units) -> enforces spatial continuity."""
    ref_gr = np.asarray(ref_gr, np.float32)
    ref_tvt = np.asarray(ref_tvt, np.float32)
    query_gr = np.asarray(query_gr, np.float32)
    nq = len(query_gr)
    nr = len(ref_gr)
    if nq == 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    if nr < max(hws) * 2 + 2:
        return np.full(nq, ref_tvt[-1] if nr else 0.0, np.float32), np.zeros(nq, np.float32)

    tvt_stack = []
    score_stack = []
    for hw in hws:
        win = 2 * hw + 1
        if nr < win + 1:
            continue
        rg = _smooth(ref_gr, 5).astype(np.float32)
        qg = _smooth(query_gr, 5).astype(np.float32)
        sts = np.arange(0, nr - win + 1, stride, dtype=np.int32)
        if len(sts) == 0:
            continue
        C = rg[sts[:, None] + np.arange(win, dtype=np.int32)[None, :]]
        Cn = (C - C.mean(1, keepdims=True)) / (C.std(1, keepdims=True) + 1e-6)
        ctr_tvt = ref_tvt[np.clip(sts + hw, 0, nr - 1)]          # [M]
        qp = np.pad(qg, hw, mode="edge")
        H = qp[np.arange(nq)[:, None] + np.arange(win)[None, :]]
        Hn = (H - H.mean(1, keepdims=True)) / (H.std(1, keepdims=True) + 1e-6)
        ncc = (Hn @ Cn.T) / win                                  # [nq, M]
        if band is not None and band_center is not None:
            dist = np.abs(ctr_tvt[None, :] - np.asarray(band_center, np.float32)[:, None])
            ncc = np.where(dist <= band, ncc, -2.0)
        best = ncc.argmax(1)
        sc = ncc.max(1).astype(np.float32)
        tvt_stack.append(ctr_tvt[best].astype(np.float32))
        score_stack.append(np.clip(sc, 0.0, None))
    if not tvt_stack:
        return np.full(nq, ref_tvt[-1], np.float32), np.zeros(nq, np.float32)
    T = np.stack(tvt_stack, 1)                                   # [nq, S]
    W = np.stack(score_stack, 1) + 1e-6
    tvt = (T * W).sum(1) / W.sum(1)
    score = W.max(1)
    return tvt.astype(np.float32), score.astype(np.float32)


# ------------------------------------------------------------------ per-well predictors
def _prep(hw, tw):
    tw = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = tw["GR"].to_numpy(float)
    ev = hw["TVT_input"].isna().to_numpy()
    kn = ~ev
    return tw_tvt, tw_gr, ev, kn


def _tvt_ps(hw, ev, training):
    ps = int(np.argmax(ev))
    col = "TVT" if training else "TVT_input"
    return float(hw[col].iloc[ps - 1]) if ps > 0 else float(hw["TVT_input"].dropna().iloc[-1])


def predict_ncc(hw, tw_tvt, tw_gr, ev, kn, tvt_ps, hws=(8, 15, 25), band=25.0,
                warmup_tau=200.0, smooth=15):
    """Recover TVT for eval rows via calibrated multi-scale NCC + continuity band + warmup damping."""
    G = hw["GR"].to_numpy(float)
    MD = hw["MD"].to_numpy(float)
    # calibrate GR onto typewell scale using heel
    kgr = G[kn]
    ktvt = hw["TVT_input"].to_numpy(float)[kn]
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
    a, b = affine_cal(kgr, tw_at_k)
    Gc = (G - b) / a                                            # GR in typewell units
    # band center = last-known TVT propagated (continuity prior)
    q = Gc[ev]
    band_center = np.full(ev.sum(), tvt_ps, np.float32)
    tvt_raw, score = multi_scale_ncc(tw_gr, tw_tvt, q, hws=hws, band=band,
                                     band_center=band_center)
    # continuity: blend toward tvt_ps early (warmup), trust NCC more as MD advances
    md_ev = MD[ev]
    md_ps = MD[kn][-1] if kn.any() else md_ev[0]
    damp = 1.0 - np.exp(-(md_ev - md_ps) / warmup_tau)
    damp = np.clip(damp, 0.0, 1.0)
    tvt = tvt_ps + damp * (tvt_raw - tvt_ps)
    tvt = _smooth(tvt, smooth)
    return tvt - tvt_ps, score


def predict_hold(hw, ev, kn, tvt_ps, training):
    """Constant-continuation baseline (dtvt=0), but continue last velocity slightly (linear)."""
    return np.zeros(int(ev.sum()), float), np.zeros(int(ev.sum()), float)


# ------------------------------------------------------------------ selector via heel hold-out
def _score_variant_on_heel(hw, tw_tvt, tw_gr, kn, holdout_frac=0.35, **kw):
    """Hide the tail of the visible heel, predict it with NCC, return RMSE vs known TVT."""
    ki = np.where(kn)[0]
    if len(ki) < 60:
        return np.inf
    cut = int(len(ki) * (1 - holdout_frac))
    if cut < 30:
        return np.inf
    hidden = ki[cut:]
    G = hw["GR"].to_numpy(float)
    tvt_true = hw["TVT_input"].to_numpy(float)
    # calibrate on retained heel
    retain = ki[:cut]
    tw_at_k = np.interp(tvt_true[retain], tw_tvt, tw_gr)
    a, b = affine_cal(G[retain], tw_at_k)
    Gc = (G - b) / a
    tvt_ps = float(tvt_true[retain[-1]])
    band_center = np.full(len(hidden), tvt_ps, np.float32)
    tvt_pred, _ = multi_scale_ncc(tw_gr, tw_tvt, Gc[hidden],
                                  hws=kw.get("hws", (8, 15, 25)),
                                  band=kw.get("band", 25.0), band_center=band_center)
    md = hw["MD"].to_numpy(float)
    damp = 1.0 - np.exp(-(md[hidden] - md[retain[-1]]) / kw.get("warmup_tau", 200.0))
    tvt_pred = tvt_ps + np.clip(damp, 0, 1) * (tvt_pred - tvt_ps)
    return float(np.sqrt(np.mean((tvt_pred - tvt_true[hidden]) ** 2)))


def predict_well(hw, tw, training=True, use_selector=True):
    tw_tvt, tw_gr, ev, kn = _prep(hw, tw)
    n = int(ev.sum())
    if n == 0:
        return np.zeros(0), 0.0
    tvt_ps = _tvt_ps(hw, ev, training)
    variants = {
        "ncc25": dict(hws=(8, 15, 25), band=25.0, warmup_tau=200.0),
        "ncc40": dict(hws=(10, 20, 35), band=40.0, warmup_tau=150.0),
        "ncc_tight": dict(hws=(6, 12, 20), band=15.0, warmup_tau=250.0),
    }
    if use_selector:
        scores = {k: _score_variant_on_heel(hw, tw_tvt, tw_gr, kn, **v)
                  for k, v in variants.items()}
        hold_rmse = _heel_hold_rmse(hw, kn)
        best = min(scores, key=scores.get)
        best_rmse = scores[best]
        if not np.isfinite(best_rmse) or best_rmse >= hold_rmse:
            return np.zeros(n), tvt_ps   # guarded fallback to const-continuation
        kw = variants[best]
    else:
        kw = variants["ncc25"]
    dtvt, _ = predict_ncc(hw, tw_tvt, tw_gr, ev, kn, tvt_ps, **kw)
    return dtvt, tvt_ps


def _heel_hold_rmse(hw, kn, holdout_frac=0.35):
    """RMSE of const-continuation on the hidden heel tail (the fallback baseline to beat)."""
    ki = np.where(kn)[0]
    if len(ki) < 60:
        return np.inf
    cut = int(len(ki) * (1 - holdout_frac))
    hidden = ki[cut:]
    tvt_true = hw["TVT_input"].to_numpy(float)
    tvt_ps = float(tvt_true[ki[cut - 1]])
    return float(np.sqrt(np.mean((tvt_true[hidden] - tvt_ps) ** 2)))


# ------------------------------------------------------------------ OOF runner
def run_oof(data_dir, training=True, limit=None, use_selector=True):
    rows = []
    pairs = list_pairs(data_dir)
    if limit:
        pairs = pairs[:limit]
    for wid, hp, tp in pairs:
        hw = pd.read_csv(hp)
        if "TVT_input" not in hw.columns:
            continue
        tw = pd.read_csv(tp)
        ev = hw["TVT_input"].isna().to_numpy()
        if training:
            ev &= hw["TVT"].notna().to_numpy()
        if ev.sum() == 0:
            continue
        dtvt, tvt_ps = predict_well(hw, tw, training=training, use_selector=use_selector)
        idx = np.where(ev)[0]
        if len(dtvt) != len(idx):
            # predict_well used full-eval mask; realign
            fullev = hw["TVT_input"].isna().to_numpy()
            fidx = np.where(fullev)[0]
            m = np.isin(fidx, idx)
            dtvt = dtvt[m]
        d = {"id": [f"{wid}_{i}" for i in idx], "well": wid,
             "dtvt_pred": dtvt, "tvt_ps": tvt_ps}
        if training:
            d["dtvt_true"] = hw["TVT"].to_numpy()[idx] - tvt_ps
        rows.append(pd.DataFrame(d))
    return pd.concat(rows, ignore_index=True)


def pooled_rmse(df):
    return float(np.sqrt(np.mean((df["dtvt_pred"] - df["dtvt_true"]) ** 2)))


# ================================================================== Z-velocity PF (Pilkwang port)
from scipy.interpolate import interp1d

_PF = dict(N=300, MOM=0.993, VN=0.005, PN=0.01, GR_SIG_MIN=10., GR_SIG_MAX=60., GR_SIG_DEF=30.,
           INIT_V_STD=0.02, INIT_SPR=0.5, RESAMP=0.5, ROUGH_P=0.2, ROUGH_V=0.003,
           GR_WIN=5, GR_WT=0.3)


def _cal_gr_sigma(kn, tw_tvt, tw_gr):
    if len(kn) < 20:
        return _PF["GR_SIG_DEF"]
    ex = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    return float(np.clip(np.std(kn["GR_c"].values - ex), _PF["GR_SIG_MIN"], _PF["GR_SIG_MAX"]))


def _z_beta(kn):
    if len(kn) < 30:
        return -1., 0., 0.1
    dz = np.diff(kn["Z"].values); dtvt = np.diff(kn["TVT_input"].values); dmd = np.diff(kn["MD"].values)
    m = dmd > 0
    if m.sum() < 10:
        return -1., 0., 0.1
    vz = dz[m] / dmd[m]; vt = dtvt[m] / dmd[m]
    A = np.column_stack([vz, np.ones_like(vz)])
    c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
    return c[0], c[1], max(np.std(vt - (c[0] * vz + c[1])), 0.001)


def _init_v(kn):
    if len(kn) < 10:
        return 0.
    tail = kn.tail(20); dtvt = np.diff(tail["TVT_input"].values); dmd = np.diff(tail["MD"].values)
    m = dmd > 0
    return 0. if m.sum() < 3 else float(np.median(dtvt[m] / dmd[m]))


def run_pf_z(hw, tw_tvt, tw_gr, N=None, seed=42, gr_wt=None):
    """Physics-grounded particle filter: TVT velocity ~ beta*Z-velocity (dip regression on heel)
    + GR likelihood against affine-calibrated typewell. Returns (pts, std) over eval rows."""
    N = N or _PF["N"]; gr_wt = _PF["GR_WT"] if gr_wt is None else gr_wt
    rng = np.random.RandomState(seed)
    tw_s = pd.Series(tw_gr).rolling(_PF["GR_WIN"], center=True, min_periods=1).mean().values
    tf_p = interp1d(tw_tvt, tw_gr, bounds_error=False, fill_value=(tw_gr[0], tw_gr[-1]))
    tf_s = interp1d(tw_tvt, tw_s, bounds_error=False, fill_value=(tw_s[0], tw_s[-1]))
    tmin, tmax = tw_tvt.min(), tw_tvt.max()
    kn = hw[hw["TVT_input"].notna()]; ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    gs = _cal_gr_sigma(kn, tw_tvt, tw_gr); beta, icpt, zsig = _z_beta(kn)
    gr_sm = hw["GR_c"].rolling(_PF["GR_WIN"], center=True, min_periods=1).mean()
    pos = float(kn["TVT_input"].iloc[-1]) + rng.normal(0, _PF["INIT_SPR"], N)
    vel = _init_v(kn) + rng.normal(0, _PF["INIT_V_STD"], N)
    w = np.ones(N) / N
    md_v = ev["MD"].values; gr_v = ev["GR_c"].values; z_v = ev["Z"].values
    pm = float(kn["MD"].iloc[-1]); pz = float(kn["Z"].iloc[-1])
    pts = np.empty(len(ev)); std = np.empty(len(ev))
    ev_pos = [hw.index.get_loc(i) for i in ev.index]
    for i, ridx in enumerate(ev.index):
        dm = max(md_v[i] - pm, 1.); dzd = (z_v[i] - pz) / dm
        ve = beta * dzd + icpt
        vel = _PF["MOM"] * vel + rng.normal(0, _PF["VN"], N)
        pos = pos + vel * dm + rng.normal(0, _PF["PN"], N)
        pos = np.clip(pos, tmin - 50, tmax + 50)
        if not np.isnan(gr_v[i]):
            ep = tf_p(pos); lp = np.exp(-0.5 * ((gr_v[i] - ep) / gs) ** 2)
            gs_sm = gr_sm.iloc[ev_pos[i]]
            if not np.isnan(gs_sm):
                es = tf_s(pos); ls = np.exp(-0.5 * ((gs_sm - es) / (gs * 1.5)) ** 2)
                lk = (1 - gr_wt) * lp + gr_wt * ls
            else:
                lk = lp
            lk = np.maximum(lk, 1e-300); w *= lk; ws = w.sum()
            w = (w / ws) if ws > 0 else np.full(N, 1. / N)
        zs = max(zsig * 2., 0.005); lz = np.exp(-0.5 * ((vel - ve) / zs) ** 2)
        lz = np.maximum(lz, 1e-300); w *= lz; ws = w.sum()
        w = (w / ws) if ws > 0 else np.full(N, 1. / N)
        ne = 1. / np.sum(w ** 2)
        if ne < _PF["RESAMP"] * N:
            cum = np.cumsum(w); u = (np.arange(N) + rng.uniform()) / N
            ix = np.searchsorted(cum, u); pos = pos[ix]; vel = vel[ix]; w[:] = 1. / N
            pos += rng.normal(0, _PF["ROUGH_P"], N); vel += rng.normal(0, _PF["ROUGH_V"], N)
        pts[i] = np.average(pos, weights=w)
        std[i] = np.sqrt(np.average((pos - pts[i]) ** 2, weights=w))
        pm = md_v[i]; pz = z_v[i]
    return pts, std


def predict_pfz(hw, tw, training=True, n_seeds=8, base_seed=42):
    """Affine-calibrate GR, run seed-averaged Z-velocity PF; return (dtvt over TVT_input-NaN rows, tvt_ps, std)."""
    tw_tvt, tw_gr, ev, kn = _prep(hw, tw)
    n = int(ev.sum())
    if n == 0:
        return np.zeros(0), 0.0, np.zeros(0)
    hw = hw.copy()
    G = hw["GR"].to_numpy(float)
    ktvt = hw["TVT_input"].to_numpy(float)[kn]
    tw_at_k = np.interp(ktvt, tw_tvt, tw_gr)
    a, b = affine_cal(G[kn], tw_at_k)
    hw["GR_c"] = (G - b) / a
    tvt_ps = _tvt_ps(hw, ev, training)
    acc = np.zeros(n); accs = np.zeros(n)
    for s in range(n_seeds):
        p, sd = run_pf_z(hw, tw_tvt, tw_gr, seed=base_seed + 17 * s)
        acc += p; accs += sd
    pts = acc / n_seeds; std = accs / n_seeds
    return pts - tvt_ps, tvt_ps, std


# ================================================================== meta-stack + budget runner
def build_features(data_dir, pf_oof_df, training=True):
    """Per-eval-row leak-free features aligned to pf_oof_df (id,well,dtvt_pred[PF],[dtvt_true],trackB_conf)."""
    df = pf_oof_df.copy()
    df["rowidx"] = df.id.str.rsplit("_", n=1).str[1].astype(int)
    out = []
    for wid, grp in df.groupby("well"):
        hp = os.path.join(data_dir, f"{wid}__horizontal_well.csv")
        tp = os.path.join(data_dir, f"{wid}__typewell.csv")
        if not os.path.exists(hp):
            continue
        hw = pd.read_csv(hp); tw = pd.read_csv(tp)
        tw_tvt, tw_gr, ev, kn = _prep(hw, tw)
        beta, icpt, zsig = _z_beta(hw[hw["TVT_input"].notna()])
        ps = int(np.argmax(hw["TVT_input"].isna().to_numpy()))
        MD = hw["MD"].to_numpy(float); Z = hw["Z"].to_numpy(float); G = hw["GR"].to_numpy(float)
        col = "TVT" if training else "TVT_input"
        tvtps = float(hw[col].iloc[ps - 1]) if ps > 0 else float(hw["TVT_input"].dropna().iloc[-1])
        ktvt = hw["TVT_input"].to_numpy(float)[kn]
        a, b = affine_cal(G[kn], np.interp(ktvt, tw_tvt, tw_gr)); Gc = (G - b) / a
        ii = grp["rowidx"].values
        md_since = MD[ii] - MD[ps - 1]; z_since = Z[ii] - Z[ps - 1]
        gr_res = np.abs(Gc[ii] - np.interp(tvtps + grp["dtvt_pred"].values, tw_tvt, tw_gr))
        d = pd.DataFrame({"id": grp.id.values, "well": wid,
            "pf_dtvt": grp["dtvt_pred"].values, "conf": grp["trackB_conf"].values,
            "md_since": md_since, "z_since": z_since, "z_rate": z_since / np.maximum(md_since, 1),
            "rowidx_since": ii - ps, "beta": beta, "icpt": icpt, "zsig": zsig,
            "n_known": kn.sum(), "gr": Gc[ii], "gr_res": gr_res, "a": a, "tvt_ps": tvtps,
            "twspan": tw_tvt.max() - tw_tvt.min(),
            "proj_dtvt": np.clip(beta * z_since + icpt * md_since, -80, 80)})
        if "dtvt_true" in grp:
            d["dtvt_true"] = grp["dtvt_true"].values
        out.append(d)
    return pd.concat(out, ignore_index=True)


META_FEATS = ["pf_dtvt", "conf", "md_since", "z_since", "z_rate", "rowidx_since",
              "beta", "icpt", "zsig", "n_known", "gr", "gr_res", "a", "tvt_ps",
              "twspan", "proj_dtvt"]
