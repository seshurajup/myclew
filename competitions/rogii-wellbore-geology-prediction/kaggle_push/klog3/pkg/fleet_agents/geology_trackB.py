"""geology_trackB — REUSABLE trajectory-track (Track B) for wellbore geosteering.

Likelihood-weighted PARTICLE FILTER over the typewell GR signature: each particle carries a stratigraphic
level (pos = TVT + Z) and a drift rate; at each MD step the GR predicted from the typewell at the particle's
TVT is compared to the observed horizontal GR to reweight, with systematic-resampling. An ensemble of
`n_seeds` filters is combined by softmax over per-filter log-likelihood (temperature `scale`). This is the
geosteering signal a per-row GBM (Track A) cannot see. Ported from the public 6.858 dual-track notebook
(run_particle_filter / run_pf_lik_ensemble). Deterministic given seeds.

Contract: `trackB_predict_well(hw_df, tw_df, n_particles, n_seeds, scale) -> full-length abs-TVT array`
(known rows carry TVT_input; eval rows filled with the PF estimate). Reduce to residual dtvt by
subtracting tvt_ps outside. `GeologyTrackB` BaseAgent wraps OOF assembly over a data dir.
"""
from __future__ import annotations
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .base import BaseAgent
except Exception:  # noqa: BLE001
    BaseAgent = object

# PF process/measurement constants (notebook-tuned)
MOM, VN, PN, RP, RR, RESAMP = 0.998, 0.002, 0.005, 0.1, 0.001, 0.5


def run_particle_filter(hw, tw, n_particles=500, seed=42):
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    out_vals = hw["TVT_input"].values.astype(float).copy()
    if len(ev) == 0 or len(kn) == 0:
        return out_vals, 0.0

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"]); last_Z = float(last["Z"]); last_MD = float(last["MD"])

    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + 4.5 * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N

    md_v = ev["MD"].values.astype(float)
    z_v = ev["Z"].values.astype(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    res = np.empty(len(ev)); prev_MD = last_MD; log_lik = 0.0
    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PN * rng.standard_normal(N)
        tvt_p = np.clip(pos - z_v[i], tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]
        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.maximum(np.exp(-0.5 * np.minimum(d ** 2, 600.0)), 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N
        if 1.0 / (w ** 2).sum() < RESAMP * N:
            cum = np.cumsum(w); u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + RP * rng.standard_normal(N)
            rate = rate[idx] + RR * rng.standard_normal(N)
            w = np.ones(N) / N
        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_ensemble_gpu(hw, tw, n_particles=500, n_seeds=128, scale=5.0, return_per_seed=False, pn=None, vn=None, gs_scale=1.0):
    """GPU (cupy) equivalent of run_pf_lik_ensemble: all n_seeds filters run as one
    (n_seeds, n_particles) tensor on the RTX 5090. Same measurement/process model & softmax-over-lik
    combine as the CPU path. Returns the full-length abs-TVT array (eval rows filled)."""
    import cupy as cp
    pn = PN if pn is None else float(pn); vn = VN if vn is None else float(vn)
    tw_s = tw.sort_values("TVT")
    tw_tvt = cp.asarray(tw_s["TVT"].values, dtype=cp.float32)
    tw_gr = cp.asarray(tw_s["GR"].fillna(tw_s["GR"].mean()).values, dtype=cp.float32)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    out_vals = hw["TVT_input"].values.astype(float).copy()
    if len(ev) == 0 or len(kn) == 0:
        return out_vals, 0.0

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"]); last_Z = float(last["Z"]); last_MD = float(last["MD"])
    tw_at_k = np.interp(kn["TVT_input"].values, cp.asnumpy(tw_tvt), cp.asnumpy(tw_gr))
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0)) * gs_scale
    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values); dz = np.diff(tail["Z"].values); dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    S, N = int(n_seeds), int(n_particles)
    rng = cp.random.default_rng(0)
    ls = last_tvt + last_Z
    pos = (ls + 4.5 * rng.standard_normal((S, N))).astype(cp.float32)
    rate = (ir + 0.01 * rng.standard_normal((S, N))).astype(cp.float32)
    w = cp.full((S, N), 1.0 / N, dtype=cp.float32)
    log_lik = cp.zeros(S)

    md_v = ev["MD"].values.astype(float)
    z_v = ev["Z"].values.astype(float)
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(float(cp.asnumpy(tw_gr.mean())))
    gr_v = gr_interp.values.astype(float)[ev.index]
    lo, hi = float(tw_tvt[0]) - 100, float(tw_tvt[-1]) + 100
    ar = cp.arange(N) / N
    n_ev = len(ev); res = cp.empty((S, n_ev)); prev_MD = last_MD
    for i in range(n_ev):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = MOM * rate + vn * rng.standard_normal((S, N))
        pos = pos + rate * dm_step + pn * rng.standard_normal((S, N))
        tvt_p = cp.clip(pos - z_v[i], lo, hi); pos = tvt_p + z_v[i]
        eg = cp.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = cp.maximum(cp.exp(-0.5 * cp.minimum(d ** 2, 600.0)), 1e-300)
        avg_lk = (w * lk).sum(axis=1)
        log_lik += cp.log(cp.maximum(avg_lk, 1e-300))
        w = w * lk
        w = w / cp.maximum(w.sum(axis=1, keepdims=True), 1e-300)
        ess = 1.0 / (w ** 2).sum(axis=1)
        mask = ess < RESAMP * N
        if bool(mask.any()):  # guarded: O(S*N^2) resample only when needed (measured faster than always-on)
            cum = cp.cumsum(w, axis=1)
            u0 = rng.uniform(0, 1.0 / N, size=(S, 1))
            pts = u0 + ar[None, :]
            idx = cp.clip((pts[:, :, None] > cum[:, None, :]).sum(axis=2), 0, N - 1)
            npos = cp.take_along_axis(pos, idx, axis=1) + RP * rng.standard_normal((S, N))
            nrate = cp.take_along_axis(rate, idx, axis=1) + RR * rng.standard_normal((S, N))
            m2 = mask[:, None]
            pos = cp.where(m2, npos, pos); rate = cp.where(m2, nrate, rate)
            w = cp.where(m2, 1.0 / N, w)
        res[:, i] = (w * (pos - z_v[i])).sum(axis=1)
        prev_MD = md_v[i]

    liks_n = log_lik - log_lik.max()
    weights = cp.exp(liks_n / scale); weights /= weights.sum()
    combined = (weights[:, None] * res).sum(axis=0)
    out_vals[list(ev.index)] = cp.asnumpy(combined)
    conf = _ensemble_confidence(cp.asnumpy(weights))
    if return_per_seed:
        # per-seed trajectories + log-likelihoods: seed-combination knobs (scale/weighting/selection) are then
        # cheap re-weightings of THESE, needing NO PF re-run. eval-row order matches list(ev.index).
        return out_vals, conf, {"res": cp.asnumpy(res), "log_lik": cp.asnumpy(log_lik),
                                 "idx": np.asarray(list(ev.index), dtype=int)}
    return out_vals, conf


def _ensemble_confidence(weights):
    """1 - normalized entropy of the seed-ensemble softmax weights: near 1 = one/few seeds dominate
    (the PF found a confident GR match to the typewell), near 0 = seeds disagree (ambiguous/weak signal).
    No leakage — computed purely from the PF's own internal likelihoods, available at real inference time."""
    w = np.asarray(weights, dtype=float)
    w = w / max(w.sum(), 1e-300)
    S = len(w)
    if S < 2:
        return 1.0
    ent = -np.sum(w * np.log(np.maximum(w, 1e-300)))
    return float(1.0 - ent / np.log(S))


def run_pf_lik_ensemble(hw, tw, n_particles=500, n_seeds=128, scale=5.0):
    preds, liks = [], []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p); liks.append(ll)
    liks = np.array(liks); liks_n = liks - liks.max()
    weights = np.exp(liks_n / scale); weights /= weights.sum()
    conf = _ensemble_confidence(weights)
    return (weights[:, None] * np.stack(preds, 0)).sum(0), conf


def _list_pairs(data_dir):
    pairs = []
    for hp in sorted(glob.glob(os.path.join(data_dir, "*__horizontal_well.csv"))):
        wid = os.path.basename(hp).split("__")[0]
        tp = os.path.join(data_dir, f"{wid}__typewell.csv")
        if os.path.exists(tp):
            pairs.append((wid, hp, tp))
    return pairs


def trackB_oof(data_dir, out_path, training=True, n_particles=500, n_seeds=128, scale=5.0,
               limit=None, gpu=False, noise_id=False, gs_scale=1.0, noise_mode="absolute"):
    """Run Track-B over every well; emit per-row eval-region predictions aligned to `<well>_<rowidx>` ids."""
    pairs = _list_pairs(data_dir)
    if limit:
        pairs = pairs[:limit]
    _rel = None
    if noise_id and noise_mode == "relative":
        from . import pf_noise_id as _nid
        ests = []
        for _wid, _hp, _tp in pairs:
            try: ests.append(_nid.identify(pd.read_csv(_hp), pd.read_csv(_tp)))
            except Exception: pass
        _nid.calibrate(ests); _rel = _nid
    rows = []
    for wid, hp, tp in pairs:
        hw = pd.read_csv(hp)
        if "TVT_input" not in hw.columns:
            continue
        tw = pd.read_csv(tp)
        pred_mask = hw["TVT_input"].isna().to_numpy()
        if training:
            pred_mask &= hw["TVT"].notna().to_numpy()
        if pred_mask.sum() == 0:
            continue
        ps = int(np.argmax(hw["TVT_input"].isna().to_numpy()))
        tvt_ps = float(hw["TVT"].iloc[ps - 1]) if training else float(hw["TVT_input"].iloc[max(ps - 1, 0)])
        _pf = run_pf_ensemble_gpu if gpu else run_pf_lik_ensemble
        if noise_id and gpu:
            from . import pf_noise_id as _nid
            est = _rel.identify_relative(hw, tw) if (noise_mode == "relative" and _rel is not None) else _nid.identify(hw, tw)
            full, conf = _pf(hw, tw, n_particles=n_particles, n_seeds=n_seeds, scale=scale,
                             pn=est["pn"], vn=est["vn"], gs_scale=1.0)
        elif gpu:
            full, conf = _pf(hw, tw, n_particles=n_particles, n_seeds=n_seeds, scale=scale, gs_scale=gs_scale)
        else:
            full, conf = _pf(hw, tw, n_particles=n_particles, n_seeds=n_seeds, scale=scale)
        idx = np.where(pred_mask)[0]
        d = pd.DataFrame({
            "id": [f"{wid}_{i}" for i in idx],
            "well": wid,
            "trackB_tvt": full[idx],
            "trackB_dtvt": full[idx] - tvt_ps,
            "trackB_conf": conf,   # per-well PF ensemble confidence (no leakage; real-inference-time signal)
        })
        if training:
            d["true_dtvt"] = hw["TVT"].to_numpy()[idx] - tvt_ps
            d["true_tvt"] = hw["TVT"].to_numpy()[idx]
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return str(out_path)


class GeologyTrackB(BaseAgent):
    name = "geology-trackb"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        dd, out = spec.get("data_dir"), spec.get("out")
        if not dd or not out:
            return self.escalate(worker, "leader", "geology-trackb needs spec ['data_dir','out']")
        p = trackB_oof(dd, out, training=bool(spec.get("training", True)),
                       n_particles=int(spec.get("n_particles", 500)),
                       n_seeds=int(spec.get("n_seeds", 128)), scale=float(spec.get("scale", 5.0)),
                       limit=spec.get("limit"))
        df = pd.read_csv(p)
        msg = f"geology-trackb: {len(df)} rows → {Path(p).name}"
        if "true_dtvt" in df:
            rmse = float(np.sqrt(((df.trackB_dtvt - df.true_dtvt) ** 2).mean()))
            msg += f" | pooled RMSE={rmse:.3f}"
        self.log(msg, kind="finding", recommendation="blend with Track-A OOF via blend-optimize")
        return self.done({"path": p}, msg)


_AGENT = GeologyTrackB() if BaseAgent is not object else None


def run(q, worker):
    return _AGENT.run(q, worker)
