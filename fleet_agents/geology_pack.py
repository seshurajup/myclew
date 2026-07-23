"""geology_pack — REUSABLE geosteering domain hook for wellbore geology comps (rogii-wellbore-*).

Assembles the per-well multi-file geosteering data (one `<id>__horizontal_well.csv` + one
`<id>__typewell.csv` per well) into a FLAT per-row tabular table the tab-* pack can train on.

Task: predict TVT (true vertical thickness = stratigraphic depth, ft) at every 1-ft MD step of a
horizontal well BEYOND the Prediction Start (PS) point. PS = first row whose `TVT_input` is NaN;
before PS, `TVT_input == TVT` (known). Metric = RMSE of (manualTVT - predictedTVT) over predicted rows.

We model ABSOLUTE tvt (so the pack writes predictions directly, zero post-processing); `tvt_ps` (the
last known TVT at PS) is fed as a feature the model anchors on, and `cand_tvt` (a typewell GR-match
anchor) supplies the geosteering signal. GR in horizontal wells is ~66% NaN → forward/back-filled.

Reusable: `geology_assemble(data_dir, out_path, training=...)` — no comp hardcoding. A thin BaseAgent
wrapper (`GeologyAssemble`) follows the pack convention so it can be driven from the fleet.
"""
from __future__ import annotations
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .base import BaseAgent
except Exception:  # noqa: BLE001 — allow standalone import in scripts/tests
    BaseAgent = object

# formation-top surfaces are train-only → never used as features
FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
BAND_FT = 120.0  # typewell GR-match search band around tvt_ps


def _list_pairs(data_dir):
    pairs = []
    for hp in sorted(glob.glob(os.path.join(data_dir, "*__horizontal_well.csv"))):
        wid = os.path.basename(hp).split("__")[0]
        tp = os.path.join(data_dir, f"{wid}__typewell.csv")
        if os.path.exists(tp):
            pairs.append((wid, hp, tp))
    return pairs


def _typewell_match(gr_ff, tvt_ps, tw):
    """cand_tvt[i] = typewell TVT (within ±BAND_FT of tvt_ps) whose GR is nearest to gr_ff[i]."""
    tw = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    t = tw["TVT"].to_numpy(float)
    g = tw["GR"].to_numpy(float)
    n = len(gr_ff)
    if len(t) < 5:
        return np.full(n, tvt_ps, float)
    band = (t >= tvt_ps - BAND_FT) & (t <= tvt_ps + BAND_FT)
    if band.sum() < 5:
        band = np.ones_like(t, dtype=bool)
    t_b, g_b = t[band], g[band]
    cand = np.full(n, tvt_ps, float)
    valid = np.isfinite(gr_ff)
    if valid.any():
        d = np.abs(gr_ff[valid][:, None] - g_b[None, :])  # (n_valid, n_band)
        cand[valid] = t_b[np.argmin(d, axis=1)]
    return cand


def _rolling(a, w, fn):
    s = pd.Series(a)
    r = s.rolling(w, min_periods=1, center=True)
    return getattr(r, fn)().to_numpy()


def _well_rows(wid, hp, tp, training):
    h = pd.read_csv(hp)
    if "TVT_input" not in h.columns:
        return None
    n = len(h)
    pred = h["TVT_input"].isna().to_numpy()
    if training:
        pred &= h["TVT"].notna().to_numpy()
    if pred.sum() == 0:
        return None
    ps = int(np.argmax(h["TVT_input"].isna().to_numpy()))  # first NaN row = PS
    j = max(ps - 1, 0)
    tvt_ps = float(h["TVT"].iloc[j]) if training else float(h["TVT_input"].iloc[j])
    if not np.isfinite(tvt_ps):
        tvt_ps = float(np.nanmedian(h["TVT_input"])) if h["TVT_input"].notna().any() else 0.0
    x_ps, y_ps, z_ps, md_ps = (float(h[c].iloc[j]) for c in ["X", "Y", "Z", "MD"])

    md = h["MD"].to_numpy(float)
    X, Y, Z = (h[c].to_numpy(float) for c in ["X", "Y", "Z"])
    gr = h["GR"].to_numpy(float)
    gr_ff = pd.Series(gr).ffill().bfill().to_numpy()
    if not np.isfinite(gr_ff).all():
        gr_ff = np.nan_to_num(gr_ff, nan=float(np.nanmedian(gr_ff)) if np.isfinite(np.nanmedian(gr_ff)) else 0.0)
    gr_ps = gr_ff[j]

    cand = _typewell_match(gr_ff, tvt_ps, pd.read_csv(tp))
    incl = np.gradient(Z, md) if n > 1 else np.zeros(n)
    gr_grad = np.gradient(gr_ff, md) if n > 1 else np.zeros(n)

    df = pd.DataFrame({
        "id": [f"{wid}_{i}" for i in range(n)],
        "well": wid,
        "dmd": md - md_ps,
        "dz": Z - z_ps,
        "horiz_disp": np.hypot(X - x_ps, Y - y_ps),
        "incl": incl,
        "abs_z": Z,
        "md_ps": md_ps,
        "tvt_ps": tvt_ps,
        "gr": gr_ff,
        "gr_isnan": np.isnan(gr).astype(np.int8),
        "gr_minus_ps": gr_ff - gr_ps,
        "gr_grad": gr_grad,
        "gr_roll5": _rolling(gr_ff, 5, "mean"),
        "gr_roll25": _rolling(gr_ff, 25, "mean"),
        "gr_std25": np.nan_to_num(_rolling(gr_ff, 25, "std")),
        "cand_tvt": cand,
    })
    if training:
        # TARGET = residual drift dtvt = TVT - tvt_ps (bounded, well-independent so trees generalize to
        # unseen wells; absolute TVT varies ~600ft across wells and cannot be extrapolated). Reconstruct
        # absolute TVT = pred + tvt_ps at submission time. RMSE(dtvt) == competition RMSE(absolute TVT).
        df["tvt"] = h["TVT"].to_numpy(float) - tvt_ps
    return df.loc[pred].reset_index(drop=True)


def geology_assemble(data_dir, out_path, training=True, limit=None):
    """Assemble a flat per-row table over the prediction region. Returns the written path."""
    pairs = _list_pairs(data_dir)
    if limit:
        pairs = pairs[:limit]
    frames = [r for wid, hp, tp in pairs if (r := _well_rows(wid, hp, tp, training)) is not None]
    if not frames:
        raise RuntimeError(f"geology_assemble: no usable wells in {data_dir}")
    df = pd.concat(frames, ignore_index=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return str(out)


FEATURES = ["dmd", "dz", "horiz_disp", "incl", "abs_z", "md_ps", "tvt_ps", "gr", "gr_isnan",
            "gr_minus_ps", "gr_grad", "gr_roll5", "gr_roll25", "gr_std25", "cand_tvt"]


class GeologyAssemble(BaseAgent):
    name = "geology-assemble"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        data_dir = spec.get("data_dir")
        out_path = spec.get("out")
        if not data_dir or not out_path:
            return self.escalate(worker, "leader",
                                 "geology-assemble needs spec keys ['data_dir','out'] — none provided")
        p = geology_assemble(data_dir, out_path, training=bool(spec.get("training", True)),
                             limit=spec.get("limit"))
        n = sum(1 for _ in open(p)) - 1
        msg = f"geology-assemble: wrote {n} rows → {Path(p).name}"
        self.log(msg, kind="finding", recommendation="feed to tab-autobaseline as cfg.data")
        return self.done({"rows": n, "path": p}, msg)


_AGENT = GeologyAssemble() if BaseAgent is not object else None


def run(q, worker):
    return _AGENT.run(q, worker)
