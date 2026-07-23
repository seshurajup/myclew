"""geology_postprocess — reusable honest-engine post-processors for wellbore geosteering (Track B).

Ported faithfully from the public 7159 notebook's transferable (non-leakage) stack:
  * robust deg-4 IRLS drift projection (`project_well_A`) — stabilises PF drift, blend 0.75 proj / 0.25 raw
  * Savitzky-Golay smoothing of the per-well eval prediction

These operate on a horizontal-well DataFrame + the PF eval prediction (absolute TVT), returning a
refined absolute-TVT eval prediction. No leakage, no train-duplicate lookup.
"""
from __future__ import annotations
import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:  # noqa: BLE001
    savgol_filter = None


def robfit(s, y, deg=4):
    """Tukey-IRLS polynomial fit (robust to wrong-branch excursions). Returns (coef, x0, xs) or const array."""
    s = np.asarray(s, float); y = np.asarray(y, float)
    m = np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    if len(s) < deg + 2:
        return np.full_like(s, np.nanmedian(y) if len(y) else 0.0)
    x0 = s[0]; xs = (s.max() - s.min()) if s.max() > s.min() else 1.0
    xn = (s - x0) / xs
    coef = np.polyfit(xn, y, deg)
    for _ in range(6):
        res = y - np.polyval(coef, xn)
        sc = 1.4826 * np.median(np.abs(res - np.median(res))) + 1e-6
        w = 1.0 / (1.0 + (res / (4.685 * sc)) ** 2)
        coef = np.polyfit(xn, y, deg, w=w)
    return coef, x0, xs


def project_well(hw, tvt_pred_ev, deg=4, w_proj=0.75):
    """Fit u = TVT+Z-anchor vs MD on the KNOWN section (robust deg-4), extrapolate over the eval
    rows, blend w_proj*projected + (1-w_proj)*raw. Returns refined abs-TVT for eval rows."""
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(kn) < deg + 2 or len(ev) == 0:
        return tvt_pred_ev
    anchor = float(kn["TVT_input"].iloc[-1])
    u_known = kn["TVT_input"].to_numpy(float) + kn["Z"].to_numpy(float) - anchor
    fit = robfit(kn["MD"].to_numpy(float), u_known, deg=deg)
    if not isinstance(fit, tuple):
        return tvt_pred_ev
    coef, x0, xs = fit
    u_ev = np.polyval(coef, (ev["MD"].to_numpy(float) - x0) / xs)
    proj = u_ev + anchor - ev["Z"].to_numpy(float)
    return w_proj * proj + (1.0 - w_proj) * np.asarray(tvt_pred_ev, float)


def sg_smooth(v, window=17, poly=3):
    """Savitzky-Golay smooth a 1-D eval prediction; no-op if too short or scipy missing."""
    v = np.asarray(v, float)
    if savgol_filter is None or len(v) < window:
        return v
    wl = window if window % 2 == 1 else window + 1
    return savgol_filter(v, wl, poly)


def refine_eval(hw, tvt_pred_ev, project=True, smooth=True, deg=4, w_proj=0.75, sg_window=17, sg_poly=3):
    out = np.asarray(tvt_pred_ev, float)
    if project:
        out = project_well(hw, out, deg=deg, w_proj=w_proj)
    if smooth:
        out = sg_smooth(out, window=sg_window, poly=sg_poly)
    return out
