"""pf_noise_id — SYSTEM IDENTIFICATION of the particle-filter's noise parameters, the mathematician's
alternative to sweeping. The PF is a linear-Gaussian state-space model in (s=TVT+Z, v=ds/dMD) with a
typewell GR measurement. On the KNOWN heel the state s is observed, so the process/measurement variances
are estimated by MLE / method-of-moments — per well, one-shot, no held-out sweep, no leak (heel only):

  q_v = mean_i (Δv_i)^2 / Δ_i         rate-innovation variance   (v_i = Δs_i/Δ_i)
  q_s = mean_i (Δ²s_i)^2 / Δ_i        curvature the const-velocity model can't explain (2nd difference)
  r   = mean_i (GR_i - h(TVT_i))^2  +  (1/12) mean_i (h'_i·δ)^2     heel residual + trapezoidal interp-var

Returns per-well (pn, vn, gs) = (sqrt(q_s), sqrt(q_v), sqrt(r)) for the PF, replacing the global constants
(PN=0.005, VN=0.002) + the empirical gs×1.3 fudge with data-driven, per-well-adaptive values.
"""
from __future__ import annotations
import numpy as np

def identify(hw, tw, pn_floor=0.005, vn_floor=0.002):
    kn = hw["TVT_input"].notna().to_numpy()
    if kn.sum() < 12:
        return dict(pn=pn_floor, vn=vn_floor, gs=None, n=int(kn.sum()))
    MD = hw["MD"].to_numpy(float); Z = hw["Z"].to_numpy(float); T = hw["TVT_input"].to_numpy(float)
    GR = hw["GR"].to_numpy(float)
    ki = np.where(kn)[0]
    s = T[ki] + Z[ki]; md = MD[ki]
    d = np.diff(md); ok = d > 0
    if ok.sum() < 8:
        return dict(pn=pn_floor, vn=vn_floor, gs=None, n=int(kn.sum()))
    dd = d[ok]
    # Integrated-random-walk system ID: q_s and q_v are ENTANGLED in the differenced series, so disentangle
    # via the autocovariance of the 2nd difference. With ~uniform step h (MD stations), unit-normalise per h:
    h = float(np.median(dd))
    d2s = (s[2:] - 2 * s[1:-1] + s[:-2]) / max(h, 1e-6)     # ~ per-unit-depth 2nd difference
    if len(d2s) < 8:
        return dict(pn=pn_floor, vn=vn_floor, gs=None, n=int(kn.sum()))
    d2s = d2s - d2s.mean()
    g0 = float(np.mean(d2s * d2s))                          # Var(Δ²s)          = q_v + 2 q_s
    g1 = float(np.mean(d2s[1:] * d2s[:-1]))                 # Cov lag-1         = -q_s
    q_s = max(-g1, 0.0)                                     # q_s = -γ₁
    q_v = max(g0 + 2.0 * g1, 0.0)                           # q_v = γ₀ + 2γ₁
    # r: heel GR residual + interpolation-error variance
    tw_s = tw.sort_values("TVT"); tvt = tw_s["TVT"].to_numpy(float); g = tw_s["GR"].to_numpy(float)
    m = np.isfinite(tvt) & np.isfinite(g); tvt, g = tvt[m], g[m]
    h_at = np.interp(T[ki], tvt, g)
    resid = GR[ki] - h_at
    res_var = float(np.nanvar(resid[np.isfinite(resid)]))
    dtvt = np.median(np.diff(tvt)) if len(tvt) > 1 else 0.5
    hp = np.gradient(g, tvt)                                # h'(TVT)
    hp_at = np.interp(T[ki], tvt, hp)
    interp_var = float(np.mean((hp_at * dtvt) ** 2) / 12.0)
    r = res_var + interp_var
    return dict(pn=max(np.sqrt(q_s), pn_floor), vn=max(np.sqrt(q_v), vn_floor),
                gs=np.sqrt(max(r, 1.0)), n=int(kn.sum()),
                q_s=q_s, q_v=q_v, res_std=np.sqrt(res_var), interp_std=np.sqrt(interp_var))


# Global medians of the physical estimates (filled by calibrate()) so per-well noise can be applied as a
# RELATIVE modulation anchored at the PF's tuned scale, not an absolute (which over-diffuses — see ledger
# exp12). PN_w = PN_opt * (q_s_w / median q_s)^0.5 clamped; keeps per-well adaptation, fixes the scale.
_MED = {"qs": None, "qv": None}
def calibrate(estimates):
    import statistics as _st
    qs = [e["pn"]**2 for e in estimates if e.get("gs") is not None]
    qv = [e["vn"]**2 for e in estimates if e.get("gs") is not None]
    _MED["qs"] = _st.median(qs) if qs else 1.0; _MED["qv"] = _st.median(qv) if qv else 1.0
    return _MED

def identify_relative(hw, tw, pn_opt=0.0135, vn_opt=0.0054, lo=0.4, hi=2.5):
    """Per-well noise as a RELATIVE modulation around the PF-tuned optimum (pn_opt≈PN*2.7, vn_opt≈VN*2.7):
    scale by sqrt(q_w/median_q), clamped to [lo,hi]. Requires calibrate() over the well set first."""
    import numpy as _np
    est = identify(hw, tw)
    mqs = _MED["qs"] or est["pn"]**2; mqv = _MED["qv"] or est["vn"]**2
    fs = float(_np.clip((est["pn"]**2/max(mqs,1e-12))**0.5, lo, hi))
    fv = float(_np.clip((est["vn"]**2/max(mqv,1e-12))**0.5, lo, hi))
    return dict(pn=pn_opt*fs, vn=vn_opt*fv, gs=est["gs"], n=est["n"], fs=fs, fv=fv)
