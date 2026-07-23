"""finance_pack — the portfolio/forecasting levers the one-by-one pass found (hull-tactical, mitsui,
jane-street). All pure numpy, offline-verified, CompConfig-agnostic:

  • portfolio-position-sizer          — turn an alpha signal into a RISK-managed allocation: volatility
                                        targeting + tanh signal→position mapping + leverage clip (hull: the
                                        winning lever was position sizing, not prediction accuracy).
  • market-odds-blend                 — ingest external betting odds: moneyline→no-vig implied probability,
                                        then tier-blend with the model prediction by confidence (march-mania).
  • forecast-drivers-then-derive      — forecast the raw driver features N steps ahead, then apply the KNOWN
                                        target formula (beat direct noisy-target prediction; mitsui).
  • label-lag-anchor-blend            — blend the model output with the mean of recently-revealed labels
                                        (persistence anchor) to survive regime shifts (mitsui streaming).
  • distributional-metric-recalibrator — fit a per-group scale+shift on predictions to correct train/test
                                        distribution shift / metric bias (CSIRO per-state, ariel mu/sigma).
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- portfolio-position-sizer
def size_positions(signal, vol=None, target_vol=0.1, lo=0.0, hi=2.0, gain=1.0, neutral=1.0):
    """Alpha signal → allocation. tanh maps signal to a direction/magnitude; inverse-vol scales it toward a
    target volatility; clip to leverage bounds. Returns allocations in [lo, hi].
    neutral: the no-signal book center (default 1.0 for a long-only [0,2] book; use 0.0 for long/short)."""
    s = np.nan_to_num(np.asarray(signal, float), nan=0.0, posinf=0.0, neginf=0.0)
    pos = np.tanh(gain * s)                                 # bounded signal→position
    if vol is not None:
        v = np.nan_to_num(np.asarray(vol, float), nan=0.0)
        pos = pos * (target_vol / np.clip(v, 1e-6, None))
    # center a long-only [0,2] book around `neutral`, scaled by the (possibly>1) signal
    alloc = neutral + pos
    return np.clip(alloc, lo, hi)


# ---------------------------------------------------------------- market-odds-blend
def no_vig(moneyline):
    """American moneyline odds → no-vig (fair) implied probabilities for a two-outcome market.
    moneyline = (ml_a, ml_b). Returns (p_a, p_b) summing to 1."""
    def implied(ml):
        return 100 / (ml + 100) if ml > 0 else (-ml) / (-ml + 100)
    pa, pb = implied(moneyline[0]), implied(moneyline[1])
    tot = pa + pb
    return pa / tot, pb / tot


def blend_market(model_p, market_p, weight=0.5):
    """Tier-blend model probability with market-implied probability (weight = trust in the market)."""
    return (1 - weight) * np.asarray(model_p, float) + weight * np.asarray(market_p, float)


# ---------------------------------------------------------------- forecast-drivers-then-derive
def derive_from_drivers(driver_forecasts, formula):
    """Apply the KNOWN target formula to forecasted driver features. formula(driver_row)->target."""
    D = np.asarray(driver_forecasts, float)
    return np.array([float(formula(row)) for row in D])


# ---------------------------------------------------------------- label-lag-anchor-blend
def anchor_blend(model_pred, recent_labels, w=0.3):
    """Blend model output with the mean of recently-revealed labels (persistence anchor for regime shift)."""
    rl = np.asarray(recent_labels, float)
    anchor = float(np.nanmean(rl)) if rl.size and np.isfinite(np.nanmean(rl)) else 0.0
    return (1 - w) * np.nan_to_num(np.asarray(model_pred, float)) + w * anchor


# ---------------------------------------------------------------- distributional-metric-recalibrator
def recalibrate_by_group(preds, y_true, groups, min_group=2):
    """Fit a per-group affine (a·pred + b) minimizing squared error to y on OOF — corrects per-group bias/scale.
    Returns (recalibrated preds, {group: (a, b)}).
    min_group: minimum samples in a group to fit a slope; smaller groups get a bias-only shift."""
    preds = np.nan_to_num(np.asarray(preds, float)); y = np.nan_to_num(np.asarray(y_true, float))
    g = np.asarray(groups)
    out = preds.copy(); params = {}
    for gid in np.unique(g):
        m = g == gid
        if m.sum() >= max(2, int(min_group)) and np.std(preds[m]) > 1e-9:
            a, b = np.polyfit(preds[m], y[m], 1)
        else:
            a, b = 1.0, (float(np.mean(y[m] - preds[m])) if m.any() else 0.0)
        out[m] = a * preds[m] + b; params[str(gid)] = (float(a), float(b))
    return out, params


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class PositionSizer(_B):
    name = "portfolio-position-sizer"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("signal",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"portfolio-position-sizer needs spec keys {missing} — none provided")
        alloc = size_positions(s["signal"], s.get("vol"), float(s.get("target_vol", 0.1)),
                                                 float(s.get("lo", 0.0)), float(s.get("hi", 2.0)),
                                                 gain=float(s.get("gain", 1.0)), neutral=float(s.get("neutral", 1.0)))
        msg = f"portfolio-position-sizer: allocations in [{alloc.min():.2f},{alloc.max():.2f}] (vol-targeted, clipped)"
        self.log(msg, kind="finding", recommendation="the risk overlay matters more than prediction accuracy (hull)")
        return self.done({"_alloc": alloc.tolist()}, msg)


class MarketOddsBlend(_B):
    name = "market-odds-blend"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("moneyline", "model_p") if k not in s]
        if missing: return self.escalate(worker, "leader", f"market-odds-blend needs spec keys {missing} — none provided")
        pa, pb = no_vig(tuple(s["moneyline"]))
        blended = blend_market(s["model_p"], pa, float(s.get("weight", 0.5)))
        msg = f"market-odds-blend: no-vig p={pa:.3f}/{pb:.3f}; blended with model (w={s.get('weight',0.5)})"
        self.log(msg, kind="finding", recommendation="trust the market more on game-specific short-horizon picks")
        return self.done({"no_vig": [pa, pb], "_blended": np.asarray(blended).tolist()}, msg)


class LabelLagAnchor(_B):
    name = "label-lag-anchor-blend"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("model_pred", "recent_labels") if k not in s]
        if missing: return self.escalate(worker, "leader", f"label-lag-anchor-blend needs spec keys {missing} — none provided")
        out = anchor_blend(s["model_pred"], s["recent_labels"], float(s.get("w", 0.3)))
        msg = "label-lag-anchor-blend: blended prediction with recent-label persistence anchor"
        self.log(msg, kind="finding", recommendation="raise anchor weight during detected regime shifts")
        return self.done({"_pred": np.asarray(out).tolist()}, msg)


class DistRecalibrator(_B):
    name = "distributional-metric-recalibrator"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("preds", "y", "groups") if k not in s]
        if missing: return self.escalate(worker, "leader", f"distributional-metric-recalibrator needs spec keys {missing} — none provided")
        out, params = recalibrate_by_group(s["preds"], s["y"], s["groups"],
                                                             min_group=int(s.get("min_group", 2)))
        msg = f"distributional-metric-recalibrator: per-group affine correction over {len(params)} groups"
        self.log(msg, kind="finding", recommendation="apply the fitted per-group (a,b) to test predictions")
        return self.done({"params": params, "_preds": out.tolist()}, msg)


class ForecastDrivers(_B):
    name = "forecast-drivers-then-derive"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("driver_forecasts", "coef") if k not in s]
        if missing: return self.escalate(worker, "leader", f"forecast-drivers-then-derive needs spec keys {missing} — none provided")
        D = np.asarray(s["driver_forecasts"], float); coef = np.asarray(s["coef"], float)
        intercept = float(s.get("intercept", 0.0))
        out = D @ coef + intercept                          # known linear target formula over forecasted drivers
        msg = f"forecast-drivers-then-derive: derived {len(out)} targets from forecasted drivers via the known formula"
        self.log(msg, kind="finding", recommendation="forecast the raw drivers (less noisy) then apply the formula")
        return self.done({"_target": np.asarray(out).tolist()}, msg)


_PS = PositionSizer(); _MO = MarketOddsBlend(); _LA = LabelLagAnchor(); _DR = DistRecalibrator(); _FD = ForecastDrivers()


def run_sizer(q, worker): return _PS.run(q, worker)
def run_odds(q, worker): return _MO.run(q, worker)
def run_anchor(q, worker): return _LA.run(q, worker)
def run_recal(q, worker): return _DR.run(q, worker)
def run_drivers(q, worker): return _FD.run(q, worker)
