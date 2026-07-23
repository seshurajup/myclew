"""calibrate — probability calibration, used on OOF by the top solutions (equity sigmoid/beta calibration,
rsna temperature scaling). Fits Platt (logistic), isotonic, or temperature calibration on OOF and reports the
Expected Calibration Error before/after so the gain is honest. Modality-agnostic; wraps math-master.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent
from . import math_master as MM


def _beta_calibrate(s, y):
    """Beta calibration (Kull et al.): logistic fit on [log p, log(1-p)] — a flexible 2-parameter map that
    generalizes Platt scaling. Falls back to Platt on any failure."""
    try:
        from sklearn.linear_model import LogisticRegression
        sc = np.clip(s, 1e-6, 1 - 1e-6)
        X = np.column_stack([np.log(sc), -np.log(1 - sc)])
        lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000).fit(X, np.asarray(y, int))
        return lr.predict_proba(X)[:, 1]
    except Exception:  # noqa: BLE001
        return MM.platt_scale(s, y)


def calibrate(scores, y_true, method="isotonic"):
    s = np.nan_to_num(np.asarray(scores, float), nan=0.5, posinf=1.0, neginf=0.0)
    y = np.asarray(y_true)
    if len(s) == 0:
        return s
    if method == "platt":
        out = MM.platt_scale(s, y)
    elif method == "beta":
        out = _beta_calibrate(s, y)
    elif method == "temperature":
        # 1-D temperature search minimizing log-loss
        from scipy.optimize import minimize_scalar
        logit = np.log(np.clip(s, 1e-6, 1 - 1e-6) / np.clip(1 - s, 1e-6, 1 - 1e-6))
        def nll(T):
            p = 1 / (1 + np.exp(-logit / max(T, 1e-3))); p = np.clip(p, 1e-9, 1 - 1e-9)
            return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        T = minimize_scalar(nll, bounds=(0.05, 10), method="bounded").x
        out = 1 / (1 + np.exp(-logit / T))
    else:
        out = MM.isotonic_calibrate(s, y)
    return np.nan_to_num(np.asarray(out, float), nan=0.5, posinf=1.0, neginf=0.0)


def ece(scores, y_true):
    try:
        return float(MM.expected_calibration_error(np.asarray(y_true), np.asarray(scores, float)))
    except Exception:  # noqa: BLE001
        # fallback ECE
        s = np.asarray(scores, float); y = np.asarray(y_true, float)
        bins = np.linspace(0, 1, 11); e = 0.0
        for i in range(10):
            m = (s >= bins[i]) & (s < bins[i + 1] if i < 9 else s <= 1.0)
            if m.sum():
                e += m.mean() * abs(s[m].mean() - y[m].mean())
        return float(e)


class Calibrate(BaseAgent):
    name = "calibrate"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        missing = [k for k in ("scores", "y") if k not in spec]
        if missing:
            return self.escalate(worker, "leader", f"calibrate needs spec keys {missing} — none provided")
        s = np.asarray(spec["scores"], float); y = np.asarray(spec["y"])
        method = spec.get("method", "isotonic")
        before = ece(s, y)
        cal = calibrate(s, y, method)
        after = ece(cal, y)
        msg = f"calibrate[{method}]: ECE {before:.4f} → {after:.4f} (Δ {before-after:+.4f})"
        self.log(msg, kind="finding", recommendation="apply the fitted calibrator to test scores if ECE improved")
        return self.done({"method": method, "ece_before": round(before, 5), "ece_after": round(after, 5),
                          "_calibrated": cal.tolist()}, msg)


_AGENT = Calibrate()


def run(q, worker):
    return _AGENT.run(q, worker)
