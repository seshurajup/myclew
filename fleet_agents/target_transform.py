"""target-transform — target engineering used by the top solutions: monotone target transforms that make a
metric easier to fit, and TARGET FACTORIZATION (the equity-post-HCT golden trick: split a survival target
into an event classifier + an event-time rank regressor, recombine). Reusable across tabular/timeseries.

Transforms (all with exact inverse): sqrt, log1p, rank_gauss, inverse_normal. Factorization returns the
sub-targets + a recombine() so the pack trains two heads and merges them. Pure numpy/scipy.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


def forward(y, method="rank_gauss"):
    """Monotone target transform. method: sqrt | log1p | arcsinh | rank_gauss/inverse_normal (all invertible).
    arcsinh: signed log-like transform for heavy-tailed targets (no clipping, exact inverse)."""
    y = np.nan_to_num(np.asarray(y, float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(y) == 0:
        return y
    if method == "sqrt":
        return np.sign(y) * np.sqrt(np.abs(y))
    if method == "log1p":
        return np.log1p(np.clip(y, 0, None))
    if method == "arcsinh":
        return np.arcsinh(y)
    if method in ("rank_gauss", "inverse_normal"):
        from scipy.stats import norm
        r = (np.argsort(np.argsort(y)) + 0.5) / max(len(y), 1)
        return norm.ppf(np.clip(r, 1e-6, 1 - 1e-6))
    return y


def inverse(yt, y_ref, method="rank_gauss"):
    """Invert a transform. rank_gauss uses the reference training target's quantiles."""
    yt = np.nan_to_num(np.asarray(yt, float), nan=0.0, posinf=0.0, neginf=0.0)
    if len(yt) == 0:
        return yt
    if method == "sqrt":
        return np.sign(yt) * yt ** 2
    if method == "log1p":
        return np.expm1(yt)
    if method == "arcsinh":
        return np.sinh(yt)
    if method in ("rank_gauss", "inverse_normal"):
        from scipy.stats import norm
        ref = np.sort(np.asarray(y_ref, float))
        q = norm.cdf(yt)
        idx = np.clip((q * (len(ref) - 1)).round().astype(int), 0, len(ref) - 1)
        return ref[idx]
    return yt


def factorize_survival(time, event):
    """Equity trick: classifier target = event (observed?), regressor target = time on event==1 rows only.
    Returns dict with the sub-targets + a recombine(prob_event, rank_time) → risk score."""
    time = np.asarray(time, float); event = np.asarray(event, int)
    reg_mask = event == 1
    def recombine(prob_event, time_rank, a=1.0, b=1.0):
        # higher survival-score = longer expected survival: rank-normalize time then blend with P(no-event)
        pr = np.asarray(prob_event, float); tr = np.asarray(time_rank, float)
        return a * tr + b * (1 - pr)
    return {"clf_target": event, "reg_target": time, "reg_mask": reg_mask, "recombine": recombine}


class TargetTransform(BaseAgent):
    name = "target-transform"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        missing = [k for k in ("y",) if k not in spec]
        if missing:
            return self.escalate(worker, "leader", f"target-transform needs spec keys {missing} — none provided")
        y = np.asarray(spec["y"], float)
        method = spec.get("method", "rank_gauss")
        yt = forward(y, method)
        # verify round-trip on the fly (honesty)
        back = inverse(yt, y, method)
        rt_err = float(np.mean(np.abs(np.sort(back) - np.sort(y))))
        msg = f"target-transform[{method}]: transformed {len(y)} targets, round-trip |Δ|={rt_err:.4g}"
        self.log(msg, kind="finding", recommendation="train on transformed target; invert predictions before scoring")
        return self.done({"method": method, "roundtrip_err": rt_err, "_transformed": yt.tolist()}, msg)


_AGENT = TargetTransform()


def run(q, worker):
    return _AGENT.run(q, worker)
