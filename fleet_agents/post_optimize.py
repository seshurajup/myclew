"""post-optimize — metric-specific POST-PROCESSING, a lever in essentially every top solution. Applies the
right transform for the competition metric:
  • qwk_round      — optimize ordinal cut-points on OOF (child-mind QWK 1st-place lever)
  • clip           — clip predictions to a safe range (s5e4: unclipped→RMSE 177; jane-street clip[-5,5])
  • temperature    — temperature-scale logits/probs (rsna T=0.91, wsdm)
  • quantile_thr   — threshold detection heatmap at a quantile of its max (byu/czii peak decision)
  • rank_average   — rank-transform predictions (isic rank-averaging before blend)
Modality-agnostic; wraps math-master. Choose the op via spec['op'] (or auto from the metric).
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent
from . import comp_config as CC
from . import math_master as MM


def auto_op(metric):
    if metric in ("quadratic_weighted_kappa",):
        return "qwk_round"
    if metric in ("rmse", "rmsle", "mae", "r2", "smape"):
        return "clip"
    return "rank_average"


def apply(op, pred, y_true=None, metric=None, **kw):
    """Returns (transformed_pred, info)."""
    p = np.nan_to_num(np.asarray(pred, float), nan=0.0, posinf=0.0, neginf=0.0)
    if p.size == 0:
        return p, {"op": "identity", "empty": True}
    if op == "qwk_round":
        mfn = lambda a, b: CC.score(metric or "quadratic_weighted_kappa", a, b)
        th, best, rounded = MM.optimized_rounder(y_true, p, mfn, n_classes=kw.get("n_classes"))
        return rounded, {"op": op, "thresholds": th, "score": best}
    if op == "clip":
        out = MM.clip_guard(p, lo=kw.get("lo"), hi=kw.get("hi"), train_y=y_true)
        return out, {"op": op, "range": [float(out.min()), float(out.max())]}
    if op == "temperature":
        T = float(kw.get("T", 1.0))
        if p.ndim == 1:
            logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
            out = 1 / (1 + np.exp(-logit / T))
        else:
            logit = np.log(np.clip(p, 1e-9, None)); e = np.exp(logit / T); out = e / e.sum(1, keepdims=True)
        return out, {"op": op, "T": T}
    if op == "quantile_thr":
        qq = float(kw.get("q", 0.56))
        thr = np.quantile(p, qq)
        return (p >= thr).astype(int), {"op": op, "q": qq, "threshold": float(thr)}
    if op == "rank_average":
        r = np.argsort(np.argsort(p)) / max(len(p) - 1, 1)
        return r, {"op": op}
    return p, {"op": "identity"}


class PostOptimize(BaseAgent):
    name = "post-optimize"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        missing = [k for k in ("pred",) if k not in spec]
        if missing:
            return self.escalate(worker, "leader", f"post-optimize needs spec keys {missing} — none provided")
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else None
        metric = spec.get("metric") or (cfg.metric if cfg else None)
        op = spec.get("op") or auto_op(metric or "")
        out, info = apply(op, np.asarray(spec["pred"], float),
                          y_true=np.asarray(spec["y"]) if "y" in spec else None, metric=metric,
                          **{k: spec[k] for k in ("lo", "hi", "T", "q", "n_classes") if k in spec})
        msg = f"post-optimize: op={op} {info}"
        self.log(msg, kind="finding", recommendation="apply the same op to the test predictions before submit")
        return self.done({"info": info, "_pred": np.asarray(out).tolist()}, msg)


_AGENT = PostOptimize()


def run(q, worker):
    return _AGENT.run(q, worker)
