"""pseudo-label — self-training / pseudo-labeling, the SINGLE most repeated lever in the 2025-26 top
solutions (birdclef 0.87→0.93 via Noisy-Student; also child-mind, s5e4/s5e11, byu, rsna, isic, wsdm, eedi).
Reusable across tabular/image/audio/llm: given a model's test predictions, select the CONFIDENT ones as
extra training labels (optionally power/temperature-transformed to denoise), and hand back the augmented
index so the pack retrains on train+pseudo.

Modality-agnostic: operates on prediction arrays from the CompConfig-driven trainer, not on any comp's raw
data. Pure numpy.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent
from . import comp_config as CC


def confidence(pred, kind="classification"):
    """Per-row confidence. Classification: max class prob (binary→|p-0.5|*2). Regression: not used (all kept)."""
    p = np.asarray(pred, float)
    if kind == "regression":
        return np.ones(len(p))
    if p.ndim == 1:
        return np.abs(p - 0.5) * 2.0            # binary prob → [0,1] confidence
    return p.max(1)


def temperature_soft(prob, T=0.5):
    """Power/temperature transform of soft labels to denoise pseudo-labels (birdclef used ~1/0.6). T<1 sharpens."""
    p = np.asarray(prob, float)
    if p.ndim == 1:
        a = np.clip(p, 1e-6, 1 - 1e-6) ** (1.0 / T)
        b = (1 - np.clip(p, 1e-6, 1 - 1e-6)) ** (1.0 / T)
        return a / (a + b)
    q = np.clip(p, 1e-6, None) ** (1.0 / T)
    return q / q.sum(1, keepdims=True)


def select_pseudo(test_pred, kind="classification", conf_threshold=0.9, max_frac=0.5, T=None,
                  confidence_thresh=None):
    """Return (idx, pseudo_labels, info). idx = test rows confident enough to pseudo-label.
    Hard labels for classification (argmax/threshold); soft (optionally temperature-sharpened) if T set.
    confidence_thresh: alias for conf_threshold (takes precedence when provided)."""
    if confidence_thresh is not None:
        conf_threshold = float(confidence_thresh)
    p = np.nan_to_num(np.asarray(test_pred, float), nan=0.0, posinf=1.0, neginf=0.0)
    if len(p) == 0:
        return np.array([], int), np.array([], float), {"n_test": 0, "n_selected": 0, "frac": 0.0,
                                                        "conf_threshold": conf_threshold, "soft": bool(T)}
    conf = confidence(p, kind)
    order = np.argsort(-conf)
    keep = conf >= conf_threshold
    # cap at max_frac of test to avoid drift
    cap = int(max_frac * len(p))
    idx = order[np.isin(order, np.where(keep)[0])][:cap] if cap > 0 else np.array([], int)
    if kind == "regression":
        labels = p[idx]
    else:
        sp = temperature_soft(p, T) if T else p
        labels = (sp[idx] if T else (sp.argmax(1) if sp.ndim > 1 else (sp[idx] >= 0.5).astype(int)))
        if T is None:
            labels = sp.argmax(1)[idx] if sp.ndim > 1 else (sp[idx] >= 0.5).astype(int)
    info = {"n_test": int(len(p)), "n_selected": int(len(idx)),
            "frac": round(len(idx) / max(len(p), 1), 4), "conf_threshold": conf_threshold, "soft": bool(T)}
    return idx, labels, info


class PseudoLabel(BaseAgent):
    name = "pseudo-label"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        missing = [k for k in ("test_pred",) if k not in spec]
        if missing:
            return self.escalate(worker, "leader", f"pseudo-label needs spec keys {missing} — none provided")
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else None
        kind = spec.get("kind") or ("regression" if (cfg and cfg.metric in ("rmse", "rmsle", "mae", "r2", "smape")) else "classification")
        thr = spec.get("confidence_thresh", spec.get("conf_threshold", 0.9))
        idx, labels, info = select_pseudo(np.asarray(spec["test_pred"], float), kind=kind,
                                          conf_threshold=float(thr),
                                          max_frac=float(spec.get("max_frac", 0.5)),
                                          T=spec.get("temperature"))
        msg = (f"pseudo-label: selected {info['n_selected']}/{info['n_test']} confident test rows "
               f"({info['frac']*100:.1f}%) @conf>={info['conf_threshold']} → add to train and retrain.")
        self.log(msg, kind="finding", recommendation="retrain on train+pseudo; keep only if paired_delta lifts CV")
        return self.done({"pseudo_idx": idx.tolist(), "pseudo_labels": np.asarray(labels).tolist(), **info}, msg)


_AGENT = PseudoLabel()


def run(q, worker):
    return _AGENT.run(q, worker)
