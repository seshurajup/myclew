"""imbalance_sampler_pack — TEMPERED class-balanced sampling, distilled from the BirdCLEF+ 2025 2nd-place
solution (VSydorskyy). Heavy class imbalance (long-tailed labels) is one of the most common Kaggle wounds;
the winning fix was NOT hard 1/freq inverse-frequency sampling (which over-samples noisy rare classes into
oblivion) but a TEMPERED interpolation between the natural distribution and full class-balance, via a single
`power` knob (their "SqrtBalancing" == power -0.5, "EqualBalancing" == power -1.0). This gives a WeightedRandom
sample-weight vector. Also ships the Cui et al. "effective number of samples" reweighting. No sampler/weight
primitive existed in the fleet (train_tricks_pack has focal loss for imbalance at the LOSS side, nothing on
the SAMPLING side). Pure-numpy, deterministic resampling for reproducible CV.

  • class-balance-sampler — per-sample sampling weights = count[class]^power (power in [-1,0]: 0=natural,
                            -0.5=sqrt-tempered, -1=fully class-balanced), optional Cui effective-number mode
                            (beta), returned normalised + a deterministic multinomial resample of row indices.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


def class_counts(labels):
    """Return (unique_classes, count_per_class_dict) for a 1-D array of integer/str class labels."""
    labels = np.asarray(labels)
    uniq, cnt = np.unique(labels, return_counts=True)
    return uniq, dict(zip(uniq.tolist(), cnt.tolist()))


def tempered_class_weight(counts, power=-0.5):
    """Per-CLASS weight = count^power. power=0 → natural (all 1 → sampling follows data frequency);
    power=-1 → 1/count (fully class-balanced: every class gets equal TOTAL sampling mass);
    power=-0.5 → sqrt-tempered (the robust middle the winner used). counts: {class: n}."""
    return {c: float(n) ** float(power) for c, n in counts.items()}


def effective_number_weight(counts, beta=0.999):
    """Cui et al. 'class-balanced loss' reweighting: w_c = (1-beta)/(1-beta^n_c), normalised to mean 1.
    Softer than 1/n for large n; beta→0 recovers uniform, beta→1 approaches inverse-frequency."""
    w = {c: (1.0 - beta) / (1.0 - beta ** int(n)) for c, n in counts.items()}
    mean = np.mean(list(w.values()))
    return {c: v / mean for c, v in w.items()} if mean > 0 else w


def sample_weights(labels, power=-0.5, mode="power", beta=0.999):
    """Map each ROW's class to its class weight → per-sample weight vector (normalised to sum=len).
    mode='power' uses tempered_class_weight; mode='effective_number' uses the Cui reweighting."""
    labels = np.asarray(labels)
    _, counts = class_counts(labels)
    cw = effective_number_weight(counts, beta) if mode == "effective_number" else tempered_class_weight(counts, power)
    w = np.array([cw[l] for l in labels.tolist()], dtype=float)
    tot = w.sum()
    return w * (len(w) / tot) if tot > 0 else np.ones_like(w)


def resample_indices(labels, n=None, power=-0.5, mode="power", beta=0.999, seed=0, replace=True):
    """Deterministic multinomial resample of ROW indices proportional to sample_weights (a numpy stand-in for
    torch WeightedRandomSampler). Returns an int index array of length n (default len(labels))."""
    labels = np.asarray(labels)
    w = sample_weights(labels, power=power, mode=mode, beta=beta)
    p = w / w.sum()
    n = len(labels) if n is None else int(n)
    rng = np.random.RandomState(int(seed))
    return rng.choice(len(labels), size=n, replace=replace, p=p)


# ---------------------------------------------------------------- agent
class ClassBalanceSampler(BaseAgent):
    name = "class-balance-sampler"; thread = "S"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("labels",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"class-balance-sampler needs spec keys {missing} — none provided")
        labels = s["labels"]
        power = float(s.get("power", -0.5)); mode = s.get("mode", "power"); beta = float(s.get("beta", 0.999))
        w = sample_weights(labels, power=power, mode=mode, beta=beta)
        _, counts = class_counts(labels)
        idx = None
        if s.get("resample", False):
            idx = resample_indices(labels, n=s.get("n"), power=power, mode=mode, beta=beta,
                                   seed=int(s.get("seed", 0)), replace=bool(s.get("replace", True))).tolist()
        tag = f"effective-number(beta={beta})" if mode == "effective_number" else f"power={power}"
        msg = (f"class-balance-sampler[{tag}]: {len(counts)} classes, imbalance "
               f"{max(counts.values())}:{min(counts.values())} → tempered sample weights "
               f"(range {w.min():.3f}–{w.max():.3f})")
        self.log(msg, kind="finding",
                 recommendation="feed weights to WeightedRandomSampler; power -0.5 (sqrt) is the robust default")
        out = {"weights": w.tolist(), "class_counts": {str(k): v for k, v in counts.items()}, "mode": mode}
        if idx is not None:
            out["resampled_indices"] = idx
        return self.done(out, msg)


_CB = ClassBalanceSampler()


def run(q, worker): return _CB.run(q, worker)
