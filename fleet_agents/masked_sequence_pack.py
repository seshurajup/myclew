"""masked_sequence_pack — LEAKAGE-FREE ops for VARIABLE-LENGTH padded sequences, distilled from the
Kaggle "CMI — Detect Behavior with Sensor Data" 2nd-place solution (Yamato-Arai). Padded batches are
everywhere (audio clips, IMU/sensor windows, event streams, our own per-track temporal features), and the
usual BatchNorm / mean-pool silently mixes PAD positions into the statistics — a real, hard-to-spot leak
that hurts every variable-length model. The winning fix is to make every reduction MASK-AWARE so padding
contributes exactly nothing. None of these existed in the fleet (train_tricks_pack has EMA/SWA/mixup/focal
etc., not masked reductions). All pure-numpy, mirrors the reference torch modules 1:1.

  • masked-sequence-norm — MaskedBatchNorm-style per-channel z-score computed over ONLY the valid timesteps
                           (count = mask.sum()), padded outputs re-zeroed. Statistics are provably identical
                           whether or not garbage padding is appended → no train/serve skew.
  • masked-sequence-pool — mean / max / attention pooling that ignore PAD: masked mean (sum/valid_count),
                           masked max (pad→ -inf), masked softmax-attention (pad→ -1e9 before softmax so
                           weights sum to 1 over valid positions only). The SE-block / attention-head trick
                           the winner used to summarise a padded sequence into one vector without leakage.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent

_NEG = -1e9


def _as_bcl(x, mask):
    """Return x as (N,C,L) float and mask as (N,L) {0,1} float, validating shapes."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 3:
        raise ValueError(f"masked-seq: x must be (N,C,L), got shape {x.shape}")
    m = np.asarray(mask, dtype=float)
    if m.ndim != 2 or m.shape[0] != x.shape[0] or m.shape[1] != x.shape[2]:
        raise ValueError(f"masked-seq: mask must be (N,L)=({x.shape[0]},{x.shape[2]}), got {m.shape}")
    return x, m


# ---------------------------------------------------------------- masked normalization (MaskedBatchNorm)
def masked_zscore(x, mask, eps=1e-5, return_stats=False):
    """Per-channel z-score over ONLY valid timesteps. x=(N,C,L), mask=(N,L). Padded outputs set to 0.
    Statistics (mean/var per channel) are pooled over batch AND time of the valid positions — exactly the
    MaskedBatchNorm training-mode computation. Leakage-free: appending mask=0 columns cannot change stats."""
    x, m = _as_bcl(x, mask)
    m3 = m[:, None, :]                                   # (N,1,L) broadcast over channels
    n = m.sum()                                          # total valid positions (scalar, shared per channel)
    if n <= 0:
        out = np.zeros_like(x)
        return (out, {"mean": np.zeros(x.shape[1]), "var": np.zeros(x.shape[1])}) if return_stats else out
    mean = (x * m3).sum(axis=(0, 2)) / n                 # (C,)
    var = (((x - mean[None, :, None]) * m3) ** 2).sum(axis=(0, 2)) / n
    out = (x - mean[None, :, None]) / np.sqrt(var[None, :, None] + eps)
    out = out * m3                                       # re-zero padding (never trust pad outputs)
    return (out, {"mean": mean, "var": var}) if return_stats else out


# ---------------------------------------------------------------- masked pooling
def masked_mean_pool(x, mask):
    """(N,C,L),(N,L) -> (N,C): mean over valid timesteps per sample (sum / valid_count)."""
    x, m = _as_bcl(x, mask)
    m3 = m[:, None, :]
    cnt = m.sum(axis=1)[:, None]                         # (N,1) valid length per sample
    cnt = np.where(cnt <= 0, 1.0, cnt)
    return (x * m3).sum(axis=2) / cnt


def masked_max_pool(x, mask):
    """(N,C,L),(N,L) -> (N,C): max over valid timesteps (padding forced to -inf so it never wins)."""
    x, m = _as_bcl(x, mask)
    neg = np.where(m[:, None, :] > 0, x, -np.inf)
    out = neg.max(axis=2)
    return np.where(np.isfinite(out), out, 0.0)          # all-pad sample -> 0


def masked_softmax(scores, mask):
    """Softmax of scores=(N,L) over the L axis, with PAD positions (mask==0) forced to weight ~0.
    Uses the -1e9 pre-softmax fill (the reference PhaseAttention trick); weights sum to 1 over valid."""
    scores = np.asarray(scores, dtype=float)
    m = np.asarray(mask, dtype=float)
    s = np.where(m > 0, scores, _NEG)
    s = s - s.max(axis=1, keepdims=True)                 # stability
    e = np.exp(s) * (m > 0)                              # kill pad exactly (exp(-1e9)~0, force 0)
    denom = e.sum(axis=1, keepdims=True)
    denom = np.where(denom <= 0, 1.0, denom)
    return e / denom


def masked_attention_pool(x, mask, scores):
    """(N,C,L),(N,L),(N,L) -> (N,C): attention pooling where the attention weights are a masked softmax of
    `scores` over valid timesteps (padding gets zero weight). Summarise a padded sequence into one vector."""
    x, m = _as_bcl(x, mask)
    w = masked_softmax(scores, m)                        # (N,L)
    return (x * w[:, None, :]).sum(axis=2)               # (N,C)


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "S"; kind = "finding"


class MaskedSequenceNorm(_B):
    name = "masked-sequence-norm"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("x", "mask") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"masked-sequence-norm needs spec keys {missing} — none provided")
        out, stats = masked_zscore(s["x"], s["mask"], eps=float(s.get("eps", 1e-5)), return_stats=True)
        C = out.shape[1]
        msg = (f"masked-sequence-norm: z-scored {C} channels over valid timesteps only "
               f"(MaskedBatchNorm) — padding excluded from mean/var → no train/serve skew")
        self.log(msg, kind="finding", recommendation="use in place of BatchNorm1d for padded variable-length batches")
        return self.done({"normalized": out.tolist(), "mean": stats["mean"].tolist(),
                          "var": stats["var"].tolist()}, msg)


class MaskedSequencePool(_B):
    name = "masked-sequence-pool"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("x", "mask") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"masked-sequence-pool needs spec keys {missing} — none provided")
        how = s.get("how", "mean")
        if how == "mean":
            pooled = masked_mean_pool(s["x"], s["mask"])
        elif how == "max":
            pooled = masked_max_pool(s["x"], s["mask"])
        elif how == "attention":
            pooled = masked_attention_pool(s["x"], s["mask"], s["scores"])
        else:
            raise ValueError(f"masked-sequence-pool: unknown how={how!r} (mean|max|attention)")
        msg = (f"masked-sequence-pool[{how}]: summarised padded sequence → {pooled.shape[1]}-d vector/sample "
               f"(padding contributes nothing)")
        self.log(msg, kind="finding", recommendation="masked pooling head for any variable-length sequence model")
        return self.done({"pooled": pooled.tolist(), "how": how}, msg)


_MN = MaskedSequenceNorm(); _MP = MaskedSequencePool()


def run_norm(q, worker): return _MN.run(q, worker)
def run_pool(q, worker): return _MP.run(q, worker)
