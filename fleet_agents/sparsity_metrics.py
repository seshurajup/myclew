"""sparsity_metrics — hidden-state SPARSITY as a difficulty / uncertainty signal (MingyuJ666/sparsityLLM,
cot/utils/rank.py::compute_sparsity_metrics). The finding: an LLM's last-token hidden state is SPARSER
(energy concentrated in few dims → high Gini / high top-k-energy / low effective rank) on easy, in-distribution
inputs and DENSER / higher-entropy on hard or OOD inputs. So per-example sparsity is a free, model-internal
difficulty score — usable to order a curriculum (easy→hard), weight samples, or flag OOD without labels.

This module lifts the metric core (pure numpy, no model) so any fleet trainer that can dump a feature/embedding
vector per example gets the six signals, plus a curriculum-ordering helper. The `get_last_hidden_state` part of
the source (which needs a live HF model) is intentionally NOT ported — the fleet computes vectors its own way.

Reusable primitives (numpy, deps = numpy):
  • sparsity_metrics(v)          — l0, top{1,5,10}pct energy ratio, Gini, effective-rank ratio for one vector.
  • batch_sparsity(V)            — the same six metrics per row of a (N, D) matrix (vectorized).
  • difficulty_score(V, ...)     — combine metrics into ONE scalar (sparse=easy → low score) per example.
  • curriculum_order(V, ...)     — argsort easy→hard for curriculum-learning sample ordering.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- single-vector metrics (mirrors the source)
def sparsity_metrics(v) -> dict:
    """Six sparsity metrics for one vector v (any length). Byte-faithful to sparsityLLM's compute_sparsity_metrics:
      l0_norm        — fraction of |dims| above mean+1σ (few big dims → small l0 → sparse).
      top{1,5,10}pct — energy (Σ|·|) share of the top 1/5/10% dims (higher → sparser).
      gini           — Gini coefficient of |v| (higher → more unequal → sparser).
      effective_rank — exp(entropy of L1-normalized |v|) / D  (lower → sparser, fewer effective dims)."""
    h = np.asarray(v, float).ravel()
    D = h.shape[0]
    ah = np.abs(h)
    total = ah.sum()
    def topk_ratio(p):
        k = max(1, int(D * p))
        return float(np.sort(ah)[-k:].sum() / total) if total > 0 else 0.0
    mean, std = ah.mean(), ah.std()
    l0 = float((ah > (mean + 1.0 * std)).mean())
    # Gini on sorted |v|:  (2 Σ i·x_i)/(n Σ x_i) - (n+1)/n
    s = np.sort(ah); n = D; idx = np.arange(1, n + 1)
    gini = float((2.0 * (idx * s).sum()) / (n * s.sum()) - (n + 1) / n) if s.sum() > 0 else 0.0
    # effective rank via L1-entropy
    p = ah / total if total > 0 else np.full(D, 1.0 / D)
    p = p + 1e-10
    eff = float(np.exp(-(p * np.log(p)).sum()) / D)
    return {"l0_norm": l0, "top1pct_ratio": topk_ratio(0.01), "top5pct_ratio": topk_ratio(0.05),
            "top10pct_ratio": topk_ratio(0.10), "gini": gini, "effective_rank": eff}


def batch_sparsity(V) -> list:
    """Per-row metrics for a (N, D) matrix. Returns a list of N metric dicts."""
    V = np.asarray(V, float)
    if V.ndim == 1:
        V = V[None, :]
    return [sparsity_metrics(row) for row in V]


# ---------------------------------------------------------------- difficulty signal + curriculum
def difficulty_score(V, weights=None) -> np.ndarray:
    """One scalar difficulty per example (higher = harder). Sparse hidden states = easy/in-distribution, so
    difficulty rises with DENSITY: we use (1-gini), (1-top5pct), effective_rank, l0 — all larger when denser.
    Returns a (N,) array, min-max normalized to [0,1] across the batch."""
    mets = batch_sparsity(V)
    g = np.array([m["gini"] for m in mets]); t5 = np.array([m["top5pct_ratio"] for m in mets])
    er = np.array([m["effective_rank"] for m in mets]); l0 = np.array([m["l0_norm"] for m in mets])
    w = weights or {"gini": 1.0, "top5": 1.0, "eff": 1.0, "l0": 1.0}
    raw = w["gini"] * (1 - g) + w["top5"] * (1 - t5) + w["eff"] * er + w["l0"] * l0
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)


def curriculum_order(V, hardest_first=False) -> np.ndarray:
    """Index order for curriculum learning. Default easy→hard (ascending difficulty), the sparsityLLM recipe."""
    d = difficulty_score(V)
    order = np.argsort(d)
    return order[::-1].copy() if hardest_first else order


# ---------------------------------------------------------------- agent
class SparsityMetrics(BaseAgent):
    name = "sparsity-metrics"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        rng = np.random.RandomState(int(s.get("seed", 0)))
        N = int(s.get("n", 200)); D = int(s.get("dim", 512))
        # synthetic: half "easy" (sparse: few large dims), half "hard" (dense gaussian) — sparsity must separate them
        easy = np.zeros((N // 2, D)); hard = rng.randn(N - N // 2, D)
        for i in range(easy.shape[0]):
            k = rng.randint(3, 10); easy[i, rng.choice(D, k, replace=False)] = rng.randn(k) * 5
        V = np.vstack([easy, hard]); label_hard = np.array([0] * easy.shape[0] + [1] * hard.shape[0])
        d = difficulty_score(V)
        # does the difficulty score rank dense(hard) above sparse(easy)?  AUC-style separation
        eh = d[label_hard == 1]; el = d[label_hard == 0]
        auc = float((eh[:, None] > el[None, :]).mean())
        msg = (f"sparsity-metrics: {N} vectors dim {D} — hidden-state density difficulty score separates "
               f"dense(hard) from sparse(easy) at AUC={auc:.3f} (sparse=easy/in-dist, dense=hard/OOD). "
               f"Six metrics per example (l0/top-k-energy/Gini/eff-rank); curriculum_order() gives easy→hard "
               f"ordering — a free label-less difficulty signal (sparsityLLM)")
        self.log(msg, kind="finding",
                 recommendation="dump a per-example embedding/feature vector and use difficulty_score for "
                                "curriculum ordering or sample weighting; high density → OOD/hard flag")
        return self.done({"auc": auc, "dim": D, "n": N}, msg)


_AGENT = SparsityMetrics()


def run_sparsity(q, worker):
    return _AGENT.run(q, worker)
