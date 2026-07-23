"""moe_quantile_balance — Kimi-K3's "Quantile Balancing" MoE router load-balancer (K3 blog, Stable
LatentMoE §): derive each expert's admission threshold directly from the QUANTILES of its own router-score
column, so expert load balances by CONSTRUCTION — no auxiliary load-balancing loss, no learned per-expert
bias that has to be nudged every step ("eliminating heuristic updates"). This is what lets K3 run an extreme
16-of-896 sparsity without a handful of experts collapsing into hot-spots.

The problem it fixes (pure routing accounting, testable with NO model):
  A softmax/top-k token-choice router sends each token to its k highest-logit experts. When some experts have
  systematically larger logits (miscalibrated columns), those experts are chosen far more often → load skews,
  the hot experts bottleneck compute and the cold experts are starved of gradient. The classic fix is an
  auxiliary balance loss (Switch/GShard) with a sensitive coefficient, or DeepSeek's per-expert bias updated
  by a hand-tuned rule — both are heuristics bolted onto the loss.

Quantile Balancing (this module): map each expert column of scores to its empirical QUANTILE (rank / N) across
the token batch BEFORE top-k. Every expert's score distribution becomes ~Uniform(0,1), so every expert is a
token's top choice equally often in expectation → marginal load is flat with no aux term. The routed set still
respects the signal (a token's relative preference ORDER within its own experts is largely preserved), it just
removes the cross-expert scale bias that caused the imbalance.

Reusable primitives (numpy, no deps):
  • topk_route(logits, k)                — baseline token-choice top-k assignment + per-expert load.
  • quantile_balance_route(logits, k)    — quantile-normalize columns, then top-k → balanced load, no aux loss.
  • load_stats(assign, n_experts)        — load fraction per expert, coefficient of variation, max-load, utilisation.
  • aux_free_saving(...)                 — reports the balance gained WITHOUT adding an aux-loss term.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- core routing math
def _topk_cols(scores, k):
    """Indices of the top-k columns per row (token). scores: (N, E). Returns (N, k) int array."""
    k = int(max(1, min(k, scores.shape[1])))
    return np.argsort(-scores, axis=1)[:, :k]


def topk_route(logits, k=2):
    """Baseline token-choice top-k router. logits: (N tokens, E experts). Returns dict with the (N,k)
    assignment and per-expert token counts. This is the router whose load SKEWS when columns are miscalibrated."""
    logits = np.asarray(logits, float)
    assign = _topk_cols(logits, k)
    return {"assign": assign, "load": np.bincount(assign.ravel(), minlength=logits.shape[1])}


def column_quantiles(logits):
    """Map every expert column to its empirical quantile (rank/N) across tokens → each column ~Uniform(0,1).
    Ties get averaged ranks so identical scores don't bias one expert. Returns (N, E)."""
    logits = np.asarray(logits, float)
    N = logits.shape[0]
    q = np.empty_like(logits)
    for e in range(logits.shape[1]):
        col = logits[:, e]
        order = np.argsort(col, kind="mergesort")
        ranks = np.empty(N, float); ranks[order] = np.arange(N)
        # average ties
        _, inv, cnt = np.unique(col, return_inverse=True, return_counts=True)
        csum = np.cumsum(cnt); starts = csum - cnt
        avg = (starts + csum - 1) / 2.0
        ranks = avg[inv]
        q[:, e] = (ranks + 0.5) / N                      # (0,1), symmetric
    return q


def quantile_balance_route(logits, k=2):
    """Kimi-K3 Quantile Balancing: top-k over QUANTILE-normalized columns instead of raw logits. Balances
    per-expert load by construction (each column ~Uniform → each expert equally likely to be a top choice)
    with NO auxiliary loss and NO learned bias. Returns the same dict shape as topk_route."""
    q = column_quantiles(logits)
    assign = _topk_cols(q, k)
    return {"assign": assign, "load": np.bincount(assign.ravel(), minlength=q.shape[1]), "qscores": q}


# ---------------------------------------------------------------- balance metrics
def load_stats(load, n_experts=None):
    """Balance diagnostics for a per-expert load vector. cv = std/mean (0 = perfectly flat); max_frac = the
    busiest expert's share of tokens (1/E is ideal); util = fraction of experts that got ≥1 token."""
    load = np.asarray(load, float)
    E = int(n_experts) if n_experts else load.shape[0]
    tot = load.sum()
    frac = load / tot if tot > 0 else load
    mean = load.mean() if load.size else 0.0
    cv = float(load.std() / mean) if mean > 0 else 0.0
    return {"cv": cv, "max_frac": float(frac.max()) if frac.size else 0.0,
            "ideal_frac": 1.0 / E, "util": float((load > 0).mean())}


def aux_free_saving(logits, k=2):
    """Compare baseline top-k vs quantile-balanced routing on the SAME logits. Returns both load-CVs and the
    reduction — the balance bought WITHOUT any auxiliary load-balancing loss term or learned bias."""
    E = np.asarray(logits).shape[1]
    base = load_stats(topk_route(logits, k)["load"], E)
    qb = load_stats(quantile_balance_route(logits, k)["load"], E)
    red = (base["cv"] - qb["cv"]) / base["cv"] if base["cv"] > 0 else 0.0
    return {"baseline_cv": base["cv"], "qbalance_cv": qb["cv"], "cv_reduction": float(red),
            "baseline_max_frac": base["max_frac"], "qbalance_max_frac": qb["max_frac"],
            "baseline_util": base["util"], "qbalance_util": qb["util"]}


# ---------------------------------------------------------------- agent
class MoEQuantileBalance(BaseAgent):
    name = "moe-quantile-balance"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        rng = np.random.RandomState(int(s.get("seed", 0)))
        N = int(s.get("n_tokens", 512)); E = int(s.get("n_experts", 16)); k = int(s.get("k", 2))
        # miscalibrated router: some experts have a systematic logit offset → baseline top-k skews to them
        bias = rng.randn(E) * float(s.get("bias_scale", 1.5))
        logits = rng.randn(N, E) + bias
        r = aux_free_saving(logits, k)
        msg = (f"moe-quantile-balance: {k}-of-{E} routing, {N} tokens — load-CV {r['baseline_cv']:.3f} "
               f"(top-k) → {r['qbalance_cv']:.3f} (quantile-balanced), {r['cv_reduction']*100:.0f}% flatter; "
               f"busiest expert {r['baseline_max_frac']*100:.1f}%→{r['qbalance_max_frac']*100:.1f}% "
               f"(ideal {100.0/E:.1f}%); util {r['baseline_util']*100:.0f}%→{r['qbalance_util']*100:.0f}% — "
               f"NO aux-loss, NO learned bias (K3 Stable LatentMoE)")
        self.log(msg, kind="finding",
                 recommendation="quantile-normalize router columns before top-k to balance a very-sparse MoE "
                                "without an aux-loss coefficient to tune; pair with moe-inference-cost for footprint")
        return self.done({"baseline_cv": r["baseline_cv"], "qbalance_cv": r["qbalance_cv"],
                          "cv_reduction": r["cv_reduction"]}, msg)


_AGENT = MoEQuantileBalance()


def run_qbalance(q, worker):
    return _AGENT.run(q, worker)
