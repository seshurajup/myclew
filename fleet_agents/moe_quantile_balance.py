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

THE OTHER ANSWER — Routing-Free MoE (https://arxiv.org/pdf/2604.00801, eqs. 8-17 and appendix B; lessons
`learning/annotated/rfm*.learning`, notes `docs/papers/routing-free-moe/routing-free-moe.md`). K3 keeps top-k
and fixes the router; that paper deletes the router: each expert gates ITSELF with ReLU(‖xA_gate,i‖₂ − b_i),
so there is no Softmax coupling, no non-differentiable top-k, and no router matrix. Balance is then two
EXPLICIT objectives — expert-balance and token-balance — interpolated by one knob μ, with the penalty weight
driven by a multiplicative sign controller instead of a hand-set coefficient.

The two designs trade differently and neither dominates, so both live here and you pick by measurement:
  – quantile balancing gives expert balance BY CONSTRUCTION and token balance for free (k is fixed);
  – routing-free gives per-token adaptive compute and gradient to unchosen experts, but token balance must
    be asked for, and its COMMUNICATION win is conditional on k + 1 > n_devices (eq. 30, `comm_delta`).
  • routing_free_route(scores, b, theta)  — per-expert self-gating: no softmax, no top-k, k_eff emerges.
  • balance_losses(gates, rho_star, mu)   — eqs. 13-15: L_EB, L_TB and their μ-interpolation.
  • lambda_controller(rho_seq, ...)       — eq. 17: λ ← λ(1+η)^sign(ρ−ρ*), the aux-weight controller.
  • comm_delta(k, n_devices, ...)         — eq. 30: the SIGN that decides if deleting the router saves bytes.
  • router_vs_routing_free(logits, k)     — both balancers on the same logits, measured side by side.
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


# ---------------------------------------------------------------- routing-free MoE (arXiv 2604.00801)
def routing_free_route(scores, b=None, theta=0.0):
    """Routing-Free MoE gating (eq. 9): G_i(x) = ReLU(‖xA_gate,i‖₂ − b_i), one INDEPENDENT decision per
    expert. `scores` (N, E) are the per-expert gate norms — a quantity the expert already computes when its
    gate projection is low-rank (eq. 6), so no router matrix exists. `b` is the per-expert learned threshold.

    Unlike top-k this returns a variable number of experts per token: k_eff is an emergent average, which is
    the point (hard tokens may use more capacity) and also the thing you must monitor, since every cost in
    appendix B is linear in it."""
    scores = np.asarray(scores, float)
    b = np.zeros(scores.shape[1]) if b is None else np.asarray(b, float)
    gates = np.maximum(scores - b, 0.0)                  # eq. 9 — no softmax, so no cross-expert coupling
    active = gates > theta                               # eq. 10 — bookkeeping only, never in the forward path
    return {"gates": gates, "active": active, "load": active.sum(0).astype(float),
            "k_eff": float(active.sum(1).mean()), "density": float(active.mean())}


def balance_losses(gates, rho_star=0.25, mu=0.5):
    """Eqs. 13-15. Deleting top-k costs you TWO guarantees, so both come back as objectives:
      L_EB = mean_e (mean_x g − ρ*)²   every expert should fire about as often as every other;
      L_TB = mean_x (mean_e g − ρ*)²   every token should activate about as many experts as every other.
    `mu` interpolates: μ=1 protects the hardware, μ=0 protects the tokens. Classic MoE fixes both silently;
    making the trade explicit is this paper's cleanest contribution. `gates` should be a bounded density
    surrogate in [0,1] (eq. 12) — raw ReLU values are unbounded and will not compare to ρ*."""
    g = np.asarray(gates, float)
    l_eb = float(((g.mean(0) - rho_star) ** 2).mean())
    l_tb = float(((g.mean(1) - rho_star) ** 2).mean())
    return {"L_EB": l_eb, "L_TB": l_tb, "L_LB": mu * l_eb + (1 - mu) * l_tb, "mu": float(mu)}


def lambda_controller(rho_seq, lam0=0.01, eta=0.05, rho_star=0.25):
    """Eq. 17: λ_{t+1} = λ_t·(1+η)^sign(ρ_t − ρ*). A CONTROLLER, not a hyper-parameter — over-activation
    multiplies the balance weight up, under-activation multiplies it down. Multiplicative so λ stays strictly
    positive by construction and moves on a log scale (it can cross orders of magnitude in a few hundred
    steps). Use this instead of a fixed aux coefficient whenever you keep an auxiliary balance term at all."""
    lam, hist = float(lam0), []
    for r in np.asarray(rho_seq, float):
        lam *= (1.0 + eta) ** (1 if r > rho_star else -1)
        hist.append(lam)
    return {"lambda_final": lam, "history": hist, "crossed_decades": float(np.log10(max(lam, 1e-30) / lam0))}


def comm_delta(k, n_devices, n_tokens=4096, d_model=7168, bytes_per_el=2, bandwidth=200e9):
    """Eq. 30: Δ_B = (k + 1 − M)·T·D·b / (M·B) — bytes SAVED per step by dropping the router.

    A standard MoE pays all-to-all twice (dispatch + combine) with a payload in k; routing-free all-gathers
    instead, so the payload is (M−1+k_eff). The sign therefore depends only on **k + 1 − M**, i.e. deleting
    the router saves communication exactly when each token selects MORE experts than you have devices. That
    is a property of your topology, not of the method — compute it before believing any speed claim. On a
    2-GPU box it is positive for any k ≥ 2; on a 64-way expert-parallel cluster with k=8 it is negative."""
    delta = (int(k) + 1 - int(n_devices)) * n_tokens * d_model * bytes_per_el / (int(n_devices) * bandwidth)
    return {"delta_seconds": float(delta), "delta_ms": float(delta * 1e3),
            "favours": "routing-free" if delta > 0 else "standard MoE",
            "condition": f"k+1>M is {int(k) + 1 > int(n_devices)} (k={int(k)}, M={int(n_devices)})"}


def router_vs_routing_free(logits, k=2, rho_star=None, mu=0.5):
    """Put both answers on the SAME logits and report what each buys. Quantile balancing wins on flatness by
    construction; routing-free buys adaptive per-token compute (k_eff) and gradient for every expert. Read
    `cv` for hardware risk and `k_eff` for the compute the ReLU gate actually chose to spend."""
    logits = np.asarray(logits, float); E = logits.shape[1]
    rho_star = float(k) / E if rho_star is None else float(rho_star)
    qb = load_stats(quantile_balance_route(logits, k)["load"], E)
    base = load_stats(topk_route(logits, k)["load"], E)
    # a routing-free gate on the same signal: |logit| as the gate norm, threshold at the ρ* quantile
    scores = np.abs(logits)
    b = np.quantile(scores, 1.0 - rho_star, axis=0)      # the threshold ρ* implies, per expert
    rf = routing_free_route(scores, b)
    rf_stats = load_stats(rf["load"], E)
    dens = np.clip(rf["gates"] / (rf["gates"].max() + 1e-12), 0, 1)
    return {"topk_cv": base["cv"], "qbalance_cv": qb["cv"], "routing_free_cv": rf_stats["cv"],
            "k_fixed": float(k), "k_eff": rf["k_eff"], "density": rf["density"], "rho_star": rho_star,
            "adaptive_compute": float(rf["active"].sum(1).std()),
            **balance_losses(dens, rho_star, mu)}


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
        # the other answer, measured on the same logits, plus the topology test that decides its cost claim
        cmp = router_vs_routing_free(logits, k)
        dev = int(s.get("n_devices", 2))
        cd = comm_delta(k, dev)
        msg += (f" | routing-free (2604.00801): CV {cmp['routing_free_cv']:.3f}, k_eff {cmp['k_eff']:.2f} vs "
                f"fixed {k} (±{cmp['adaptive_compute']:.2f} adaptive), L_LB {cmp['L_LB']:.5f}; "
                f"comm Δ on M={dev} favours {cd['favours']} ({cd['condition']})")
        self.log(msg, kind="finding",
                 recommendation="quantile-normalize router columns before top-k to balance a very-sparse MoE "
                                "without an aux-loss coefficient to tune; pair with moe-inference-cost for footprint. "
                                "Consider routing-free self-gating ONLY where comm_delta is positive (k+1>M) or where "
                                "per-token adaptive compute is worth losing top-k's free token balance")
        return self.done({"baseline_cv": r["baseline_cv"], "qbalance_cv": r["qbalance_cv"],
                          "cv_reduction": r["cv_reduction"], "routing_free_cv": cmp["routing_free_cv"],
                          "k_eff": cmp["k_eff"], "comm_delta_ms": cd["delta_ms"],
                          "comm_favours": cd["favours"]}, msg)


_AGENT = MoEQuantileBalance()


def run_qbalance(q, worker):
    return _AGENT.run(q, worker)
