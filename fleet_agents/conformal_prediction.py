"""conformal_prediction — distribution-free, finite-sample COVERAGE guarantees the fleet lacked (we had
`calibrate` for scalar probability calibration + ECE, but nothing that emits set/interval predictions with a
provable 1-alpha coverage). Split conformal + adaptive prediction sets generalise across EVERY modality:
regression → prediction intervals, classification → prediction SETS, both with guaranteed marginal coverage
and optional class-/group-conditional (Mondrian) coverage under imbalance.

Papers (2026 frontier):
  • AdaptNC: "Adaptive Nonconformity Scores for Conformal Prediction under Distribution Shift",
    arXiv:2602.01629 (2026) — online adaptation of the score + threshold with a replay buffer.
  • "Kandinsky Conformal Prediction: Beyond Class- and Covariate-Conditional Coverage", arXiv:2502.17264.
  • "Conformal Prediction Adaptive to Unknown Subpopulation Shifts", arXiv:2506.05583 (2026).
  • Foundations: Vovk et al. (conformal), Romano/Angelopoulos APS & RAPS (adaptive prediction sets).

Reusable: wrap ANY trained model's holdout scores. classification uses APS/RAPS nonconformity (cumulative
sorted softmax mass) → adaptive set sizes (bigger sets on ambiguous inputs); regression uses absolute
residual quantiles → symmetric intervals. Mondrian mode computes a per-group threshold for group-wise coverage.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- conformal quantile
def conformal_quantile(scores, alpha):
    """The finite-sample-valid (1-alpha) conformal quantile of calibration nonconformity `scores`:
    the ceil((n+1)(1-alpha))/n empirical quantile (returns +inf if the level exceeds 1)."""
    s = np.sort(np.asarray(scores, float)); n = len(s)
    if n == 0:
        return np.inf
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    if k > n:
        return np.inf
    return float(s[k - 1])


# ---------------------------------------------------------------- regression: split conformal intervals
def split_conformal_interval(cal_pred, cal_y, test_pred, alpha=0.1):
    """Symmetric split-conformal interval. Returns (lo, hi, qhat). Guarantees P(y in [lo,hi]) >= 1-alpha."""
    res = np.abs(np.asarray(cal_y, float) - np.asarray(cal_pred, float))
    q = conformal_quantile(res, alpha); tp = np.asarray(test_pred, float)
    return tp - q, tp + q, q


def coverage(y, lo, hi):
    y = np.asarray(y, float)
    return float(np.mean((y >= np.asarray(lo, float)) & (y <= np.asarray(hi, float))))


# ---------------------------------------------------------------- classification: APS / RAPS
def _aps_scores(probs, labels, rng, randomize=True, k_reg=0, lam=0.0):
    """APS/RAPS nonconformity for each calibration row: cumulative sorted-softmax mass up to (and including,
    with a uniform tie-break) the TRUE label, plus an optional RAPS regularisation penalty encouraging small
    sets. probs [n,C], labels [n]."""
    probs = np.asarray(probs, float); labels = np.asarray(labels, int); n, C = probs.shape
    order = np.argsort(-probs, axis=1)                       # high→low prob
    ranks = np.argsort(order, axis=1)                        # rank of each class
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1)
    out = np.empty(n)
    for i in range(n):
        r = ranks[i, labels[i]]                              # 0-based position of true label in sorted order
        u = rng.rand() if randomize else 1.0
        base = cum[i, r] - u * sorted_p[i, r]                # subtract uniform fraction of the true-class mass
        reg = lam * max(0, (r + 1) - k_reg)                  # RAPS penalty (0 if k_reg large / lam=0 → pure APS)
        out[i] = base + reg
    return out


def _aps_predict(probs, qhat, rng, randomize=True, k_reg=0, lam=0.0):
    """Build prediction SETS: include classes (high→low prob) until cumulative mass exceeds qhat."""
    probs = np.asarray(probs, float); n, C = probs.shape
    order = np.argsort(-probs, axis=1); sorted_p = np.take_along_axis(probs, order, axis=1)
    cum = np.cumsum(sorted_p, axis=1); sets = []
    for i in range(n):
        u = rng.rand() if randomize else 1.0
        inc = np.zeros(C, bool)
        for j in range(C):
            reg = lam * max(0, (j + 1) - k_reg)
            val = cum[i, j] - u * sorted_p[i, j] + reg
            inc[order[i, j]] = True
            if val > qhat:
                break
        sets.append(np.where(inc)[0])
    return sets


def conformal_classify(cal_probs, cal_labels, test_probs, alpha=0.1, seed=0, randomize=True, k_reg=0, lam=0.0):
    """Split-conformal APS/RAPS. Returns (sets, qhat). Marginal coverage >= 1-alpha; set size adapts to
    input difficulty. Set k_reg>0, lam>0 for RAPS (smaller, more stable sets)."""
    rng = np.random.RandomState(seed)
    s = _aps_scores(cal_probs, cal_labels, rng, randomize=randomize, k_reg=k_reg, lam=lam)
    q = conformal_quantile(s, alpha)
    sets = _aps_predict(test_probs, q, rng, randomize=randomize, k_reg=k_reg, lam=lam)
    return sets, q


def mondrian_classify(cal_probs, cal_labels, test_probs, test_groups, cal_groups, alpha=0.1, seed=0):
    """Class-/group-conditional (Mondrian) conformal: a SEPARATE threshold per group so each group gets
    >= 1-alpha coverage (fixes marginal-only coverage collapsing on minority groups). Non-randomised APS."""
    rng = np.random.RandomState(seed)
    cal_probs = np.asarray(cal_probs, float); cal_labels = np.asarray(cal_labels, int)
    cal_groups = np.asarray(cal_groups); test_groups = np.asarray(test_groups)
    qhat = {}
    for g in np.unique(cal_groups):
        m = cal_groups == g
        s = _aps_scores(cal_probs[m], cal_labels[m], rng, randomize=False)
        qhat[g] = conformal_quantile(s, alpha)
    default = max(qhat.values()) if qhat else np.inf
    sets = []
    for i in range(len(test_probs)):
        q = qhat.get(test_groups[i], default)
        sets.append(_aps_predict(np.asarray(test_probs)[i:i + 1], q, rng, randomize=False)[0])
    return sets, qhat


def set_coverage(sets, y_true):
    y = np.asarray(y_true, int)
    return float(np.mean([y[i] in sets[i] for i in range(len(y))]))


def mean_set_size(sets):
    return float(np.mean([len(s) for s in sets]))


# ---------------------------------------------------------------- agent
class ConformalPredict(BaseAgent):
    name = "conformal-predict"
    thread = "S"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q); alpha = float(s.get("alpha", 0.1)); seed = int(s.get("seed", 0))
        task = s.get("task", "auto")
        if "cal_probs" not in s and "cal_pred" not in s:     # no data → self-demo on synthetic regression
            rng = np.random.RandomState(seed); x = rng.uniform(-3, 3, 3000); y = np.sin(x) + 0.3 * rng.randn(3000)
            pred = np.sin(x)
            s = dict(s); s.update(task="regression", cal_pred=pred[:1500], cal_y=y[:1500],
                                  test_pred=pred[1500:], test_y=y[1500:]); task = "regression"
        if task in ("classification", "auto") and "cal_probs" in s and task != "regression":
            sets, qh = conformal_classify(s["cal_probs"], s["cal_labels"], s["test_probs"], alpha=alpha,
                                          seed=seed, k_reg=int(s.get("k_reg", 0)), lam=float(s.get("lam", 0.0)))
            cov = set_coverage(sets, s["test_labels"]) if "test_labels" in s else None
            msg = (f"conformal-predict[cls]: qhat={qh:.3f}, mean set size={mean_set_size(sets):.2f}, "
                   f"coverage={cov if cov is None else round(cov,3)} (target {1-alpha:.2f})")
            self.log(msg, kind="finding", recommendation="use RAPS (k_reg,lam) for smaller sets; Mondrian for per-group coverage")
            return self.done({"qhat": qh, "mean_set_size": mean_set_size(sets), "coverage": cov,
                              "sets": [x.tolist() for x in sets[:32]]}, msg)
        # regression path
        lo, hi, qh = split_conformal_interval(s["cal_pred"], s["cal_y"], s["test_pred"], alpha=alpha)
        cov = coverage(s["test_y"], lo, hi) if "test_y" in s else None
        width = float(np.mean(np.asarray(hi) - np.asarray(lo)))
        msg = (f"conformal-predict[reg]: qhat={qh:.4f}, mean interval width={width:.4f}, "
               f"coverage={cov if cov is None else round(cov,3)} (target {1-alpha:.2f})")
        self.log(msg, kind="finding", recommendation="feed residuals of ANY model; use Mondrian for group coverage")
        return self.done({"qhat": qh, "mean_width": width, "coverage": cov,
                          "lo": np.asarray(lo)[:32].tolist(), "hi": np.asarray(hi)[:32].tolist()}, msg)


_AGENT = ConformalPredict()


def run(q, worker):
    return _AGENT.run(q, worker)
