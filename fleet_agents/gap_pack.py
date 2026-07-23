"""gap_pack — cross-cutting reusable agents mined from the FULL 61-comp gap-scan (techniques the 19-comp
sample missed). Each is competition-agnostic and offline-verifiable:

  • subset-classifier-router      — classify each item into a family → route to the specialist (waveform/
                                    vesuvius/mitsui inference-time mixture-of-experts).
  • analysis-by-synthesis-refiner — test-time gradient refinement so forward(pred)≈observed with a KNOWN
                                    forward operator (waveform 28.8→7.6; ariel transit fit) — no retraining.
  • checkpoint-merger             — weight-space model merging (linear / TIES) — beats prediction blending
                                    and cuts tokens (aimo mergekit).
  • constrained-label-assignment  — Hungarian / joint-MLE decode under per-group COUNT constraints (cmi +0.02).
  • lb-shift-prober               — fit an affine/offset correction from a probe grid (polymer: caught a
                                    °C/°F unit bug + a constant offset) — a systematic grader-diagnosis probe.

Pure numpy/scipy/sklearn. Verified by test_fleet_agents/gap_pack_test.py.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- subset-classifier-router
def route(X_train, family, X_test, clf=None):
    """Fit a family classifier on (X_train, family) and predict the family of each test item → route index."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    clf = clf or HistGradientBoostingClassifier(max_iter=200, random_state=0)
    clf.fit(np.asarray(X_train, float), np.asarray(family))
    return clf.predict(np.asarray(X_test, float)), clf


# ---------------------------------------------------------------- analysis-by-synthesis-refiner
def refine(pred0, A, obs, steps=200, lr=None, prior_tv=0.0, tol=0.0):
    """Refine x so A@x ≈ obs (least squares + optional TV prior) by gradient descent — the inverse-problem
    test-time refinement. A = linear forward operator (m×n). Returns (x_refined, residual_before, after).
    tol: early-stop when the residual improvement between steps drops below this (0 = run all steps)."""
    A = np.nan_to_num(np.asarray(A, float)); obs = np.nan_to_num(np.asarray(obs, float))
    x = np.nan_to_num(np.asarray(pred0, float)).copy()
    r0 = float(np.linalg.norm(A @ x - obs))
    if lr is None:
        try:
            s = np.linalg.svd(A, compute_uv=False); lr = 1.0 / (s[0] ** 2 + 1e-9)  # 1/Lipschitz
        except Exception:  # noqa: BLE001
            lr = 1e-3
    prev = r0
    for _ in range(int(steps)):
        grad = A.T @ (A @ x - obs)
        if prior_tv:
            d = np.diff(x, prepend=x[:1]); grad = grad + prior_tv * np.sign(d)
        x = x - lr * grad
        if tol > 0:
            cur = float(np.linalg.norm(A @ x - obs))
            if abs(prev - cur) < tol:
                break
            prev = cur
    return x, r0, float(np.linalg.norm(A @ x - obs))


# ---------------------------------------------------------------- checkpoint-merger
def merge_linear(params, weights=None):
    """Weighted average of parameter vectors (linear model-soup). params = list of np arrays (same shape)."""
    P = [np.asarray(p, float) for p in params]
    w = np.ones(len(P)) / len(P) if weights is None else np.asarray(weights, float) / np.sum(weights)
    return sum(w[i] * P[i] for i in range(len(P)))


def merge_ties(params, base=None, density=0.5):
    """TIES merge: sparsify each task-delta to its top-|density| magnitudes, elect the majority sign per
    coordinate, average only the agreeing deltas. base = pretrained params (0 if None)."""
    P = [np.asarray(p, float) for p in params]
    base = np.zeros_like(P[0]) if base is None else np.asarray(base, float)
    deltas = [p - base for p in P]
    trimmed = []
    for d in deltas:
        k = max(1, int(density * d.size)); thr = np.sort(np.abs(d))[::-1][k - 1]
        trimmed.append(np.where(np.abs(d) >= thr, d, 0.0))
    stack = np.stack(trimmed)
    sign = np.sign(np.sum(np.sign(stack), axis=0))                 # elected sign per coord
    agree = (np.sign(stack) == sign) & (sign != 0)
    num = np.sum(np.where(agree, stack, 0.0), axis=0)
    den = np.maximum(np.sum(agree, axis=0), 1)
    return base + num / den


# ---------------------------------------------------------------- constrained-label-assignment
def assign_constrained(logprob, counts):
    """Assign each item a label maximizing total log-prob s.t. label l is used exactly counts[l] times
    (per-group combinatorial structure, cmi). Solved as a rectangular assignment by expanding label slots."""
    from scipy.optimize import linear_sum_assignment
    L = np.asarray(logprob, float); n, k = L.shape
    assert int(sum(counts)) == n, "counts must sum to n_items"
    # expand: one column per slot (label l repeated counts[l] times); cost = -logprob
    slot_label = np.concatenate([[l] * int(counts[l]) for l in range(k)])
    cost = -L[:, slot_label]
    ri, ci = linear_sum_assignment(cost)
    out = np.empty(n, int)
    for i, j in zip(ri, ci):
        out[i] = slot_label[j]
    return out


# ---------------------------------------------------------------- lb-shift-prober
def fit_offset(offsets, scores, maximize=False):
    """Given probe offsets and their observed scores, fit a parabola and return the optimal offset (vertex).
    polymer lever: submit constants ±k·σ, recover the affine/units correction. maximize=True for a score metric."""
    x = np.nan_to_num(np.asarray(offsets, float)); y = np.nan_to_num(np.asarray(scores, float))
    if len(x) < 3:                                     # need >=3 points for a quadratic fit
        return float(x[np.argmax(y) if maximize else np.argmin(y)]) if len(x) else 0.0
    a, b, c = np.polyfit(x, y, 2)
    if abs(a) < 1e-12:
        return float(x[np.argmax(y) if maximize else np.argmin(y)])
    vertex = -b / (2 * a)
    return float(vertex)


# ---------------------------------------------------------------- agents
class _Base(BaseAgent):
    thread = "M"; kind = "finding"


class SubsetRouter(_Base):
    name = "subset-classifier-router"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("X_train", "family", "X_test") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"subset-classifier-router needs spec keys {missing} — none provided")
        fam, _ = route(s["X_train"], s["family"], s["X_test"])
        msg = f"subset-classifier-router: routed {len(fam)} items into {len(set(fam))} families for specialist dispatch"
        self.log(msg, kind="finding", recommendation="apply per-family model/postproc/weights, then merge")
        return self.done({"family": np.asarray(fam).tolist()}, msg)


class AnalysisRefiner(_Base):
    name = "analysis-by-synthesis-refiner"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("pred0", "A", "obs") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"analysis-by-synthesis-refiner needs spec keys {missing} — none provided")
        x, r0, r1 = refine(s["pred0"], s["A"], s["obs"], steps=int(s.get("steps", 200)),
                           lr=s.get("lr"), prior_tv=float(s.get("tv", 0.0)), tol=float(s.get("tol", 0.0)))
        msg = f"analysis-by-synthesis-refiner: residual {r0:.4g} → {r1:.4g} via forward-operator gradient refinement"
        self.log(msg, kind="finding", recommendation="keep only if the metric improves on held-out")
        return self.done({"_refined": np.asarray(x).tolist(), "residual_before": r0, "residual_after": r1}, msg)


class CheckpointMerger(_Base):
    name = "checkpoint-merger"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("params",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"checkpoint-merger needs spec keys {missing} — none provided")
        method = s.get("method", "linear")
        m = merge_ties(s["params"], base=s.get("base")) if method == "ties" else merge_linear(s["params"], s.get("weights"))
        msg = f"checkpoint-merger[{method}]: merged {len(s['params'])} checkpoints in weight space"
        self.log(msg, kind="finding", recommendation="eval merged weights on hidden-label set before keeping")
        return self.done({"_merged": np.asarray(m).tolist()}, msg)


class ConstrainedAssign(_Base):
    name = "constrained-label-assignment"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("logprob", "counts") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"constrained-label-assignment needs spec keys {missing} — none provided")
        lab = assign_constrained(s["logprob"], s["counts"])
        msg = f"constrained-label-assignment: decoded {len(lab)} labels under count constraints (joint-MLE)"
        self.log(msg, kind="finding", recommendation="use when per-group label counts are known/fixed")
        return self.done({"labels": np.asarray(lab).tolist()}, msg)


class LbShiftProber(_Base):
    name = "lb-shift-prober"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("offsets", "scores") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"lb-shift-prober needs spec keys {missing} — none provided")
        opt = fit_offset(s["offsets"], s["scores"], maximize=bool(s.get("maximize", False)))
        msg = f"lb-shift-prober: optimal correction offset ≈ {opt:.5g} (fit from probe grid) — checks for units/offset bugs"
        self.log(msg, kind="finding", recommendation="apply the offset to ALL predictions; human-gated probe submits")
        return self.done({"optimal_offset": opt}, msg)


_ROUTER = SubsetRouter(); _REFINER = AnalysisRefiner(); _MERGER = CheckpointMerger()
_ASSIGN = ConstrainedAssign(); _PROBER = LbShiftProber()


def run_router(q, worker): return _ROUTER.run(q, worker)
def run_refiner(q, worker): return _REFINER.run(q, worker)
def run_merger(q, worker): return _MERGER.run(q, worker)
def run_assign(q, worker): return _ASSIGN.run(q, worker)
def run_prober(q, worker): return _PROBER.run(q, worker)
