"""robustness_pack — recurring PURE agents from the one-by-one pass covering distribution-shift, decoding,
and domain-constraint levers. All numpy/sklearn, offline-verified, CompConfig-agnostic:

  • shift-adapt              — adversarial train-vs-test discriminator → per-sample importance weights +
                              a shift-aligned holdout (test-most-similar rows) as local CV (s5e12).
  • geospatial-fe           — grid-cell target-encoding + spatial-KNN class-fraction for lat/lon / RA-Dec (s6e6).
  • linear-constraint-projector — project predictions onto a known linear manifold Ax=b (mass balance /
                              Einthoven) — minimal change, constraint satisfied (CSIRO, ECG).
  • runtime-budget-router   — under a time/compute budget, send only the highest-value items to the expensive
                              model, rest to a fast fallback (orbit-wars, LLM cascades, biohub T4).
  • mbr-consensus-selector  — Minimum Bayes Risk: pick the candidate most similar to the pool (translation MBR).
  • noisy-label-cleaner     — resolve conflicting/duplicate (input,label) supervision → majority + soft target.
  • knn-label-transfer      — retrieval-based label transfer via similarity-weighted neighbor vote (CAFA homology).
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


def _fin2d(x):
    """Coerce to a finite 2-D float array (nan/inf → 0) so no stray value poisons a fit."""
    a = np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)
    return a if a.ndim >= 2 else a.reshape(-1, 1)


# ---------------------------------------------------------------- shift-adapt
def shift_weights(X_train, X_test, clip=20.0, seed=0, n_splits=3):
    """Fit a train-vs-test discriminator; importance weight = p(test)/p(train) per train row (density ratio).
    seed: RNG for the discriminator; n_splits: CV folds (auto-capped to the smaller side). None if sklearn absent."""
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_predict
    except Exception:  # noqa: BLE001 — never crash the fleet on a missing optional dep
        return None
    Xtr = _fin2d(X_train); Xte = _fin2d(X_test)
    if len(Xtr) == 0 or len(Xte) == 0:
        return None
    cv = int(max(2, min(n_splits, len(Xtr), len(Xte))))
    X = np.vstack([Xtr, Xte]); y = np.r_[np.zeros(len(Xtr)), np.ones(len(Xte))]
    p = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=seed), X, y,
                          cv=cv, method="predict_proba")[:len(Xtr), 1]
    p = np.clip(p, 1e-3, 1 - 1e-3)
    w = p / (1 - p)
    return np.clip(w / (w.mean() + 1e-12), 1.0 / clip, clip)


def shift_aligned_holdout(X_train, X_test, frac=0.1):
    """Return indices of the train rows MOST similar to test (a shift-aligned local validation set)."""
    w = shift_weights(X_train, X_test)
    if w is None:
        return np.array([], dtype=int)
    k = max(1, int(frac * len(w)))
    return np.argsort(-w)[:k]


# ---------------------------------------------------------------- geospatial-fe
def geo_features(coords_tr, y, coords_te, n_bins=16, k=10):
    """Grid-cell mean-target encoding + spatial-KNN target-mean for 2-D coordinates. Leak-safe on test."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:  # noqa: BLE001
        return None, None
    ctr = _fin2d(coords_tr); cte = _fin2d(coords_te)
    y = np.nan_to_num(np.asarray(y, float), nan=0.0, posinf=0.0, neginf=0.0)
    lo = ctr.min(0); hi = ctr.max(0) + 1e-9
    def cell(c): return tuple((((c - lo) / (hi - lo)) * n_bins).astype(int).clip(0, n_bins - 1))
    gmean = {}
    from collections import defaultdict
    acc = defaultdict(list)
    for c, t in zip(ctr, y):
        acc[cell(c)].append(t)
    gmean = {kk: np.mean(v) for kk, v in acc.items()}
    glob = float(y.mean())
    tr_cell = np.array([gmean.get(cell(c), glob) for c in ctr])
    te_cell = np.array([gmean.get(cell(c), glob) for c in cte])
    nn = NearestNeighbors(n_neighbors=min(k, len(ctr))).fit(ctr)
    _, idx = nn.kneighbors(cte); te_knn = y[idx].mean(1)
    return np.column_stack([tr_cell]), np.column_stack([te_cell, te_knn])


# ---------------------------------------------------------------- linear-constraint-projector
def project_constraints(preds, A, b):
    """Project each row of preds onto {x : A x = b} (min ||x - pred||). preds (n,d), A (m,d), b (m,)."""
    P = _fin2d(preds); A = np.atleast_2d(np.asarray(A, float)); b = np.asarray(b, float).ravel()
    AAt_inv = np.linalg.pinv(A @ A.T)
    resid = (P @ A.T) - b                                  # (n, m)
    return P - resid @ AAt_inv @ A


# ---------------------------------------------------------------- runtime-budget-router
def budget_route(item_costs, budget, quality_gain=None):
    """Choose which items get the EXPENSIVE model under a total budget, maximizing summed quality_gain
    (greedy by gain/cost). Returns a boolean mask; unmasked items use the fast fallback."""
    c = np.asarray(item_costs, float); n = len(c)
    g = np.ones(n) if quality_gain is None else np.asarray(quality_gain, float)
    order = np.argsort(-(g / np.maximum(c, 1e-9)))
    mask = np.zeros(n, bool); spent = 0.0
    for i in order:
        if spent + c[i] <= budget:
            mask[i] = True; spent += c[i]
    return mask, float(spent)


# ---------------------------------------------------------------- mbr-consensus-selector
def mbr_select(candidates, sim):
    """Minimum Bayes Risk: pick the candidate with the highest mean similarity to all others. sim(a,b)->float."""
    m = len(candidates)
    scores = [np.mean([sim(candidates[i], candidates[j]) for j in range(m) if j != i]) for i in range(m)]
    return int(np.argmax(scores)), scores


# ---------------------------------------------------------------- noisy-label-cleaner
def clean_labels(keys, labels):
    """Resolve conflicting supervision: for each identical key, majority label + soft target (positive ratio)."""
    from collections import defaultdict
    acc = defaultdict(list)
    for k, l in zip(keys, labels):
        acc[k].append(l)
    hard = {k: max(set(v), key=v.count) for k, v in acc.items()}
    soft = {k: float(np.mean(v)) for k, v in acc.items()}
    return hard, soft


# ---------------------------------------------------------------- knn-label-transfer
def knn_transfer(X_train, y_train, X_test, k=5):
    """Similarity-weighted neighbor label vote (retrieval/homology-style label transfer). None if sklearn absent."""
    try:
        from sklearn.neighbors import NearestNeighbors
    except Exception:  # noqa: BLE001
        return None
    Xtr = _fin2d(X_train); y = np.asarray(y_train); Xte = _fin2d(X_test)
    if len(Xtr) == 0:
        return np.array([])
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr))).fit(Xtr)
    d, idx = nn.kneighbors(Xte); w = 1.0 / (d + 1e-9)
    out = []
    for row_idx, row_w in zip(idx, w):
        labs = y[row_idx]; vote = {}
        for l, ww in zip(labs, row_w):
            vote[l] = vote.get(l, 0.0) + ww
        out.append(max(vote, key=vote.get))
    return np.array(out)


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class ShiftAdapt(_B):
    name = "shift-adapt"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("X_train", "X_test") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"shift-adapt needs spec keys {missing} — none provided")
        w = shift_weights(s["X_train"], s["X_test"])
        if w is None:
            return self.escalate(worker, "researcher", f"[{worker}] shift-adapt: sklearn unavailable or empty input.")
        hold = shift_aligned_holdout(s["X_train"], s["X_test"], float(s.get("frac", 0.1)))
        msg = f"shift-adapt: importance weights (mean {w.mean():.2f}, max {w.max():.2f}) + {len(hold)}-row shift-aligned holdout"
        self.log(msg, kind="finding", recommendation="train with sample_weight=w; validate on the shift-aligned holdout")
        return self.done({"_weights": w.tolist(), "holdout_idx": hold.tolist()}, msg)


class GeospatialFe(_B):
    name = "geospatial-fe"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("coords_train", "y", "coords_test") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"geospatial-fe needs spec keys {missing} — none provided")
        tr, te = geo_features(s["coords_train"], s["y"], s["coords_test"],
                                                int(s.get("n_bins", 16)), int(s.get("k", 10)))
        if tr is None:
            return self.escalate(worker, "researcher", f"[{worker}] geospatial-fe: sklearn unavailable.")
        msg = f"geospatial-fe: grid-cell TE + spatial-KNN features ({te.shape[1]} test cols)"
        self.log(msg, kind="finding", recommendation="add to tab-train; the only lift when spatial structure is preserved")
        return self.done({"_train": tr.tolist(), "_test": te.tolist()}, msg)


class ConstraintProjector(_B):
    name = "linear-constraint-projector"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("preds", "A", "b") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"linear-constraint-projector needs spec keys {missing} — none provided")
        out = project_constraints(s["preds"], s["A"], s["b"])
        msg = "linear-constraint-projector: predictions projected onto the linear constraint manifold Ax=b"
        self.log(msg, kind="finding", recommendation="enforce known relations (mass balance / Einthoven) post-hoc")
        return self.done({"_preds": np.asarray(out).tolist()}, msg)


class RuntimeBudgetRouter(_B):
    name = "runtime-budget-router"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("item_costs", "budget") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"runtime-budget-router needs spec keys {missing} — none provided")
        mask, spent = budget_route(s["item_costs"], float(s["budget"]), s.get("quality_gain"))
        msg = f"runtime-budget-router: {int(mask.sum())}/{len(mask)} items → expensive model (spent {spent:.1f}/{s['budget']})"
        self.log(msg, kind="finding", recommendation="fast fallback for the rest; keeps you under the runtime limit")
        return self.done({"expensive_mask": mask.tolist(), "spent": spent}, msg)


class MbrSelector(_B):
    name = "mbr-consensus-selector"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("candidates",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"mbr-consensus-selector needs spec keys {missing} — none provided")
        cands = s["candidates"]
        best, scores = mbr_select([np.asarray(c, float) for c in cands],
                                  lambda a, b: -float(np.linalg.norm(a - b)))
        msg = f"mbr-consensus-selector: chose candidate {best} (highest mutual agreement)"
        self.log(msg, kind="finding", recommendation="use for text/structure generation candidate selection")
        return self.done({"chosen": best, "scores": [float(x) for x in scores]}, msg)


class NoisyLabelCleaner(_B):
    name = "noisy-label-cleaner"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("keys", "labels") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"noisy-label-cleaner needs spec keys {missing} — none provided")
        hard, soft = clean_labels([tuple(k) if isinstance(k, list) else k for k in s["keys"]], s["labels"])
        msg = f"noisy-label-cleaner: resolved {len(hard)} unique keys (majority + soft target)"
        self.log(msg, kind="finding", recommendation="train on soft targets with a noise-robust loss (GCE/Tsallis)")
        return self.done({"n_keys": len(hard)}, msg)


class KnnLabelTransfer(_B):
    name = "knn-label-transfer"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("X_train", "y_train", "X_test") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"knn-label-transfer needs spec keys {missing} — none provided")
        out = knn_transfer(s["X_train"], s["y_train"], s["X_test"], int(s.get("k", 5)))
        if out is None:
            return self.escalate(worker, "researcher", f"[{worker}] knn-label-transfer: sklearn unavailable.")
        msg = f"knn-label-transfer: transferred labels to {len(out)} items via similarity-weighted vote"
        self.log(msg, kind="finding", recommendation="parameter-free predictor; blend with learned models")
        return self.done({"labels": out.tolist()}, msg)


_SH = ShiftAdapt(); _GEO = GeospatialFe(); _CP = ConstraintProjector(); _RB = RuntimeBudgetRouter()
_MBR = MbrSelector(); _NL = NoisyLabelCleaner(); _KT = KnnLabelTransfer()


def run_shift(q, worker): return _SH.run(q, worker)
def run_geo(q, worker): return _GEO.run(q, worker)
def run_project(q, worker): return _CP.run(q, worker)
def run_router(q, worker): return _RB.run(q, worker)
def run_mbr(q, worker): return _MBR.run(q, worker)
def run_cleanlabel(q, worker): return _NL.run(q, worker)
def run_transfer(q, worker): return _KT.run(q, worker)
