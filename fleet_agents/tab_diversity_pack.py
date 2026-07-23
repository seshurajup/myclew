"""tab_diversity_pack — the recurring PURE tabular levers the full 67-comp pass surfaced across essentially
every Playground winner. All sklearn/numpy, offline-verifiable, CompConfig-agnostic (operate on arrays):

  • synth-artifact-fe        — generator-fingerprint FE for synthetic-from-original comps: digit/decimal
                               extraction, is-round flags, snap-to-nearest-original + diff, frequency-ratio,
                               and original-dataset target priors (append-as-columns).
  • oof-diversity-prune      — build the OOF error-correlation matrix, prune near-twin models, keep the
                               decorrelated legs (winners: the weak orthogonal tail drives the lift).
  • feature-select           — consensus feature importance (multi-GBDT gain + permutation) → stable top-K.
  • residual-boost           — fit model B on the residuals of a baseline A; final = A + B (cdeotte lever).
  • knn-feature              — leak-safe OOF kNN target-mean + distance meta-features.
  • full-retrain-calibrator  — the 100%-train retrain iteration count iters×(1+1/(K-1)) + seed-averaging.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- synth-artifact-fe
def synth_artifact_features(x, original=None):
    """Generator-fingerprint features from a 1-D numeric column. original = the source column to snap to."""
    x = np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0); feats = {"raw": x}
    scaled = np.round(x * 1000).astype(np.int64)
    feats["n_decimals"] = np.array([len(str(v).split(".")[1].rstrip("0")) if "." in str(v) else 0 for v in x], float)
    feats["is_round"] = (np.abs(x - np.round(x)) < 1e-9).astype(float)
    feats["last_digit"] = (scaled % 10).astype(float)
    feats["mod100"] = (scaled % 100).astype(float)
    if original is not None and len(original):
        orig = np.sort(np.unique(np.asarray(original, float)))
        idx = np.clip(np.searchsorted(orig, x), 0, len(orig) - 1)
        nearest = orig[idx]
        feats["snap_diff"] = x - nearest
        feats["snap_is_exact"] = (np.abs(x - nearest) < 1e-9).astype(float)
        vc = {v: c for v, c in zip(*np.unique(np.asarray(original, float), return_counts=True))}
        feats["orig_freq"] = np.array([vc.get(round(v, 6), 0) for v in np.round(x, 6)], float)
    names = list(feats); X = np.column_stack([feats[k] for k in names])
    return np.nan_to_num(X).astype(np.float32), names


# ---------------------------------------------------------------- oof-diversity-prune
def diversity_prune(oof_dict, corr_threshold=0.999):
    """Keep a decorrelated subset: drop any model whose OOF correlation to an already-kept model exceeds
    the threshold (keeps the first seen). Returns (kept names, pairwise corr matrix)."""
    names = list(oof_dict)
    if not names:
        return [], np.zeros((0, 0))
    mats = np.column_stack([np.nan_to_num(np.asarray(oof_dict[n], float), nan=0.0, posinf=0.0, neginf=0.0)
                            for n in names])
    C = np.nan_to_num(np.corrcoef(mats.T), nan=0.0)     # constant column → 0 corr (treated decorrelated)
    C = np.atleast_2d(C)
    kept = []
    for i, n in enumerate(names):
        if all(abs(C[i, names.index(k)]) < corr_threshold for k in kept):
            kept.append(n)
    return kept, C


# ---------------------------------------------------------------- feature-select
def consensus_select(X, y, top_k=None, task="regression", n_estimators=200, permutation=False):
    """Rank features by consensus of multiple GBDT gain importances; return the top-K indices. Leak-safe
    (importance fit on train). top_k defaults to sqrt(n_features)·2.
    n_estimators: forest size for the importance estimator.
    permutation: if True, average impurity importance with permutation importance (more robust ranking)."""
    from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
    X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0, neginf=0.0); y = np.asarray(y)
    n_feat = X.shape[1] if X.ndim == 2 else 0
    if n_feat == 0:
        return [], np.zeros(0)
    top_k = top_k or max(2, int(2 * np.sqrt(n_feat)))
    et = (ExtraTreesClassifier if task == "classification" else ExtraTreesRegressor)(
        n_estimators=int(n_estimators), random_state=0, n_jobs=1).fit(X, y)
    imp = et.feature_importances_
    if permutation:
        try:
            from sklearn.inspection import permutation_importance
            pi = permutation_importance(et, X, y, n_repeats=5, random_state=0, n_jobs=1).importances_mean
            pi = pi / (pi.sum() + 1e-12)
            imp = 0.5 * (imp / (imp.sum() + 1e-12)) + 0.5 * pi
        except Exception:  # noqa: BLE001
            pass
    order = np.argsort(-imp)
    return order[:top_k].tolist(), imp


# ---------------------------------------------------------------- residual-boost
def residual_boost(base_model, boost_model, X, y, Xte=None):
    """final = base(X) + boost(residual). Returns (oof_final, test_final|None). Both models cloned/fit here."""
    from sklearn.base import clone
    X = np.nan_to_num(np.asarray(X, float)); y = np.nan_to_num(np.asarray(y, float))
    b = clone(base_model).fit(X, y); base_pred = b.predict(X)
    resid = y - base_pred
    r = clone(boost_model).fit(X, resid)
    oof = base_pred + r.predict(X)
    test = None
    if Xte is not None:
        Xte = np.nan_to_num(np.asarray(Xte, float)); test = b.predict(Xte) + r.predict(Xte)
    return oof, test


# ---------------------------------------------------------------- knn-feature
def knn_target_feature(X, y, folds, k=15, Xte=None):
    """Leak-safe OOF kNN target-mean + mean-distance features (regression/prob). Uses the CV folds."""
    from sklearn.neighbors import NearestNeighbors
    X = np.nan_to_num(np.asarray(X, float)); y = np.nan_to_num(np.asarray(y, float)); n = len(y)
    oof_mean = np.zeros(n); oof_dist = np.zeros(n)
    for tr, va in folds:
        nn = NearestNeighbors(n_neighbors=max(1, min(k, len(tr)))).fit(X[tr])
        d, idx = nn.kneighbors(X[va])
        oof_mean[va] = y[tr][idx].mean(1); oof_dist[va] = d.mean(1)
    te = None
    if Xte is not None:
        nn = NearestNeighbors(n_neighbors=max(1, min(k, n))).fit(X)
        d, idx = nn.kneighbors(np.nan_to_num(np.asarray(Xte, float)))
        te = np.column_stack([y[idx].mean(1), d.mean(1)])
    return np.column_stack([oof_mean, oof_dist]), te


# ---------------------------------------------------------------- full-retrain-calibrator
def retrain_iterations(mean_best_iter, k_folds):
    """The 100%-train retrain iteration count: mean early-stop iter × (1 + 1/(K-1)) (more data → more iters)."""
    return int(round(mean_best_iter * (1 + 1.0 / max(k_folds - 1, 1))))


def seed_average(pred_fn, seeds):
    """Average predictions over seeds (variance reduction for rank/threshold metrics). pred_fn(seed)->array."""
    preds = [np.asarray(pred_fn(s), float) for s in seeds]
    return np.mean(preds, axis=0)


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class SynthArtifactFe(_B):
    name = "synth-artifact-fe"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("x",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"synth-artifact-fe needs spec keys {missing} — none provided")
        X, names = synth_artifact_features(s["x"], s.get("original"))
        msg = f"synth-artifact-fe: {X.shape[1]} generator-fingerprint features ({names})"
        self.log(msg, kind="finding", recommendation="append original-data priors as columns; also add as extra rows")
        return self.done({"n_features": int(X.shape[1]), "names": names, "_X": X.tolist()}, msg)


class OofDiversityPrune(_B):
    name = "oof-diversity-prune"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("oof",) if k not in s]
        if missing: return self.escalate(worker, "leader", f"oof-diversity-prune needs spec keys {missing} — none provided")
        kept, C = diversity_prune(s["oof"], float(s.get("corr_threshold", 0.999)))
        msg = f"oof-diversity-prune: kept {len(kept)}/{len(s['oof'])} decorrelated models → {kept}"
        self.log(msg, kind="finding", recommendation="stack the kept decorrelated legs; weak orthogonal ones add most")
        return self.done({"kept": kept, "n_dropped": len(s["oof"]) - len(kept)}, msg)


class FeatureSelect(_B):
    name = "feature-select"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("X", "y") if k not in s]
        if missing: return self.escalate(worker, "leader", f"feature-select needs spec keys {missing} — none provided")
        idx, imp = consensus_select(s["X"], s["y"], s.get("top_k"), s.get("task", "regression"),
                                                      n_estimators=int(s.get("n_estimators", 200)),
                                                      permutation=bool(s.get("permutation", False)))
        msg = f"feature-select: kept top {len(idx)} features by consensus importance"
        self.log(msg, kind="finding", recommendation="train on the selected subset; reduces overfit on small data")
        return self.done({"selected": idx}, msg)


class ResidualBoost(_B):
    name = "residual-boost"
    def run(self, q, worker):
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import HistGradientBoostingRegressor
        s = self.spec(q)
        missing = [k for k in ("X", "y") if k not in s]
        if missing: return self.escalate(worker, "leader", f"residual-boost needs spec keys {missing} — none provided")
        oof, test = residual_boost(Ridge(1.0), HistGradientBoostingRegressor(max_iter=300),
                                   np.asarray(s["X"], float), np.asarray(s["y"], float),
                                   np.asarray(s["Xte"], float) if "Xte" in s else None)
        msg = "residual-boost: final = baseline + residual-model"
        self.log(msg, kind="finding", recommendation="use when a strong baseline/generating-function exists")
        return self.done({"_oof": oof.tolist(), "_test": None if test is None else test.tolist()}, msg)


class FullRetrainCalibrator(_B):
    name = "full-retrain-calibrator"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("mean_best_iter", "k_folds") if k not in s]
        if missing: return self.escalate(worker, "leader", f"full-retrain-calibrator needs spec keys {missing} — none provided")
        n = retrain_iterations(float(s["mean_best_iter"]), int(s["k_folds"]))
        msg = f"full-retrain-calibrator: retrain on 100% train with {n} iterations (was {s['mean_best_iter']} @ {s['k_folds']}-fold) + seed-avg"
        self.log(msg, kind="finding", recommendation="multi-seed average the full-data models")
        return self.done({"retrain_iterations": n}, msg)


_SA = SynthArtifactFe(); _DP = OofDiversityPrune(); _FS = FeatureSelect(); _RB = ResidualBoost(); _FR = FullRetrainCalibrator()


def run_synth(q, worker): return _SA.run(q, worker)
def run_prune(q, worker): return _DP.run(q, worker)
def run_fselect(q, worker): return _FS.run(q, worker)
def run_resboost(q, worker): return _RB.run(q, worker)
def run_retrain(q, worker): return _FR.run(q, worker)
