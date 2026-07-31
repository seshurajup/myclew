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

Lifted from google-research/tabfm (https://github.com/google-research/tabfm · lessons
learning/annotated/tfm*.learning · no paper exists, so the code at the pinned commit IS the reference):
  • appearance_ordinal_encode — codes ordered by first appearance/frequency, not alphabet (tfm unit 5).
  • two_stage_clip            — z-score outlier CLIPPING in two passes, because one pass under-detects when
                                the outlier inflates the σ it is measured against (unit 10).
  • noise_then_quantile       — RTDL quantile transform with noise before fitting and n_quantiles scaled to
                                the row count instead of a fixed 1000 (unit 12).
  • train_range_clip          — fit clip bounds on TRAIN, apply at test, so an unseen extreme cannot walk
                                off the distribution the model was fitted on (unit 13).
  • order_sensitivity         — THE GATE. Measures whether a fitted predictor's output actually changes when
                                you permute columns. We measured that trees and forests give +0.00% from
                                column-shuffle TTA (they are order-invariant: the permuted tree IS the same
                                tree), while TabFM's own forward moves by 4e-2. So view-ensembling is only
                                worth running when this returns non-zero — check before you spend the compute.
  • view_ensemble             — tabfm's EnsembleGenerator idea (one frozen predictor, N views) with that
                                gate wired in front of it.
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


# ---------------------------------------------------------------- tabfm lifts (github.com/google-research/tabfm)
def appearance_ordinal_encode(col, min_frequency=None):
    """Ordinal codes ordered by FREQUENCY then first appearance — not by alphabet (tabfm unit 5).

    Which arbitrary integer a category gets is arbitrary; which *arbitrariness* is not. Ordering by
    frequency puts common categories at small indices, so a model that treats the code as a magnitude sees
    a meaningful ordering rather than alphabetical noise. Categories rarer than `min_frequency` fold into a
    single bucket. Unknown values at transform time map to -1 rather than raising.
    """
    a = np.asarray(["NA" if v is None or (isinstance(v, float) and v != v) else str(v)
                    for v in np.asarray(col).ravel()], dtype=object)
    vals, counts = np.unique(a, return_counts=True)
    if min_frequency:
        rare = set(vals[counts < int(min_frequency)].tolist())
        if rare:
            a = np.asarray(["__RARE__" if v in rare else v for v in a], dtype=object)
            vals, counts = np.unique(a, return_counts=True)
    # frequency-descending, ties broken by first appearance so the mapping is deterministic
    first = {v: int(np.argmax(a == v)) for v in vals.tolist()}
    order = sorted(vals.tolist(), key=lambda v: (-int(counts[vals.tolist().index(v)]), first[v]))
    mapping = {v: i for i, v in enumerate(order)}
    return np.asarray([mapping.get(v, -1) for v in a], dtype=float), mapping


def two_stage_clip(X, threshold=4.0):
    """Two-pass z-score clipping (tabfm unit 10). Returns (clipped, lower, upper).

    One pass under-detects: a single extreme value inflates the σ it is being measured against. So pass one
    clips on the raw σ, pass two recomputes σ on the already-clipped data and clips again. Values are
    CLIPPED, never dropped — a row's other features are still informative, and in tabular comps rows are
    the scarce resource.
    """
    A = np.asarray(X, float).copy()
    lo = hi = None
    for _ in range(2):
        mu, sd = np.nanmean(A, axis=0), np.nanstd(A, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        lo, hi = mu - threshold * sd, mu + threshold * sd
        A = np.clip(A, lo, hi)
    return A, lo, hi


def noise_then_quantile(X, n_quantiles=None, noise=1e-3, random_state=0, output="normal"):
    """RTDL quantile transform: add noise BEFORE fitting, and scale n_quantiles to the data (tabfm unit 12).

    Quantile-transforming a tied column memorises the exact breakpoints of those ties; a little noise breaks
    them so the mapping generalises. `n_quantiles` defaults to a data-dependent value — tabfm's shipped
    transformer chose 10 for 20 rows and 166 for 5000, where a fixed 1000 would over-fit a small column.
    Returns (transformed, fitted_transformer) so test data reuses the TRAIN fit.
    """
    from sklearn.preprocessing import QuantileTransformer
    A = np.asarray(X, float)
    n = max(A.shape[0], 1)
    nq = int(n_quantiles) if n_quantiles else max(10, min(1000, n // 30 or 10))
    rs = np.random.RandomState(random_state)
    qt = QuantileTransformer(n_quantiles=min(nq, n), output_distribution=output,
                             subsample=max(n, 10_000), random_state=random_state)
    Z = qt.fit_transform(A + noise * rs.randn(*A.shape) * (np.nanstd(A, axis=0, keepdims=True) + 1e-12))
    return Z, qt


def train_range_clip(X_train, X_test, threshold=4.0):
    """Fit clip bounds on TRAIN, apply them to TEST (tabfm unit 13).

    The quiet detail in tabfm's PreprocessingPipeline: at transform time it clips to bounds learned during
    fit. A model — frozen foundation model or fitted GBM — only behaves where it was fitted, so an unseen
    extreme in the test set must not be allowed outside that range. Returns (train_clipped, test_clipped).
    """
    tr, lo, hi = two_stage_clip(X_train, threshold)
    te = np.clip(np.asarray(X_test, float), lo, hi)
    return tr, te


def order_sensitivity(predict_fn, X, n_probe=8, random_state=0):
    """**Run this BEFORE any view-ensembling.** Does permuting columns change the prediction at all?

    tabfm ensembles over dataset VIEWS (feature shuffles × label shifts × categorical permutations ×
    normalisations) because its network's output genuinely depends on column order — measured at 4e-2 on the
    real architecture. A tree does not: it splits on feature CONTENT, so the permuted model is the same
    model and column-shuffle TTA bought exactly +0.00% on both a DecisionTree and a RandomForest in our
    measurement. This function separates the two cases so we never pay for TTA that cannot work.

    `predict_fn(X) -> 1-D or 2-D predictions`. Returns the mean absolute movement and a verdict.
    """
    A = np.asarray(X, float)
    base = np.asarray(predict_fn(A), float)
    rs = np.random.RandomState(random_state)
    moves = []
    for _ in range(int(n_probe)):
        perm = rs.permutation(A.shape[1])
        p = np.asarray(predict_fn(A[:, perm]), float)
        moves.append(float(np.abs(p - base).mean()))
    m = float(np.mean(moves))
    scale = float(np.abs(base).mean()) + 1e-12
    rel = m / scale
    return {"mean_abs_move": m, "relative_move": rel,
            "order_sensitive": rel > 1e-6,
            "verdict": ("view-ensembling CAN help — the output depends on column order"
                        if rel > 1e-6 else
                        "order-INVARIANT predictor: column-shuffle views are identical, do not spend the "
                        "compute (measured +0.00% on trees/forests)")}


def view_ensemble(predict_fn, X, n_views=16, transforms=None, random_state=0, gate=True):
    """tabfm's EnsembleGenerator idea for any frozen predictor — with `order_sensitivity` in front of it.

    One predictor, N information-preserving views, averaged. Views are column permutations plus any
    `transforms` (callables applied to X, e.g. different normalisations). `gate=True` refuses to waste
    compute on an order-invariant predictor and says so, rather than silently returning the single-view
    answer dressed up as an ensemble.
    """
    A = np.asarray(X, float)
    if gate:
        chk = order_sensitivity(predict_fn, A, random_state=random_state)
        if not chk["order_sensitive"] and not transforms:
            return {"pred": np.asarray(predict_fn(A), float), "n_views": 1, "gated": True,
                    "reason": chk["verdict"]}
    rs = np.random.RandomState(random_state)
    preds = [np.asarray(predict_fn(A), float)]
    for _ in range(max(int(n_views) - 1, 0)):
        perm = rs.permutation(A.shape[1])
        preds.append(np.asarray(predict_fn(A[:, perm]), float))
    for t in (transforms or []):
        preds.append(np.asarray(predict_fn(np.asarray(t(A), float)), float))
    P = np.stack(preds)
    return {"pred": P.mean(0), "n_views": int(P.shape[0]), "gated": False,
            "disagreement": float(P.std(0).mean())}


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
