"""misc_domain_pack — the remaining PURE domain post-processing / FE levers from the one-by-one pass. All
stdlib/numpy/sklearn, offline-verified:

  • hierarchy-consistency-postproc — enforce ontology-DAG consistency (parent ≥ max(children)) on multi-label
                                     probabilities (CAFA GO-term propagation).
  • invariance-feature-normalizer  — egocentric / frame-canonical coordinates (translate to a reference point,
                                     rotate to a reference axis) so a model transfers across sources (MABe/NFL).
  • template-retrieval-reranker    — retrieve candidates by similarity + rerank (RNA templates, retrieval tasks).
  • calendar-holiday-fe            — day-of-week / month / is-weekend + cyclical encodings from dates (+ optional
                                     `holidays` lib) (playground-s5e1 sticker sales).
  • annotation-error-corrector     — flag likely-wrong ground-truth via model disagreement (high OOF loss) for
                                     human review (BYU GT correction: the biggest data-side lever).
  • binary-size-compressor         — deflate an artifact (weights/binary) and report the byte win under a size cap.
  • knn-feature                    — leak-safe OOF kNN target-mean meta-features (wraps tab_diversity_pack).
"""
from __future__ import annotations
import numpy as np
import zlib
from datetime import date
from .base import BaseAgent


# ---------------------------------------------------------------- hierarchy-consistency-postproc
def propagate_hierarchy(probs, edges, max_iter=None):
    """probs = (n_samples, n_terms). edges = list of (child, parent) term indices. Enforce parent >= max child
    probability (a term implies all its ancestors). Iterated to a fixed point over the DAG.
    max_iter: cap on fixed-point passes (default = n_terms, enough to reach any ancestor)."""
    P = np.nan_to_num(np.asarray(probs, float), nan=0.0, posinf=1.0, neginf=0.0).copy()
    if P.ndim != 2 or P.size == 0:
        return P
    n_pass = int(max_iter) if max_iter else P.shape[1]
    for _ in range(max(1, n_pass)):
        changed = False
        for child, parent in edges:
            newp = np.maximum(P[:, parent], P[:, child])
            if np.any(newp > P[:, parent] + 1e-12):
                P[:, parent] = newp; changed = True
        if not changed:
            break
    return P


# ---------------------------------------------------------------- invariance-feature-normalizer
def egocentric_normalize(points, origin, axis_angle):
    """Translate points to `origin` and rotate by -axis_angle → frame-invariant coordinates. points (n,2)."""
    p = np.asarray(points, float) - np.asarray(origin, float)
    c, s = np.cos(-axis_angle), np.sin(-axis_angle)
    R = np.array([[c, -s], [s, c]])
    return p @ R.T


# ---------------------------------------------------------------- template-retrieval-reranker
def retrieve_rerank(query, templates, k=5, score=None):
    """Retrieve top-k templates by cosine similarity to query, then rerank by `score(query, template)` (default
    similarity). Returns ranked template indices."""
    Q = np.asarray(query, float); T = np.asarray(templates, float)
    qn = Q / (np.linalg.norm(Q) + 1e-9); Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    sim = Tn @ qn
    top = np.argsort(-sim)[:k]
    if score is None:
        return top.tolist()
    reranked = sorted(top, key=lambda i: -score(Q, T[i]))
    return list(reranked)


# ---------------------------------------------------------------- calendar-holiday-fe
def calendar_features(iso_dates, country=None):
    """From ISO date strings → [dow, month, is_weekend, sin_doy, cos_doy, is_holiday]. Uses the `holidays`
    library if available + a country; otherwise is_holiday=0."""
    hol = set()
    if country:
        try:
            import holidays as _h
            years = sorted({int(d[:4]) for d in iso_dates})
            hol = set(_h.country_holidays(country, years=years).keys())
        except Exception:  # noqa: BLE001
            hol = set()
    rows = []
    names = ["dow", "month", "is_weekend", "sin_doy", "cos_doy", "is_holiday"]
    for ds in iso_dates:
        try:
            y, m, d = (int(x) for x in str(ds).split("-")[:3])
            dt = date(y, m, d); doy = dt.timetuple().tm_yday
            rows.append([dt.weekday(), m, 1.0 if dt.weekday() >= 5 else 0.0,
                         np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                         1.0 if dt in hol else 0.0])
        except Exception:  # noqa: BLE001 — unparseable date → zero row (keeps alignment)
            rows.append([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    return np.array(rows, float) if rows else np.zeros((0, 6), float), names


# ---------------------------------------------------------------- annotation-error-corrector
def flag_label_errors(y_true, oof_pred, z=3.0):
    """Flag samples whose OOF residual is an outlier (|resid| > z·std) — likely mislabeled GT for human review."""
    y = np.nan_to_num(np.asarray(y_true, float)); p = np.nan_to_num(np.asarray(oof_pred, float))
    r = np.abs(y - p)
    if r.size == 0:
        return [], 0.0
    thr = float(r.mean() + z * r.std())
    idx = np.where(r > thr)[0]
    return idx.tolist(), thr


# ---------------------------------------------------------------- binary-size-compressor
def compress_artifact(data_bytes, cap=None):
    o = len(data_bytes); c = len(zlib.compress(bytes(data_bytes), 9))
    return {"orig_bytes": o, "compressed_bytes": c, "ratio": round(c / max(o, 1), 4),
            "under_cap": (c <= cap) if cap else None}


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class HierarchyPostproc(_B):
    name = "hierarchy-consistency-postproc"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("probs", "edges") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"hierarchy-consistency-postproc needs spec keys {missing} — none provided")
        P = propagate_hierarchy(s["probs"], [tuple(e) for e in s["edges"]])
        msg = f"hierarchy-consistency-postproc: enforced parent≥child over {len(s['edges'])} DAG edges"
        self.log(msg, kind="finding", recommendation="apply before scoring hierarchical multi-label (CAFA)")
        return self.done({"_probs": P.tolist()}, msg)


class InvarianceNormalizer(_B):
    name = "invariance-feature-normalizer"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("points", "origin", "axis_angle") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"invariance-feature-normalizer needs spec keys {missing} — none provided")
        out = egocentric_normalize(s["points"], s["origin"], float(s["axis_angle"]))
        msg = "invariance-feature-normalizer: egocentric frame-canonical coordinates"
        self.log(msg, kind="finding", recommendation="use invariant coords so one model transfers across sources")
        return self.done({"_points": np.asarray(out).tolist()}, msg)


class TemplateReranker(_B):
    name = "template-retrieval-reranker"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("query", "templates") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"template-retrieval-reranker needs spec keys {missing} — none provided")
        ranked = retrieve_rerank(s["query"], s["templates"], int(s.get("k", 5)))
        msg = f"template-retrieval-reranker: top-{len(ranked)} templates {ranked[:5]}"
        self.log(msg, kind="finding", recommendation="transfer the top template's structure/label")
        return self.done({"ranked": ranked}, msg)


class CalendarFe(_B):
    name = "calendar-holiday-fe"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("dates",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"calendar-holiday-fe needs spec keys {missing} — none provided")
        X, names = calendar_features(s["dates"], s.get("country"))
        msg = f"calendar-holiday-fe: {X.shape[1]} calendar features ({names})"
        self.log(msg, kind="finding", recommendation="add to forecasting/tabular; holidays help retail/sales")
        return self.done({"names": names, "_X": X.tolist()}, msg)


class AnnotationErrorCorrector(_B):
    name = "annotation-error-corrector"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("y", "oof_pred") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"annotation-error-corrector needs spec keys {missing} — none provided")
        idx, thr = flag_label_errors(s["y"], s["oof_pred"], float(s.get("z", 3.0)))
        msg = f"annotation-error-corrector: flagged {len(idx)} likely-mislabeled samples (|resid|>{thr:.3f})"
        self.log(msg, kind="finding", recommendation="human-review the flagged GT (biggest data-side lever, BYU)")
        return self.done({"flagged_idx": idx, "threshold": thr}, msg)


class BinaryCompressor(_B):
    name = "binary-size-compressor"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("data",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"binary-size-compressor needs spec keys {missing} — none provided")
        data = s["data"].encode() if isinstance(s["data"], str) else bytes(s["data"])
        res = compress_artifact(data, s.get("cap"))
        msg = f"binary-size-compressor: {res['orig_bytes']} → {res['compressed_bytes']} bytes (ratio {res['ratio']})"
        self.log(msg, kind="finding", recommendation="self-extract at runtime when under a hard size cap")
        return self.done(res, msg)


class KnnFeature(_B):
    name = "knn-feature"
    def run(self, q, worker):
        from . import tab_diversity_pack as TD
        from sklearn.model_selection import KFold
        s = self.spec(q)
        missing = [k for k in ("X", "y") if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"knn-feature needs spec keys {missing} — none provided")
        X = np.asarray(s["X"], float); y = np.asarray(s["y"], float)
        folds = list(KFold(int(s.get("folds", 5)), shuffle=True, random_state=0).split(X))
        feat, _ = TD.knn_target_feature(X, y, folds, k=int(s.get("k", 15)))
        msg = f"knn-feature: OOF kNN target-mean + distance meta-features ({feat.shape[1]} cols)"
        self.log(msg, kind="finding", recommendation="append to tab-train; strong on geometric/embedding spaces")
        return self.done({"_feat": feat.tolist()}, msg)


_HP = HierarchyPostproc(); _IN = InvarianceNormalizer(); _TR = TemplateReranker(); _CF = CalendarFe()
_AE = AnnotationErrorCorrector(); _BC = BinaryCompressor(); _KF = KnnFeature()


def run_hier(q, worker): return _HP.run(q, worker)
def run_invar(q, worker): return _IN.run(q, worker)
def run_template(q, worker): return _TR.run(q, worker)
def run_calendar(q, worker): return _CF.run(q, worker)
def run_annoterr(q, worker): return _AE.run(q, worker)
def run_bincompress(q, worker): return _BC.run(q, worker)
def run_knnfeat(q, worker): return _KF.run(q, worker)
