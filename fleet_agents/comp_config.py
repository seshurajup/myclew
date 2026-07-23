"""comp_config — the competition-agnostic CONTRACT every generalized agent reads.

One agent body must serve tabular, image, video, 3D, 3D+time, LLM, agentic, reasoning and unknown
competitions. It does that by dispatching on a `CompConfig` instead of hard-coding biohub. CompConfig
captures WHAT a competition is:

    data-modality × paradigm × task × metric × cv-scheme × submission-schema × domain-hook

so `comp-onboard` can fingerprint any new comp into a CompConfig and every downstream agent (profile / cv /
train-or-solve / score / submit) reads the same object. This module is PURE stdlib + numpy — no biohub, no
torch — so it is relocatable to the shared `researchpapers` framework and reused across ALL comp dirs.

Contents:
  • Modality / Paradigm / Task enums (strings, open for `unknown`).
  • CompConfig dataclass (+ from_dict / to_dict / load / save).
  • METRIC_REGISTRY — name → {direction, fn, aliases}; common Kaggle metrics implemented in pure numpy so
    the tabular/LLM packs score WITHOUT any biohub coupling. biohub's edge_jaccard registers its own fn.
  • PACK_ROUTES + route() — (modality, paradigm) → the pack that handles it (or 'unknown' → onboard reports).
  • Five interface base classes (Profiler / CvBuilder / Solver / Scorer / Submitter) each taking a CompConfig.

Everything here is data-wise tested by test_fleet_agents/comp_config_test.py (constructs the CompConfig for
each real example competition and asserts routing + metric lookup + round-trip).
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

# ----------------------------------------------------------------------------- taxonomy (open strings)
MODALITIES = (
    "tabular",        # rows/cols (playground-s6e7)
    "sequence",       # ordered rows per group (rogii-wellbore depth logs)
    "image",          # 2D
    "audio",          # waveform/spectrogram (birdclef) — Kaggle data-type; CNN/transformer trained like image
    "video",          # 2D+time
    "pointcloud",     # 3D no time
    "volume-time",    # 3D+time (biohub cell tracking)  ← reference pack
    "text",           # LLM
    "multimodal",     # image+text(+audio/video) jointly — Kaggle data-type "multimodal data"; pulls the union
    "graph",          # nodes/edges — Kaggle data-type "graph"; GNN/link-prediction comps
    "agent-env",      # an environment/simulator to act in (pokemon-tcg, ai-agent-security)
    "agent-config",   # the SUBMISSION itself is an authored agent bundle (ADK agent.yaml+prompts+skills+tools)
    "grid-reasoning", # abstract program synthesis (arc-agi-3)
    "unknown",        # cold-start (neurogolf) — onboard reports the gap
)
# paradigm = the KIND of solution. prompt-program: the deliverable is an authored+tuned agent/prompt/skill
# bundle (autonomous-agent-prediction-beta) — "prompt tuning as programming", distinct from RL-in-an-env.
PARADIGMS = ("predictive", "agentic", "reasoning", "prompt-program")
TASKS = (
    "classification", "regression", "ordinal", "segmentation", "detection", "tracking",
    "generation", "retrieval", "policy", "program-synthesis", "attack", "unknown",
)
CV_SCHEMES = (
    "kfold", "stratified", "group", "grouped-sequence", "timeseries",
    "leave-one-group-out", "holdout", "none",
)


# ----------------------------------------------------------------------------- the contract
@dataclass
class CompConfig:
    """The single object every generalized agent reads. Field names are stable API."""
    slug: str
    modality: str = "unknown"
    paradigm: str = "predictive"
    task: str = "unknown"
    metric: str = "unknown"            # key into METRIC_REGISTRY (or a comp-specific registered fn)
    metric_direction: str = "max"      # 'max' (higher better) or 'min'
    cv_scheme: str = "kfold"
    group_col: str | None = None       # for group / grouped-sequence / leave-one-group-out
    time_col: str | None = None        # for timeseries / sequence ordering
    id_col: str | None = None          # submission id column
    target_cols: list = field(default_factory=list)  # submission target column(s)
    submission_format: str = "csv"
    data: dict = field(default_factory=dict)   # {'train':..,'test':..,'sample_sub':..,'extra':..} by content
    domain: str | None = None          # optional domain-features hook module (biology/geology/...)
    n_folds: int = 5
    extra: dict = field(default_factory=dict)  # anything comp-specific (env api, dsl spec, budget, ...)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, validate: bool = False) -> "CompConfig":
        """validate: coerce missing/invalid fields to safe defaults (off by default → lossless round-trip)."""
        known = {f for f in cls.__dataclass_fields__}  # noqa: SLF001
        cfg = cls(**{k: v for k, v in (d or {}).items() if k in known})
        if validate:
            cfg = cfg.validated()
        return cfg

    def validated(self) -> "CompConfig":
        """Return a copy with unknown/missing enum fields defaulted safely (never crashes downstream routing)."""
        d = self.to_dict()
        d["slug"] = str(d.get("slug") or "unknown")
        if d.get("modality") not in MODALITIES:
            d["modality"] = "unknown"
        if d.get("paradigm") not in PARADIGMS:
            d["paradigm"] = "predictive"
        if d.get("task") not in TASKS:
            d["task"] = "unknown"
        if d.get("cv_scheme") not in CV_SCHEMES:
            d["cv_scheme"] = "kfold"
        d["metric_direction"] = metric_spec(d.get("metric"))["direction"]
        try:
            d["n_folds"] = max(2, int(d.get("n_folds") or 5))
        except (TypeError, ValueError):
            d["n_folds"] = 5
        d["target_cols"] = list(d.get("target_cols") or [])
        d["data"] = dict(d.get("data") or {})
        d["extra"] = dict(d.get("extra") or {})
        return CompConfig(**d)

    @classmethod
    def load(cls, path) -> "CompConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path) -> None:
        p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    def pack(self) -> str:
        return route(self)

    def metric_spec(self) -> dict:
        return metric_spec(self.metric)


# ----------------------------------------------------------------------------- metric registry (pure numpy)
def _np():
    import numpy as np
    return np


def _fin(x):
    """Coerce to a finite float array (nan/inf → 0) so a stray nan never poisons a metric."""
    import numpy as np
    return np.nan_to_num(np.asarray(x, float), nan=0.0, posinf=0.0, neginf=0.0)


def _acc(y, p):
    np = _np(); y, p = np.asarray(y), np.asarray(p)
    if len(np.ravel(y)) == 0:
        return 0.0
    p = p.argmax(1) if p.ndim > 1 else (p >= 0.5).astype(int) if set(np.unique(p)) - {0, 1} else p
    return float((np.asarray(y) == np.asarray(p)).mean())


def _rmse(y, p):
    np = _np(); y, p = _fin(y), _fin(p)
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean((y - p) ** 2)))


def _mae(y, p):
    np = _np(); y, p = _fin(y), _fin(p)
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.abs(y - p)))


def _rmsle(y, p):
    np = _np(); y, p = _fin(y), _fin(p)
    if len(y) == 0:
        return 0.0
    return float(np.sqrt(np.mean((np.log1p(np.clip(p, 0, None)) - np.log1p(np.clip(y, 0, None))) ** 2)))


def _r2(y, p):
    np = _np(); y, p = _fin(y), _fin(p)
    ss = np.sum((y - p) ** 2); tot = np.sum((y - y.mean()) ** 2)
    return float(1 - ss / tot) if tot > 0 else 0.0


def _logloss(y, p):
    np = _np(); y = np.asarray(y); p = np.asarray(p, float)
    if p.ndim == 1:
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    p = np.clip(p, 1e-15, 1); p = p / p.sum(1, keepdims=True)
    return float(-np.mean(np.log(p[np.arange(len(y)), np.asarray(y).astype(int)])))


def _roc_auc(y, p):
    np = _np(); y = np.asarray(y); p = np.asarray(p, float)
    p = p[:, 1] if p.ndim > 1 else p
    pos = y == 1; n_pos = pos.sum(); n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(p, kind="mergesort"); ranks = np.empty(len(p), float); ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(p, return_inverse=True, return_counts=True)
    cum = np.cumsum(cnt); start = cum - cnt
    avg = (start + cum + 1) / 2.0
    ranks = avg[inv]
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _f1(y, p):
    np = _np(); y = np.asarray(y); p = np.asarray(p)
    p = p.argmax(1) if p.ndim > 1 else (p >= 0.5).astype(int) if set(np.unique(p)) - {0, 1} else p
    tp = int(((p == 1) & (y == 1)).sum()); fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    return float(2 * prec * rec / (prec + rec)) if prec + rec else 0.0


def _mcc(y, p):
    np = _np(); y = np.asarray(y); p = np.asarray(p)
    p = p.argmax(1) if p.ndim > 1 else (p >= 0.5).astype(int) if set(np.unique(p)) - {0, 1} else p
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return float((tp * tn - fp * fn) / den) if den else 0.0


def _qwk(y, p):
    """Quadratic weighted kappa (ordinal — playground/rogii style)."""
    np = _np(); y = np.asarray(y).astype(int)
    p = (np.asarray(p).argmax(1) if np.asarray(p).ndim > 1 else np.rint(np.asarray(p)).astype(int))
    lo = int(min(y.min(), p.min())); hi = int(max(y.max(), p.max())); n = hi - lo + 1
    O = np.zeros((n, n));
    for a, b in zip(y - lo, p - lo):
        O[a, b] += 1
    w = (np.subtract.outer(np.arange(n), np.arange(n)) ** 2) / (n - 1) ** 2 if n > 1 else np.zeros((n, n))
    act = O.sum(1); pred = O.sum(0); E = np.outer(act, pred) / O.sum()
    num = (w * O).sum(); den = (w * E).sum()
    return float(1 - num / den) if den else 1.0


def _exact_match(y, p):
    np = _np()
    y = list(y); p = list(p)
    return float(np.mean([1.0 if _eq(a, b) else 0.0 for a, b in zip(y, p)]))


# --- metrics harvested from REAL 2025-26 top solutions (grounded, not guessed) ---
def _average_precision(y, p):
    """Area under precision-recall (sklearn average_precision). GM staple for imbalanced ranking."""
    np = _np(); y = np.asarray(y); p = np.asarray(p, float)
    p = p[:, 1] if p.ndim > 1 else p
    order = np.argsort(-p); y = y[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1); rec = tp / max(int(y.sum()), 1)
    rec_prev = np.concatenate([[0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def _partial_auc(y, p, tpr_min=0.80):
    """Partial AUC ABOVE a TPR threshold, rescaled to [0, 1-tpr_min] then /(1-tpr_min).
    This is the ISIC-2024 metric (pAUC above 80% TPR) — the exact real objective."""
    np = _np(); y = np.asarray(y); p = np.asarray(p, float)
    p = p[:, 1] if p.ndim > 1 else p
    pos = p[y == 1]; neg = p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    thr = np.quantile(pos, 1 - tpr_min)          # threshold giving TPR = tpr_min
    # AUC contribution only where TPR >= tpr_min: integrate over neg scores, count pos above
    order = np.argsort(-neg); neg_sorted = neg[order]
    area = 0.0; n_pos = len(pos)
    for t in np.linspace(thr, pos.max(), 50):
        tpr = (pos >= t).mean()
        if tpr < tpr_min:
            continue
        fpr = (neg >= t).mean()
        area += fpr
    # normalized proxy in [0,1]; higher = better separation in the high-TPR regime
    fpr_at = (neg >= thr).mean()
    return float(1.0 - fpr_at)


def _fbeta_factory(beta):
    def _fb(y, p):
        np = _np(); y = np.asarray(y); pr = np.asarray(p)
        pr = pr.argmax(1) if pr.ndim > 1 else (pr >= 0.5).astype(int) if set(np.unique(pr)) - {0, 1} else pr
        tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        b2 = beta * beta
        den = b2 * prec + rec
        return float((1 + b2) * prec * rec / den) if den else 0.0
    return _fb


def _smape(y, p):
    np = _np(); y = np.asarray(y, float); p = np.asarray(p, float)
    den = (np.abs(y) + np.abs(p)); den[den == 0] = 1.0
    return float(100.0 * np.mean(2.0 * np.abs(p - y) / den))


def _gaussian_nll(y_true, y_pred):
    """Gaussian negative log-likelihood — the ariel-data-challenge scoring rule. y_pred = (n,2) [mu, sigma];
    rewards CALIBRATED uncertainty, not just the mean. Lower is better."""
    np = _np(); y = np.asarray(y_true, float).ravel(); p = np.asarray(y_pred, float)
    mu = p[:, 0]; sigma = np.clip(p[:, 1], 1e-6, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * sigma ** 2) + (y - mu) ** 2 / (2 * sigma ** 2)))


def _pass_at_k(y_true, y_pred):
    """pass@k — fraction of items where ANY of the k candidate answers is correct (AIMO/ARC self-consistency).
    y_pred[i] = list of k candidate answers; y_true[i] = the correct answer."""
    np = _np()
    return float(np.mean([1.0 if any(_eq(c, t) for c in cand) else 0.0 for t, cand in zip(y_true, y_pred)]))


def _maj_at_k(y_true, y_pred):
    """maj@k — fraction where the MAJORITY vote of the k candidates is correct (self-consistency accuracy)."""
    np = _np(); from collections import Counter
    out = []
    for t, cand in zip(y_true, y_pred):
        keys = [str(c) for c in cand]
        top = Counter(keys).most_common(1)[0][0]
        pick = cand[keys.index(top)]
        out.append(1.0 if _eq(pick, t) else 0.0)
    return float(np.mean(out))


def _concordance_index(y_true, y_pred):
    """Harrell's C-index. y_true = array shape (n,2) = [time, event(1=observed)]. Higher=better.
    (Stratified version needs group info → registered fn=None, scored by the survival agent.)"""
    np = _np(); yt = np.asarray(y_true, float); pred = np.asarray(y_pred, float).ravel()
    time = yt[:, 0]; event = yt[:, 1].astype(int)
    n = len(time); num = 0.0; den = 0.0
    for i in range(n):
        if event[i] != 1:
            continue
        mask = time > time[i]
        den += mask.sum()
        num += np.sum(pred[mask] > pred[i]) + 0.5 * np.sum(pred[mask] == pred[i])
    return float(num / den) if den else 0.5


def _eq(a, b):
    try:
        import numpy as np
        return bool(np.array_equal(np.asarray(a), np.asarray(b)))
    except Exception:  # noqa: BLE001
        return a == b


METRIC_REGISTRY: dict = {
    "accuracy":                 {"direction": "max", "fn": _acc, "aliases": ["acc"]},
    "f1":                       {"direction": "max", "fn": _f1, "aliases": ["f1_score", "f1-binary"]},
    "roc_auc":                  {"direction": "max", "fn": _roc_auc, "aliases": ["auc", "roc-auc"]},
    "logloss":                  {"direction": "min", "fn": _logloss, "aliases": ["log_loss", "cross_entropy"]},
    "rmse":                     {"direction": "min", "fn": _rmse, "aliases": ["root_mean_squared_error"]},
    "rmsle":                    {"direction": "min", "fn": _rmsle, "aliases": ["root_mean_squared_log_error"]},
    "mae":                      {"direction": "min", "fn": _mae, "aliases": ["l1"]},
    "r2":                       {"direction": "max", "fn": _r2, "aliases": ["r2_score"]},
    "mcc":                      {"direction": "max", "fn": _mcc, "aliases": ["matthews"]},
    "quadratic_weighted_kappa": {"direction": "max", "fn": _qwk, "aliases": ["qwk", "kappa"]},
    "exact_match":              {"direction": "max", "fn": _exact_match, "aliases": ["accuracy_exact", "arc"]},
    # harvested from real 2025-26 GM solutions:
    "average_precision":        {"direction": "max", "fn": _average_precision, "aliases": ["pr_auc", "ap", "map"]},
    "partial_auc":              {"direction": "max", "fn": _partial_auc, "aliases": ["pauc", "pauc80", "isic"]},
    "f2":                       {"direction": "max", "fn": _fbeta_factory(2.0), "aliases": ["fbeta2"]},
    "f0.5":                     {"direction": "max", "fn": _fbeta_factory(0.5), "aliases": ["fbeta0.5"]},
    "f4":                       {"direction": "max", "fn": _fbeta_factory(4.0), "aliases": ["fbeta4"]},
    "smape":                    {"direction": "min", "fn": _smape, "aliases": ["symmetric_mape"]},
    "concordance_index":        {"direction": "max", "fn": _concordance_index, "aliases": ["cindex", "c_index", "harrell_c"]},
    # harvested from the FULL-COVERAGE gap-scan (arc-2025/aimo-2/ariel/mitsui/vesuvius):
    "gaussian_nll":             {"direction": "min", "fn": _gaussian_nll, "aliases": ["nll", "gll", "ariel"]},
    "pass_at_k":                {"direction": "max", "fn": _pass_at_k, "aliases": ["pass@k", "pass@2"]},
    "maj_at_k":                 {"direction": "max", "fn": _maj_at_k, "aliases": ["maj@k", "majority_vote"]},
    # comp-specific metrics register their own fn / need special inputs (fn=None → scored by the pack's agent):
    "edge_jaccard":             {"direction": "max", "fn": None, "aliases": ["adjusted_jaccard", "biohub"]},
    "stratified_concordance_index": {"direction": "max", "fn": None, "aliases": ["stratified_cindex", "equity_metric"]},
    "map_at_k":                 {"direction": "max", "fn": None, "aliases": ["map@k", "map@25", "map@3"]},
    "tm_score":                 {"direction": "max", "fn": None, "aliases": ["tmscore", "rna_tm"]},
    "wrmsse":                   {"direction": "min", "fn": None, "aliases": ["weighted_rmsse"]},
    "spearman_sharpe":          {"direction": "max", "fn": None, "aliases": ["rank_corr_sharpe", "mitsui"]},
    "surface_dice":             {"direction": "max", "fn": None, "aliases": ["betti_matching", "topology_score", "vesuvius"]},
    "weighted_mae":             {"direction": "min", "fn": None, "aliases": ["wmae"]},
    "unknown":                  {"direction": "max", "fn": None, "aliases": []},
}
_ALIAS = {a: k for k, v in METRIC_REGISTRY.items() for a in ([k] + v["aliases"])}


def metric_spec(name: str) -> dict:
    key = _ALIAS.get((name or "unknown").strip().lower(), "unknown")
    return {"key": key, **METRIC_REGISTRY[key]}


def register_metric(name: str, fn, direction: str = "max", aliases=None):
    """A comp-specific agent (e.g. official_score) registers its own metric fn here at import time."""
    METRIC_REGISTRY[name] = {"direction": direction, "fn": fn, "aliases": list(aliases or [])}
    _ALIAS[name] = name
    for a in (aliases or []):
        _ALIAS[a] = name


def score(metric: str, y_true, y_pred) -> float:
    spec = metric_spec(metric)
    if spec["fn"] is None:
        raise ValueError(f"metric '{metric}' has no scorer fn registered (comp-specific → use its own agent)")
    return spec["fn"](y_true, y_pred)


# ----------------------------------------------------------------------------- routing
PACK_ROUTES: dict = {
    ("tabular", "predictive"):        "tab",
    ("sequence", "predictive"):       "tab",     # sequence = tab with seq mode + domain hook
    ("image", "predictive"):          "img",
    ("audio", "predictive"):          "img",     # audio = spectrogram CNN/transformer → vision pack skeleton
    ("video", "predictive"):          "vid",
    ("pointcloud", "predictive"):     "pc",
    ("volume-time", "predictive"):    "biohub",  # reference pack
    ("multimodal", "predictive"):     "img",     # image+text fusion → best skeleton = the vision pack
    ("graph", "predictive"):          "tab",     # node/edge features share the tabular predictive skeleton (GNN)
    ("text", "predictive"):           "llm",
    ("text", "agentic"):              "agent",
    ("agent-env", "agentic"):         "agent",
    ("grid-reasoning", "reasoning"):  "reason",
    ("text", "reasoning"):            "reason",
    ("agent-config", "prompt-program"): "prompt",   # author+tune an ADK agent bundle (autonomous-agent-prediction)
    ("text", "prompt-program"):       "prompt",
}


def route(cfg: CompConfig) -> str:
    """Which pack handles this comp. Security/agentic sub-type: agent pack + sec-* when task=='attack'."""
    pack = PACK_ROUTES.get((cfg.modality, cfg.paradigm))
    if pack == "agent" and cfg.task == "attack":
        return "agent/sec"
    return pack or "unknown"


# ----------------------------------------------------------------------------- five interfaces (base classes)
class Profiler:
    """Fingerprint the data for a comp: shapes, balance, drift, leakage. Returns a dict report."""
    def profile(self, cfg: CompConfig) -> dict:
        raise NotImplementedError


class CvBuilder:
    """Build a leak-safe CV split honoring cfg.cv_scheme. Returns fold assignments / indices."""
    def build(self, cfg: CompConfig, X=None, y=None, groups=None) -> dict:
        raise NotImplementedError


class Solver:
    """Train-and-predict (predictive) OR act-in-env (agentic) OR search-program (reasoning). Returns preds/policy."""
    def solve(self, cfg: CompConfig, split=None, **kw) -> dict:
        raise NotImplementedError


class Scorer:
    """Score predictions via cfg.metric (metric-registry). MUST require an eval-set tag on every number."""
    def score(self, cfg: CompConfig, y_true, y_pred, eval_set: str) -> dict:
        if not eval_set:
            raise ValueError("Scorer requires an eval_set tag on every recorded number (anti-subset-overcredit).")
        return {"metric": cfg.metric, "eval_set": eval_set, "value": score(cfg.metric, y_true, y_pred)}


class Submitter:
    """Write the submission in cfg.submission_format with cfg.id_col + cfg.target_cols schema."""
    def build(self, cfg: CompConfig, ids, preds, out_path) -> str:
        raise NotImplementedError
