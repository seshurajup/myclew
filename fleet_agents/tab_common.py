"""tab_common — shared, REUSABLE tabular helpers used by every tab-* agent (load / basic-FE / CV / submit).

Kept in ONE place so no tab agent duplicates data loading, feature engineering, fold construction, or the
submission write — the reusability discipline. All functions take a `CompConfig` and are pure w.r.t. the
comp identity (no hard-coded slug/paths). Backends are handled elsewhere (tab_train); this file is data
plumbing + the leak-safe CV that a generalized `split-build` will later delegate to.

Pure pandas/numpy/sklearn — no torch, no biohub.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


def _safe_read_csv(path, **kw):
    """Read a CSV robustly; return None on any failure (missing file / parse error) instead of crashing."""
    try:
        return pd.read_csv(path, **kw)
    except Exception:  # noqa: BLE001
        return None


def load_frames(cfg, read_kw=None):
    """Read train/test frames from cfg.data ({'train':path,'test':path,'sample_sub':path}). Returns
    (df_train, df_test, df_sample) — df_test/df_sample may be None.
    read_kw: optional dict of extra kwargs forwarded to pandas.read_csv (e.g. {'sep':';'})."""
    d = cfg.data or {}
    read_kw = dict(read_kw or {})
    df_train = _safe_read_csv(d["train"], **read_kw) if d.get("train") else None
    df_test = _safe_read_csv(d["test"], **read_kw) if d.get("test") else None
    df_sample = _safe_read_csv(d["sample_sub"], **read_kw) if d.get("sample_sub") else None
    return df_train, df_test, df_sample


def _safe_median(series):
    """Median that never returns NaN (all-NaN / empty column → 0.0) — keeps downstream matrices finite."""
    try:
        m = float(series.median())
        return m if np.isfinite(m) else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def _target_name(cfg, df_train):
    if cfg.target_cols:
        return cfg.target_cols[0]
    # fall back: last column not the id
    cols = [c for c in df_train.columns if c != cfg.id_col]
    return cols[-1]


def basic_features(df_train, df_test, cfg, drop_constant=False, clip_inf=True):
    """Leak-safe basic FE: drop id, label-encode categoricals (leakage-free), median-impute numerics
    (fit on train). Returns X_train(df), y(np), X_test(df|None), feature_names, target_name, encoders.
    This is the turnkey default; richer FE is a tab-fe refinement layered on top.
    drop_constant: if True, drop zero-variance (constant) numeric columns (no signal, safe to remove).
    clip_inf: if True, replace ±inf with the column median before imputation (keeps matrices finite)."""
    tgt = _target_name(cfg, df_train)
    gcol = getattr(cfg, "group_col", None)
    drop = [c for c in [cfg.id_col, tgt, gcol] if c and c in df_train.columns]
    feats = [c for c in df_train.columns if c not in drop]
    Xtr = df_train[feats].copy()
    Xte = df_test[feats].copy() if df_test is not None else None
    encoders = {}
    kept = []
    for c in feats:
        if Xtr[c].dtype == object or str(Xtr[c].dtype).startswith("category"):
            # label-encode on the UNION of categories (leakage-free — mapping ≠ target)
            cats = pd.Index(Xtr[c].astype(str).fillna("NA").unique())
            if Xte is not None:
                cats = cats.union(pd.Index(Xte[c].astype(str).fillna("NA").unique()))
            mapping = {v: i for i, v in enumerate(cats)}
            encoders[c] = mapping
            Xtr[c] = Xtr[c].astype(str).fillna("NA").map(mapping).astype("float32")
            if Xte is not None:
                Xte[c] = Xte[c].astype(str).fillna("NA").map(mapping).fillna(-1).astype("float32")
        else:
            if clip_inf:
                Xtr[c] = Xtr[c].replace([np.inf, -np.inf], np.nan)
                if Xte is not None:
                    Xte[c] = Xte[c].replace([np.inf, -np.inf], np.nan)
            med = _safe_median(Xtr[c])
            encoders[c] = {"__median__": med}
            Xtr[c] = Xtr[c].fillna(med).astype("float32")
            if Xte is not None:
                Xte[c] = Xte[c].fillna(med).astype("float32")
            if drop_constant and float(np.nanstd(Xtr[c].to_numpy())) <= 1e-12:
                continue
        kept.append(c)
    if drop_constant and len(kept) < len(feats):
        feats = kept
        Xtr = Xtr[feats]
        if Xte is not None:
            Xte = Xte[feats]
    y = df_train[tgt].to_numpy() if tgt in df_train.columns else None
    return Xtr, y, Xte, feats, tgt, encoders


def make_cv(cfg, y, groups=None, seed=42, n_folds=None):
    """Leak-safe fold list honoring cfg.cv_scheme. Returns [(train_idx, val_idx), ...].
    This IS the generalized split logic; split-build will later delegate here for tabular/sequence comps.
    n_folds: optional override of cfg.n_folds (clamped to a valid range for the data size)."""
    from sklearn.model_selection import (KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit)
    n = len(y) if y is not None else 0
    k = int(n_folds if n_folds is not None else (cfg.n_folds or 5))
    k = max(2, min(k, max(2, n)))                       # never request more folds than samples
    scheme = cfg.cv_scheme or "kfold"
    idx = np.arange(n)
    if n < 2:                                           # single-row / empty → degenerate single fold
        return [(idx, idx)]
    try:
        if scheme in ("stratified",) and y is not None:
            yb = y if _is_discrete(y) else _bin(y)
            _, cnt = np.unique(yb, return_counts=True)
            k = max(2, min(k, int(cnt.min())))          # StratifiedKFold needs k <= smallest class count
            return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(idx, yb))
        if scheme in ("group", "leave-one-group-out", "grouped-sequence") and groups is not None:
            # group / grouped-sequence → K disjoint groups per fold; LOGO → one group per fold
            ng = len(np.unique(groups))
            kk = ng if scheme == "leave-one-group-out" else min(k, ng)
            return list(GroupKFold(max(2, kk)).split(idx, y, groups))
        if scheme in ("timeseries", "grouped-sequence"):
            return list(TimeSeriesSplit(k).split(idx))
        if scheme in ("purged-embargo", "purged"):
            return _purged_embargo_folds(n, k, embargo=max(1, int(0.01 * n)))
        return list(KFold(k, shuffle=True, random_state=seed).split(idx))
    except Exception:  # noqa: BLE001 — any splitter constraint violation → plain KFold fallback
        return list(KFold(k, shuffle=True, random_state=seed).split(idx))


def _purged_embargo_folds(n, k, embargo=1):
    """Combinatorial-purged K-fold with an embargo gap — the mitsui/finance CV for autocorrelated targets:
    each contiguous block is a validation fold; training excludes the block PLUS an embargo buffer on both
    sides so horizon-overlapping (leaky) samples don't leak into train."""
    import numpy as np
    idx = np.arange(n); bounds = np.linspace(0, n, k + 1).astype(int); folds = []
    for i in range(k):
        va = idx[bounds[i]:bounds[i + 1]]
        lo = max(0, bounds[i] - embargo); hi = min(n, bounds[i + 1] + embargo)
        tr = np.concatenate([idx[:lo], idx[hi:]])
        folds.append((tr, va))
    return folds


def _is_discrete(y):
    y = np.asarray(y)
    return y.dtype.kind in "iub" or (y.dtype.kind == "f" and np.all(np.equal(np.mod(y[~np.isnan(y)], 1), 0)) and len(np.unique(y)) <= 50)


def _bin(y, bins=10):
    y = np.nan_to_num(np.asarray(y, float), nan=0.0, posinf=0.0, neginf=0.0)
    q = np.quantile(y, np.linspace(0, 1, bins + 1))
    return np.clip(np.digitize(y, np.unique(q)[1:-1]), 0, bins - 1)


def write_submission(cfg, ids, preds, out_path, sanitize=True):
    """Generic tabular submission writer: id_col + target_cols. Multiclass preds → argmax unless the
    sample schema wants probabilities. Returns the path written.
    sanitize: if True, replace NaN/±inf in numeric predictions with 0 so the CSV is always valid."""
    out = Path(out_path); out.parent.mkdir(parents=True, exist_ok=True)
    preds = np.asarray(preds)
    if sanitize and preds.dtype.kind in "fc":
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
    tgt = cfg.target_cols or ["target"]
    df = pd.DataFrame({cfg.id_col or "id": np.asarray(ids)})
    if preds.ndim == 1 or len(tgt) == 1:
        col = tgt[0]
        df[col] = preds if preds.ndim == 1 else preds.argmax(1)
    else:
        for j, c in enumerate(tgt):
            df[c] = preds[:, j]
    df.to_csv(out, index=False)
    return str(out)
