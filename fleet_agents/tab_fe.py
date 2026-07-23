"""tab-fe — the GRANDMASTER feature-engineering layer for tabular comps. This is where our tabular quality
moves from "working baseline" to "matches a Kaggle GM solution". All transforms are LEAK-SAFE:

  • OUT-OF-FOLD target (mean) encoding with smoothing — the #1 tabular GM lever for categoricals /
    high-cardinality IDs. Each row's encoding uses ONLY the other folds; test uses full-train means.
  • Frequency / count encoding — leak-free (not target-dependent).
  • Row aggregates across numerics (mean/std/min/max/median) — cheap signal GMs always add.
  • Top pairwise interactions (products of the highest-variance numerics).

Uses the SAME CV folds as tab-train (nested-safe). Returns augmented numeric matrices ready to train on.
Pure pandas/numpy. tab-train calls this when fe is on; a plain run falls back to tab_common.basic_features.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseAgent
from . import comp_config as CC
from . import tab_common as TC


def _oof_target_encode(col_tr, y, folds, col_te=None, smoothing=10.0):
    """Leak-safe OOF mean-target encoding of one categorical column. Returns (oof_tr array, te array|None)."""
    col_tr = col_tr.astype(str).fillna("NA").to_numpy()
    y = np.nan_to_num(np.asarray(y, float), nan=0.0, posinf=0.0, neginf=0.0)
    global_mean = float(y.mean()) if len(y) else 0.0
    smoothing = max(float(smoothing), 1e-9)
    oof = np.full(len(col_tr), global_mean, float)
    for tr_idx, va_idx in folds:
        df = pd.DataFrame({"c": col_tr[tr_idx], "y": y[tr_idx]})
        agg = df.groupby("c")["y"].agg(["sum", "count"])
        enc = (agg["sum"] + smoothing * global_mean) / (agg["count"] + smoothing)
        m = enc.to_dict()
        oof[va_idx] = [m.get(c, global_mean) for c in col_tr[va_idx]]
    te = None
    if col_te is not None:
        col_te = col_te.astype(str).fillna("NA").to_numpy()
        df = pd.DataFrame({"c": col_tr, "y": y})
        agg = df.groupby("c")["y"].agg(["sum", "count"])
        enc = ((agg["sum"] + smoothing * global_mean) / (agg["count"] + smoothing)).to_dict()
        te = np.array([enc.get(c, global_mean) for c in col_te], float)
    return oof, te


def engineer(df_train, df_test, cfg, y, folds, max_interactions=6,
             target_encode=True, freq_encode=True, row_aggs=True, smoothing=10.0):
    """Return (Xtr np, Xte np|None, feature_names). Leak-safe GM FE on top of basic imputation/encoding.
    target_encode: add OOF mean-target encoding of categoricals (the #1 GM categorical lever).
    freq_encode: add leak-free frequency/count encoding of categoricals.
    row_aggs: add per-row aggregate stats (mean/std/min/max/median) over the numeric block.
    smoothing: Bayesian smoothing strength for target encoding (higher = more shrink to the global mean)."""
    tgt = TC._target_name(cfg, df_train)
    feats = [c for c in df_train.columns if c not in [cfg.id_col, tgt]]
    cat = [c for c in feats if df_train[c].dtype == object or str(df_train[c].dtype).startswith("category")]
    num = [c for c in feats if c not in cat]
    parts_tr, parts_te, names = [], [], []

    def _med(col):
        m = float(col.median())
        return m if np.isfinite(m) else 0.0

    # numerics: median impute (fit on train)
    for c in num:
        med = _med(df_train[c].replace([np.inf, -np.inf], np.nan))
        parts_tr.append(df_train[c].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(float)[:, None])
        names.append(c)
        if df_test is not None:
            parts_te.append(df_test[c].replace([np.inf, -np.inf], np.nan).fillna(med).to_numpy(float)[:, None])

    # categoricals: OOF target-encode + frequency-encode (leak-safe)
    for c in cat:
        if target_encode:
            oof, te = _oof_target_encode(df_train[c], y, folds, df_test[c] if df_test is not None else None,
                                         smoothing=smoothing)
            parts_tr.append(oof[:, None]); names.append(f"{c}__te")
            if df_test is not None:
                parts_te.append(te[:, None])
        if freq_encode:
            freq = df_train[c].astype(str).fillna("NA").value_counts(normalize=True).to_dict()
            parts_tr.append(df_train[c].astype(str).fillna("NA").map(freq).fillna(0).to_numpy(float)[:, None])
            names.append(f"{c}__freq")
            if df_test is not None:
                parts_te.append(df_test[c].astype(str).fillna("NA").map(freq).fillna(0).to_numpy(float)[:, None])

    if not parts_tr:                                   # no usable features → single zero column
        Xtr = np.zeros((len(df_train), 1), float); names = ["__const__"]
        Xte = np.zeros((len(df_test), 1), float) if df_test is not None else None
        return Xtr.astype("float32"), (Xte.astype("float32") if Xte is not None else None), names
    Xtr = np.hstack(parts_tr); Xte = np.hstack(parts_te) if df_test is not None else None

    # row aggregates over the numeric block (cheap GM signal)
    if row_aggs and len(num) >= 3:
        nb_tr = df_train[num].fillna(df_train[num].median()).to_numpy(float)
        aggs_tr = np.column_stack([nb_tr.mean(1), nb_tr.std(1), nb_tr.min(1), nb_tr.max(1), np.median(nb_tr, 1)])
        Xtr = np.hstack([Xtr, aggs_tr]); names += ["row_mean", "row_std", "row_min", "row_max", "row_med"]
        if df_test is not None:
            nb_te = df_test[num].fillna(df_train[num].median()).to_numpy(float)
            aggs_te = np.column_stack([nb_te.mean(1), nb_te.std(1), nb_te.min(1), nb_te.max(1), np.median(nb_te, 1)])
            Xte = np.hstack([Xte, aggs_te])

    # top pairwise interactions (highest-variance numerics)
    if len(num) >= 2 and max_interactions > 0:
        var = df_train[num].var().sort_values(ascending=False)
        top = list(var.index[:max(2, int(np.sqrt(2 * max_interactions)) + 1)])
        pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]][:max_interactions]
        for a, b in pairs:
            ma, mb = float(df_train[a].median()), float(df_train[b].median())
            Xtr = np.hstack([Xtr, (df_train[a].fillna(ma) * df_train[b].fillna(mb)).to_numpy(float)[:, None]])
            names.append(f"{a}*{b}")
            if df_test is not None:
                Xte = np.hstack([Xte, (df_test[a].fillna(ma) * df_test[b].fillna(mb)).to_numpy(float)[:, None]])
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    if Xte is not None:
        Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    return Xtr.astype("float32"), (Xte.astype("float32") if Xte is not None else None), names


class TabFe(BaseAgent):
    name = "tab-fe"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        if "config" not in spec and "config_file" not in spec:
            return self.escalate(worker, "leader", "tab-fe needs spec keys ['config' or 'config_file'] — none provided")
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else CC.CompConfig.load(spec["config_file"])
        df_train, df_test, _ = TC.load_frames(cfg)
        tgt = TC._target_name(cfg, df_train)
        y = df_train[tgt].to_numpy().astype(float)
        folds = TC.make_cv(cfg, y, seed=int(spec.get("seed", 42)))
        Xtr, Xte, names = engineer(df_train, df_test, cfg, y, folds,
                                   max_interactions=int(spec.get("max_interactions", 6)),
                                   target_encode=bool(spec.get("target_encode", True)),
                                   freq_encode=bool(spec.get("freq_encode", True)),
                                   row_aggs=bool(spec.get("row_aggs", True)),
                                   smoothing=float(spec.get("smoothing", 10.0)))
        added = [n for n in names if "__te" in n or "__freq" in n or "row_" in n or "*" in n]
        msg = (f"tab-fe {cfg.slug}: {Xtr.shape[1]} features ({len(added)} engineered: "
               f"target/freq-enc + row-aggs + interactions). Leak-safe OOF encoding.")
        self.log(msg, kind="finding", recommendation="train with fe=True; verify uplift vs basic via paired_delta")
        return self.done({"n_features": int(Xtr.shape[1]), "engineered": added, "feature_names": names}, msg)


_AGENT = TabFe()


def run(q, worker):
    return _AGENT.run(q, worker)
