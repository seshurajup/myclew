"""tab-profile — fingerprint a tabular/sequence competition's data so downstream agents (and a human) know
what they're dealing with BEFORE training: shape, dtypes, cardinality, missingness, target balance, a
cheap train/test drift signal, and a leakage sniff (a feature that predicts the target too perfectly).

Reads the CompConfig (no hard-coded slug). Reuses tab_common for loading. Pure pandas/numpy/sklearn.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import BaseAgent
from . import comp_config as CC
from . import tab_common as TC


def profile(cfg, top_k=5):
    """Fingerprint the comp data. top_k: how many top missing/drift features to list in the report."""
    df_train, df_test, _ = TC.load_frames(cfg)
    if df_train is None or len(df_train) == 0:         # missing/empty train → minimal safe report
        return {"slug": cfg.slug, "n_train": 0, "n_test": 0, "n_features": 0, "target": None,
                "n_categorical": 0, "n_numeric": 0, "high_cardinality": [], "top_missing": {},
                "cv_scheme": cfg.cv_scheme, "metric": f"{cfg.metric}({cfg.metric_direction})",
                "leakage_suspects": []}
    top_k = max(1, int(top_k))
    tgt = TC._target_name(cfg, df_train)
    feats = [c for c in df_train.columns if c not in [cfg.id_col, tgt]]
    rep = {"slug": cfg.slug, "n_train": int(len(df_train)),
           "n_test": int(len(df_test)) if df_test is not None else 0,
           "n_features": len(feats), "target": tgt}
    # dtypes / cardinality / missing
    cat = [c for c in feats if df_train[c].dtype == object or str(df_train[c].dtype).startswith("category")]
    num = [c for c in feats if c not in cat]
    rep["n_categorical"] = len(cat); rep["n_numeric"] = len(num)
    rep["high_cardinality"] = [c for c in cat if df_train[c].nunique() > 50]
    miss = (df_train[feats].isna().mean()).sort_values(ascending=False) if feats else pd.Series(dtype=float)
    rep["top_missing"] = {c: round(float(miss[c]), 4) for c in miss.index[:top_k] if miss[c] > 0}
    # target balance / range
    y = df_train[tgt]
    if y.nunique() <= 20:
        vc = y.value_counts(normalize=True)
        rep["target_balance"] = {str(k): round(float(v), 4) for k, v in vc.items()}
        rep["target_type"] = "discrete"
    else:
        rep["target_type"] = "continuous"
        rep["target_stats"] = {"mean": round(float(y.mean()), 4), "std": round(float(y.std()), 4),
                               "min": float(y.min()), "max": float(y.max())}
    # cheap drift: standardized mean shift per numeric feature (train vs test)
    if df_test is not None and num:
        drift = {}
        for c in num:
            if c not in df_test.columns:
                continue
            a = df_train[c].astype(float).replace([np.inf, -np.inf], np.nan)
            b = df_test[c].astype(float).replace([np.inf, -np.inf], np.nan)
            s = float(a.std())
            s = s if np.isfinite(s) and s > 1e-12 else 1.0
            dv = abs(float(a.mean() - b.mean()) / s)
            drift[c] = dv if np.isfinite(dv) else 0.0
        top = sorted(drift.items(), key=lambda kv: -kv[1])[:top_k]
        rep["top_drift"] = {c: round(v, 4) for c, v in top}
        rep["max_drift"] = round(max(drift.values()), 4) if drift else 0.0
    # leakage sniff: numeric feature with |corr|>0.999 to target
    if rep["target_type"] == "discrete" or True:
        leak = []
        try:
            yv = np.nan_to_num(y.astype(float).to_numpy(), nan=0.0, posinf=0.0, neginf=0.0)
            for c in num:
                col = np.nan_to_num(df_train[c].astype(float).fillna(0).to_numpy(), posinf=0.0, neginf=0.0)
                if np.std(col) <= 1e-12:
                    continue
                cc = np.corrcoef(col, yv)[0, 1]
                if np.isfinite(cc) and abs(cc) > 0.999:
                    leak.append(c)
        except Exception:  # noqa: BLE001
            pass
        rep["leakage_suspects"] = leak
    # recommended cv scheme sanity
    rep["cv_scheme"] = cfg.cv_scheme
    rep["metric"] = f"{cfg.metric}({cfg.metric_direction})"
    return rep


class TabProfile(BaseAgent):
    name = "tab-profile"
    thread = "A"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        if "config" not in spec and "config_file" not in spec:
            return self.escalate(worker, "leader", "tab-profile needs spec keys ['config' or 'config_file'] — none provided")
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else CC.CompConfig.load(spec["config_file"])
        rep = profile(cfg, top_k=int(spec.get("top_k", 5)))
        warn = []
        if rep.get("max_drift", 0) > 1.0:
            warn.append(f"train/test drift high (max {rep['max_drift']}σ)")
        if rep.get("leakage_suspects"):
            warn.append(f"LEAKAGE suspects: {rep['leakage_suspects']}")
        msg = (f"tab-profile {cfg.slug}: {rep['n_train']}×{rep['n_features']} (cat={rep['n_categorical']}/"
               f"num={rep['n_numeric']}), target={rep['target']}({rep['target_type']}), metric={rep['metric']}"
               + (f" ⚠ {'; '.join(warn)}" if warn else ""))
        self.log(msg, kind="finding", recommendation="; ".join(warn) or "clean — proceed to tab-train")
        return self.done({"profile": rep, "warnings": warn}, msg)


_AGENT = TabProfile()


def run(q, worker):
    return _AGENT.run(q, worker)
