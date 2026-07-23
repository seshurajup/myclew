"""tab-train — CV-train the tabular backends that are installed (LightGBM/XGBoost/CatBoost/HistGBM),
producing out-of-fold + test predictions and an HONEST CV score via the CompConfig metric-registry.

Backends are OPTIONAL: the agent uses whatever the env provides (xgboost + sklearn-HistGBM are the
guaranteed set here; lightgbm/catboost join automatically in the kaggle_tabular env) and logs which ran —
so the same agent is reusable across envs without edits. It does NOT re-implement CV, FE, or scoring: it
CALLS tab_common (fold + FE) and comp_config.score (metric). GPU is used automatically for xgboost.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent
from . import comp_config as CC
from . import tab_common as TC


_LGBM_CUDA = None


def _lgbm_cuda_ok():
    """True iff the installed LightGBM build has the CUDA tree learner (a CPU-only PyPI wheel accepts
    device='cuda' at construction but FAILS at fit). Probed once with a tiny fit and cached."""
    global _LGBM_CUDA
    if _LGBM_CUDA is None:
        try:
            import lightgbm as lgb, numpy as _np
            lgb.LGBMRegressor(device="cuda", n_estimators=1, min_child_samples=1, verbose=-1).fit(
                _np.zeros((4, 2)), _np.arange(4))
            _LGBM_CUDA = True
        except Exception:  # noqa: BLE001
            _LGBM_CUDA = False
    return _LGBM_CUDA


def _cuda(gpu=None):
    """True if GPU should be used. gpu=None → auto-detect via torch; gpu=True/False forces it."""
    if gpu is not None:
        return bool(gpu)
    try:
        import torch; return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


def _task_kind(cfg, y):
    if cfg.task in ("regression",) or cfg.metric in ("rmse", "rmsle", "mae", "r2"):
        return "regression"
    return "classification"


def available_backends():
    b = []
    try:
        import xgboost  # noqa: F401
        b.append("xgb")
    except Exception:  # noqa: BLE001
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
        b.append("histgbm")
    except Exception:  # noqa: BLE001
        pass
    try:
        import lightgbm  # noqa: F401
        b.append("lgbm")
    except Exception:  # noqa: BLE001
        pass
    try:
        import catboost  # noqa: F401
        b.append("catboost")
    except Exception:  # noqa: BLE001
        pass
    return b


def _make(backend, kind, n_classes, seed, gpu=None, params=None):
    """Build a backend model. gpu: force GPU on/off (None=auto). params: per-backend kwarg override dict."""
    dev = "cuda" if _cuda(gpu) else "cpu"
    ov = dict((params or {}).get(backend, {}) if isinstance(params, dict) else {})
    if backend == "xgb":
        import xgboost as xgb
        kw = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", device=dev, random_state=seed, n_jobs=0)
        kw.update(ov)
        return xgb.XGBRegressor(**kw) if kind == "regression" else xgb.XGBClassifier(**kw, eval_metric="logloss")
    if backend == "histgbm":
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        kw = dict(max_iter=300, learning_rate=0.05, max_depth=None, random_state=seed)
        kw.update(ov)
        return HistGradientBoostingRegressor(**kw) if kind == "regression" else HistGradientBoostingClassifier(**kw)
    if backend == "lgbm":
        import lightgbm as lgb
        kw = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8,
                  colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
        # LightGBM v4.7.0 fast CUDA backend (device="cuda"): saturate the 5090 per the always-GPU rule.
        # CUDA histograms cap max_bin at 255; honor the low-bit rule (bits→max_bin: 8→255, 4→15) via `ov`.
        if _cuda(gpu) and _lgbm_cuda_ok():                           # only when the build truly has CUDA
            kw["device"] = "cuda"
            kw.setdefault("max_bin", 255)                             # CUDA hist limit / low-bit default
        kw.update(ov)
        return lgb.LGBMRegressor(**kw) if kind == "regression" else lgb.LGBMClassifier(**kw)
    if backend == "catboost":
        from catboost import CatBoostRegressor, CatBoostClassifier
        kw = dict(iterations=400, learning_rate=0.05, depth=6, random_seed=seed, verbose=0,
                  task_type="GPU" if _cuda(gpu) else "CPU")
        kw.update(ov)
        return CatBoostRegressor(**kw) if kind == "regression" else CatBoostClassifier(**kw)
    raise ValueError(backend)


def _predict(model, kind, n_classes, X):
    if kind == "regression":
        return np.asarray(model.predict(X), float)
    proba = model.predict_proba(X)
    return proba[:, 1] if n_classes == 2 else proba  # binary→1D positive-prob; multiclass→2D


def _fit_es(model, backend, Xtr, ytr, Xva, yva):
    """Fit with early stopping on (Xva,yva) when the backend supports it; else plain fit. Never raises."""
    try:
        if backend == "xgb":
            try:
                model.set_params(early_stopping_rounds=30)
            except Exception:  # noqa: BLE001
                pass
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            return model
        if backend == "lgbm":
            import lightgbm as lgb
            cbs = [lgb.early_stopping(30, verbose=False)]
            model.fit(Xtr, ytr, eval_set=[(Xva, yva)], callbacks=cbs)
            return model
        if backend == "catboost":
            model.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=30, verbose=0)
            return model
    except Exception:  # noqa: BLE001
        pass
    model.fit(Xtr, ytr)
    return model


def train_backends(cfg, backends=None, seed=42, fe=False, n_folds=None, gpu=None,
                   early_stopping=False, params=None):
    """Returns {backend: {'oof':array,'test':array|None,'cv':float}} + shared y/ids/test_ids.
    fe=True routes through the grandmaster tab-fe layer (fold-safe target/freq encoding + interactions),
    using the SAME folds as CV so target encoding stays leak-safe.
    n_folds: override the CV fold count (else cfg.n_folds).
    gpu: force GPU on/off for backends that support it (None=auto-detect).
    early_stopping: if True, hold out the fold's validation block as an early-stopping eval set (GBDTs).
    params: per-backend hyperparameter override dict, e.g. {'xgb': {'max_depth': 8}}."""
    df_train, df_test, df_sample = TC.load_frames(cfg)
    tgt = TC._target_name(cfg, df_train)
    y = df_train[tgt].to_numpy()
    kind = _task_kind(cfg, y)
    n_classes = int(len(np.unique(y))) if kind == "classification" else 1
    # well-disjoint / grouped CV: extract the group column, then drop it so it is never a feature
    gcol = getattr(cfg, "group_col", None)
    groups = None
    if gcol and df_train is not None and gcol in df_train.columns:
        groups = df_train[gcol].to_numpy()
        df_train = df_train.drop(columns=[gcol])
        if df_test is not None and gcol in df_test.columns:
            df_test = df_test.drop(columns=[gcol])
    folds = TC.make_cv(cfg, y, groups=groups, seed=seed, n_folds=n_folds)
    if fe:
        from . import tab_fe as FE
        y_te = y if kind == "regression" else y.astype(float)  # target-encode on numeric target
        Xtr_v, Xte_v, feats = FE.engineer(df_train, df_test, cfg, y_te, folds)
    else:
        Xtr, _y, Xte, feats, _tgt, _ = TC.basic_features(df_train, df_test, cfg)
        Xtr_v = Xtr.to_numpy(); Xte_v = Xte.to_numpy() if Xte is not None else None
    # sanitize feature matrices — no NaN/inf reaches the backends
    Xtr_v = np.nan_to_num(np.asarray(Xtr_v, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if Xte_v is not None:
        Xte_v = np.nan_to_num(np.asarray(Xte_v, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    backends = backends or available_backends()
    out = {}
    for b in backends:
        oof = np.zeros(len(y)) if (kind == "regression" or n_classes == 2) else np.zeros((len(y), n_classes))
        test_acc = None
        for tr, va in folds:
            m = _make(b, kind, n_classes, seed, gpu=gpu, params=params)
            if early_stopping and b in ("xgb", "lgbm", "catboost"):
                _fit_es(m, b, Xtr_v[tr], y[tr], Xtr_v[va], y[va])
            else:
                m.fit(Xtr_v[tr], y[tr])
            oof[va] = _predict(m, kind, n_classes, Xtr_v[va])
            if Xte_v is not None:
                tp = _predict(m, kind, n_classes, Xte_v)
                test_acc = tp if test_acc is None else test_acc + tp
        test_pred = (test_acc / len(folds)) if test_acc is not None else None
        cv = CC.score(cfg.metric, y, oof) if CC.metric_spec(cfg.metric)["fn"] else float("nan")
        out[b] = {"oof": oof, "test": test_pred, "cv": cv}
    ids = df_test[cfg.id_col].to_numpy() if (df_test is not None and cfg.id_col in df_test.columns) else (
        np.arange(len(df_test)) if df_test is not None else None)
    return out, {"y": y, "test_ids": ids, "kind": kind, "n_classes": n_classes, "n_feats": len(feats)}


class TabTrain(BaseAgent):
    name = "tab-train"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        if "config" not in spec and "config_file" not in spec:
            return self.escalate(worker, "leader", "tab-train needs spec keys ['config' or 'config_file'] — none provided")
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else CC.CompConfig.load(spec["config_file"])
        res, meta = train_backends(cfg, backends=spec.get("backends"), seed=int(spec.get("seed", 42)),
                                   fe=bool(spec.get("fe", False)),
                                   n_folds=spec.get("n_folds"), gpu=spec.get("gpu"),
                                   early_stopping=bool(spec.get("early_stopping", False)),
                                   params=spec.get("params"))
        cvs = {b: r["cv"] for b, r in res.items()}
        best = max(cvs, key=lambda b: cvs[b] if cfg.metric_direction == "max" else -cvs[b])
        # keep arrays out of the board msg; return them in data for chaining
        data = {"per_backend_cv": cvs, "best_backend": best, "best_cv": cvs[best], "meta": meta,
                "_preds": res}
        msg = (f"tab-train {cfg.slug}: backends={list(res)} CV({cfg.metric})={ {b: round(v,5) for b,v in cvs.items()} } "
               f"→ best={best} {cvs[best]:.5f}")
        self.log(msg, kind="finding", recommendation=f"blend with tab-stack; best single = {best}")
        return self.done(data, msg)


_AGENT = TabTrain()


def run(q, worker):
    return _AGENT.run(q, worker)
