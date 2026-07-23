"""geology_trackD — Track D for wellbore geosteering: Google **TabFM** (in-context tabular foundation
model, https://github.com/google-research/tabfm) as a zero-shot/no-training regressor over the same
flat feature table `geology_pack.geology_assemble` builds (dmd, dz, horiz_disp, incl, abs_z, md_ps,
tvt_ps, gr, gr_isnan, gr_minus_ps, gr_grad, gr_roll5/25, gr_std25, cand_dtvt -> residual dtvt).

TabFM has NO training step: at inference it reads a bounded "context" of (X_train, y_train) rows and
predicts on X_test directly (in-context learning, TabPFN-style). Its row-attention is O(context^2), so
passing the full 3.78M-row train set as context is infeasible — this module builds a SUBSAMPLED,
well-disjoint context per fold (see `_build_context`), capped at `max_context` rows (empirically
calibrated — see `docs/tabfm_context_ceiling` measurement in `trackD_run.py`'s smoke test).

Contract: `trackD_oof(train_csv, test_csv, out_oof, out_test, params, log=print) -> cv`. Standardized
ledgers (id, well, dtvt_pred[, dtvt_true]) so it drops straight into the blend like every other track.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["dmd", "dz", "horiz_disp", "incl", "abs_z", "md_ps", "tvt_ps", "gr", "gr_isnan",
            "gr_minus_ps", "gr_grad", "gr_roll5", "gr_roll25", "gr_std25", "cand_tvt"]


def _build_context(train_df, exclude_wells, max_context, rng):
    """A representative, well-disjoint context subsample: stratify by well (equal rows/well, capped)
    so no single long well dominates the in-context examples, then cap total at max_context."""
    pool = train_df[~train_df["well"].isin(exclude_wells)]
    wells = pool["well"].unique()
    if len(wells) == 0:
        return pool.iloc[:0]
    per_well = max(1, max_context // len(wells))
    parts = []
    for w in wells:
        sub = pool[pool["well"] == w]
        if len(sub) > per_well:
            idx = rng.choice(len(sub), size=per_well, replace=False)
            parts.append(sub.iloc[idx])
        else:
            parts.append(sub)
    ctx = pd.concat(parts, ignore_index=True)
    if len(ctx) > max_context:
        idx = rng.choice(len(ctx), size=max_context, replace=False)
        ctx = ctx.iloc[idx]
    return ctx


def _predict_fold(reg_cls, model, ctx, query_df, max_context, batch_rows=2000):
    """Fit TabFMRegressor on the context once, predict the query rows in batches (query batching is
    just inference chunking — the model's context stays fixed per batch)."""
    Xc = ctx[FEATURES].to_numpy(dtype=np.float32)
    yc = ctx["dtvt"].to_numpy(dtype=np.float32)
    reg = reg_cls(model=model)
    reg.fit(Xc, yc)
    preds = np.empty(len(query_df), dtype=np.float32)
    Xq_all = query_df[FEATURES].to_numpy(dtype=np.float32)
    for s in range(0, len(query_df), batch_rows):
        preds[s:s + batch_rows] = reg.predict(Xq_all[s:s + batch_rows])
    return preds


def trackD_oof(train_csv, test_csv, out_oof, out_test, params, log=print):
    from sklearn.model_selection import GroupKFold
    max_context = int(params.get("max_context", 2000))
    n_folds = int(params.get("folds", 5))
    seed = int(params.get("seed", 42))
    limit_rows = params.get("limit_rows")  # smoke-test row cap, independent of max_context

    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    tr["dtvt"] = tr["tvt"].astype(float) - tr["tvt_ps"].astype(float)  # geology_assemble writes ABSOLUTE tvt
    if limit_rows:
        tr = tr.sample(n=min(limit_rows, len(tr)), random_state=seed).reset_index(drop=True)
    for c in FEATURES:
        tr[c] = tr[c].astype(float).fillna(0.0)
        te[c] = te[c].astype(float).fillna(0.0)

    backend = params.get("backend", "pytorch")
    if backend == "pytorch":
        from tabfm import tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    else:
        from tabfm import tabfm_v1_0_0_jax as tabfm_v1_0_0
    from tabfm import TabFMRegressor
    model = tabfm_v1_0_0.load(model_type="regression")

    rng = np.random.default_rng(seed)
    wells = tr["well"].to_numpy()
    gkf = GroupKFold(n_splits=n_folds)
    oof = np.zeros(len(tr), dtype=np.float32)
    for i, (_, va_idx) in enumerate(gkf.split(tr, tr["dtvt"], groups=wells)):
        va = tr.iloc[va_idx]
        va_wells = set(va["well"].unique())
        ctx = _build_context(tr, va_wells, max_context, rng)
        log(f"  [D] fold {i+1}/{n_folds}: context={len(ctx)} rows, query={len(va)} rows")
        oof[va_idx] = _predict_fold(TabFMRegressor, model, ctx, va, max_context)

    cv = float(np.sqrt(np.mean((oof - tr["dtvt"].to_numpy()) ** 2)))
    oof_df = pd.DataFrame({"id": tr["id"], "well": tr["well"], "dtvt_pred": oof,
                          "dtvt_true": tr["dtvt"].to_numpy()})
    oof_df.to_csv(out_oof, index=False)

    ctx_full = _build_context(tr, set(), max_context, rng)
    log(f"  [D] test: context={len(ctx_full)} rows, query={len(te)} rows")
    test_pred = _predict_fold(TabFMRegressor, model, ctx_full, te, max_context)
    test_df = pd.DataFrame({"id": te["id"], "dtvt_pred": test_pred, "tvt_ps": te["tvt_ps"]})
    test_df.to_csv(out_test, index=False)
    return cv
