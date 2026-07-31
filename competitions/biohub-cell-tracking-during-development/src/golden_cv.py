"""GOLDEN CV — FROZEN 2026-06-30. Do not change without re-validating vs the 9 public LBs.

The single source of truth for scoring any config. Aligns +0.90 with public LBs;
stratified-5-fold alignment 0.910±0.033 (lowest variance of all tested protocols).

PROTOCOL (frozen):
  metric      : official adjusted edge-Jaccard + 0.1*division (src.metric.official_*)
  aggregation : MICRO (weight-avg by w=TP+FP+FN)  [NOT loeo-min]
  data        : full 199-dataset TRAIN folder only (never the dummy test/)
  CV value    : AVERAGE micro over repeated stratified 5-fold (embryo+size balanced), >=20 seeds
  guardrail   : VALID only if predicted node-density <= best-public (xiaoleilian) level
                (44b6 <= ~190/frame, 6bba <= ~95/frame). Denser => over-detection => CV unreliable.
  significance: a change is real only if it beats prior golden CV by > ~0.03.
"""
from __future__ import annotations
import numpy as np, pandas as pd
from . import metric

# best-validated (xiaoleilian, LB 0.720) predicted density ceiling, per frame, +small margin
DENSITY_CAP = {"44b6": 190.0, "6bba": 95.0}
N_SEEDS = 20
K = 5


def _strat_folds(meta: pd.DataFrame, k: int, seed: int):
    """Stratified folds: balance embryo AND dataset size (w) per fold."""
    folds = [[] for _ in range(k)]
    for _, g in meta.groupby("embryo"):
        g = g.sample(frac=1, random_state=seed).sort_values("w", ascending=False)
        for i, d in enumerate(g["dataset"]):
            folds[i % k].append(d)
    return [set(f) for f in folds]


def golden_cv(per_dataset: pd.DataFrame, frames: pd.DataFrame | None = None,
              n_seeds: int = N_SEEDS, k: int = K) -> dict:
    """per_dataset: official_counts rows (cols: dataset, embryo, adj_jaccard, w, t_pred, ...).
    frames: optional dataset->frames for the density guardrail.
    Returns {golden_cv, std, valid, density, ...}."""
    df = per_dataset.set_index("dataset") if "dataset" in per_dataset.columns else per_dataset
    meta = pd.DataFrame({"dataset": df.index, "embryo": df["embryo"].values, "w": df["w"].values})

    def micro(ds):
        g = df.loc[df.index.intersection(ds)]
        return (g.w * g.adj_jaccard).sum() / g.w.sum() if g.w.sum() > 0 else np.nan

    fold_vals = []
    for s in range(n_seeds):
        for fold in _strat_folds(meta, k, s):
            fold_vals.append(micro(fold))
    fold_vals = np.array([v for v in fold_vals if np.isfinite(v)])
    cv = float(fold_vals.mean())

    valid, density = True, {}
    if frames is not None:
        d = df.join(frames["frames"]) if "frames" not in df.columns else df
        for emb, g in d.groupby("embryo"):
            dens = float((g.t_pred / g.frames).mean()); density[emb] = round(dens, 1)
            if dens > DENSITY_CAP.get(emb, 1e9) * 1.0:
                valid = False
    return {"golden_cv": round(cv, 4), "fold_std": round(float(fold_vals.std()), 4),
            "valid": valid, "density": density}
