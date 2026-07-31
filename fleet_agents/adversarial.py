"""Adversarial-validation agent — confirm the CV axis matches the hidden test (embryo-disjoint).

Deterministic: for the active split, assert no embryo appears in both train & test of any fold, and
report the leak axis. The EDA established adversarial AUC 44b6-vs-6bba = 0.98 → embryo IS the leak
axis, so leave-one-embryo-out is correct and golden-12 is a secondary (leaky) anchor only.
"""
from __future__ import annotations

import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
DEFAULT_SPLIT = COMP / "learning" / "ensemble_work" / "finetune" / "splits_loeo_density.json"
ADVERSARIAL_AUC = 0.98  # 44b6 vs 6bba, from the competition EDA


def _emb(x):
    return x.split("_")[0]


def report(q, worker):
    # `n_splits`: optional expected fold count (flags a mismatch). `seed`: accepted for provenance (no-op here).
    spec = (q or {}).get("spec", {}) or {}
    split_path = COMP / spec.get("split", "learning/ensemble_work/finetune/splits_loeo_density.json")
    if not split_path.exists():
        split_path = DEFAULT_SPLIT
    if not split_path.exists():
        return ("escalated", {"reason": "no split file"}, "researcher",
                f"[{worker}] ADVERSARIAL-VAL: no split file to validate — build one via the cv-build agent first.")
    try:
        folds = json.loads(split_path.read_text())
    except Exception as e:  # noqa: BLE001 — malformed/empty split file → clean escalate, not a crash
        return ("escalated", {"reason": f"unreadable split ({type(e).__name__})"}, "researcher",
                f"[{worker}] ADVERSARIAL-VAL: split file {split_path.name} is unreadable/invalid JSON.")
    if not isinstance(folds, list):
        return ("escalated", {"reason": "split not a list of folds"}, "researcher",
                f"[{worker}] ADVERSARIAL-VAL: split file {split_path.name} is not a list of folds.")
    probs, fold_desc = [], []
    for i, f in enumerate(folds):
        f = f or {}
        tr = {_emb(x) for x in f.get("train", [])}
        te = {_emb(x) for x in f.get("test", [])}
        if tr & te:
            probs.append(f"fold{i}: embryo overlap {sorted(tr & te)}")
        fold_desc.append(f"fold{i}: train{sorted(tr)}/test{sorted(te)}")
    n_splits = spec.get("n_splits")
    if n_splits is not None:
        try:
            if int(n_splits) != len(folds):
                probs.append(f"fold-count {len(folds)} != expected n_splits={int(n_splits)}")
        except (TypeError, ValueError):
            pass    # an unparsable n_splits is a spec typo, not a leak signal — the fold checks above still ran
    ok = not probs
    verdict = ("EMBRYO-DISJOINT ✓ (correct CV axis)" if ok else "LEAK: " + "; ".join(probs))
    return ("done", {"embryo_disjoint": ok, "adversarial_auc": ADVERSARIAL_AUC, "folds": fold_desc,
                     "n_folds": len(folds)}, "all",
            f"[{worker}] ADVERSARIAL-VAL ({split_path.name}): {verdict}. "
            f"embryo separability AUC={ADVERSARIAL_AUC} → embryo is the leak axis; LOEO is correct, "
            f"golden-12 is a secondary (leaky) anchor only. {' | '.join(fold_desc)}")
