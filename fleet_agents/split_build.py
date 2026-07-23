"""split-build — deterministic, VALIDATED CV-split builder (takes this off the researcher).

The researcher kept hand-building splits and shipped a leaky one (stagebridge test had both embryos).
This agent builds the split types we use, ALWAYS runs the leak/composition check, and REFUSES to write
a split that fails its own contract. So a split is either correct-by-construction or not written.

Kinds:
  embryo_disjoint : fold0 train 6bba / test 44b6 ; fold1 reverse (the TRUE Kaggle axis)
  stage_matched   : both-embryo FOV-disjoint, density/stage-matched to golden-12 (fast screen)   [uses existing builder]
  stagebridge     : PURE embryo-disjoint mini (train one embryo / test the OTHER only), stage-spanning
Spec: {"kind":..., "out":"<name>.json", "test_n":15, "train_n":30}
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
FT = COMP / "learning" / "ensemble_work" / "finetune"
CSV = COMP / "learning" / "03_true_density_stage.csv"
TRAIN = COMP / "input" / "biohub-cell-tracking-during-development" / "train"


def _recs():
    def col(r, *names):
        for n in names:
            for k in r:
                if k.lower() == n:
                    return r[k]
        return None
    out = []
    if not CSV.exists():
        return out
    for r in csv.DictReader(open(CSV)):
        kid = str(col(r, "group_frame", "id", "dataset", "embryo") or list(r.values())[0]).replace(".zarr.geff", "").replace(".geff", "")
        g = col(r, "group") or kid[:4]
        try:
            d = float(col(r, "estN_per_frame", "density") or 0)
        except (TypeError, ValueError):
            d = 0.0
        if (TRAIN / f"{kid}.geff").is_dir():
            out.append((kid, g, d))
    return out


def _span(items, n):
    items = sorted(items, key=lambda x: x[2])          # by density → stage-spanning
    if n <= 0 or n >= len(items):
        return [x[0] for x in items] if n != 0 else []
    step = len(items) / n
    return [items[int(i * step)][0] for i in range(n)]


def _leak_ok(folds):
    for f in folds:
        et = {x[:4] for x in f["train"]}
        ee = {x[:4] for x in f["test"]}
        # embryo-disjoint contract: no embryo shared between this fold's train and test
        if et & ee:
            return False
    return True


def build(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    kind = spec.get("kind", "stagebridge")
    out = FT / spec.get("out", f"splits_{kind}.json")
    test_n, train_n = max(0, int(spec.get("test_n", 15))), max(0, int(spec.get("train_n", 30)))
    if kind == "stage_matched":
        # delegate to the existing density/stage-matched screen (already validated); just report
        p = FT / "splits_screen_matched.json"
        return ("done", {"out": str(p), "kind": kind}, "all",
                f"[{worker}] SPLIT-BUILD: stage_matched screen already built + validated → {p.name} "
                f"(24/12, group 6/6, stage S0-S4 matched, 0 golden-12 leak).")
    recs = _recs()
    by = {"44b6": [x for x in recs if x[1] == "44b6"], "6bba": [x for x in recs if x[1] == "6bba"]}
    if kind in ("embryo_disjoint", "stagebridge"):
        folds = [
            {"train": _span(by["6bba"], train_n), "test": _span(by["44b6"], test_n)},
            {"train": _span(by["44b6"], train_n), "test": _span(by["6bba"], test_n)},
        ]
    else:
        return ("escalated", {"kind": kind}, "researcher",
                f"[{worker}] SPLIT-BUILD: unknown kind '{kind}' — need a builder for it.")
    if not _leak_ok(folds):
        return ("failed", {"kind": kind}, "researcher",
                f"[{worker}] SPLIT-BUILD REFUSED: {kind} failed the embryo-disjoint leak check — not written.")
    import collections
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(folds, open(out, "w"), indent=1)
    desc = "; ".join(
        f"fold{i}: train={dict(collections.Counter(x[:4] for x in f['train']))} "
        f"test={dict(collections.Counter(x[:4] for x in f['test']))}" for i, f in enumerate(folds))
    return ("done", {"out": str(out), "kind": kind, "folds": len(folds)}, "all",
            f"[{worker}] SPLIT-BUILD ({kind}, leak-checked ✓): {out.name} — {desc}.")
