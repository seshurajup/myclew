"""Stage-0 CV contract assertions — the gate EVERY run must pass before it is trusted.

Enforces the frozen embryo-disjoint LOEO contract on a split file (list of {"train":[...],"test":[...]}):
  1. LEAK-ASSERT   : no embryo (id prefix before "_") appears in both train and test of ANY fold.
  2. VALIDITY      : every id resolves to a real train geff (a zarr dir) — catches phantom/malformed ids
                     like "<ds>.zarr" that don't exist on disk (the bug that was in fleet_loeo_mini.json).
  3. NO-DUP        : no id repeated within a fold's train or test.
  4. TRAIN∩TEST=∅  : train and test of a fold are disjoint at the id level.

Reuses the competition's own embryo axis (src.io.embryo_id) so the leak axis matches the pipeline.
CLI exits non-zero on any violation:
    research/cellmot_venv/bin/python -m fleet_agents.cv_contract \
        learning/ensemble_work/finetune/fleet_loeo_mini.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
TRAIN = COMP / "input" / "biohub-cell-tracking-during-development" / "train"


def _embryo(ds: str) -> str:
    return str(ds).split("_")[0]


def check_contract(split_path, train_dir=TRAIN) -> list[str]:
    """Return a list of violation strings (empty == contract holds)."""
    train_dir = Path(train_dir)
    with open(split_path) as _fh:
        folds = json.load(_fh)
    if not isinstance(folds, list):
        return [f"split is not a list of folds (got {type(folds).__name__})"]
    errs: list[str] = []
    seen_test_embryos = {}
    for i, f in enumerate(folds):
        if not isinstance(f, dict):
            errs.append(f"fold{i}: not an object with train/test")
            continue
        tr, te = list(f.get("train", [])), list(f.get("test", []))
        # 1. leak-assert: embryo-disjoint train vs test
        overlap = {_embryo(x) for x in tr} & {_embryo(x) for x in te}
        if overlap:
            errs.append(f"fold{i}: LEAK — embryo(s) in BOTH train & test: {sorted(overlap)}")
        # 2. validity: every id resolves to a real train geff
        for scope, ids in (("train", tr), ("test", te)):
            missing = [x for x in ids if not (train_dir / f"{x}.geff").is_dir()]
            if missing:
                errs.append(f"fold{i}.{scope}: {len(missing)} id(s) do not resolve to a train geff "
                            f"(e.g. {missing[:3]})")
        # 3. no-dup within scope
        for scope, ids in (("train", tr), ("test", te)):
            if len(ids) != len(set(ids)):
                dups = sorted({x for x in ids if ids.count(x) > 1})
                errs.append(f"fold{i}.{scope}: duplicate id(s): {dups[:5]}")
        # 4. train ∩ test == ∅
        both = set(tr) & set(te)
        if both:
            errs.append(f"fold{i}: id(s) in BOTH train & test: {sorted(both)[:5]}")
        # (informational) an embryo used as test twice is fine for LOEO; record for the report
        for e in {_embryo(x) for x in te}:
            seen_test_embryos.setdefault(e, []).append(i)
    return errs


def assert_contract(split_path, train_dir=TRAIN) -> None:
    errs = check_contract(split_path, train_dir)
    if errs:
        raise AssertionError("CV contract VIOLATED:\n  - " + "\n  - ".join(errs))


def main(argv=None):
    argv = argv or sys.argv[1:]
    split = argv[0] if argv else str(COMP / "learning/ensemble_work/finetune/fleet_loeo_mini.json")
    with open(split) as _fh:
        folds = json.load(_fh)
    errs = check_contract(split)
    print(f"CV contract check: {split}")
    for i, f in enumerate(folds if isinstance(folds, list) else []):
        tr, te = list(f.get("train", [])), list(f.get("test", []))
        print(f"  fold{i}: train={len(tr)}({_embryo(tr[0]) if tr else '—'}) "
              f"test={len(te)}({_embryo(te[0]) if te else '—'})")
    if errs:
        print(f"FAIL ({len(errs)} violation(s)):")
        for e in errs:
            print("  - " + e)
        return 1
    print("PASS: embryo-disjoint, all ids resolve, no dups, train∩test=∅.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
