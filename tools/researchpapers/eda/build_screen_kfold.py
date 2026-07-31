#!/usr/bin/env python
"""Build a K-fold stage-matched screening CV split for the Biohub cell-tracking comp.

Extends the 2-fold `splits_screen_matched.json` design (see
`eda/screen_miniset_analysis.md`) to K folds. Each fold's VAL mirrors the
golden-12 judge's joint (group x stage) distribution and, within each
(group,stage) cell, is picked by nearest-density to the corresponding golden-12
FOVs. VAL sets are pairwise disjoint across folds and never overlap golden-12.
TRAIN is a density-spanning, group-balanced set disjoint from that fold's VAL
and from golden-12 (train MAY overlap across folds).

Deterministic: pure sort-based selection, tie-broken by id and a fixed seed.

Usage:
    python build_screen_kfold.py [K] \
        --csv learning/03_true_density_stage.csv \
        --golden learning/ensemble_work/finetune/splits_ft.json \
        --out learning/ensemble_work/finetune/splits_screen_matched_k6.json \
        --seed 20260705
"""
import argparse
import csv
import json
import os
import statistics
from collections import Counter, defaultdict

# golden-12 recipe (the distribution every screen VAL must mirror).
GROUP_TARGET = {"44b6": 6, "6bba": 6}
STAGE_TARGET = {"S0": 3, "S1": 1, "S2": 2, "S3": 2, "S4": 4}
TRAIN_PER_GROUP = 12  # 12 x 44b6 + 12 x 6bba = 24, mirrors the 2-fold train.


def load_rows(csv_path):
    rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            r["density"] = float(r["estN_per_frame"])
            rows[r["dataset"]] = r
    return rows


def load_golden(golden_path):
    data = json.load(open(golden_path))
    # golden-12 = fold-0 test of splits_ft.json
    return list(data[0]["test"])


def quantile_spaced(ids_sorted_by_density, k):
    """Pick k evenly (quantile) spaced ids from a density-sorted list."""
    n = len(ids_sorted_by_density)
    if n <= k:
        return list(ids_sorted_by_density)
    idx = sorted({round(j * (n - 1) / (k - 1)) for j in range(k)})
    # round() can collide near the ends; backfill to guarantee exactly k.
    picked = [i for i in idx]
    j = 0
    while len(picked) < k and j < n:
        if j not in picked:
            picked.append(j)
        j += 1
    picked = sorted(picked)[:k]
    return [ids_sorted_by_density[i] for i in picked]


def build(rows, golden, k, seed):
    golden_set = set(golden)
    pool = [i for i in rows if i not in golden_set]

    # golden-12 joint (group,stage) -> list of target densities
    golden_targets = defaultdict(list)
    for gid in golden:
        r = rows[gid]
        golden_targets[(r["group"], r["stage"])].append(r["density"])
    for key in golden_targets:
        golden_targets[key].sort()

    # pool grouped by (group,stage)
    pool_cell = defaultdict(list)
    for pid in pool:
        r = rows[pid]
        pool_cell[(r["group"], r["stage"])].append(pid)

    # feasibility + val assignment
    val = [[] for _ in range(k)]
    warnings = []
    for cell, targets in sorted(golden_targets.items()):
        need = len(targets) * k
        avail = len(pool_cell[cell])
        if avail < need:
            warnings.append(
                f"cell {cell}: need {need} ({len(targets)}/fold x {k}) but pool has {avail}"
            )
        used = set()
        # process each golden target in this cell; assign k nearest-unused pool
        # embryos to the k folds. Rotate the fold-order per target so no single
        # fold always gets the tightest match.
        for t_idx, tgt in enumerate(targets):
            cand = [pid for pid in pool_cell[cell] if pid not in used]
            # deterministic order: nearest density, then density, then id (+seed).
            cand.sort(key=lambda pid: (abs(rows[pid]["density"] - tgt),
                                       rows[pid]["density"],
                                       str(seed) + pid))
            chosen = cand[:k]
            for pos, pid in enumerate(chosen):
                fold = (pos + t_idx) % k
                val[fold].append(pid)
                used.add(pid)

    val = [sorted(v) for v in val]

    # train per fold: density-spanning, group-balanced, excl golden-12 & this val.
    folds = []
    for f in range(k):
        excl = golden_set | set(val[f])
        train = []
        for g in ("44b6", "6bba"):
            elig = sorted((pid for pid in pool
                           if pid not in excl and rows[pid]["group"] == g),
                          key=lambda pid: (rows[pid]["density"], str(seed) + pid))
            train.extend(quantile_spaced(elig, TRAIN_PER_GROUP))
        folds.append({"train": sorted(train), "test": val[f]})

    return folds, warnings


def verify(rows, golden, folds):
    golden_set = set(golden)
    g_dens = sorted(rows[i]["density"] for i in golden)
    g_med = statistics.median(g_dens)
    print(f"\ngolden-12: group={dict(sorted(Counter(rows[i]['group'] for i in golden).items()))} "
          f"stage={dict(sorted(Counter(rows[i]['stage'] for i in golden).items()))} "
          f"median_density={g_med:.1f}")

    header = f"{'fold':>4} | {'44b6/6bba':>9} | {'S0 S1 S2 S3 S4':>14} | {'val∩gold':>8} | {'val_med_dens':>12} | {'Δmed':>6}"
    print("\n" + header)
    print("-" * len(header))
    all_val = []
    ok = True
    for f, fold in enumerate(folds):
        v = fold["test"]
        gc = Counter(rows[i]["group"] for i in v)
        sc = Counter(rows[i]["stage"] for i in v)
        leak = len(set(v) & golden_set)
        vmed = statistics.median(sorted(rows[i]["density"] for i in v))
        grp_ok = gc.get("44b6", 0) == 6 and gc.get("6bba", 0) == 6
        stage_ok = all(sc.get(s, 0) == STAGE_TARGET[s] for s in STAGE_TARGET)
        if not (grp_ok and stage_ok and leak == 0):
            ok = False
        stage_str = " ".join(f"{sc.get(s,0):2d}" for s in ["S0", "S1", "S2", "S3", "S4"])
        print(f"{f:>4} | {gc.get('44b6',0):>4}/{gc.get('6bba',0):<4} | {stage_str:>14} | "
              f"{leak:>8} | {vmed:>12.1f} | {vmed-g_med:>+6.1f}")
        all_val.append(set(v))

    # pairwise val disjointness
    print("\npairwise val∩val (must all be 0):")
    max_overlap = 0
    for i in range(len(all_val)):
        for j in range(i + 1, len(all_val)):
            o = len(all_val[i] & all_val[j])
            max_overlap = max(max_overlap, o)
            if o:
                print(f"  fold {i} ∩ fold {j} = {o}  <-- OVERLAP!")
                ok = False
    print(f"  max pairwise overlap = {max_overlap}")

    # train leak checks
    print("\ntrain leak checks (must all be 0):")
    for f, fold in enumerate(folds):
        tr = set(fold["train"])
        tv = len(tr & set(fold["test"]))
        tg = len(tr & golden_set)
        gc = Counter(rows[i]["group"] for i in fold["train"])
        tdens = sorted(rows[i]["density"] for i in fold["train"])
        if tv or tg:
            ok = False
        print(f"  fold {f}: n_train={len(tr)} group={gc.get('44b6',0)}/{gc.get('6bba',0)} "
              f"train∩val={tv} train∩gold={tg} dens[{tdens[0]:.0f}..{tdens[-1]:.0f}]")

    print("\nALL CHECKS PASS" if ok else "\n*** CHECKS FAILED ***")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("k", nargs="?", type=int, default=6, help="number of folds (default 6)")
    root = "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development"
    ap.add_argument("--csv", default=os.path.join(root, "learning/03_true_density_stage.csv"))
    ap.add_argument("--golden", default=os.path.join(
        root, "learning/ensemble_work/finetune/splits_ft.json"))
    ap.add_argument("--out", default=os.path.join(
        root, "learning/ensemble_work/finetune/splits_screen_matched_k6.json"))
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()

    rows = load_rows(args.csv)
    golden = load_golden(args.golden)
    folds, warnings = build(rows, golden, args.k, args.seed)

    if warnings:
        print("STAGE-RECIPE WARNINGS:")
        for w in warnings:
            print("  -", w)

    ok = verify(rows, golden, folds)

    with open(args.out, "w") as f:
        json.dump(folds, f, indent=1)
    print(f"\nwrote {len(folds)} folds -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
