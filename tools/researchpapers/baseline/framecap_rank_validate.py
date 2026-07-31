#!/usr/bin/env python
"""STAGE 1 (SCREEN-V1) — frame-cap rank-validation (Policy B gate).

A frame cap N (`--max-frames`) is usable for cheap screening ONLY if the frame-capped official score
PRESERVES the ranking of a known-delta pair. Our measured full-T known pair is pilkwang-FULL (0.8373) vs
canqiang-FULL (0.7926), +0.0447 on both density folds.

This tool does that check CPU-ONLY, by frame-FILTERING the EXISTING full-T gate geffs to the first-N-frames
window (t < N) — no GPU re-predict. It scores each detector on the t<N window with the same official
primitives (`src.metric`, which reproduced evaluate.py exactly for canqiang), aggregating per fold then
mean. NOTE (GT-time-sparsity caveat): GT annotations occupy a per-embryo t-window (e.g. t=11..50), so a small
N includes fewer GT frames and fewer contributing embryos; embryos with no GT in t<N are excluded. The cap is
a SCREENING PROXY — the final judge is always full-T + official evaluate.py.

Accept the SMALLEST N where pilk-full > canqiang-full on BOTH folds (rank preserved).

Usage:
  python baseline/framecap_rank_validate.py --caps 24,40,64            # full validation
  python baseline/framecap_rank_validate.py --caps 64 --limit 2 --dry-run   # minimal dry-run
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARENT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = PARENT / "input/biohub-cell-tracking-during-development/train"
PRED = PARENT / "research/official_repo/predictions/seshu"
SPLITS = PARENT / "learning/ensemble_work/finetune/splits_loeo_density.json"
sys.path.insert(0, str(PARENT))
from src import io                                    # noqa: E402
from src.config import Config                         # noqa: E402
from src.metric import official_counts, official_score  # noqa: E402

SRC = Config()
METHODS = {"pilk_full": "pilk_full_loeodens", "canqiang_full": "canqiang_full"}


def _cap(nodes: pd.DataFrame, edges: pd.DataFrame, n: int | None):
    """Restrict a node/edge table to t < n (n=None -> full)."""
    if n is None:
        return nodes, edges
    keep = nodes[nodes["t"] < n]
    ids = set(keep["node_id"].tolist())
    e = edges[edges["source_id"].isin(ids) & edges["target_id"].isin(ids)] if len(edges) else edges
    return keep, e


def score_method_fold(method_dir: str, fold_stems: list[str], cap: int | None, limit: int | None):
    rows = []
    stems = fold_stems[:limit] if limit else fold_stems
    for ds in stems:
        pred_p = PRED / method_dir / (f"split_0/{ds}.geff" if ds.startswith("44b6") else f"split_1/{ds}.geff")
        if not pred_p.exists():
            continue
        gn, ge = io.read_geff(TRAIN / f"{ds}.geff")
        pn, pe = io.read_geff(pred_p)
        gn_c, ge_c = _cap(gn, ge, cap)
        pn_c, pe_c = _cap(pn, pe, cap)
        if len(gn_c) == 0:          # no GT in this window -> embryo can't be scored at this cap
            continue
        tt = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
        r = official_counts(gn_c, ge_c, pn_c, pe_c, SRC.SCALE, SRC.MATCH_GATE_UM, t_true=tt)
        r["dataset"] = ds
        rows.append(r)
    if not rows:
        return None, 0
    df = pd.DataFrame(rows)
    return float(official_score(df.to_dict("records"))["score"]), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--caps", default="24,40,64", help="comma frame caps to test (+ full-T reference)")
    ap.add_argument("--limit", type=int, default=None, help="score only first N embryos/fold (dry-run)")
    ap.add_argument("--dry-run", action="store_true", help="wiring check only; do not decide N")
    ap.add_argument("--pilk-method", default="pilk_full_loeodens",
                    help="pilk geff method dir under predictions/seshu/ (e.g. pilk_full_cap24 for the "
                         "PREDICT-capped confirmation)")
    args = ap.parse_args()
    METHODS["pilk_full"] = args.pilk_method

    import json
    sp = json.load(open(SPLITS))
    fold_stems = [[s.replace(".zarr", "") for s in f["test"]] for f in sp]
    caps = [int(x) for x in args.caps.split(",")] + [None]   # None = full-T reference

    print(f"{'cap':>6} | {'pilk f0':>8} {'pilk f1':>8} {'pilk mean':>9} | "
          f"{'canq f0':>8} {'canq f1':>8} {'canq mean':>9} | {'Δmean':>8} | rank", flush=True)
    chosen = None
    for cap in caps:
        pf = [score_method_fold(METHODS["pilk_full"], fold_stems[f], cap, args.limit) for f in (0, 1)]
        cf = [score_method_fold(METHODS["canqiang_full"], fold_stems[f], cap, args.limit) for f in (0, 1)]
        pvals = [x[0] for x in pf]; cvals = [x[0] for x in cf]
        if any(v is None for v in pvals + cvals):
            print(f"{str(cap or 'full'):>6} | insufficient GT in window (n_scored pilk={[x[1] for x in pf]})", flush=True)
            continue
        pm, cm = float(np.mean(pvals)), float(np.mean(cvals))
        both = pvals[0] > cvals[0] and pvals[1] > cvals[1]
        rank = "PILK>CANQ both" if both else ("pilk>canq mean only" if pm > cm else "INVERTED")
        print(f"{str(cap or 'full'):>6} | {pvals[0]:8.4f} {pvals[1]:8.4f} {pm:9.4f} | "
              f"{cvals[0]:8.4f} {cvals[1]:8.4f} {cm:9.4f} | {pm-cm:+8.4f} | {rank}", flush=True)
        if cap is not None and both and chosen is None:
            chosen = cap
    if args.dry_run:
        print("\n[dry-run] wiring GREEN — caps parsed, geffs load, windowed official scores computed. "
              "Not deciding N.", flush=True)
    elif chosen is not None:
        print(f"\nCHOSEN N = {chosen} (smallest cap preserving pilk>canq on BOTH folds). "
              "Frame cap for all SCREEN-V1 early rungs. Final judge stays full-T + official evaluate.py.", flush=True)
    else:
        print("\nNO cap in the set preserved the both-fold rank — screen at full-T or test larger caps.", flush=True)


if __name__ == "__main__":
    main()
