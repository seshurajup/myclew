#!/usr/bin/env python
"""Convert a pilkwang_full/pipeline.py `submission.csv` into evaluate.py-ready per-dataset geffs.

pipeline.py bakes the FULL pilkwang pipeline (pilk_post + best.pt fusion + gap-recovery + postproc) into a
single `submission.csv` (row_type node|edge, per `dataset`) — it does NOT emit geffs. This converter turns
that CSV into `predictions/{user}/{method}/split_{fold}/{dataset}.geff` so the OFFICIAL scorer
(research/official_repo/scripts/evaluate.py) can score the full pipeline like-for-like vs canqiang_full.

Generalises learning/ensemble_work/pilkwang_full/score_full.py (which hardcodes split_0 + ./submission.csv
+ pilk_full890). Used for the fully-fair CV gate (EXP-CVGATE-FAIR).

Usage:
  python baseline/pilk_submission_to_geff.py --submission <submission.csv> --fold 0 --method pilk_full_loeodens
  python baseline/pilk_submission_to_geff.py --submission <submission.csv> --fold 0 --method <m> --limit 1  # dry-run
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
import polars as pl

PARENT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
PRED_ROOT = PARENT / "research/official_repo/predictions/seshu"
sys.path.insert(0, str(PARENT))
import tracksdata as td   # noqa: E402
K = td.DEFAULT_ATTR_KEYS


def convert(submission: Path, fold: int, method: str, limit: int | None) -> int:
    df = pd.read_csv(submission)
    out_dir = PRED_ROOT / method / f"split_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    datasets = sorted(df["dataset"].unique())
    if limit:
        datasets = datasets[:limit]
    n = 0
    for ds in datasets:
        sub = df[df["dataset"] == ds]
        nodes = sub[sub.row_type == "node"]
        edges = sub[sub.row_type == "edge"]
        g = td.graph.IndexedRXGraph()
        for k in (K.T, K.Z, K.Y, K.X):
            try:
                g.add_node_attr_key(k, pl.Float64, default_value=0.0)
            except Exception:  # noqa: BLE001
                pass
        idmap = {}
        for r in nodes.itertuples(index=False):
            idmap[int(r.node_id)] = g.add_node(
                {K.T: int(r.t), K.Z: float(r.z), K.Y: float(r.y), K.X: float(r.x)}, index=int(r.node_id))
        for r in edges.itertuples(index=False):
            s, t = int(r.source_id), int(r.target_id)
            if s in idmap and t in idmap:
                g.add_edge(idmap[s], idmap[t], {})
        out = out_dir / f"{ds}.geff"
        if out.exists():
            shutil.rmtree(out)
        g.to_geff(str(out))
        print(f"  {ds:18s} nodes={len(nodes):6d} edges={len(edges):6d} -> {method}/split_{fold}/{out.name}",
              flush=True)
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True, help="path to a pipeline.py submission.csv")
    ap.add_argument("--fold", type=int, required=True, help="OUR fold label (0=44b6 test, 1=6bba test)")
    ap.add_argument("--method", required=True, help="predictions/seshu/<method>/split_<fold> namespace")
    ap.add_argument("--limit", type=int, default=None, help="convert first N datasets only (dry-run)")
    args = ap.parse_args()
    sub = Path(args.submission)
    if not sub.exists():
        ap.error(f"submission not found: {sub}")
    n = convert(sub, args.fold, args.method, args.limit)
    print(f"done — wrote {n} geff(s) to {PRED_ROOT / args.method / f'split_{args.fold}'}", flush=True)


if __name__ == "__main__":
    main()
