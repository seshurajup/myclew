#!/usr/bin/env python
"""Convert canqiang's persisted density-CV predictions (node/edge CSVs) into evaluate.py-ready geffs.

The density-CV gate runner (baseline/run_canqiang_loeodens.py) saved per-dataset predictions as CSVs but
NOT geffs, so the OFFICIAL scorer (research/official_repo/scripts/evaluate.py) — which discovers
predictions/{user}/{method}/split_{fold}/*.geff — cannot see them. This CPU-only converter (no GPU
re-predict) rebuilds geffs in that layout so canqiang can be scored with the SAME official metric as
pilkwang, closing the thread-A proxy hole for the fully-fair gate (EXP-CVGATE-FAIR).

Mirrors learning/ensemble_work/pilkwang_full/score_full.py's IndexedRXGraph -> to_geff pattern.
Fold mapping from splits_loeo_density.json: fold0 test = 44b6*, fold1 test = 6bba*.

Usage:
  python baseline/canqiang_csv_to_geff.py            # all 15 datasets, both folds
  python baseline/canqiang_csv_to_geff.py --limit 1  # minimal dry-run (first dataset only)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import polars as pl

PARENT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
ENS = PARENT / "learning/ensemble_work"
NODES = ENS / "canqiang_loeodens_nodes"
EDGES = ENS / "canqiang_loeodens_tracks"
OUT_ROOT = PARENT / "research/official_repo/predictions/seshu/canqiang_full"
FForEmbryo = {"44b6": 0, "6bba": 1}   # matches splits_loeo_density test folds

sys.path.insert(0, str(PARENT))
import tracksdata as td   # noqa: E402
K = td.DEFAULT_ATTR_KEYS


def convert_one(ds: str) -> Path:
    ndf = pd.read_csv(NODES / f"{ds}.csv")
    edf = pd.read_csv(EDGES / f"{ds}_edges.csv")
    g = td.graph.IndexedRXGraph()
    for k in (K.T, K.Z, K.Y, K.X):
        try:
            g.add_node_attr_key(k, pl.Float64, default_value=0.0)
        except Exception:  # noqa: BLE001
            pass
    idmap = {}
    for r in ndf.itertuples(index=False):
        idmap[int(r.node_id)] = g.add_node(
            {K.T: int(r.t), K.Z: float(r.z), K.Y: float(r.y), K.X: float(r.x)}, index=int(r.node_id))
    for r in edf.itertuples(index=False):
        s, t = int(r.source_id), int(r.target_id)
        if s in idmap and t in idmap:
            g.add_edge(idmap[s], idmap[t], {})
    fold = FForEmbryo[ds.split("_")[0]]
    out_dir = OUT_ROOT / f"split_{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{ds}.geff"
    if out.exists():                      # idempotent re-run (geff is a zarr dir)
        import shutil
        shutil.rmtree(out)
    g.to_geff(str(out))
    print(f"  {ds:18s} nodes={len(ndf):5d} edges={len(edf):5d} -> split_{fold}/{out.name}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="convert only the first N datasets (dry-run)")
    args = ap.parse_args()
    stems = sorted(p.stem for p in NODES.glob("*.csv"))
    if args.limit:
        stems = stems[:args.limit]
    print(f"converting {len(stems)} canqiang density predictions -> {OUT_ROOT}", flush=True)
    for ds in stems:
        convert_one(ds)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
