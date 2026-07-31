"""HONEST golden-12 scorer — canonical tracking_cellmot.metrics (NOT the src.metric proxy).

Leader mandate (2026-07-09): reconcile all prior golden-12 numbers against the OFFICIAL scorer.
src/metric.py is a self-admitted PROXY ("official aggregation weights are not public"); every
0.87xx we quoted came from it. This script scores golden-12 with the SAME metric code the official
evaluate.py uses — tracking_cellmot.metrics.{evaluate,node_recall,per_sample_metrics,summarise} —
so the number is the real thing.

Pipeline per dataset:
  cached pilk pred geff --geff_to_dicts--> pilk_post.filter_output_graph (env-driven gap/mtl/etc)
  --> write temp geff --> td IndexedRXGraph.from_geff --> evaluate(pred, GT.tracks, scale, 7um)
  --> per_sample_metrics(er, estimated_number_of_nodes, node_recall) --> summarise.

Params come from BIOHUB_* env vars (read by pilk_post at import), so the fleet/grid driver sets
BIOHUB_OUTPUT_FILTER_SHORT_TRACKS / BIOHUB_OUTPUT_MIN_TRACK_LEN / BIOHUB_GAP_CLOSE_UM and runs this
as a fresh subprocess per combination.

Prints ONE json line: {score, adj_edge_jaccard, edge_jaccard, node_recall, division_jaccard, n, ...}.

Usage:
  research/cellmot_venv/bin/python scripts/score_golden12_official.py [--limit N] [--raw] [--tag STR]
    --limit N : embryo-balanced fast screen on N datasets (0/absent = full golden-12)
    --raw     : score the RAW pilk pred (skip filter_output_graph) — sanity/anchor baseline
    --tag     : echoed into the json line for the grid driver
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP / "research" / "official_repo" / "src"))
sys.path.insert(0, str(COMP / "learning" / "ensemble_work"))
sys.path.insert(0, str(COMP / "tools" / "researchpapers" / "baseline" / "postproc"))

import tracksdata as td  # noqa: E402
from geff import GeffMetadata  # noqa: E402
from tracking_cellmot.io import open_dataset  # noqa: E402
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate as compute_metric,
    node_recall,
    per_sample_metrics,
    summarise,
)

from score_pilkwang import geff_to_dicts, GOLDEN12  # noqa: E402
import pilk_post as P  # noqa: E402  (reads BIOHUB_* env at import)
import zarr  # noqa: E402


def _prop_meta(ident, dtype):
    return {"identifier": ident, "dtype": dtype, "varlength": False,
            "unit": None, "name": None, "description": None}


def write_geff(out_path: Path, nodes_by_id: dict, edges: list):
    """Write a GEFF v1.1 group that the installed geff_spec ACCEPTS (needs node_props_metadata /
    edge_props_metadata — the shared _geff_glue.write_geff omits them, so canonical from_geff rejects
    its output; the src.metric proxy's io.read_geff tolerated it, which is why proxy numbers exist but
    were never canonically validated)."""
    out_path = Path(out_path)
    if out_path.exists():
        import shutil
        shutil.rmtree(out_path)
    ids = sorted(nodes_by_id)
    t = np.array([int(nodes_by_id[i]["t"]) for i in ids], dtype=np.int32)
    z = np.array([float(nodes_by_id[i]["z"]) for i in ids], dtype=np.float64)
    y = np.array([float(nodes_by_id[i]["y"]) for i in ids], dtype=np.float64)
    x = np.array([float(nodes_by_id[i]["x"]) for i in ids], dtype=np.float64)
    node_ids = np.array(ids, dtype=np.uint64)
    if edges:
        e_arr = np.array([[int(e["source_id"]), int(e["target_id"])] for e in edges], dtype=np.uint64)
        e_prob = np.array([float(e["edge_prob"]) if e.get("edge_prob") is not None else 0.0
                           for e in edges], dtype=np.float64)
    else:
        e_arr = np.zeros((0, 2), dtype=np.uint64)
        e_prob = np.zeros((0,), dtype=np.float64)
    g = zarr.open_group(str(out_path), mode="w")
    g.create_array("nodes/ids", data=node_ids)
    g.create_array("nodes/props/t/values", data=t)
    g.create_array("nodes/props/z/values", data=z)
    g.create_array("nodes/props/y/values", data=y)
    g.create_array("nodes/props/x/values", data=x)
    g.create_array("edges/ids", data=e_arr)
    g.create_array("edges/props/edge_prob/values", data=e_prob)
    axes = [{"name": n, "type": "time" if n == "t" else "space",
             "min": float(a.min()) if len(a) else 0.0, "max": float(a.max()) if len(a) else 0.0}
            for n, a in (("t", t), ("z", z), ("y", y), ("x", x))]
    g.attrs["geff"] = {
        "geff_version": "1.1", "directed": True, "axes": axes,
        "node_props_metadata": {k: _prop_meta(k, d) for k, d in
                                (("t", "int32"), ("z", "float64"), ("y", "float64"), ("x", "float64"))},
        "edge_props_metadata": {"edge_prob": _prop_meta("edge_prob", "float64")},
        "extra": {"estimated_number_of_nodes": int(len(ids))},
    }
    return out_path

SCALE = (1.625, 0.40625, 0.40625)          # z, y, x µm/voxel (fixed data contract)
MATCH_GATE = 7.0
TRAIN = COMP / "input" / "biohub-cell-tracking-during-development" / "train"
PILK = COMP / "research" / "pilkwang_support_pack" / "repo" / "predictions" / "seshu" / "unet_transformer" / "split_0"


def _estimated_n(geff_path: Path) -> float:
    try:
        meta = GeffMetadata.read(geff_path)
    except Exception:
        return float("nan")
    val = (meta.extra or {}).get("estimated_number_of_nodes")
    return float(val) if val is not None else float("nan")


def _subset(limit: int):
    ds_all = sorted(GOLDEN12)
    if limit <= 0 or limit >= len(ds_all):
        return ds_all
    by_emb = {}
    for d in ds_all:
        by_emb.setdefault(d.split("_")[0], []).append(d)
    picked, i = [], 0
    while len(picked) < limit:
        for emb in sorted(by_emb):
            if i < len(by_emb[emb]) and len(picked) < limit:
                picked.append(by_emb[emb][i])
        i += 1
        if i > max(len(v) for v in by_emb.values()):
            break
    return sorted(picked)


def _find_in_dir(pred_dir: Path, ds: str):
    """Locate an already-built prediction geff for ds in pred_dir (GPU-pipeline outputs)."""
    for cand in (pred_dir / f"{ds}.geff", pred_dir / f"{ds}.zarr.geff"):
        if cand.exists():
            return cand
    return None


def _pred_geff_for(ds: str, tmp: Path, raw: bool, pred_dir: Path | None) -> Path:
    """Return the prediction geff for one dataset.

    --pred-dir MODE: score an already-built prediction geff verbatim (GPU model / pipeline output) —
    NO pilk_post, NO BIOHUB_* env. This is how GPU re-detect / external_train outputs clear the SAME
    canonical gate. Otherwise: build from the cached pilk blobs via env-driven filter_output_graph.
    """
    if pred_dir is not None:
        p = _find_in_dir(pred_dir, ds)
        if p is None:
            raise FileNotFoundError(f"no prediction geff for {ds} in {pred_dir}")
        return p
    src_geff = PILK / f"{ds}.zarr.geff"
    if raw:
        return src_geff
    nbi, raw_edges = geff_to_dicts(src_geff)
    nbi2, edges2, _ = P.filter_output_graph(dict(nbi), list(raw_edges), dataset=ds)
    out = tmp / f"{ds}.geff"
    write_geff(out, nbi2, edges2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--pred-dir", type=str, default=None,
                    help="score already-built prediction geffs from this dir (GPU/pipeline output); "
                         "bypasses the cached-pilk postproc build")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--split-file", type=str, default=None,
                    help="score a LOEO fold's test set from this split file (list of {train,test}) instead of golden-12")
    ap.add_argument("--fold", type=int, default=0, help="fold index into --split-file")
    args = ap.parse_args()

    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    if args.split_file:
        folds = json.loads(Path(args.split_file).read_text())
        datasets = sorted(t.replace(".zarr", "").replace(".geff", "") for t in folds[args.fold]["test"])
    else:
        datasets = _subset(args.limit)
    if pred_dir is not None:
        datasets = [d for d in datasets if _find_in_dir(pred_dir, d) is not None]
    rows = []
    with tempfile.TemporaryDirectory() as td_dir:
        tmp = Path(td_dir)
        for i, ds in enumerate(datasets, 1):
            gt_geff = TRAIN / f"{ds}.geff"
            print(f"PROGRESS {i}/{len(datasets)} {ds}", file=sys.stderr, flush=True)
            gt_ds = open_dataset(TRAIN / ds, require_tracks=True, load_image=False, device="cpu")
            pred_path = _pred_geff_for(ds, tmp, args.raw, pred_dir)
            pred_res = td.graph.IndexedRXGraph.from_geff(pred_path)
            pred_graph = pred_res[0] if isinstance(pred_res, tuple) else pred_res
            er = compute_metric(pred_graph, gt_ds.tracks, scale=SCALE, max_distance=MATCH_GATE)
            if pred_graph.num_edges() > 0 and pred_graph.num_nodes() > 0:
                recall = node_recall(pred_graph, gt_ds.tracks)
            else:
                recall = 0.0
            rows.append(per_sample_metrics(er, _estimated_n(gt_geff), recall))

    s = summarise(rows)
    out = {
        "tag": args.tag,
        "metric": "tracking_cellmot.metrics.canonical",   # self-identifying: this is the REAL metric, not src.metric proxy
        "eval_set": "golden_12",
        "raw": args.raw,
        "score": round(float(s["score"]), 4),
        "adj_edge_jaccard": round(float(s["adj_edge_jaccard"]), 4),
        "edge_jaccard": round(float(s["edge_jaccard"]), 4),
        "node_recall": round(float(s["node_recall"]), 4) if s["node_recall"] == s["node_recall"] else None,
        "division_jaccard": (round(float(s["division_jaccard"]), 4)
                             if s["division_jaccard"] == s["division_jaccard"] else None),
        "n": s["n"],
        "source": ("pred_dir:" + str(pred_dir)) if pred_dir is not None else ("pilk_cache_raw" if args.raw else "pilk_cache_postproc"),
        "gap_close_um": os.environ.get("BIOHUB_GAP_CLOSE_UM", "6.0") if pred_dir is None else None,
        "min_track_len": os.environ.get("BIOHUB_OUTPUT_MIN_TRACK_LEN", "4") if pred_dir is None else None,
        "filter_short": os.environ.get("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "0") if pred_dir is None else None,
    }
    print(json.dumps(out))


if __name__ == "__main__":
    main()
