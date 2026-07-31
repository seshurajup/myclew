"""Standalone post-proc module 5: line-fit trajectory smoothing.

Smooths node COORDINATES along each linear track interior by fitting a line to a
+/- window neighbourhood (single-predecessor / single-successor chain) and blending
the fitted position with the original by OUTPUT_LINEFIT_WEIGHT. Graph TOPOLOGY is
unchanged: node and edge counts are identical before/after; only z,y,x move.

The function is COPIED VERBATIM from
  learning/ensemble_work/pilkwang_full/pipeline.py
    - linefit_smooth_output_graph  (pipeline.py lines 1930-2001)
(identical body also in learning/public_pull/ravi_lineage_0891/code.py:1580.)

Params preserved verbatim from pipeline.py:125-127. Default LINEFIT_WEIGHT is 0.8
(pipeline global); the ravi score_push config uses 0.72 with window 2
(ravi code.py:147-148). We expose both as CLI so either can be A/B'd:
  BIOHUB_OUTPUT_LINEFIT_SMOOTH   (default 1)
  BIOHUB_OUTPUT_LINEFIT_WEIGHT   (default 0.8; ravi score_push = 0.72) -> --weight
  BIOHUB_OUTPUT_LINEFIT_WINDOW   (default 2)                           -> --window

Per-track ordering is built from the edge graph (consecutive-frame single-parent /
single-child chains), exactly as the verbatim function does. Only the GEFF <->
nodes_by_id/edges I/O glue is adapted.

CLI:  python line_fit_smooth.py --in-dir <pred geff dir> --out-dir <dir> [--weight W] [--window N]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _geff_glue as glue  # noqa: E402

# ---- params preserved verbatim from pipeline.py lines 125-127 -------------------
OUTPUT_LINEFIT_SMOOTH = os.environ.get("BIOHUB_OUTPUT_LINEFIT_SMOOTH", "1") != "0"
OUTPUT_LINEFIT_WEIGHT = float(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WEIGHT", "0.8"))
OUTPUT_LINEFIT_WINDOW = int(os.environ.get("BIOHUB_OUTPUT_LINEFIT_WINDOW", "2"))


# ===========================================================================
# VERBATIM from pipeline.py (linefit_smooth_output_graph, lines 1930-2001)
# ===========================================================================
def linefit_smooth_output_graph(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    """Smooth linear track interiors without changing graph topology."""
    if not OUTPUT_LINEFIT_SMOOTH or OUTPUT_LINEFIT_WEIGHT <= 0 or OUTPUT_LINEFIT_WINDOW <= 0 or not edges:
        return nodes_by_id

    predecessor: dict[int, list[int]] = {}
    successor: dict[int, list[int]] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        source = nodes_by_id.get(source_id)
        target = nodes_by_id.get(target_id)
        if source is None or target is None:
            continue
        if int(target["t"]) != int(source["t"]) + 1:
            continue
        successor.setdefault(source_id, []).append(target_id)
        predecessor.setdefault(target_id, []).append(source_id)

    original_pos = {
        node_id: np.array([float(node["z"]), float(node["y"]), float(node["x"])], dtype=np.float64)
        for node_id, node in nodes_by_id.items()
    }
    updated_pos: dict[int, np.ndarray] = {}
    weight = float(np.clip(OUTPUT_LINEFIT_WEIGHT, 0.0, 1.0))

    for node_id in sorted(nodes_by_id):
        neighbourhood: list[tuple[int, int]] = [(0, node_id)]

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            prev_ids = predecessor.get(current, [])
            if len(prev_ids) != 1:
                break
            current = prev_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((-step, current))

        current = node_id
        for step in range(1, OUTPUT_LINEFIT_WINDOW + 1):
            next_ids = successor.get(current, [])
            if len(next_ids) != 1:
                break
            current = next_ids[0]
            if current not in original_pos:
                break
            neighbourhood.append((step, current))

        if len(neighbourhood) < 3:
            stats["linefit_skipped_nodes"] += 1
            continue

        dts = np.array([delta for delta, _ in neighbourhood], dtype=np.float64)
        coords = np.stack([original_pos[nid] for _, nid in neighbourhood])
        fitted = np.array([np.polyval(np.polyfit(dts, coords[:, axis], 1), 0.0) for axis in range(3)], dtype=np.float64)
        if not np.isfinite(fitted).all():
            stats["linefit_skipped_nodes"] += 1
            continue
        updated_pos[node_id] = (1.0 - weight) * original_pos[node_id] + weight * fitted

    for node_id, pos in updated_pos.items():
        nodes_by_id[node_id]["z"] = float(pos[0])
        nodes_by_id[node_id]["y"] = float(pos[1])
        nodes_by_id[node_id]["x"] = float(pos[2])

    stats["linefit_smoothed_nodes"] = len(updated_pos)
    return nodes_by_id


# ===========================================================================
# I/O glue driver
# ===========================================================================
def _coord_shift_um(before: dict, after: dict) -> float:
    import math
    scale = glue.VOXEL_SCALE_UM
    dz = (float(before["z"]) - float(after["z"])) * scale[0]
    dy = (float(before["y"]) - float(after["y"])) * scale[1]
    dx = (float(before["x"]) - float(after["x"])) * scale[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def process_geff(in_geff: Path, out_geff: Path):
    nodes_by_id, edges, _learned = glue.read_pred_geff(in_geff, filter_solution=True)
    stats: dict[str, int] = Counter()

    n_before, e_before = len(nodes_by_id), len(edges)
    before_pos = {nid: dict(n) for nid, n in nodes_by_id.items()}
    nodes_by_id = linefit_smooth_output_graph(nodes_by_id, edges, stats)
    n_after, e_after = len(nodes_by_id), len(edges)

    moved = sum(1 for nid in nodes_by_id if _coord_shift_um(before_pos[nid], nodes_by_id[nid]) > 1e-9)
    shifts = [_coord_shift_um(before_pos[nid], nodes_by_id[nid]) for nid in nodes_by_id]
    mean_shift = float(np.mean(shifts)) if shifts else 0.0
    max_shift = float(np.max(shifts)) if shifts else 0.0

    glue.write_geff(out_geff, nodes_by_id, edges)
    print(f"  {in_geff.name}: nodes {n_before}->{n_after}  edges {e_before}->{e_after}"
          f"  [smoothed={stats.get('linefit_smoothed_nodes', 0)} moved={moved}"
          f" mean_shift={mean_shift:.4f}um max_shift={max_shift:.4f}um]")
    return n_before, n_after, e_before, e_after


def main(argv=None):
    # CLI override of the verbatim function's module globals (resolved at call-time).
    global OUTPUT_LINEFIT_WEIGHT, OUTPUT_LINEFIT_WINDOW
    ap = argparse.ArgumentParser(description="line-fit trajectory smoothing on prediction geffs (moves coords, keeps topology)")
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--weight", type=float, default=OUTPUT_LINEFIT_WEIGHT,
                    help="blend weight of fitted vs original position (default 0.8 pipeline; ravi score_push=0.72)")
    ap.add_argument("--window", type=int, default=OUTPUT_LINEFIT_WINDOW,
                    help="+/- neighbourhood window along the track (default 2)")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_LINEFIT_WEIGHT = float(args.weight)
    OUTPUT_LINEFIT_WINDOW = int(args.window)

    geffs = glue.list_pred_geffs(args.in_dir)
    print(f"line_fit_smooth: {len(geffs)} geff(s) from {args.in_dir}  "
          f"(weight={OUTPUT_LINEFIT_WEIGHT}, window={OUTPUT_LINEFIT_WINDOW})")
    tot_nb = tot_na = tot_eb = tot_ea = 0
    for stem, path in geffs:
        nb, na, eb, ea = process_geff(path, args.out_dir / f"{stem}.geff")
        tot_nb += nb; tot_na += na; tot_eb += eb; tot_ea += ea
    print(f"TOTAL nodes {tot_nb} -> {tot_na}   edges {tot_eb} -> {tot_ea}  (topology unchanged; coords smoothed)")


if __name__ == "__main__":
    main()
