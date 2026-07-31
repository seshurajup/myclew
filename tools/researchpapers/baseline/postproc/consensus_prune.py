"""Standalone post-proc module 3: ultra-tiny edge-consensus precision prune.

Drops only a small, capped (hard cap ~50) set of weak RELAXED motion-only edges that are
physically long, motion-inconsistent, AND unsupported by the learned edge scorer. Branch,
gap-closed, gap2-recovered and safe-division edges are protected. Writes refined GEFFs.

The consensus prune does NOT exist in ravi; it is COPIED VERBATIM from the beicicc kernel
  scratchpad/kernels/beicicc_biohub-exp029-lb884-prune50/*.txt
    - edge_consensus_precision_filter  (kernel lines 1925-1997)
Params are the kernel's `ultra_tiny_edge_prune_50` preset (kernel lines 223-231):
  MIN_DISTANCE_UM=9.9, MIN_MOTION_UM=7.5, MAX_LEARNED_PROB=0.006,
  MAX_DROP_FRAC=0.00035, MAX_DROP_ABS=50 (the ~50 hard cap), PROTECT_BRANCH=1.

The filter keys on `motion_pass`=="relaxed" and `motion_distance_um`, which only the
constant-velocity motion relinker produces. So (as in the real pipeline, where prune runs
after motion relink) this module first runs the VERBATIM motion_relink_edges from
motion_relink_gap.py to build the motion-tagged graph, then applies the verbatim prune.
Only the GEFF <-> nodes_by_id / edges I/O glue is adapted.

CLI:  python consensus_prune.py --in-dir <pred geff dir> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _geff_glue as glue  # noqa: E402
from motion_relink_gap import motion_relink_edges, edge_distance_um  # verbatim relink + helper

# beicicc `ultra_tiny_edge_prune_50` preset (kernel lines 223-231), applied through the
# kernel's exact env-override derivation (kernel lines 353-359).
_PRUNE_50 = {
    "BIOHUB_OUTPUT_EDGE_CONSENSUS_PRUNE": "1",
    "BIOHUB_EDGE_CONSENSUS_MIN_DISTANCE_UM": "9.9",
    "BIOHUB_EDGE_CONSENSUS_MIN_MOTION_UM": "7.5",
    "BIOHUB_EDGE_CONSENSUS_MAX_LEARNED_PROB": "0.006",
    "BIOHUB_EDGE_CONSENSUS_MAX_DROP_FRAC": "0.00035",
    "BIOHUB_EDGE_CONSENSUS_MAX_DROP_ABS": "50",
    "BIOHUB_EDGE_CONSENSUS_PROTECT_BRANCH": "1",
}
for _key, _value in _PRUNE_50.items():
    os.environ.setdefault(_key, str(_value))

OUTPUT_EDGE_CONSENSUS_PRUNE = os.environ.get("BIOHUB_OUTPUT_EDGE_CONSENSUS_PRUNE", "0") != "0"
EDGE_CONSENSUS_MIN_DISTANCE_UM = float(os.environ.get("BIOHUB_EDGE_CONSENSUS_MIN_DISTANCE_UM", "8.8"))
EDGE_CONSENSUS_MIN_MOTION_UM = float(os.environ.get("BIOHUB_EDGE_CONSENSUS_MIN_MOTION_UM", "6.2"))
EDGE_CONSENSUS_MAX_LEARNED_PROB = float(os.environ.get("BIOHUB_EDGE_CONSENSUS_MAX_LEARNED_PROB", "0.015"))
EDGE_CONSENSUS_MAX_DROP_FRAC = float(os.environ.get("BIOHUB_EDGE_CONSENSUS_MAX_DROP_FRAC", "0.0011"))
EDGE_CONSENSUS_MAX_DROP_ABS = int(os.environ.get("BIOHUB_EDGE_CONSENSUS_MAX_DROP_ABS", "160"))
EDGE_CONSENSUS_PROTECT_BRANCH = os.environ.get("BIOHUB_EDGE_CONSENSUS_PROTECT_BRANCH", "1") != "0"


# ===========================================================================
# VERBATIM from beicicc kernel (edge_consensus_precision_filter, lines 1925-1997)
# ===========================================================================
def edge_consensus_precision_filter(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> list[dict[str, object]]:
    """Prune only a tiny capped set of weak relaxed motion-only edges.

    This is intentionally conservative. It does not add nodes or edges and it
    protects gap-closed, gap2-recovered, safe-division, and branch-touching
    edges. A candidate must be all of the following:
      * a relaxed motion-relink edge,
      * physically long,
      * motion-inconsistent,
      * unsupported by the learned edge scorer.
    """
    if not OUTPUT_EDGE_CONSENSUS_PRUNE or not edges:
        return edges

    out_count: dict[int, int] = {}
    in_count: dict[int, int] = {}
    for edge in edges:
        out_count[int(edge["source_id"])] = out_count.get(int(edge["source_id"]), 0) + 1
        in_count[int(edge["target_id"])] = in_count.get(int(edge["target_id"]), 0) + 1

    candidates: list[tuple[float, int, dict[str, object]]] = []
    for idx, edge in enumerate(edges):
        if edge.get("gap_closed") or edge.get("gap2_recovered") or edge.get("safe_division"):
            continue
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        if source_id not in nodes_by_id or target_id not in nodes_by_id:
            continue
        if EDGE_CONSENSUS_PROTECT_BRANCH and (out_count.get(source_id, 0) > 1 or in_count.get(target_id, 0) > 1):
            continue
        if str(edge.get("motion_pass", "")).lower() != "relaxed":
            continue
        dist = float(edge.get("distance_um", edge_distance_um(nodes_by_id[source_id], nodes_by_id[target_id])))
        motion = float(edge.get("motion_distance_um", dist))
        prob_raw = edge.get("edge_prob", 0.0)
        try:
            prob = float(prob_raw)
        except (TypeError, ValueError):
            prob = 0.0
        if not np.isfinite(prob):
            prob = 0.0
        if dist < EDGE_CONSENSUS_MIN_DISTANCE_UM:
            continue
        if motion < EDGE_CONSENSUS_MIN_MOTION_UM:
            continue
        if prob > EDGE_CONSENSUS_MAX_LEARNED_PROB:
            continue
        # Higher score means more suspicious. Favor long, motion-inconsistent,
        # learned-unsupported relaxed links.
        score = dist + 0.75 * motion - 8.0 * prob
        candidates.append((score, idx, edge))

    stats["edge_consensus_candidates"] = len(candidates)
    if not candidates:
        return edges

    cap_frac = max(0, int(round(len(edges) * EDGE_CONSENSUS_MAX_DROP_FRAC))) if EDGE_CONSENSUS_MAX_DROP_FRAC > 0 else 0
    cap = min(EDGE_CONSENSUS_MAX_DROP_ABS, cap_frac) if EDGE_CONSENSUS_MAX_DROP_ABS > 0 else cap_frac
    cap = max(0, cap)
    stats["edge_consensus_drop_cap"] = cap
    if cap <= 0:
        return edges

    candidates.sort(key=lambda item: item[0], reverse=True)
    drop_idx = {idx for _, idx, _ in candidates[:cap]}
    if not drop_idx:
        return edges
    stats["edge_consensus_dropped"] = len(drop_idx)
    return [edge for idx, edge in enumerate(edges) if idx not in drop_idx]


# ===========================================================================
# I/O glue driver
# ===========================================================================
def process_geff(in_geff: Path, out_geff: Path):
    from collections import Counter
    nodes_by_id, ilp_edges, learned = glue.read_pred_geff(in_geff, filter_solution=True)
    stats: dict[str, int] = Counter()

    # Build the motion-tagged graph the consensus filter operates on (real-pipeline
    # order: prune runs after motion relink). motion_relink_edges is the verbatim relinker.
    edges = motion_relink_edges(nodes_by_id, stats, learned)
    e_before = len(edges)
    edges = edge_consensus_precision_filter(nodes_by_id, edges, stats)
    e_after = len(edges)

    glue.write_geff(out_geff, nodes_by_id, edges)
    print(f"  {in_geff.name}: edges {e_before}(relinked)->{e_after}(pruned)  dropped={stats.get('edge_consensus_dropped',0)}"
          f"  [candidates={stats.get('edge_consensus_candidates',0)} cap={stats.get('edge_consensus_drop_cap',0)}"
          f" relaxed_pool={stats.get('motion_relink_relaxed_edges',0)}]")
    return e_before, e_after


def main(argv=None):
    ap = argparse.ArgumentParser(description="ultra-tiny edge-consensus precision prune (cap ~50) on prediction geffs")
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    geffs = glue.list_pred_geffs(args.in_dir)
    print(f"consensus_prune: {len(geffs)} geff(s) from {args.in_dir}")
    tot_b = tot_a = 0
    for stem, path in geffs:
        eb, ea = process_geff(path, args.out_dir / f"{stem}.geff")
        tot_b += eb; tot_a += ea
    print(f"TOTAL edges {tot_b} -> {tot_a}  (dropped {tot_b - tot_a})")


if __name__ == "__main__":
    main()
