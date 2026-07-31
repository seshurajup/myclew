"""Standalone post-proc module 6: boristown T+2-aware safe-division ranking.

Adds conservative SECOND daughters to single-child parents, but RE-RANKS which
candidates spend the capped additions first using a t+2 daughter-coherence bonus:
a division is preferred when BOTH the existing child and the new candidate each
continue to a distinct node at t+2 with short per-step motion and a short sister
distance at t+2. This is DIFFERENT from the plain geometric add_safe_divisions
(confirmed dead in our metric) — the geometry gates are the same, but the ranking
key gets a -SAFE_DIV_T2_BONUS_UM discount for t+2-supported pairs.

The function is COPIED VERBATIM from the boristown "V14 T+2-Aware Division Ranking"
kernel:
  learning/public_pull/boristown_agi-biohub-cell-tracking/agi-biohub-cell-tracking.txt
    - add_safe_divisions_postlink  (kernel .txt lines 1625-1745)
    - _single_successor_map        (kernel .txt lines 1461-1465)
    - edge_distance_um             (kernel .txt lines 1102-1106)

Params preserved verbatim from the kernel .txt (lines 249-251, 261-266):
  BIOHUB_OUTPUT_SAFE_DIVISIONS   (default 1)
  BIOHUB_SAFE_DIV_MAX_UM=4.7  BIOHUB_SAFE_DIV_SISTER_MAX_UM=7.2
  BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM=7.8
  BIOHUB_SAFE_DIV_FRAME_FRAC_CAP=0.008  BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP=0.004
  BIOHUB_SAFE_DIV_T2_BONUS_UM=0.90 (soft rank bonus)
  BIOHUB_SAFE_DIV_T2_STEP_MAX_UM=6.5  BIOHUB_SAFE_DIV_T2_SISTER_MAX_UM=9.0

NOTE: division is largely a phantom in our metric (div_tp ~ 0 even for pilkwang),
so this is LOW priority and may not move the score — authored + dry-runnable for A/B.
Only the GEFF <-> nodes_by_id/edges I/O glue is adapted.

CLI:  python boristown_t2_division.py --in-dir <pred geff dir> --out-dir <dir>
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _geff_glue as glue  # noqa: E402

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)

# ---- params preserved verbatim from kernel .txt lines 261-266, 249-251 ----------
OUTPUT_SAFE_DIVISIONS = os.environ.get("BIOHUB_OUTPUT_SAFE_DIVISIONS", "1") != "0"
SAFE_DIV_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_MAX_UM", "4.7"))
SAFE_DIV_SISTER_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_SISTER_MAX_UM", "7.2"))
SAFE_DIV_EXISTING_CHILD_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM", "7.8"))
SAFE_DIV_FRAME_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_FRAME_FRAC_CAP", "0.008"))
SAFE_DIV_GLOBAL_FRAC_CAP = float(os.environ.get("BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP", "0.004"))
SAFE_DIV_T2_BONUS_UM = float(os.environ.get("BIOHUB_SAFE_DIV_T2_BONUS_UM", "0.90"))
SAFE_DIV_T2_STEP_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_T2_STEP_MAX_UM", "6.5"))
SAFE_DIV_T2_SISTER_MAX_UM = float(os.environ.get("BIOHUB_SAFE_DIV_T2_SISTER_MAX_UM", "9.0"))


# ===========================================================================
# VERBATIM helpers from boristown kernel .txt
# ===========================================================================
def edge_distance_um(source: dict[str, object], target: dict[str, object]) -> float:  # .txt lines 1102-1106
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def _single_successor_map(edges: list[dict[str, object]]) -> dict[int, int]:  # .txt lines 1461-1465
    by_source: dict[int, list[int]] = {}
    for edge in edges:
        by_source.setdefault(int(edge["source_id"]), []).append(int(edge["target_id"]))
    return {source: targets[0] for source, targets in by_source.items() if len(targets) == 1}


# ===========================================================================
# VERBATIM from boristown kernel .txt (add_safe_divisions_postlink, lines 1625-1745)
# ===========================================================================
def add_safe_divisions_postlink(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> list[dict[str, object]]:
    """Add conservative second daughters, ranking t+2-supported events first.

    This remains a soft ranking signal rather than a hard eligibility gate. It
    therefore preserves division recall near sequence ends while using future
    continuation to spend the existing per-frame/global caps on more coherent
    daughter pairs when that evidence exists.
    """
    if not OUTPUT_SAFE_DIVISIONS or not edges or not nodes_by_id:
        return edges

    out_by_source: dict[int, list[dict[str, object]]] = {}
    incoming: set[int] = set()
    for edge in edges:
        out_by_source.setdefault(int(edge["source_id"]), []).append(edge)
        incoming.add(int(edge["target_id"]))

    successor = _single_successor_map(edges)
    ids_by_t: dict[int, list[int]] = {}
    for node_id, node in nodes_by_id.items():
        ids_by_t.setdefault(int(node["t"]), []).append(node_id)

    existing_edges = {(int(edge["source_id"]), int(edge["target_id"])) for edge in edges}
    global_cap = max(1, int(round(max(1, len(edges)) * SAFE_DIV_GLOBAL_FRAC_CAP)))
    added: list[dict[str, object]] = []
    used_targets: set[int] = set()

    for t in sorted(ids_by_t):
        child_frame_ids = ids_by_t.get(t + 1, [])
        if not child_frame_ids:
            continue
        source_ids = [node_id for node_id in ids_by_t[t] if len(out_by_source.get(node_id, [])) == 1]
        candidate_ids = [node_id for node_id in child_frame_ids if node_id not in incoming and node_id not in used_targets]
        if not source_ids or not candidate_ids:
            continue

        frame_cap = max(1, int(round(len(source_ids) * SAFE_DIV_FRAME_FRAC_CAP)))
        proposals: list[tuple[float, int, int, float, float, bool]] = []
        for source_id in source_ids:
            source = nodes_by_id[source_id]
            existing_child_edge = out_by_source[source_id][0]
            existing_child_id = int(existing_child_edge["target_id"])
            existing_child = nodes_by_id.get(existing_child_id)
            if existing_child is None or int(existing_child["t"]) != t + 1:
                continue
            child_dist = edge_distance_um(source, existing_child)
            if child_dist > SAFE_DIV_EXISTING_CHILD_MAX_UM:
                continue
            for candidate_id in candidate_ids:
                if (source_id, candidate_id) in existing_edges:
                    continue
                candidate = nodes_by_id[candidate_id]
                parent_dist = edge_distance_um(source, candidate)
                if parent_dist > SAFE_DIV_MAX_UM:
                    continue
                sister_dist = edge_distance_um(existing_child, candidate)
                if sister_dist > SAFE_DIV_SISTER_MAX_UM:
                    continue

                t2_supported = False
                existing_next_id = successor.get(existing_child_id)
                candidate_next_id = successor.get(candidate_id)
                if (
                    existing_next_id is not None
                    and candidate_next_id is not None
                    and existing_next_id != candidate_next_id
                    and existing_next_id in nodes_by_id
                    and candidate_next_id in nodes_by_id
                    and int(nodes_by_id[existing_next_id]["t"]) == t + 2
                    and int(nodes_by_id[candidate_next_id]["t"]) == t + 2
                ):
                    existing_step = edge_distance_um(existing_child, nodes_by_id[existing_next_id])
                    candidate_step = edge_distance_um(candidate, nodes_by_id[candidate_next_id])
                    sister_t2 = edge_distance_um(nodes_by_id[existing_next_id], nodes_by_id[candidate_next_id])
                    t2_supported = (
                        existing_step <= SAFE_DIV_T2_STEP_MAX_UM
                        and candidate_step <= SAFE_DIV_T2_STEP_MAX_UM
                        and sister_t2 <= SAFE_DIV_T2_SISTER_MAX_UM
                    )
                    if t2_supported:
                        stats["safe_division_t2_supported_candidates"] += 1

                score = parent_dist + 0.15 * sister_dist
                if t2_supported:
                    score -= SAFE_DIV_T2_BONUS_UM
                proposals.append((score, source_id, candidate_id, parent_dist, sister_dist, t2_supported))

        stats["safe_division_candidates"] += len(proposals)
        if not proposals:
            continue
        proposals.sort(key=lambda item: item[0])
        added_this_frame = 0
        for _, source_id, candidate_id, parent_dist, _, t2_supported in proposals:
            if len(added) >= global_cap:
                stats["safe_division_skipped_cap"] += 1
                break
            if added_this_frame >= frame_cap:
                break
            if candidate_id in used_targets or candidate_id in incoming:
                continue
            added.append({
                "source_id": source_id,
                "target_id": candidate_id,
                "edge_prob": None,
                "distance_um": parent_dist,
                "safe_division": 1,
                "safe_division_t2_supported": int(t2_supported),
            })
            if t2_supported:
                stats["safe_division_t2_supported_added"] += 1
            used_targets.add(candidate_id)
            added_this_frame += 1

    if added:
        stats["safe_divisions_added"] = len(added)
        return [*edges, *added]
    return edges


# ===========================================================================
# I/O glue driver
# ===========================================================================
def process_geff(in_geff: Path, out_geff: Path):
    nodes_by_id, edges, _learned = glue.read_pred_geff(in_geff, filter_solution=True)
    stats: dict[str, int] = Counter()

    n_before, e_before = len(nodes_by_id), len(edges)
    edges = add_safe_divisions_postlink(nodes_by_id, edges, stats)
    n_after, e_after = len(nodes_by_id), len(edges)

    glue.write_geff(out_geff, nodes_by_id, edges)
    print(f"  {in_geff.name}: nodes {n_before}->{n_after}  edges {e_before}->{e_after}"
          f"  [div_added={stats.get('safe_divisions_added', 0)}"
          f" candidates={stats.get('safe_division_candidates', 0)}"
          f" t2_added={stats.get('safe_division_t2_supported_added', 0)}"
          f" t2_candidates={stats.get('safe_division_t2_supported_candidates', 0)}]")
    return n_before, n_after, e_before, e_after


def main(argv=None):
    ap = argparse.ArgumentParser(description="boristown T+2-aware safe-division ranking on prediction geffs (adds division edges)")
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    geffs = glue.list_pred_geffs(args.in_dir)
    print(f"boristown_t2_division: {len(geffs)} geff(s) from {args.in_dir}  "
          f"(t2_bonus={SAFE_DIV_T2_BONUS_UM}um step<={SAFE_DIV_T2_STEP_MAX_UM}um sister_t2<={SAFE_DIV_T2_SISTER_MAX_UM}um)")
    tot_nb = tot_na = tot_eb = tot_ea = 0
    for stem, path in geffs:
        nb, na, eb, ea = process_geff(path, args.out_dir / f"{stem}.geff")
        tot_nb += nb; tot_na += na; tot_eb += eb; tot_ea += ea
    print(f"TOTAL nodes {tot_nb} -> {tot_na}   edges {tot_eb} -> {tot_ea}  (added {tot_ea - tot_eb} division edges)")


if __name__ == "__main__":
    main()
