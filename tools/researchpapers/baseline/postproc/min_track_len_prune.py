"""Standalone post-proc module 4: minimum-track-length component prune.

Removes connected track components shorter than ``OUTPUT_MIN_TRACK_LEN`` frames,
dropping BOTH the member nodes AND their edges (reduces over-prediction). This is
the highest-value output filter in the public writeups (~+0.011).

The function is COPIED VERBATIM from
  learning/ensemble_work/pilkwang_full/pipeline.py
    - filter_short_track_components  (pipeline.py lines 1870-1927,
      immediately after add_safe_divisions_postlink)
(identical body also in learning/public_pull/ravi_lineage_0891/code.py:1518 and
 boristown ...agi-biohub-cell-tracking.txt:1748.)

Params preserved verbatim from pipeline.py:121-123 / ravi code.py:266-268:
  BIOHUB_OUTPUT_FILTER_SHORT_TRACKS   (pipeline default 0; forced ON here since
                                       running this module IS the filter)
  BIOHUB_OUTPUT_MIN_TRACK_LEN         (default 4  -> --min-track-len)
  BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS (default 1; dividing components kept even
                                          when short, per pipeline)

IMPORTANT: this DROPS NODES as well as edges, so the module writes the reduced
nodes_by_id AND the reduced edges. Only the GEFF <-> nodes_by_id/edges I/O glue is
adapted; all pipeline math is verbatim.

CLI:  python min_track_len_prune.py --in-dir <pred geff dir> --out-dir <dir> [--min-track-len N]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _geff_glue as glue  # noqa: E402

# Force the filter on by default: running this module IS the short-track prune.
# (pipeline.py leaves BIOHUB_OUTPUT_FILTER_SHORT_TRACKS default "0" and turns it on
#  only in specific configs; here it is the module's whole purpose.)
os.environ.setdefault("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "1")

# ---- params preserved verbatim from pipeline.py lines 121-123 -------------------
OUTPUT_FILTER_SHORT_TRACKS = os.environ.get("BIOHUB_OUTPUT_FILTER_SHORT_TRACKS", "0") != "0"
OUTPUT_MIN_TRACK_LEN = int(os.environ.get("BIOHUB_OUTPUT_MIN_TRACK_LEN", "4"))
OUTPUT_KEEP_DIVISION_COMPONENTS = os.environ.get("BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS", "1") != "0"


# ===========================================================================
# VERBATIM from pipeline.py (filter_short_track_components, lines 1870-1927)
# ===========================================================================
def filter_short_track_components(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> tuple[dict[int, dict[str, object]], list[dict[str, object]]]:
    if not OUTPUT_FILTER_SHORT_TRACKS or OUTPUT_MIN_TRACK_LEN <= 1 or not edges:
        return nodes_by_id, edges

    parent = {node_id: node_id for node_id in nodes_by_id}

    def find(node_id: int) -> int:
        while parent[node_id] != node_id:
            parent[node_id] = parent[parent[node_id]]
            node_id = parent[node_id]
        return node_id

    def union(a: int, b: int) -> None:
        if a not in parent or b not in parent:
            return
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[ra] = rb

    out_count: dict[int, int] = {}
    for edge in edges:
        source_id = int(edge["source_id"])
        target_id = int(edge["target_id"])
        union(source_id, target_id)
        out_count[source_id] = out_count.get(source_id, 0) + 1

    components: dict[int, list[int]] = {}
    for node_id in nodes_by_id:
        components.setdefault(find(node_id), []).append(node_id)

    keep: set[int] = set()
    for members in components.values():
        has_division = any(out_count.get(node_id, 0) >= 2 for node_id in members)
        if len(members) >= OUTPUT_MIN_TRACK_LEN or (OUTPUT_KEEP_DIVISION_COMPONENTS and has_division):
            keep.update(members)

    if not keep:
        stats["short_track_filter_skipped_all"] += 1
        return nodes_by_id, edges

    removed_nodes = len(nodes_by_id) - len(keep)
    if removed_nodes <= 0:
        return nodes_by_id, edges

    kept_nodes = {node_id: node for node_id, node in nodes_by_id.items() if node_id in keep}
    kept_edges = [
        edge for edge in edges
        if int(edge["source_id"]) in kept_nodes and int(edge["target_id"]) in kept_nodes
    ]
    stats["short_track_components_removed"] = sum(1 for members in components.values() if not (set(members) & keep))
    stats["short_track_nodes_removed"] = removed_nodes
    stats["short_track_edges_removed"] = len(edges) - len(kept_edges)
    return kept_nodes, kept_edges


# ===========================================================================
# I/O glue driver
# ===========================================================================
def process_geff(in_geff: Path, out_geff: Path):
    nodes_by_id, edges, _learned = glue.read_pred_geff(in_geff, filter_solution=True)
    stats: dict[str, int] = Counter()

    n_before, e_before = len(nodes_by_id), len(edges)
    nodes_by_id, edges = filter_short_track_components(nodes_by_id, edges, stats)
    n_after, e_after = len(nodes_by_id), len(edges)

    glue.write_geff(out_geff, nodes_by_id, edges)
    print(f"  {in_geff.name}: nodes {n_before}->{n_after}  edges {e_before}->{e_after}"
          f"  [components_removed={stats.get('short_track_components_removed', 0)}"
          f" nodes_removed={stats.get('short_track_nodes_removed', 0)}"
          f" edges_removed={stats.get('short_track_edges_removed', 0)}]")
    return n_before, n_after, e_before, e_after


def main(argv=None):
    # CLI override of the verbatim function's module globals (resolved at call-time).
    global OUTPUT_MIN_TRACK_LEN
    ap = argparse.ArgumentParser(description="minimum-track-length component prune on prediction geffs (drops nodes+edges)")
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--min-track-len", type=int, default=OUTPUT_MIN_TRACK_LEN,
                    help="drop connected components shorter than this many nodes (default 4 = ravi/pipeline)")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_MIN_TRACK_LEN = int(args.min_track_len)

    geffs = glue.list_pred_geffs(args.in_dir)
    print(f"min_track_len_prune: {len(geffs)} geff(s) from {args.in_dir}  "
          f"(min_track_len={OUTPUT_MIN_TRACK_LEN}, keep_division_components={OUTPUT_KEEP_DIVISION_COMPONENTS})")
    tot_nb = tot_na = tot_eb = tot_ea = 0
    for stem, path in geffs:
        nb, na, eb, ea = process_geff(path, args.out_dir / f"{stem}.geff")
        tot_nb += nb; tot_na += na; tot_eb += eb; tot_ea += ea
    print(f"TOTAL nodes {tot_nb} -> {tot_na}  (dropped {tot_nb - tot_na})   "
          f"edges {tot_eb} -> {tot_ea}  (dropped {tot_eb - tot_ea})")


if __name__ == "__main__":
    main()
