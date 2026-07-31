"""I/O glue for the standalone post-processing modules.

This is the ONLY adaptation layer between the verbatim-copied ravi/tamer/beicicc
pipeline functions (which operate on a ``nodes_by_id`` dict + an ``edges`` list of
dicts) and prediction GEFFs on disk. No pipeline math lives here.

read_pred_geff:  GEFF group -> (nodes_by_id, edges, learned_edge_probs)
                 filters solution==True (raw-ILP geffs carry a `solution` bool),
                 mirroring fleet_agents/official_scorer.py::_load_pilk_solution.
write_geff:      (nodes_by_id, edges) -> GEFF group (re-readable by src.io.read_geff)

Voxel scale = (1.625, 0.40625, 0.40625) um  (z, y, x).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import zarr

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def list_pred_geffs(in_dir: Path) -> list[tuple[str, Path]]:
    """Return [(dataset_stem, geff_path)] for every *.geff group in a prediction dir.

    Handles both `<ds>.zarr.geff` (pilkwang raw-ILP) and `<ds>.geff` (decomp_C)."""
    in_dir = Path(in_dir)
    out: list[tuple[str, Path]] = []
    for p in sorted(in_dir.iterdir()):
        if not p.name.endswith(".geff") or not p.is_dir():
            continue
        stem = p.name[:-len(".geff")]
        if stem.endswith(".zarr"):
            stem = stem[:-len(".zarr")]
        out.append((stem, p))
    return out


def read_pred_geff(geff_path: Path, filter_solution: bool = True):
    """Read a prediction GEFF into the (nodes_by_id, edges, learned_edge_probs) shape
    the pipeline functions expect.

    Returns
    -------
    nodes_by_id : dict[int, dict]  keys node_id,t,z,y,x
    edges       : list[dict]       keys source_id,target_id,edge_prob,distance_um
    learned_edge_probs : dict[(src,tgt) -> float]
    """
    g = zarr.open_group(str(geff_path), mode="r")
    node_ids = np.asarray(g["nodes/ids"][:], dtype=np.int64)
    t = np.asarray(g["nodes/props/t/values"][:], dtype=np.int64)
    z = np.asarray(g["nodes/props/z/values"][:], dtype=np.float64)
    y = np.asarray(g["nodes/props/y/values"][:], dtype=np.float64)
    x = np.asarray(g["nodes/props/x/values"][:], dtype=np.float64)

    n_keep = np.ones(len(node_ids), dtype=bool)
    if filter_solution and "nodes/props/solution/values" in _group_keys(g, "nodes/props"):
        n_keep = np.asarray(g["nodes/props/solution/values"][:], dtype=bool)

    nodes_by_id: dict[int, dict] = {}
    for i in range(len(node_ids)):
        if not n_keep[i]:
            continue
        nid = int(node_ids[i])
        nodes_by_id[nid] = {
            "node_id": nid,
            "t": int(t[i]),
            "z": float(z[i]),
            "y": float(y[i]),
            "x": float(x[i]),
        }

    edges_raw = np.asarray(g["edges/ids"][:], dtype=np.int64).reshape(-1, 2)
    edge_keep = np.ones(len(edges_raw), dtype=bool)
    if filter_solution and "edges/props/solution/values" in _group_keys(g, "edges/props"):
        edge_keep = np.asarray(g["edges/props/solution/values"][:], dtype=bool)
    edge_prob = None
    if "edges/props/edge_prob/values" in _group_keys(g, "edges/props"):
        edge_prob = np.asarray(g["edges/props/edge_prob/values"][:], dtype=np.float64)

    edges: list[dict] = []
    learned_edge_probs: dict[tuple[int, int], float] = {}
    for i in range(len(edges_raw)):
        if not edge_keep[i]:
            continue
        s = int(edges_raw[i, 0]); d = int(edges_raw[i, 1])
        if s not in nodes_by_id or d not in nodes_by_id:
            continue
        prob = float(edge_prob[i]) if edge_prob is not None else None
        edges.append({
            "source_id": s,
            "target_id": d,
            "edge_prob": prob,
            "distance_um": _edge_distance_um(nodes_by_id[s], nodes_by_id[d]),
        })
        if prob is not None:
            learned_edge_probs[(s, d)] = prob
    return nodes_by_id, edges, learned_edge_probs


def _group_keys(g, subpath: str) -> set[str]:
    try:
        sub = g[subpath]
    except KeyError:
        return set()
    return {f"{subpath}/{k}/values" for k in sub.keys()}


def _edge_distance_um(source: dict, target: dict) -> float:
    import math
    dz = (float(source["z"]) - float(target["z"])) * VOXEL_SCALE_UM[0]
    dy = (float(source["y"]) - float(target["y"])) * VOXEL_SCALE_UM[1]
    dx = (float(source["x"]) - float(target["x"])) * VOXEL_SCALE_UM[2]
    return math.sqrt(dz * dz + dy * dy + dx * dx)


def write_geff(out_path: Path, nodes_by_id: dict, edges: list[dict]):
    """Write a (nodes_by_id, edges) graph as a GEFF group re-readable by src.io.read_geff.

    Writes nodes/ids, nodes/props/{t,z,y,x}/values, edges/ids, edges/props/edge_prob/values
    and a minimal geff v1.1 attribute block."""
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
        e_prob = np.array([
            float(e["edge_prob"]) if e.get("edge_prob") is not None else 0.0 for e in edges
        ], dtype=np.float64)
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

    axes = []
    for name, arr in (("t", t), ("z", z), ("y", y), ("x", x)):
        axes.append({
            "name": name,
            "type": "time" if name == "t" else "space",
            "min": float(arr.min()) if len(arr) else 0.0,
            "max": float(arr.max()) if len(arr) else 0.0,
        })
    g.attrs["geff"] = {
        "geff_version": "1.1",
        "directed": True,
        "axes": axes,
        "extra": {"estimated_number_of_nodes": int(len(ids))},
    }
    return out_path
