"""Assemble and validate submission.csv.

Schema: id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
 - node rows: node_id,t,z,y,x set; source_id=target_id=-1
 - edge rows: source_id,target_id set; node_id,t,z,y,x=-1
node_id is unique within a dataset.
"""
from __future__ import annotations
from typing import List, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def dataset_rows(dataset: str, nodes: List[dict], edges: List[Tuple[int, int]]) -> List[dict]:
    rows = []
    for n in nodes:
        rows.append({"dataset": dataset, "row_type": "node", "node_id": n["node_id"],
                     "t": n["t"], "z": n["z"], "y": n["y"], "x": n["x"],
                     "source_id": -1, "target_id": -1})
    for s, d in edges:
        rows.append({"dataset": dataset, "row_type": "edge", "node_id": -1,
                     "t": -1, "z": -1, "y": -1, "x": -1,
                     "source_id": int(s), "target_id": int(d)})
    return rows


def finalize(all_rows: List[dict], path: str) -> pd.DataFrame:
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df.index.name = "id"
    validate(df)
    df.to_csv(path)
    return df


def from_geff_dir(pred_dir, datasets: List[str], out_csv: str) -> pd.DataFrame:
    """Build submission.csv from a directory of predicted .geff graphs (one per dataset).

    Coords are rounded to integer voxels (submission requires ints). Every dataset in
    `datasets` must have a geff (`<ds>.zarr.geff` or `<ds>.geff`).
    """
    import zarr
    pred_dir = Path(pred_dir)
    all_rows: List[dict] = []
    for ds in datasets:
        gp = pred_dir / f"{ds}.zarr.geff"
        if not gp.exists():
            gp = pred_dir / f"{ds}.geff"
        if not gp.exists():
            raise FileNotFoundError(f"no prediction geff for {ds} in {pred_dir}")
        g = zarr.open_group(str(gp), mode="r")
        ids = np.asarray(g["nodes/ids"][:]).astype(np.int64)
        t = np.asarray(g["nodes/props/t/values"][:]).astype(np.int64)
        z = np.rint(np.asarray(g["nodes/props/z/values"][:])).astype(np.int64)
        y = np.rint(np.asarray(g["nodes/props/y/values"][:])).astype(np.int64)
        x = np.rint(np.asarray(g["nodes/props/x/values"][:])).astype(np.int64)
        nodes = [{"node_id": int(ids[i]), "t": int(t[i]), "z": int(z[i]), "y": int(y[i]), "x": int(x[i])}
                 for i in range(len(ids))]
        e = np.asarray(g["edges/ids"][:]).reshape(-1, 2).astype(np.int64) if "edges/ids" in g else np.zeros((0, 2), np.int64)
        edges = [(int(s), int(d)) for s, d in e]
        all_rows.extend(dataset_rows(ds, nodes, edges))
    return finalize(all_rows, out_csv)


def validate(df: pd.DataFrame) -> None:
    assert list(df.columns) == COLUMNS, f"bad columns: {list(df.columns)}"
    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    assert (nodes[["z", "y", "x"]] >= 0).all().all(), "negative node coordinates"
    # no dangling edges: every source/target must exist as a node within the same dataset
    for ds, g in df.groupby("dataset"):
        nid = set(g[g["row_type"] == "node"]["node_id"])
        e = g[g["row_type"] == "edge"]
        missing = (set(e["source_id"]) | set(e["target_id"])) - nid
        assert not missing, f"{ds}: dangling edge endpoints {list(missing)[:5]}"
