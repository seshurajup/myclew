"""I/O for Zarr image movies and GEFF label graphs.

Image arrays: Zarr v3, group path `0/`, shape (T,Z,Y,X), one timepoint per chunk at
`0/c/{t}/0/0/0` (blosc2-compressed). We read a single timepoint at a time to stay RAM-safe.

Labels: `<id>.geff` (a zarr group) with nodes/ids, nodes/props/<axis>/values, edges/ids.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd


def list_datasets(split_dir: Path) -> List[str]:
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        return []
    return sorted(p.name[:-5] for p in split_dir.iterdir() if p.name.endswith(".zarr"))


def embryo_id(dataset_id: str) -> str:
    """Embryo prefix used for embryo-disjoint CV (hidden test is embryo-disjoint)."""
    return dataset_id.split("_")[0]


def read_array_meta(zarr_path: Path) -> Tuple[Path, Tuple[int, ...], np.dtype]:
    """Return (array_dir=`<zarr>/0`, shape (T,Z,Y,X), dtype)."""
    zarr_path = Path(zarr_path)
    meta = json.load(open(zarr_path / "0" / "zarr.json"))
    shape = tuple(meta["shape"])
    dtype = np.dtype(meta["data_type"])
    return zarr_path / "0", shape, dtype


def load_volume(array_dir: Path, shape: Tuple[int, ...], dtype: np.dtype, t: int) -> np.ndarray:
    """Load a single timepoint (Z,Y,X). Fast path = raw chunk + blosc2; zarr fallback."""
    chunk_path = Path(array_dir) / "c" / str(t) / "0" / "0" / "0"
    if chunk_path.exists():
        import blosc2
        raw = open(chunk_path, "rb").read()
        buf = blosc2.decompress(raw)
        return np.frombuffer(buf, dtype=dtype).reshape(shape[1:])
    # fallback: let zarr resolve sharding/compression
    import zarr
    arr = zarr.open_array(str(Path(array_dir)), mode="r")
    return np.asarray(arr[t])


def read_geff(geff_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Read a GEFF label graph -> (nodes_df[node_id,t,z,y,x], edges_df[source_id,target_id])."""
    import zarr
    g = zarr.open_group(str(geff_path), mode="r")
    nodes = pd.DataFrame({
        "node_id": np.asarray(g["nodes/ids"][:], dtype=np.uint64),
        "t": np.asarray(g["nodes/props/t/values"][:], dtype=np.int64),
        "z": np.asarray(g["nodes/props/z/values"][:], dtype=np.int64),
        "y": np.asarray(g["nodes/props/y/values"][:], dtype=np.int64),
        "x": np.asarray(g["nodes/props/x/values"][:], dtype=np.int64),
    })
    edges_raw = np.asarray(g["edges/ids"][:], dtype=np.uint64).reshape(-1, 2)
    edges = pd.DataFrame(edges_raw, columns=["source_id", "target_id"])
    return nodes, edges


def geff_estimated_nodes(geff_path: Path) -> int | None:
    try:
        meta = json.load(open(Path(geff_path) / "zarr.json"))
        return meta["attributes"]["geff"]["extra"].get("estimated_number_of_nodes")
    except Exception:
        return None
