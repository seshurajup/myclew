"""End-to-end per-dataset pipeline: detect -> link -> divisions -> gap-close -> prune -> rows."""
from __future__ import annotations
from pathlib import Path
from typing import List
import numpy as np

from .config import Config
from . import io, detect, link, submission


def process_dataset(zarr_path: Path, dataset: str, cfg: Config, max_frames: int | None = None) -> List[dict]:
    array_dir, shape, dtype = io.read_array_meta(zarr_path)
    T = shape[0] if max_frames is None else min(shape[0], max_frames)

    frames = []
    for t in range(T):
        vol = io.load_volume(array_dir, shape, dtype, t)
        coords, _scores = detect.detect_cells(vol, cfg)
        frames.append(coords)
        del vol

    nodes, edges = link.track_dataset(frames, cfg)
    if cfg.PRUNE_ISOLATED_NODES:
        nodes, edges = link.prune_isolated(nodes, edges)
    return submission.dataset_rows(dataset, nodes, edges)


def run(test_dir: Path, out_path: str, cfg: Config, max_frames: int | None = None,
        datasets: list[str] | None = None, verbose: bool = True):
    test_dir = Path(test_dir)
    names = datasets or io.list_datasets(test_dir)
    all_rows: List[dict] = []
    for ds in names:
        rows = process_dataset(test_dir / f"{ds}.zarr", ds, cfg, max_frames=max_frames)
        all_rows.extend(rows)
        if verbose:
            nn = sum(1 for r in rows if r["row_type"] == "node")
            ne = sum(1 for r in rows if r["row_type"] == "edge")
            print(f"  {ds}: {nn} nodes, {ne} edges")
    df = submission.finalize(all_rows, out_path)
    if verbose:
        print(f"Wrote {len(df)} rows -> {out_path}")
    return df
