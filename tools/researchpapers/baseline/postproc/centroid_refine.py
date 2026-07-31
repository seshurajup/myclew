"""Standalone post-proc module 2: intensity-centroid node refinement.

Snaps EVERY node of a prediction GEFF to its local intensity center-of-mass in the
full-res frame (topology preserved; edges unchanged), capped at +/-2um so a node
cannot be dragged onto a neighbouring cell. Writes refined GEFFs.

The pipeline function below is COPIED VERBATIM from the tamerlanomralinov kernel
  scratchpad/kernels/tamerlanomralinov_biohub-learned-centroid-refine-ilp-div-public/*.txt
    - refine_all_nodes  (kernel lines 1847-1896)
    - VOXEL_SCALE_UM    (kernel line 1095)
Params are the kernel defaults (kernel lines 236-239):
  NODE_REFINE_WIN_Z=1, NODE_REFINE_WIN_YX=4, NODE_REFINE_MAX_SHIFT_UM=2.0  (the +/-2um cap).
The center-of-mass math (20th-percentile baseline, intensity-weighted mean, um-cap) is
byte-identical to the kernel; only read_test_frame (image access) is adapted to load the
image via tracking_cellmot.io.open_dataset with downsample=(1,4,4).

CLI:  python centroid_refine.py --in-dir <pred geff dir> --out-dir <dir> [--image-dir <train zarr dir>]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _geff_glue as glue  # noqa: E402

_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "research/official_repo/src"))

DEFAULT_IMAGE_DIR = _REPO / "input/biohub-cell-tracking-during-development/train"

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)

# tamer kernel defaults (lines 236-239)
OUTPUT_REFINE_NODES = os.environ.get("BIOHUB_OUTPUT_REFINE_NODES", "1") != "0"
NODE_REFINE_WIN_Z = int(os.environ.get("BIOHUB_NODE_REFINE_WIN_Z", "1"))
NODE_REFINE_WIN_YX = int(os.environ.get("BIOHUB_NODE_REFINE_WIN_YX", "4"))
NODE_REFINE_MAX_SHIFT_UM = float(os.environ.get("BIOHUB_NODE_REFINE_MAX_SHIFT_UM", "2.0"))


# ---------------------------------------------------------------------------
# I/O glue: image frame access (replaces the kernel's read_test_frame).
# open_dataset returns a normalized (T,Z,Y,X) tensor whose spatial index space
# matches the geff node coords; we serve numpy frames per timepoint.
# ---------------------------------------------------------------------------
_CURRENT_IMAGE = None  # torch.Tensor | np.ndarray of shape (T,Z,Y,X)


def _load_image(dataset_stem: str, image_dir: Path):
    from tracking_cellmot.io import open_dataset
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"
    ds = open_dataset(image_dir / dataset_stem, load_image=True, normalize=True,
                      device=device, downsample=(1, 4, 4))
    return ds.image


def read_test_frame(dataset: str, t: int, frame_cache: dict[int, np.ndarray]) -> np.ndarray:
    if t in frame_cache:
        return frame_cache[t]
    frame = _CURRENT_IMAGE[t]
    try:
        frame = frame.detach().cpu().numpy()
    except AttributeError:
        frame = np.asarray(frame)
    frame = np.asarray(frame)
    frame_cache[t] = frame
    return frame


# ===========================================================================
# VERBATIM from tamer kernel (refine_all_nodes, lines 1847-1896)
# ===========================================================================
def refine_all_nodes(nodes_by_id, dataset, stats):
    """Feature 2: snap every node to its local intensity centroid in the full-res
    frame, undoing the xy-downsample-by-4 quantization of the learned detections.
    Coordinate-only (topology preserved); each shift is capped so a node cannot be
    dragged onto a neighbouring cell. Runs before line-fit smoothing."""
    if not OUTPUT_REFINE_NODES or dataset is None or not nodes_by_id:
        return nodes_by_id
    frame_cache: dict[int, np.ndarray] = {}
    refined = rejected = failed = 0
    for node in nodes_by_id.values():
        t = int(node["t"])
        try:
            frame = read_test_frame(dataset, t, frame_cache)
        except Exception:
            failed += 1
            continue
        z = int(round(float(node["z"])))
        y = int(round(float(node["y"])))
        x = int(round(float(node["x"])))
        z0 = max(0, z - NODE_REFINE_WIN_Z); z1 = min(frame.shape[0], z + NODE_REFINE_WIN_Z + 1)
        y0 = max(0, y - NODE_REFINE_WIN_YX); y1 = min(frame.shape[1], y + NODE_REFINE_WIN_YX + 1)
        x0 = max(0, x - NODE_REFINE_WIN_YX); x1 = min(frame.shape[2], x + NODE_REFINE_WIN_YX + 1)
        patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
        if patch.size == 0:
            failed += 1
            continue
        baseline = float(np.percentile(patch, 20.0))
        weights = np.maximum(patch - baseline, 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            failed += 1
            continue
        zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
        yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
        xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
        rz = float((weights * zz).sum() / total)
        ry = float((weights * yy).sum() / total)
        rx = float((weights * xx).sum() / total)
        dz = (rz - float(node["z"])) * VOXEL_SCALE_UM[0]
        dy = (ry - float(node["y"])) * VOXEL_SCALE_UM[1]
        dx = (rx - float(node["x"])) * VOXEL_SCALE_UM[2]
        if (dz * dz + dy * dy + dx * dx) ** 0.5 > NODE_REFINE_MAX_SHIFT_UM:
            rejected += 1
            continue
        node["z"] = rz; node["y"] = ry; node["x"] = rx
        refined += 1
    stats["nodes_refined"] = refined
    stats["nodes_refine_rejected"] = rejected
    stats["nodes_refine_failed"] = failed
    return nodes_by_id


# ===========================================================================
# I/O glue driver
# ===========================================================================
def process_geff(in_geff: Path, out_geff: Path, dataset_stem: str, image_dir: Path):
    from collections import Counter
    global _CURRENT_IMAGE
    nodes_by_id, edges, _ = glue.read_pred_geff(in_geff, filter_solution=True)
    n_nodes, n_edges = len(nodes_by_id), len(edges)
    stats: dict[str, int] = Counter()

    _CURRENT_IMAGE = _load_image(dataset_stem, image_dir)
    nodes_by_id = refine_all_nodes(nodes_by_id, dataset_stem, stats)
    _CURRENT_IMAGE = None

    glue.write_geff(out_geff, nodes_by_id, edges)
    print(f"  {in_geff.name}: nodes={n_nodes} edges {n_edges}->{n_edges} (topology preserved)"
          f"  refined={stats.get('nodes_refined',0)} rejected(cap)={stats.get('nodes_refine_rejected',0)}"
          f" failed={stats.get('nodes_refine_failed',0)}")
    return n_edges, n_edges


def main(argv=None):
    ap = argparse.ArgumentParser(description="intensity-centroid node refinement (cap +/-2um) on prediction geffs")
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR,
                    help="dir with <ds>.zarr movies (default: competition train dir)")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    geffs = glue.list_pred_geffs(args.in_dir)
    print(f"centroid_refine: {len(geffs)} geff(s) from {args.in_dir}  (images: {args.image_dir})")
    for stem, path in geffs:
        process_geff(path, args.out_dir / f"{stem}.geff", stem, args.image_dir)


if __name__ == "__main__":
    main()
