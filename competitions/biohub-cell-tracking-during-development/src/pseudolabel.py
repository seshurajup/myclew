"""Pseudo-label generation: run Cellpose-SAM-3D (teacher) on a representative subset of train
volumes → save per-(dataset,frame) nucleus centroids. These pseudo-labels train the fast distilled
student (Kaggle-feasible). Teacher is offline-only (32s/frame); we sample a subset, not all 19,900.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import io

TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
OUT = ROOT / "results/pseudolabels"
OUT.mkdir(parents=True, exist_ok=True)


def gpu_temp() -> int:
    """Read GPU temp (no sudo needed)."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip().split("\n")[0]
        return int(out)
    except Exception:
        return -1


def thermal_guard(hot=72, cool=64, max_wait=120):
    """Software thermal throttle (no sudo): if GPU is hot, sleep until it cools. Caps temp by
    duty-cycling — the only thermal control available without root for -pl/-lgc."""
    import time
    t = gpu_temp()
    if t < hot:
        return
    waited = 0
    while gpu_temp() >= cool and waited < max_wait:
        time.sleep(5); waited += 5
    print(f"    [thermal] paused {waited}s (was {t}C, now {gpu_temp()}C)", flush=True)


def centroids_from_masks(masks: np.ndarray) -> np.ndarray:
    """Label volume -> (N,3) int centroids (z,y,x). Vectorized via center_of_mass."""
    from scipy import ndimage
    n = int(masks.max())
    if n == 0:
        return np.zeros((0, 3), np.float32)
    cs = ndimage.center_of_mass(np.ones_like(masks), masks, range(1, n + 1))
    return np.array(cs, dtype=np.float32)


def run(datasets, frames_per: int = 10, workers_note: str = ""):
    from cellpose import models
    import torch
    model = models.CellposeModel(gpu=torch.cuda.is_available())
    print(f"Cellpose teacher | GPU={torch.cuda.is_available()} | {len(datasets)} datasets x {frames_per} frames")
    rows = []
    for di, ds in enumerate(datasets):
        out_f = OUT / f"{ds}.parquet"
        if out_f.exists():
            print(f"  [{di}] {ds}: cached"); continue
        ad, shape, dt = io.read_array_meta(TRAIN / f"{ds}.zarr")
        T = shape[0]
        ts = np.unique(np.linspace(0, T - 1, min(frames_per, T)).astype(int))
        ds_rows = []
        t0 = time.time()
        for t in ts:
            thermal_guard()  # software throttle (no sudo) — keep the 5090 under ~72C
            vol = io.load_volume(ad, shape, dt, int(t)).astype(np.float32)
            masks, _, _ = model.eval(vol, do_3D=True, z_axis=0, anisotropy=4.0,
                                     normalize=True, batch_size=64)
            cs = centroids_from_masks(masks)
            for z, y, x in cs:
                ds_rows.append({"dataset": ds, "t": int(t), "z": float(z), "y": float(y), "x": float(x)})
            del vol, masks
        pd.DataFrame(ds_rows).to_parquet(out_f)
        rows.extend(ds_rows)
        print(f"  [{di}] {ds}: {len(ds_rows)} pseudo-cells over {len(ts)} frames ({time.time()-t0:.0f}s)")
    print(f"\nTotal pseudo-cells: {len(rows)} -> {OUT}")


def diverse_subset(n_per_embryo: int = 8):
    """Pick diverse datasets per embryo (deterministic spread)."""
    by = {"44b6": [], "6bba": []}
    for p in sorted(TRAIN.glob("*.geff")):
        by.setdefault(p.stem.split("_")[0], []).append(p.stem)
    out = []
    for emb, lst in by.items():
        idx = np.linspace(0, len(lst) - 1, min(n_per_embryo, len(lst))).astype(int)
        out += [lst[i] for i in idx]
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-embryo", type=int, default=8)
    ap.add_argument("--frames-per", type=int, default=10)
    args = ap.parse_args()
    ds = diverse_subset(args.n_per_embryo)
    print("subset:", ds)
    run(ds, frames_per=args.frames_per)
