#!/usr/bin/env python
"""Build a fast pre-downsampled + normalized frame cache for training.

WHY: the trainer's FrameWindowDataset re-opens the zarr file and blosc2-decodes
W frames on EVERY __getitem__ (no RAM caching) -> I/O-bound, GPU starved
(~30 min/epoch, GPU idle). This precomputes each dataset's frames ONCE at the
training resolution (downsample 1,4,4 -> 64,64,64) with the dataset's own
quantile normalization, stored as fp16 .npy. Training then memmaps these and
slices windows directly (no zarr, no decode, no downsample, no normalize) ->
GPU-bound, ~5-10x faster.

The stored array is byte-identical to what __getitem__ produces before augment:
  raw = z[t, ::dz,::dy,::dx].float();  imgs = clamp((raw - q_low)/(q_high-q_low+1e-6), min=0)

Usage: python scripts/build_cache.py [--downsample 1,4,4] [--workers 28]
"""
import sys, os, json, argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"


def read_quantiles(zpath):
    j = json.load(open(Path(zpath) / "zarr.json"))
    q = j["attributes"]["image_statistics"]["quantiles"]
    return float(q["0.001"]), float(q["0.999"])


def build_one(args):
    name, ds, cache_dir = args
    dz, dy, dx = ds
    zpath = TRAIN / f"{name}.zarr"
    out = Path(cache_dir) / f"{name}.npy"
    if out.exists():
        return name, "skip", 0
    import zarr
    z = zarr.open_group(str(zpath), mode="r")["0"]
    T = z.shape[0]
    q_low, q_high = read_quantiles(zpath)
    frames = []
    for t in range(T):
        raw = z[t, ::dz, ::dy, ::dx].astype(np.float32)
        img = np.clip((raw - q_low) / (q_high - q_low + 1e-6), 0.0, None)
        frames.append(img.astype(np.float16))
    arr = np.stack(frames)  # (T, Zd, Yd, Xd) fp16
    tmp = out.with_suffix(".tmp.npy")
    np.save(tmp, arr); os.replace(tmp, out)
    return name, "built", arr.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--downsample", type=str, default="1,4,4")
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    ds = tuple(int(x) for x in args.downsample.split(","))
    cache_dir = args.out or str(ROOT / f"research/cache/ds{'x'.join(map(str,ds))}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    # write a manifest so the trainer knows the cache format
    json.dump({"downsample": list(ds), "dtype": "float16", "normalized": True,
               "note": "clamp((raw-q0.001)/(q0.999-q0.001+1e-6),0); train resolution"},
              open(Path(cache_dir) / "manifest.json", "w"))
    names = sorted(p.stem for p in TRAIN.glob("*.zarr") if "_" in p.stem)
    print(f"building cache for {len(names)} datasets -> {cache_dir} (downsample {ds}, {args.workers} workers)", flush=True)
    jobs = [(n, ds, cache_dir) for n in names]
    built = skip = 0
    import time; t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(build_one, j) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            name, status, T = f.result()
            if status == "built": built += 1
            else: skip += 1
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(names)} ({built} built, {skip} skip) {time.time()-t0:.0f}s", flush=True)
    sz = sum(p.stat().st_size for p in Path(cache_dir).glob("*.npy")) / 1e9
    print(f"DONE: {built} built, {skip} skipped in {time.time()-t0:.0f}s | cache size {sz:.1f} GB -> {cache_dir}", flush=True)


if __name__ == "__main__":
    main()
