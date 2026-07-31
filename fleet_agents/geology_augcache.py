"""geology_augcache — precompute & cache the PS-augmented aligner features ONCE in a fast binary format so
experiments stop re-parsing 1,546 CSVs + re-running polyfit every run. Stores, per well, the raw survey arrays
(md,gr,z,tvt,tvti) as float32 in a single .npz, plus the real-PS index. feats_for_split() then rebuilds any
split cheaply in-memory. Rebuild only when the raw data changes.

  build: python -c "import fleet_agents.geology_augcache as c; c.build()"
  load : cache = geology_augcache.load()   # dict well -> arrays  (parse-free, ~instant)
"""
from __future__ import annotations
import numpy as np, pandas as pd, glob, os
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent / "results" / "aug_cache.npz"


def build(data_dir="input/train", out=CACHE):
    hs = sorted(glob.glob(os.path.join(data_dir, "*__horizontal_well.csv")))
    store = {}
    for hp in hs:
        w = os.path.basename(hp).split("__")[0]
        try:
            hw = pd.read_csv(hp)
        except Exception:
            continue
        if "TVT" not in hw.columns:
            continue
        tvti = hw["TVT_input"].to_numpy(float)
        if not np.isfinite(tvti).any():
            continue
        n = len(hw)
        if n < 80:
            continue
        real_ps = int(np.where(np.isfinite(tvti))[0][-1])
        arr = np.stack([hw["MD"].to_numpy(float), hw["GR"].to_numpy(float), hw["Z"].to_numpy(float),
                        hw["TVT"].to_numpy(float), tvti], 0).astype(np.float32)   # (5, n)
        store[w] = arr
        store[w + "|ps"] = np.int32(real_ps)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **store)
    print(f"cached {sum(1 for k in store if '|' not in k)} wells -> {out} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def load(path=CACHE):
    z = np.load(path)
    wells = [k for k in z.files if "|" not in k]
    out = {}
    for w in wells:
        md, gr, zc, tvt, tvti = z[w]
        out[w] = dict(md=md, gr=gr, z=zc, tvt=tvt, tvti=tvti, n=len(md), real_ps=int(z[w + "|ps"]))
    return out


if __name__ == "__main__":
    build()
