"""Where do pilkwang and canqiang DETECTIONS agree/disagree vs GT?
Determines whether a rule-based fusion can help:
  - consensus (both agree) precision
  - complementary recall (GT cells found by exactly one)
  - false positives unique to each (what a union would add)
Matching in physical µm (voxel scale z=1.625, y=x=0.40625), gate 5µm.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
import zarr

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
EW = ROOT / "learning/ensemble_work"
VOX = np.array([1.625, 0.40625, 0.40625])
GATE = 5.0  # µm


def gt_nodes(stem):
    b = EW  # read GT t,z,y,x from the geff node props
    g = TRAIN / f"{stem}.geff/nodes/props"
    t = np.asarray(zarr.open(str(g / "t/values"), mode="r")[:])
    z = np.asarray(zarr.open(str(g / "z/values"), mode="r")[:])
    y = np.asarray(zarr.open(str(g / "y/values"), mode="r")[:])
    x = np.asarray(zarr.open(str(g / "x/values"), mode="r")[:])
    return pd.DataFrame({"t": t, "z": z, "y": y, "x": x})


def match(a, b, gate=GATE):
    """fraction of `a` rows with a `b` node within gate µm, per frame. returns (n_a, n_matched)."""
    if len(a) == 0:
        return 0, 0
    n_m = 0
    for t, ga in a.groupby("t"):
        gb = b[b.t == t]
        if len(gb) == 0:
            continue
        tree = cKDTree(gb[["z", "y", "x"]].values * VOX)
        d, _ = tree.query(ga[["z", "y", "x"]].values * VOX, k=1)
        n_m += int((d <= gate).sum())
    return len(a), n_m


rows = []
stems = sorted(p.stem for p in (EW / "pilkwang_nodes").glob("*.csv"))
for stem in stems:
    P = pd.read_csv(EW / "pilkwang_nodes" / f"{stem}.csv")
    C = pd.read_csv(EW / "canqiang_nodes" / f"{stem}.csv")
    G = gt_nodes(stem)
    nG = len(G)
    # recall of each vs GT
    _, pP = match(G, P); _, pC = match(G, C)          # GT cells found by P / by C
    # union recall
    PC = pd.concat([P[["t", "z", "y", "x"]], C[["t", "z", "y", "x"]]])
    _, pU = match(G, PC)
    # agreement: pilkwang dets that canqiang also has
    nP, agree = match(P[["t", "z", "y", "x"]], C)
    rows.append({
        "stem": stem, "emb": stem[:4], "nGT": nG,
        "nP": len(P), "nC": len(C),
        "recP": round(pP / nG, 3), "recC": round(pC / nG, 3), "recUnion": round(pU / nG, 3),
        "P_agree_C_%": round(100 * agree / max(len(P), 1), 1),
    })

df = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print(df.to_string(index=False))
print("\n=== summary ===")
print(f"mean GT recall  pilkwang={df.recP.mean():.3f}  canqiang={df.recC.mean():.3f}  UNION={df.recUnion.mean():.3f}")
print(f"union recall gain over best single: {df.recUnion.mean() - max(df.recP.mean(), df.recC.mean()):+.3f}")
print(f"pilkwang dets also in canqiang (agreement): {df['P_agree_C_%'].mean():.1f}%")
print(f"pilkwang det count {df.nP.mean():.0f}/frame-set vs canqiang {df.nC.mean():.0f}  (canqiang denser => more FP among unlabeled)")
