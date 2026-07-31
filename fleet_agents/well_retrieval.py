"""well_retrieval — spatial top-k nearest-WELL search (the "see top-k walls, not one" idea). For a query well's
trajectory (X,Y), return the k train wells whose trajectories pass closest — the neighbours that carry the local
formation structure. Powers (a) the structural-surface reconstruction (use only nearby wells → denser, more
accurate than global kNN) and (b) the Gemini training flow (each example = target well + its k neighbours as
context). Fast: KDTree over all train (X,Y) points tagged by well; a query's neighbours = wells owning its
nearest points. Reusable via build(train_dir) + query(hw, k)."""
from __future__ import annotations
import os, glob, numpy as np, pandas as pd
from scipy.spatial import cKDTree

class WellIndex:
    def __init__(self, train_dir, sub_per_well=200):
        self.wells = []; pts = []; owner = []
        for hp in sorted(glob.glob(os.path.join(train_dir, "*__horizontal_well.csv"))):
            w = os.path.basename(hp).split("__")[0]; hw = pd.read_csv(hp)
            if "TVT" not in hw.columns: continue
            i = len(self.wells); self.wells.append(w)
            X = hw.X.to_numpy(float); Y = hw.Y.to_numpy(float)
            idx = np.linspace(0, len(X)-1, min(sub_per_well, len(X))).astype(int)
            pts.append(np.c_[X[idx], Y[idx]]); owner.append(np.full(len(idx), i))
        self.P = np.vstack(pts); self.owner = np.concatenate(owner); self.tree = cKDTree(self.P)
    def query(self, hw, k=8, exclude=None):
        X = hw.X.to_numpy(float); Y = hw.Y.to_numpy(float)
        idx = np.linspace(0, len(X)-1, min(150, len(X))).astype(int)
        dist, ii = self.tree.query(np.c_[X[idx], Y[idx]], k=30)
        # min distance from query trajectory to each candidate well
        best = {}
        for row_d, row_i in zip(dist, ii):
            for dd, jj in zip(row_d, row_i):
                w = self.wells[self.owner[jj]]
                if exclude is not None and w == exclude: continue
                if w not in best or dd < best[w]: best[w] = dd
        ranked = sorted(best.items(), key=lambda kv: kv[1])[:k]
        return [w for w, _ in ranked], [d for _, d in ranked]

def build(train_dir): return WellIndex(train_dir)
