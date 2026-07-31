"""Working code for me01 — the official metric. Runs the real _jaccard and a real 7um
match on real data.
"""
from pathlib import Path
import sys
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning"))
from lessonkit import build_lesson
REPO = ROOT / "research/pilkwang_support_pack/repo"
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="me01", order=20, title="The official metric",
            subtitle="Adjusted edge-Jaccard + 0.1·division, 7µm matching — run on real data",
            source="research/pilkwang_support_pack/repo/src/biohub_tracking/metrics.py")

CELLS = [
    dict(note="""## What we're scored on
`score = adjusted_edge_Jaccard + 0.1 · division_Jaccard`. An **edge** = a cell linked across two
frames; a prediction is credited only if BOTH endpoints match a real cell **within 7 µm**.
Jaccard = `TP / (TP + FP + FN)`. Below, the real `_jaccard` and the real 7 µm match run on real
`6bba_062c8d37` data."""),

    dict(note="""### The real Jaccard, on the real GT edge count
**[PyTorch]** `_jaccard` is `tp/(tp+fp+fn)`. **[Domain]** read the real number of GT edges in an
embryo, then see what score a detector that recovers 90% of them (with 5% spurious) would get.""",
         code="""import numpy as np, zarr                                             # tools
sys.path.insert(0, f"{REPO}/src")                                        # reach the real metric code
from biohub_tracking.metrics import _jaccard                             # the REAL jaccard function
gt_edges = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.geff/edges/ids")[:])  # real GT edges
n_gt = len(gt_edges)                                                     # real edge count
tp = int(0.90 * n_gt); fn = n_gt - tp; fp = int(0.05 * n_gt)            # a 90%-recall / 5%-FP scenario
{"real GT edges": n_gt, "jaccard": round(_jaccard(tp, fp, fn), 3)}       # the real metric value""",
         image="learning/assets/metric_match.png\nThe 7µm matching rule: a predicted node counts only if within 7µm of a real GT node; an edge is a true-positive only if BOTH its endpoints match."),

    dict(note="""### The 7 µm match — on real coordinates
**[Domain]** why 7 µm? A nucleus is ~5–10 µm, so a prediction within 7 µm is 'the same cell'. Take
a real GT cell, perturb it by a few µm (a plausible prediction), and check the physical distance
against the 7 µm gate — in real physical units (voxel z=1.625, y=x=0.40625 µm).""",
         code="""zc = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.geff/nodes/props/z/values")[:])  # real z
yc = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.geff/nodes/props/y/values")[:])  # real y
xc = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.geff/nodes/props/x/values")[:])  # real x
voxel = np.array([1.625, 0.40625, 0.40625])                            # physical voxel size (µm)
gt0 = np.array([zc[0], yc[0], xc[0]])                                   # a real GT cell (voxels)
pred = gt0 + np.array([2, 6, 6])                                        # a plausible prediction, a few voxels off
dist_um = np.sqrt((((pred - gt0) * voxel) ** 2).sum())                  # physical distance in µm
{"distance_µm": round(float(dist_um), 2), "matches (≤7µm)": bool(dist_um <= 7.0)}  # real match decision"""),

    dict(note="""**[Recap]** score = adjusted edge-Jaccard + 0.1·division-Jaccard, edges matched at
7 µm. The 'adjusted' part only counts predicted edges touching GT-tracked nodes (sparse-label
correction) — the reason our local CV can mislead (see rs02). **Next → pp01: post-processing.**"""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"), {"ROOT": ROOT, "REPO": REPO, "TRAIN": TRAIN, "sys": sys})
