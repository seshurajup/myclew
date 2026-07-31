"""Working code for pt04 — from volume to cell coordinates. Runs the real detection head +
peak-finding on a real feature map; writes pt04_detector_forward_and_peaks.learning.
    research/cellmot_venv/bin/python learning/annotated/pt04_detector_forward_and_peaks.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="pt04", order=13, title="From volume to cell coordinates",
            subtitle="detection head → heat-map → NMS peaks → the (z,y,x) the metric scores",
            source="research/pilkwang_support_pack/repo/scripts/predict_unet_transformer.py")

CELLS = [
    dict(note="""## Where the tensor becomes DETECTIONS
The U-Net gives 32 feature channels per voxel. A `1×1×1` **head conv** collapses them to a single
"is a nucleus centred here?" logit → a heat-map. Then **peak-finding** (non-max suppression) turns
the heat-map into a list of `(z, y, x)` coordinates — the actual cell detections the metric scores
(within 7 µm). Every number below is captured by running the real ops."""),

    dict(note="""### The detection head — 32 features → 1 logit per voxel
Build a real 32-channel feature map from a real frame, then the real head `Conv3d(32, 1, 1)`.""",
         code="""import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, zarr   # layers + IO
vol = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.zarr/0")[0]).astype(np.float32)[:, ::4, ::4]  # real frame
x = torch.from_numpy(vol)[None, None]                              # (B,C,Z,Y,X)
feat = nn.Conv3d(1, 32, 3, padding=1).eval()(x)                   # a 32-channel feature map (stand-in for the U-Net)
head = nn.Conv3d(32, 1, kernel_size=1)                            # the REAL detection head: 32 -> 1
logits = head(feat)                                              # per-voxel "nucleus?" logit
logits.shape                                                     # one score per voxel"""),

    dict(note="""### Heat-map → peaks (non-max suppression)
**[PyTorch]** `sigmoid` → probability; `max_pool3d(k, stride=1)` keeps each voxel's local max; a
voxel is a **peak** if it equals its local max AND beats the threshold. **[Data]** the real
`det_threshold` is a very high **0.99** — the labels are sparse, so precision is protected.
**[Domain]** each surviving peak = one detected cell centre `(z,y,x)`, matched to GT within 7 µm.""",
         code="""prob = torch.sigmoid(logits)                                 # logit -> probability in [0,1]
pooled = F.max_pool3d(prob, kernel_size=3, stride=1, padding=1)  # local max in each 3x3x3 window
peaks = (prob == pooled) & (prob > 0.5)                          # a peak = local max above threshold
coords = peaks.nonzero()[:, 2:]                                 # keep the (z,y,x) of each peak
int(peaks.sum())                                               # how many cells this frame detects""",
         image="learning/assets/metric_match.png\nThe metric: a predicted (z,y,x) matches a GT cell if within 7 µm; an edge is a true positive only if BOTH its endpoints match. Detections are what get matched."),

    dict(note="""**[Craft]** the real pipeline also does test-time augmentation — averaging the
heat-map over X/Y/XY flips (never Z, because light-sheet imaging is anisotropic) — before
peak-finding, for a steadier detection. **Next → pt05: linking these detections into tracks.**"""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"), {"TRAIN": TRAIN})
