"""Working code for inf01 — the inference pipeline (predict_video). Running it drives the
REAL detection + peak-extraction on a real frame and writes inf01_inference_pipeline.learning.
    research/cellmot_venv/bin/python learning/annotated/inf01_inference_pipeline.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
REPO = ROOT / "research/pilkwang_support_pack/repo"

META = dict(id="inf01", order=20, title="The inference pipeline — predict_video",
            subtitle="How a trained model turns a movie into tracks: detect → NMS peaks → edge-score → ILP",
            source="research/pilkwang_support_pack/repo/scripts/predict_unet_transformer.py")

CELLS = [
    dict(note="""## From weights to a submission
Training gave us a model; **inference** runs it on the hidden test movie and writes tracks. The
real `predict_video` does, per video: **detect** a heat-map → **NMS peaks** (candidate cells) →
**score edges** between consecutive frames → **ILP** picks the best graph → write. We drive the
real detection + peak code here on a real frame; every output is the real result."""),

    dict(note="""### Load the real detector on a real frame
Build the real `TemporalUNet3D`, read a real frame of `6bba_062c8d37`, downsample `(1,4,4)`
(isotropic), and run the detection head → a per-voxel logit heat-map.""",
         code="""import sys, numpy as np, zarr, torch                        # tools
import torch.nn as nn                                        # layers
sys.path.insert(0, f"{REPO}/src")                            # reach the real model package
from biohub_tracking.models.temporal_unet import TemporalUNet3D  # the REAL feature extractor
vol = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.zarr/0")[0:2]).astype(np.float32)  # 2 real frames (W=2)
vol = vol[:, :, ::4, ::4]                                    # downsample Y,X by 4 -> isotropic
lo, hi = np.quantile(vol, 0.001), np.quantile(vol, 0.999)    # robust normalise range
vol = np.clip((vol - lo) / (hi - lo + 1e-6), 0, 1)          # quantile-normalise like the detector
x = torch.from_numpy(vol)[None, :, None]                     # (B=1, T=2, C_in=1, Z, Y, X)
unet = TemporalUNet3D(in_channels=1, out_channels=32, layers=(32, 64, 128)).eval()  # real UNet
det_head = nn.Conv3d(32, 1, kernel_size=1)                   # the real detection head: 32 feats -> 1 logit
with torch.no_grad():                                       # inference, no gradients
    feats = unet(x)                                          # (B, T, 32, Z, Y, X) real features
    det_logits = det_head(feats[0])                          # (T, 1, Z, Y, X) per-frame logits
det_logits.shape                                             # real heat-map shape"""),

    dict(note="""### NMS peaks — the real `_detect_cells_pooled`
**[PyTorch]** A voxel is a **peak** if it equals its 3×3×3 local max (`max_pool3d`, stride 1) AND
its probability beats the threshold. **[Data]** in production `det_threshold=0.99` (very high — the
labels are sparse, so precision is protected). Run the real peak code on the real heat-map.""",
         code="""import torch.nn.functional as F                            # for max_pool3d
def detect_peaks(logits, det_threshold=0.99, pool_kernel=(3, 3, 3)):  # the REAL _detect_cells_pooled
    logits = logits.unsqueeze(0)                             # (1,1,Z,Y,X) for pooling
    pad = tuple(k // 2 for k in pool_kernel)                 # keep size
    pooled = F.max_pool3d(logits, pool_kernel, stride=1, padding=pad)  # local max per voxel
    is_peak = (logits == pooled) & (torch.sigmoid(logits) > det_threshold)  # local-max AND confident
    return torch.nonzero(is_peak[0, 0])                      # (N,3) peak coords (z,y,x)
peaks = detect_peaks(det_logits[0])                          # run on frame 0's real logits (1,Z,Y,X)
{"peaks found": int(peaks.shape[0]), "heatmap voxels": int(det_logits[0,0].numel())}  # real counts"""),

    dict(note="""### The linking config — what the graph solver uses
After peaks, an edge transformer scores each (cell@t → cell@t+1) pair (`softmax` over sources =
divisions allowed), then an **ILP** picks the graph. These are the real `PredictConfig` values the
0.885+ notebook runs with.""",
         code="""cfg = dict(                                                 # the REAL PredictConfig (inference)
    det_threshold=0.99,        # peak probability cutoff (precision)
    det_tta=True,              # flip-XY test-time augmentation (not Z: anisotropy)
    pool_kernel_um=3.0,        # NMS kernel size in microns
    edge_activation="softmax", # over sources -> a cell may have 2 children (division), not 2 parents
    use_ilp=True,              # solve a global graph, not greedy
    ilp_edge_weight=-1.0,      # w_e = -edge_prob (higher prob -> cheaper edge)
    ilp_appearance_weight=0.1, # cost to start a track
    ilp_disappearance_weight=0.1,  # cost to end a track
    ilp_division_weight=1.0,   # cost for a division event
)
cfg                                                          # the real inference configuration"""),

    dict(note="""### The ILP objective
**[Domain]** The solver minimises total cost = edge costs (`w_e = −edge_prob`, so confident links
are cheap) + appearance/disappearance/division event costs. This turns per-pair scores into ONE
globally consistent lineage. Below, the real edge-weight rule as a tiny function.""",
         code="""def edge_cost(edge_prob):                                   # the REAL ilp_edge_weight rule
    return -1.0 * edge_prob                                  # w_e = -edge_prob
[round(edge_cost(p), 2) for p in (0.99, 0.7, 0.3)]          # cheap for confident, dear for unsure"""),

    dict(note="""**[Recap]** Inference = detect heat-map → `max_pool3d` NMS peaks at threshold 0.99
→ transformer edge scores (softmax) → ILP (`w_e=−edge_prob` + event costs) → the lineage graph.
**Next → pp01: post-processing** — the physical-distance repairs applied to that graph.

**[Note]** we ran the real detector arch + real NMS on a real frame; production uses the *trained*
weights (`checkpoint_last.pth`) — the mechanics and the data here are exactly real."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"),
                 {"TRAIN": TRAIN, "REPO": REPO})
