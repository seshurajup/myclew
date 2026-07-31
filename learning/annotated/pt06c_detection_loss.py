"""Working code for pt06c — the DETECTION loss (the recall lever), on REAL data.
Runs the real compute_detection_loss against a target built from real cell coords.
    research/cellmot_venv/bin/python learning/annotated/pt06c_detection_loss.py
"""
from pathlib import Path
import sys
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning"))
from lessonkit import build_lesson

REPO = ROOT / "research/pilkwang_support_pack/repo"
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="pt06c", order=15, title="The detection loss — the recall lever",
            subtitle="compute_detection_loss: class-balanced BCE tuned for RECALL, on real cells",
            source="research/pilkwang_support_pack/repo/scripts/train_unet_transformer.py")

CELLS = [
    dict(note="""## The other loss — teaching the detector to FIND cells
pt06 was the *edge* loss. But the detector head is trained separately with a **detection loss**:
per-voxel BCE where a GT cell's voxel is positive. **[Why it's THE lever]** our score
≈ node_recall² · edge_precision, so recovering more cells matters most — and the detection loss is
tuned toward **recall** via a low negative weight. Everything below runs on real cells of
`6bba_062c8d37`."""),

    dict(note="""### Real detector logits on a real frame
Load a real frame, downsample `(1,4,4)`, run a detection head (Conv3d→1) → per-voxel logits.""",
         code="""import sys, numpy as np, zarr, torch, torch.nn as nn                 # real imports
sys.path.insert(0, f"{REPO}/src"); sys.path.insert(0, f"{REPO}/scripts")     # the real repo
vol = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.zarr/0")[0]).astype(np.float32)  # real frame (Z,Y,X)
vol = vol[:, ::4, ::4]                                                        # downsample Y,X -> detector grid
x = torch.from_numpy(vol)[None, None]                                        # (B,C,Z,Y,X)
det_head = nn.Conv3d(1, 1, kernel_size=1)                                     # the 1x1 detection head
det_logits = det_head(x)                                                      # per-voxel "is a cell here?" logits
det_logits.shape                                                              # (B,1,Z,Y,X) — a score at every voxel"""),

    dict(note="""### Build the target from REAL cell coordinates
Read the real cells at frame 0 from the `.geff`, place them (in the downsampled grid) as the
positive voxels the detector must fire on.""",
         code="""nt = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.geff/nodes/props/t/values")[:])  # node frames
zc = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.geff/nodes/props/z/values")[:])  # node z
yc = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.geff/nodes/props/y/values")[:])  # node y
xc = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.geff/nodes/props/x/values")[:])  # node x
m = nt == 0                                                                   # cells at frame 0
coords = np.stack([zc[m], yc[m] / 4, xc[m] / 4], axis=1)                      # to the downsampled grid
coords = torch.tensor(coords, dtype=torch.float32)[None]                      # (B, n_nodes, 3)
mask = torch.ones(1, coords.shape[1], dtype=torch.bool)                       # all real (non-padded)
int(mask.sum())                                                              # number of real GT cells"""),

    dict(note="""### Run the REAL compute_detection_loss
**[PyTorch]** GT voxels get weight `1/n_pos`; all others get `neg_weight/n_neg`. **[The lever]**
`neg_weight=0.1` (not 1.0) makes missing a cell cost far more than a false positive → the detector
is pushed toward **recall**. Run it on the real logits + real target.""",
         code="""from train_unet_transformer import compute_detection_loss    # the REAL detection loss
loss = compute_detection_loss(det_logits, coords, mask, neg_weight=0.1)  # recall-tilted BCE
float(loss)                                                             # the real loss value"""),

    dict(note="""**[Recap]** The detector learns from this recall-tilted BCE (`neg_weight=0.1`),
the edge model from `compute_loss` (pt06). Together they train the whole tracker. **Next → pt07:
the Dataset/DataLoader** that feeds these real frame windows in."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"),
                 {"ROOT": ROOT, "REPO": REPO, "TRAIN": TRAIN})
