"""Working code for pt04b — the full model (UNetNodeTransformer), on REAL data.
Instantiates the real model and runs its UNet on a real 2-frame window.
    research/cellmot_venv/bin/python learning/annotated/pt04b_full_model.py
"""
from pathlib import Path
import sys
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning"))
from lessonkit import build_lesson

REPO = ROOT / "research/pilkwang_support_pack/repo"
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="pt04b", order=13, title="The full model — UNetNodeTransformer",
            subtitle="How the detector (pt01-04) and the edge model (pt05) become ONE network",
            source="research/pilkwang_support_pack/repo/scripts/train_unet_transformer.py")

CELLS = [
    dict(note="""## Wiring the two halves together
We built the **detector** (pt01–04) and the **edge model** (pt05) separately. The real
`UNetNodeTransformer` combines them into one network: the UNet encodes a **2-frame window** into
features, a `detect_head` (1×1 conv) reads out the per-voxel heatmap, and the transformer scores
links between the detected cells. Everything below runs on real frames of `6bba_062c8d37`."""),

    dict(note="""### Build the real model
The real `UNetNodeTransformer(unet, unet_out_channels, pos_feat_dim)` holds a `TemporalUNet3D`, a
`detect_head` Conv3d, and the `SimpleNodeTransformer`. Count its parameters.""",
         code="""import sys, torch                                                    # tensors + repo path
sys.path.insert(0, f"{REPO}/src"); sys.path.insert(0, f"{REPO}/scripts")     # the real repo
from biohub_tracking.models.temporal_unet import TemporalUNet3D              # the real detector backbone
from train_unet_transformer import UNetNodeTransformer                       # the real combined model
unet = TemporalUNet3D(in_channels=1, out_channels=32)                        # the encoder (pt01-03)
model = UNetNodeTransformer(unet, unet_out_channels=32, pos_feat_dim=32)     # detector + edge model in one
sum(p.numel() for p in model.parameters())                                   # total trainable params"""),

    dict(note="""### Its three parts
The combined model is literally the detector backbone + a 1×1 detection head + the edge
transformer — the pieces from the previous lessons.""",
         code="""[type(model.unet).__name__,          # the 3D temporal U-Net encoder (pt01-03)
 type(model.detect_head).__name__,   # the 1x1 conv that reads the heatmap (pt04)
 type(model.transformer).__name__]   # the cross-attention edge scorer (pt05)"""),

    dict(note="""### Run the encoder on a REAL 2-frame window
**[PyTorch]** `TemporalUNet3D.forward` takes `(B, T, C, Z, Y, X)` — a **window of T=2 frames** —
and returns per-voxel features for each. **[Domain]** two frames together so the temporal attention
(pt03) can use frame `t` to help detect the same cell at `t+1`. Feed two real frames.""",
         code="""import numpy as np, zarr                                              # arrays + reader
z = zarr.open(f"{TRAIN}/6bba_062c8d37.zarr/0")                            # the real movie
w = np.stack([np.asarray(z[0]), np.asarray(z[1])]).astype(np.float32)     # frames 0 and 1 (T,Z,Y,X)
w = w[:, :, ::4, ::4]                                                     # downsample Y,X -> detector grid
x = torch.from_numpy(w)[None, :, None]                                    # (B=1, T=2, C=1, Z, Y, X)
with torch.no_grad():                                                     # inference only
    feats = model.unet(x)                                                 # UNet features for both frames
feats.shape                                                               # (B, T, 32, Z, Y, X) real features"""),

    dict(note="""### The detection heatmap for one frame
Apply the real `detect_head` to a frame's features → the per-voxel cell heatmap (as in pt04).""",
         code="""with torch.no_grad():                                                # inference only
    heat = model.detect_head(feats[:, 0])                                # 1x1 conv on frame-0 features
heat.shape                                                               # (B, 1, Z, Y, X) real heatmap"""),

    dict(note="""**[Recap]** One `UNetNodeTransformer` = detector backbone + detect head + edge
transformer, trained jointly by `compute_detection_loss` (pt06c) + `compute_loss` (pt06). **Next →
pt07: the Dataset** that feeds it real frame windows."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"),
                 {"ROOT": ROOT, "REPO": REPO, "TRAIN": TRAIN})
