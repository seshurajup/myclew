"""Working code for pt03 — temporal attention. Runs the real _TemporalAttention on a real
2-frame feature tensor and writes pt03_temporal_attention.learning with real shapes.
    research/cellmot_venv/bin/python learning/annotated/pt03_temporal_attention.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="pt03", order=12, title="Temporal attention",
            subtitle="_TemporalAttention — let each voxel look across the 2-frame window",
            source="research/pilkwang_support_pack/repo/src/biohub_tracking/models/temporal_unet.py")

CELLS = [
    dict(note="""## Why look across time inside the detector
A cell that's faint in one frame is often clearer in the next. The detector processes a small
**window of W=2 frames** and, at each voxel, lets the two timepoints **attend to each other** — so
evidence at `t` helps detect the same cell at `t+1`. This is the real `_TemporalAttention`; the
shapes below are captured by running it on real features."""),

    dict(note="""### Build real 2-frame features
Take 2 consecutive real frames (a small crop for speed), conv each to 8 feature channels, and
stack into `(B, T=2, C, Z, Y, X)` — the input the attention block expects.""",
         code="""import torch, torch.nn as nn, numpy as np, zarr                    # layers + IO
z = zarr.open(f"{TRAIN}/6bba_062c8d37.zarr/0")                       # the 4-D movie
crop = np.asarray(z[0:2, :, 96:128, 96:128]).astype(np.float32)[:, ::2]  # 2 frames, small crop (T,Z,Y,X)
conv = nn.Conv3d(1, 8, 3, padding=1).eval()                         # tiny conv to make 8 features
feats = torch.stack([conv(torch.from_numpy(crop[t])[None, None]) for t in range(2)], dim=1)  # (B,T,C,Z,Y,X)
feats.shape                                                         # the real 2-frame feature tensor"""),

    dict(note="""### The real _TemporalAttention
**[PyTorch]** it reshapes `(B,T,C,Z,Y,X)` so every voxel becomes a length-`T` sequence
`(B·S, T, C)`, runs `LayerNorm` + `MultiheadAttention` (4 heads) across time, adds the result back
(**residual**). **[Domain]** attention lets frame `t` and `t+1` share evidence about the same cell.
Output shape = input shape (it *enriches* features, doesn't resize).""",
         code="""import math                                                   # for prod(spatial)
class TemporalAttention(nn.Module):                             # the real _TemporalAttention
    def __init__(self, channels, n_heads=4):                    # per-channel attention config
        super().__init__()
        self.norm = nn.LayerNorm(channels)                      # normalise features before attention
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)  # attention across time
    def forward(self, x):                                       # x: (B,T,C,Z,Y,X)
        B, T, C = x.shape[:3]                                   # batch, time, channels
        spatial = x.shape[3:]                                   # (Z,Y,X)
        S = math.prod(spatial)                                  # number of voxels
        h = x.reshape(B, T, C, S).permute(0, 3, 1, 2).reshape(B * S, T, C)  # one length-T seq per voxel
        h = self.norm(h)                                        # LayerNorm
        h, _ = self.attn(h, h, h, need_weights=False)           # self-attention across the T axis
        h = h.reshape(B, S, T, C).permute(0, 2, 3, 1).reshape(B, T, C, *spatial)  # back to grid
        return x + h                                            # residual: enrich, don't replace
out = TemporalAttention(8).eval()(feats)                        # run it on the real features
out.shape                                                       # same shape as input (enriched)"""),

    dict(note="""**[Craft]** at full resolution this per-voxel attention is expensive, so the real
`TemporalUNet3D` replaces it with `nn.Identity` on the first (full-res) stage — a deliberate
speed/memory trade. **Next → pt04: turning these features into actual cell detections.**"""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"), {"TRAIN": TRAIN})
