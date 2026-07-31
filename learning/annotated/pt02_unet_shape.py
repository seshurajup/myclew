"""Working code for pt02 — the U-Net shape. Runs the real down/up/skip operations on a
real frame and writes pt02_unet_shape.learning with the real captured shapes.
    research/cellmot_venv/bin/python learning/annotated/pt02_unet_shape.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="pt02", order=11, title="The U-Net shape",
            subtitle="MaxPool3d down, Upsample up, torch.cat skips — real shrinking/growing shapes",
            source="research/pilkwang_support_pack/repo/src/biohub_tracking/models/temporal_unet.py")

CELLS = [
    dict(note="""## Why a U-Net
One conv block sees a tiny neighbourhood. A cell's *context* (isolated? dense tissue?) needs a
wide view, but the answer must be **voxel-precise** (the metric matches within 7 µm). A U-Net does
both: an **encoder** shrinks the volume to see context, a **decoder** grows it back, and
**skip connections** carry the sharp early detail across. Real `layers=(32,64,128)` → 3 stages.
Every shape below is captured by running the real ops."""),

    dict(note="""### Encoder stage 0 — full resolution
Load a real frame and run the first conv block: `1 → 32` channels, spatial size kept.""",
         code="""import torch, torch.nn as nn, numpy as np, zarr                    # layers + IO
def conv_block(cin, cout):                                          # the real _conv_block
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm3d(cout),
                         nn.ReLU(True), nn.Conv3d(cout, cout, 3, padding=1, bias=False),
                         nn.BatchNorm3d(cout), nn.ReLU(True))       # Conv-BN-ReLU x2
vol = np.asarray(zarr.open(f"{TRAIN}/6bba_062c8d37.zarr/0")[0]).astype(np.float32)[:, ::4, ::4]  # real frame, downsampled
x = torch.from_numpy(vol)[None, None]                              # (B,C,Z,Y,X)
f0 = conv_block(1, 32).eval()(x)                                   # encoder stage 0
f0.shape                                                           # full-res feature shape"""),

    dict(note="""### Go down — MaxPool3d halves space, next block doubles channels
**[PyTorch]** `MaxPool3d(2)` keeps the max in each 2×2×2 block → half the spatial size. Then a
conv block `32 → 64`. **[Domain]** each pooled voxel now summarises a wider region (more context).""",
         code="""pool = nn.MaxPool3d(2)                        # halve Z,Y,X by taking the max of each 2x2x2 block
f1 = conv_block(32, 64).eval()(pool(f0))     # down then stage 1: 32 -> 64 channels
f1.shape                                     # half the space, double the channels"""),

    dict(note="""### The bottleneck — deepest, smallest, widest
Pool again and run the deepest block `64 → 128`. Smallest spatial size = widest context.""",
         code="""f2 = conv_block(64, 128).eval()(pool(f1))    # down again then stage 2 (bottleneck): 64 -> 128
f2.shape                                     # smallest space, most channels (widest view)"""),

    dict(note="""### Go up — Upsample + torch.cat skip
**[PyTorch]** `Upsample(scale_factor=2, mode='trilinear')` doubles the spatial size; `torch.cat`
glues the **skip** feature (`f1`, sharp full-detail from the encoder) onto the upsampled bottleneck
along the channel axis. **[Domain]** this is how the decoder recovers voxel-precise cell positions.
The diagram shows the whole U.""",
         code="""up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)  # double space back up
u1 = up(f2)                                  # bottleneck upsampled: 128 channels, space doubled
cat = torch.cat([u1, f1], dim=1)             # skip: concat encoder f1 (64) -> 128+64 channels
cat.shape                                    # channels summed, ready for the decoder block""",
         image="learning/assets/unet_shape.png\nThe U-Net with our real shapes: encoder shrinks space & grows channels (32→64→128), decoder upsamples back, orange dashed = the torch.cat skips."),

    dict(note="""**[Recap]** encoder: space ↓, channels ↑ · bottleneck: smallest+widest · decoder:
upsample + skip-cat → back to full resolution. A `head` conv then turns those channels into the
per-voxel cell heat-map — **next → pt03: temporal attention**, then pt04 makes the detections."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"), {"TRAIN": TRAIN})
