"""Working code for pt05b — positional features & embeddings, on REAL cell coords.
Runs the real extract_pos_features on real .geff coordinates.
    research/cellmot_venv/bin/python learning/annotated/pt05b_pos_features.py
"""
from pathlib import Path
import sys
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning"))
from lessonkit import build_lesson

REPO = ROOT / "research/pilkwang_support_pack/repo"
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="pt05b", order=14, title="Positional features & embeddings",
            subtitle="extract_pos_features: turning a cell's (t,z,y,x) into what the edge model reads",
            source="research/pilkwang_support_pack/repo/scripts/train_unet_transformer.py")

CELLS = [
    dict(note="""## The edge model needs to know WHERE cells are
pt05's transformer scores links between cells. But two cells far apart in space almost never link,
so the model must **see position**. Raw `(t,z,y,x)` numbers are hard to learn from; instead each
coordinate is turned into a **sinusoidal embedding** (like a transformer's positional encoding).
Run the real `extract_pos_features` on real cells of `6bba_062c8d37`."""),

    dict(note="""### Read real cell coordinates
Pull `(t,z,y,x)` for every annotated cell from the `.geff`.""",
         code="""import numpy as np, zarr                                              # arrays + on-disk reader
G = f"{TRAIN}/6bba_062c8d37.geff/nodes/props"                            # the node properties
coords = np.stack([np.asarray(zarr.open(f"{G}/{a}/values")[:])          # stack t,z,y,x per node
                   for a in ("t", "z", "y", "x")], axis=1)              # -> (N, 4)
coords[:3]                                                              # first 3 real cells (t,z,y,x)"""),

    dict(note="""### Sinusoidal positional embedding
**[PyTorch]** For each axis, `extract_pos_features` normalises the value by the image size, then
builds `sin/cos` at several frequencies (`2**arange(dim//2) * π`). **[Domain]** this gives the edge
model a smooth, multi-scale sense of *where* each cell sits — so it can learn "cells this close can
link, this far cannot". Run it on the real coords.""",
         code="""import sys                                                            # to reach the real repo
sys.path.insert(0, f"{REPO}/src"); sys.path.insert(0, f"{REPO}/scripts")  # the real repo
from train_unet_transformer import extract_pos_features                  # the REAL function
image_shape = (100, 64, 256, 256)                                        # (T,Z,Y,X) of a real movie
pos = extract_pos_features(coords, image_shape)                          # sinusoidal embeddings
pos.shape                                                                # (N, 4 * pos_embed_dim) real feature vectors"""),

    dict(note="""**[Recap]** Every cell → a fixed-length positional feature vector. The edge model
(pt05) concatenates this with the cell's **UNet feature** (its appearance) and attends over pairs.
**Next → pt06: the loss** that trains those edge scores."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"),
                 {"ROOT": ROOT, "REPO": REPO, "TRAIN": TRAIN})
