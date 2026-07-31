"""Working code for pt05 — the edge model, run on REAL cells. Reads real cell coordinates
from a real .geff, samples real features from a real frame, and runs the real
SimpleNodeTransformer to get a real edge-logit matrix.
    research/cellmot_venv/bin/python learning/annotated/pt05_edge_model.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
REPO = ROOT / "research/pilkwang_support_pack/repo/src"

META = dict(id="pt05", order=14, title="The edge model — scoring links",
            subtitle="SimpleNodeTransformer on REAL cells: is i@t the same cell as j@t+1?",
            source="research/pilkwang_support_pack/repo/src/biohub_tracking/models/simple_node_transformer.py")

CELLS = [
    dict(note="""## From detections to a graph
Tracking = **linking** cells across frames: for every cell `i` at frame `t` and `j` at `t+1`, are
they the same cell? The real `SimpleNodeTransformer` scores every `(i, j)` pair → an **edge-logit
matrix** ("the learned graph"). Here we feed it **real cells** read from a real `.geff` — every
shape below is captured on our data."""),

    dict(note="""### Read real cells + real features
Read cell `(t,z,y,x)` from a real `.geff`; make a 32-channel feature map from the real frame and
**sample it at each real cell location** → real per-cell features.""",
         code="""import torch, torch.nn as nn, numpy as np, zarr, sys                # layers + IO
gp = f"{TRAIN}/6bba_062c8d37.geff/nodes/props"                       # the graph's node properties
t  = np.asarray(zarr.open(f"{gp}/t/values")[:])                     # frame index of every real cell
zc = np.asarray(zarr.open(f"{gp}/z/values")[:]).astype(int)         # real z of every cell
yc = np.asarray(zarr.open(f"{gp}/y/values")[:]).astype(int)         # real y
xc = np.asarray(zarr.open(f"{gp}/x/values")[:]).astype(int)         # real x
mov = zarr.open(f"{TRAIN}/6bba_062c8d37.zarr/0")                    # the real movie
enc = nn.Conv3d(1, 32, 3, padding=1).eval()                        # a 32-channel feature map maker
def cells(fr):                                                      # real cells + features at frame fr
    vol = np.asarray(mov[fr]).astype(np.float32)[:, ::4, ::4]       # real frame, downsampled (1,4,4)
    fm = enc(torch.from_numpy(vol)[None, None])[0]                  # (32, Z, Y, X) real feature map
    m = t == fr                                                     # pick this frame's cells
    zz = np.clip(zc[m], 0, fm.shape[1]-1)                           # z on the feature grid
    yy = np.clip(yc[m]//4, 0, fm.shape[2]-1)                        # y//4 (matches the (1,4,4) downsample)
    xx = np.clip(xc[m]//4, 0, fm.shape[3]-1)                        # x//4
    feats = fm[:, zz, yy, xx].T                                     # (N, 32) real features at real cells
    coords = torch.tensor(np.stack([zc[m], yc[m], xc[m]], 1), dtype=torch.float32)  # real (z,y,x)
    return feats, coords
feat_t, coords_t = cells(0)                                         # real cells at frame 0
feat_t1, coords_t1 = cells(1)                                       # real cells at frame 1
(feat_t.shape, feat_t1.shape)                                       # (#cells@t, 32), (#cells@t+1, 32)""",
         image="learning/assets/tracking_graph.png\nThe tracking graph the edges build: nodes are cells per frame (columns = time), edges link the same cell across frames, a branch = a division."),

    dict(note="""### Run the real edge model → the edge-logit matrix
**[PyTorch]** projects 32-dim features → 128 (`Linear`), 4 cross-attention blocks
(`MultiheadAttention`+`GELU`+`Dropout`), then a pairwise `Linear` MLP scores each `(i,j)`.
**[Domain]** the output is a score for **every possible link** between the real cells.""",
         code="""sys.path.insert(0, str(REPO))                                # so we can import the real model
from biohub_tracking.models.simple_node_transformer import SimpleNodeTransformer  # the real class
model = SimpleNodeTransformer(feat_dim=32).eval()            # real edge model (32-dim features)
edge_logits = model(feat_t, feat_t1, coords_t, coords_t1)   # score every real (i,j) link
edge_logits.shape                                           # (#cells@t, #cells@t+1) — one score per edge"""),

    dict(note="""**Next → pt06: the loss** — how this edge model is trained on the real graph, and
the single line (`weight[div_rows]`) that decides whether it ever learns divisions."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"), {"TRAIN": TRAIN, "REPO": REPO})
