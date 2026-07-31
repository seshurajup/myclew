"""Working code for pt06 — the loss & the division weight, on REAL data. Builds the real edge
target from the real .geff, runs the real compute_loss, and shows the division-weight lever.
    research/cellmot_venv/bin/python learning/annotated/pt06_loss_division.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="pt06", order=15, title="The loss & the division weight",
            subtitle="compute_loss on the real graph — and the one line that decides divisions",
            source="research/pilkwang_support_pack/repo/scripts/train_unet_transformer.py")

CELLS = [
    dict(note="""## Training the edge model — from the REAL graph
The edge model learns from the **real tracking graph**: for a frame pair, the target is a binary
matrix `target[i,j]=1` where cell `i@t` really links to `j@t+1`. We build that from the real
`.geff` edges, then run the real `compute_loss`. The last cell exposes **the lever** of our whole
project."""),

    dict(note="""### Build the real target matrix (frame 0 → 1)
Read real nodes (id, frame) and real edges (source→target) from the `.geff`; mark which
frame-0 cell links to which frame-1 cell.""",
         code="""import numpy as np, zarr, torch                                    # arrays, IO, tensors
G = f"{TRAIN}/6bba_062c8d37.geff"                                    # a real embryo's graph
nid = np.asarray(zarr.open(f"{G}/nodes/ids")[:])                    # every node's id
nt  = np.asarray(zarr.open(f"{G}/nodes/props/t/values")[:])        # every node's frame
edges = np.asarray(zarr.open(f"{G}/edges/ids")[:])                 # real edges: (source_id, target_id)
c0 = nid[nt == 0]                                                  # real cells at frame 0
c1 = nid[nt == 1]                                                  # real cells at frame 1
i0 = {n: i for i, n in enumerate(c0)}                              # id -> row index
i1 = {n: j for j, n in enumerate(c1)}                              # id -> col index
target = torch.zeros(len(c0), len(c1))                             # the edge target matrix
for s, d in edges:                                                # every real edge
    if s in i0 and d in i1:                                        # if it goes frame 0 -> 1
        target[i0[s], i1[d]] = 1.0                                 # mark the real link
int(target.sum())                                                 # number of real frame-0->1 links"""),

    dict(note="""### The real compute_loss — and the division line
**[PyTorch]** `softmax(dim=0)` (divisions allowed, merges not), focal weighting `((1-p_t)**2)`,
BCE on the annotated rows/cols. **[The lever]** `div_rows = target.sum(dim=1) > 1` are sources with
**2 children = a division**; `weight[div_rows]` scales their loss. It defaults to **1.0** — so the
rare divisions are drowned out and never learned (div_jaccard≈0). Our fine-tune raises it (5–20).""",
         code="""import torch.nn.functional as F, os                            # loss fns + env
def compute_loss(logits, target):                               # the REAL compute_loss
    active_rows = target.sum(dim=1) > 0                          # rows with any label
    active_cols = target.sum(dim=0) > 0                          # cols with any label
    mask = active_rows.unsqueeze(1) | active_cols.unsqueeze(0)   # score only annotated pairs (sparse GT)
    probs = torch.softmax(logits, dim=0)                        # per-target link prob (divisions allowed)
    bce = F.binary_cross_entropy(probs, target, reduction="none")  # BCE per pair
    p_t = probs * target + (1 - probs) * (1 - target)           # focal p_t
    loss = ((1 - p_t) ** 2) * bce                                # focal-weighted BCE
    div_rows = target.sum(dim=1) > 1                             # a source with >1 child = a DIVISION
    weight = torch.ones_like(loss)                              # default weight 1 everywhere
    weight[div_rows] = float(os.environ.get("BIOHUB_DIV_LOSS_WEIGHT", "1.0"))  # THE LEVER
    return (loss * weight)[mask].mean()                         # mean over annotated pairs
logits = torch.zeros_like(target)                               # (placeholder logits: untrained -> real loss value)
{"loss": round(float(compute_loss(logits, target)), 4),        # the real loss on real target
 "n_divisions_this_pair": int((target.sum(dim=1) > 1).sum())}  # real divisions in frames 0->1""",
         image="learning/assets/division_event.png\nA division: a parent at t links to TWO children at t+1 that both continue. These are the rows `weight[div_rows]` scales — rare, so at weight 1.0 the model never learns them."),

    dict(note="""**[Domain]** whether `n_divisions` here is 0 or a few, the point holds: at weight
1.0 those rows carry no more weight than the thousands of ordinary edges, so the model won't learn
them. That single default is why the +0.049 division headroom sits unclaimed — and why our
approach is to up-weight it. **Next → pt07: the Dataset that feeds this.**"""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"), {"TRAIN": TRAIN})
