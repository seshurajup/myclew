"""Working code for ours01 — our fine-tune edits. Shows the REAL effect of the division
loss weight using real division counts from the data.
"""
from pathlib import Path
import sys
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning"))
from lessonkit import build_lesson
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"

META = dict(id="ours01", order=40, title="Our fine-tune code",
            subtitle="The division-loss weight edit — why weight 1.0 buries the rare divisions",
            source="research/pilkwang_support_pack/repo/scripts/train_unet_transformer.py (our edits)")

CELLS = [
    dict(note="""## The one-line edit that matters
The metric pays `+0.1×` for divisions, but pilkwang's loss weights every training row equally:

```python
div_rows = target.sum(dim=1) > 1     # rows that are a division (parent -> 2 children)
weight = torch.ones_like(loss)
weight[div_rows] = 1.0               # <-- OUR SMOKING GUN: divisions weighted the SAME as everything
```

Our fine-tune exposes this as `BIOHUB_DIV_LOSS_WEIGHT` and raises it (5–20). Below we show, on
**real** data, why weight 1.0 fails: divisions are so rare they barely register in the loss."""),

    dict(note="""### Count real divisions vs normal links
Across real frame-pairs, count division rows (a cell → 2 children) vs normal rows. Divisions are a
tiny fraction — so at weight 1.0 they contribute almost nothing to the gradient.""",
         code="""import numpy as np, zarr                                             # tools
n_normal, n_div = 0, 0                                                   # real link vs division counts
for g in sorted(TRAIN.glob("6bba_*.geff"))[:30]:                        # 30 real embryos
    src = np.asarray(zarr.open(f"{g}/edges/ids")[:])[:, 0]              # source id of every real edge
    _, counts = np.unique(src, return_counts=True)                     # out-degree per source cell
    n_normal += int((counts == 1).sum())                               # normal continuations
    n_div += int((counts >= 2).sum())                                   # divisions (out-degree >= 2)
{"normal links": n_normal, "divisions": n_div,                          # real counts
 "division fraction %": round(100 * n_div / (n_normal + n_div), 2)}     # how rare divisions are"""),

    dict(note="""### What up-weighting does
At weight `w`, divisions get `w × n_div` of the loss 'mass' vs `n_normal` for the rest. Show the
real division share of the loss at weight 1 vs weight 10 — up-weighting multiplies their influence.""",
         code="""share = lambda w: round(100 * w * n_div / (n_normal + w * n_div), 1)  # division share of the loss at weight w
{"div share @ weight 1": share(1), "div share @ weight 10": share(10)}   # our fine-tune raises this"""),

    dict(note="""**[Recap]** divisions are ~1–2% of links, so at weight 1.0 the model optimises them
into the ground — `div_jaccard ≈ 0`. Raising the weight is our fix (validated on golden-12; see
rs04). **Next → ours02: our golden-CV harness.**"""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"), {"TRAIN": TRAIN})
