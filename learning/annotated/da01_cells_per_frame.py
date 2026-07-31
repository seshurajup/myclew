"""Working code for da01 — FULL cells-per-frame EDA. Running it explores the real data
and writes da01_cells_per_frame.learning with every real output attached.
    research/cellmot_venv/bin/python learning/annotated/da01_cells_per_frame.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

TRAIN = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/"
             "input/biohub-cell-tracking-during-development/train")

META = dict(id="da01", order=1, title="How many cells per frame?",
            subtitle="Full EDA of the tracking graph — counts, distribution, and the sparse-label trap",
            source="learning/annotated/da01_cells_per_frame.py")

CELLS = [
    dict(note="""## The very first question
Each training embryo is a `.geff` **tracking graph**; every **node = one annotated cell at one
timepoint**, and node property `t` is the frame index. So "cells in frame `t`" = how many nodes
share that `t`. Below, we explore this fully on the real data — every output is the real result of
running the code."""),

    dict(note="""### Cells per frame, for one embryo
Read `t` for every node of `6bba_09961292` and count per frame.""",
         code="""t = np.asarray(zarr.open(f"{TRAIN}/6bba_09961292.geff/nodes/props/t/values")[:])  # frame of every node
cpf = pd.Series(t).value_counts().sort_index()             # cells per frame
cpf.head(6).rename_axis("frame").rename("cells").reset_index()  # first 6 frames"""),

    dict(note="""### Scan ALL 199 embryos
Loop every `.geff`, and for each record its group, #frames, total nodes, and the min/mean/max
cells-per-frame. We keep every per-frame count too (for the distribution below).""",
         code="""import numpy as np, pandas as pd, zarr, json                # tools
rows, all_cpf = [], []                                       # per-embryo stats, and every frame's count
for g in sorted(TRAIN.glob("*.geff")):                       # each embryo
    t = np.asarray(zarr.open(f"{g}/nodes/props/t/values")[:])  # node frames
    vc = pd.Series(t).value_counts()                          # cells per frame
    all_cpf.extend(vc.tolist())                               # remember every frame's count
    estN = json.load(open(f"{g}/zarr.json"))["attributes"]["geff"]["extra"]["estimated_number_of_nodes"]  # TRUE count
    rows.append({"group": g.name[:4], "n_frames": len(vc), "n_nodes": int(len(t)),
                 "cpf_min": int(vc.min()), "cpf_mean": round(vc.mean(), 1),
                 "cpf_max": int(vc.max()), "estN": int(estN)})  # this embryo's stats
eda = pd.DataFrame(rows)                                      # 199-row EDA table
eda.describe().loc[["min", "50%", "max"]].round(1)           # overall min/median/max of each column"""),

    dict(note="""### The two groups — annotated density
Group by the 4-char embryo prefix. `44b6` is sparsely annotated (~3/frame), `6bba` denser (~9).""",
         code="""(eda.groupby("group")                                    # 44b6 vs 6bba
    .agg(n=("group", "size"), n_frames_median=("n_frames", "median"),
         cpf_mean=("cpf_mean", "mean"), cpf_max=("cpf_max", "max"))
    .round(1).reset_index())                                 # per-group table"""),

    dict(note="""### The distribution — how sparse is it really?
Bin every frame's cell count. Almost all frames have very few annotated cells — a max of ~33
anywhere, and ~78% of frames under 10.""",
         code="""bins = [0, 10, 25, 50, 1000]                              # count ranges
labels = ["1-10", "11-25", "26-50", "50+"]                   # readable labels
dist = pd.cut(all_cpf, bins=bins, labels=labels).value_counts().reindex(labels)  # frames per range
dist.rename_axis("cells/frame").rename("n_frames").reset_index()  # the real histogram"""),

    dict(note="""### The trap — annotated ≠ real
**[Domain]** A max of ~33 cells/frame is impossibly low for a mid-development zebrafish embryo.
The `.geff` only **sparsely labels** a subset of tracks. `estN` (`estimated_number_of_nodes`) is
the organisers' TRUE estimate. Compare: the annotated fraction is tiny.""",
         code="""eda["annot_frac_%"] = (100 * eda["n_nodes"] / eda["estN"]).round(1)  # labelled / true
eda["annot_frac_%"].describe().loc[["min", "50%", "max"]].round(1)   # real label-fraction range"""),

    dict(note="""**[Recap]** 199 embryos, two groups (44b6 sparse-annotated, 6bba denser), ~78% of
frames have <10 *annotated* cells, and only a few % of real cells are labelled. That sparse-label
fact is the single most important thing to remember — it's why our local CV can mislead (see rs02).

**Next → da02: following time** — how counts change across the movie, and when embryos start."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"), {"TRAIN": TRAIN})
