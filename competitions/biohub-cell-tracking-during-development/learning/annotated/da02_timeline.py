"""Working code for da02 — FULL timeline/timing EDA. Running it builds the timeline on
real data, saves the heatmap, and writes da02_timeline.learning with real outputs.
    research/cellmot_venv/bin/python learning/annotated/da02_timeline.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
ASSETS = ROOT / "learning/assets"

META = dict(id="da02", order=2, title="Following time — the timeline",
            subtitle="Full EDA of the time axis: growth, start/end timing, and movie length",
            source="learning/annotated/da02_timeline.py")

CELLS = [
    dict(note="""## Time is the whole point
This is a *tracking* competition — we follow each cell **across frames**. So we explore the time
axis fully: how the count changes over a movie (cells divide), how long movies are, and when each
embryo's recording begins and ends. Every output below is the real result of running the code."""),

    dict(note="""### Build the per-embryo timeline
For each embryo, count cells per frame onto a 0..99 timeline (`NaN` = no data). Also record its
start (`t_min`), end (`t_max`), length, and early-vs-late mean count (for the growth check).""",
         code="""import numpy as np, pandas as pd, zarr                   # tools
files = sorted(TRAIN.glob("*.geff"))                          # every embryo
mat = np.full((len(files), 100), np.nan)                      # 199 x 100 timeline, NaN = no data
info = []                                                     # per-embryo timing/growth
for i, g in enumerate(files):                                 # each embryo
    t = np.asarray(zarr.open(f"{g}/nodes/props/t/values")[:])  # node frames
    vc = pd.Series(t).value_counts().sort_index()             # cells per frame
    mat[i, vc.index.values] = vc.values                       # place on the timeline
    early = vc[vc.index < vc.index.min() + 20].mean()         # mean count in first 20 frames
    late = vc[vc.index > vc.index.max() - 20].mean()          # mean count in last 20 frames
    info.append({"group": g.name[:4], "t_min": int(t.min()), "t_max": int(t.max()),
                 "length": len(vc), "early": round(early, 1), "late": round(late, 1)})
ti = pd.DataFrame(info)                                       # timing table
mat.shape                                                     # (embryos, frames)"""),

    dict(note="""### The timeline heatmap
Sort rows sparse→dense and draw it: **grey = no data**, colour = cells that frame. **[Domain]**
dense rows brighten left→right — embryos *gain* cells over time (division). The grey staircase on
the left = different start frames.""",
         code="""import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
order = np.argsort(np.nan_to_num(mat).mean(axis=1))           # sort embryos by mean density
cmap = plt.cm.viridis.copy(); cmap.set_bad("#f2f2f2")         # NaN -> grey
fig, ax = plt.subplots(figsize=(11, 7))                       # the canvas
ax.imshow(mat[order], aspect="auto", cmap=cmap, vmin=0, vmax=25)  # the heatmap
ax.set_xlabel("frame t (the timeline →)"); ax.set_ylabel("199 embryos (sparse → dense)")
fig.savefig(f"{ASSETS}/da02_timeline.png", dpi=110, bbox_inches="tight", facecolor="white")
print("saved da02_timeline.png")""",
         image="learning/assets/da02_timeline.png\nReal timeline: each row an embryo, x = frame, colour = cells that frame, grey = no data. Cells grow left→right (division); grey staircase = different start frames."),

    dict(note="""### Do cells actually grow over time?
Compare each embryo's mean count in its **first 20 frames** vs its **last 20 frames**. If late >
early, the embryo gained cells (division) — the biological signal a tracker must follow.""",
         code="""grew = (ti["late"] > ti["early"]).mean()                 # fraction of embryos that grew
pd.Series({"mean early count": round(ti["early"].mean(), 1),   # avg count at movie start
           "mean late count": round(ti["late"].mean(), 1),     # avg count at movie end
           "embryos that grew %": round(100 * grew, 0)})        # how many gained cells"""),

    dict(note="""### Movie length & when embryos start/end
Frames per embryo, and the spread of start (`t_min`) and end (`t_max`) frames — many embryos are
time-cropped, which matters for how we split data.""",
         code="""pd.DataFrame({                                          # a small timing summary
    "length": ti["length"].describe().loc[["min", "50%", "max"]],   # frames per embryo
    "t_min":  ti["t_min"].describe().loc[["min", "50%", "max"]],    # start frame
    "t_max":  ti["t_max"].describe().loc[["min", "50%", "max"]],    # end frame
}).astype(int)                                               # min/median/max of each"""),

    dict(note="""### How many start at frame 0?
Most embryos begin at t=0; a minority start later (cropped windows).""",
         code="""pd.Series({"start at t=0": int((ti["t_min"] == 0).sum()),   # begin at frame 0
           "start later": int((ti["t_min"] > 0).sum()),            # begin later
           "latest start": int(ti["t_min"].max())})                # latest starting frame"""),

    dict(note="""**[Recap]** Cells grow across the movie (division), movies are up to ~100 frames,
and embryos start/end at different frames. Time is a first-class axis here.

**Next → da03: density → developmental stage** — turning counts into stage, and the confound
that shapes how we validate."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"),
                 {"TRAIN": TRAIN, "ASSETS": ASSETS})
