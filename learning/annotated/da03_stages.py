"""Working code for da03 — FULL density→stage EDA. Running it reads the true counts,
clusters into stages, saves the plot, and writes da03_stages.learning with real outputs.
    research/cellmot_venv/bin/python learning/annotated/da03_stages.py
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lessonkit import build_lesson

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
ASSETS = ROOT / "learning/assets"

META = dict(id="da03", order=3, title="Density → developmental stage",
            subtitle="Full EDA: TRUE density, the annotation inversion, the S0–S4 stages, and the confound",
            source="learning/annotated/da03_stages.py")

CELLS = [
    dict(note="""## From counts to biology
da01's counts were *annotated* cells (sparse). Each `.geff` also stores the organisers' TRUE
estimate — `estimated_number_of_nodes` (estN). Divide by frames → real cells/frame, which tracks
the **developmental stage** (gastrula → segmentation). We explore that fully below — real outputs."""),

    dict(note="""### Read the TRUE density for every embryo
Pull estN from each geff's attributes, divide by its frame count → true cells/frame.""",
         code="""import json, numpy as np, pandas as pd, zarr              # tools
rows = []                                                     # per-embryo true density
for g in sorted(TRAIN.glob("*.geff")):                        # every embryo
    attrs = json.load(open(f"{g}/zarr.json"))["attributes"]["geff"]  # geff metadata
    estN = attrs["extra"]["estimated_number_of_nodes"]        # organisers' TRUE node estimate
    t = np.asarray(zarr.open(f"{g}/nodes/props/t/values")[:])  # annotated node frames
    n_frames = len(np.unique(t))                              # frames this embryo spans
    rows.append({"group": g.name[:4], "true_per_frame": round(estN / n_frames)})  # true cells/frame
df = pd.DataFrame(rows)                                       # 199-row density table
df["true_per_frame"].describe().loc[["min", "50%", "max"]].astype(int)  # the real range 38..1015"""),

    dict(note="""### The inversion — annotation density is BACKWARDS from biology
`44b6` has FEW annotated cells (da01: ~3/frame) but the HIGHEST true density → the biologically
**late/dense** group. `6bba` is the opposite. So "6bba looks denser" (labels) reverses the truth.""",
         code="""(df.groupby("group")                                    # 44b6 vs 6bba
   .agg(n=("group", "size"),
        true_per_frame_median=("true_per_frame", "median"),
        true_per_frame_max=("true_per_frame", "max"))
   .astype(int).reset_index())                                # the real per-group TRUE density"""),

    dict(note="""### Cluster into 5 developmental stages (S0–S4)
Split the log of true density into 5 bins — a simple stand-in for the developmental stages
(gastrula → segmentation). Count how many embryos land in each stage.""",
         code="""x = np.log10(df["true_per_frame"].clip(lower=1))         # log density (spans ~1.5 decades)
edges = np.linspace(x.min(), x.max() + 1e-9, 6)              # 5 equal log-width stage bins
df["stage"] = pd.cut(x, bins=edges, labels=["S0", "S1", "S2", "S3", "S4"], include_lowest=True)
(df.groupby("stage", observed=True)                          # per stage
   .agg(n=("group", "size"),
        density_min=("true_per_frame", "min"),
        density_max=("true_per_frame", "max")).reset_index())  # stage -> count + density range"""),

    dict(note="""### The confound — stage is tangled with embryo
Cross-tabulate stage × group. `6bba` dominates the early stages (S0–S1), `44b6` the late (S3–S4).
**[Why it matters]** an embryo-disjoint split secretly tests a **stage shift** — so we prefer
leave-one-STAGE-out validation (see rs02).""",
         code="""pd.crosstab(df["stage"], df["group"])                   # embryos per (stage, group)"""),

    dict(note="""### Stage plot — every embryo on the density axis
Sort by true density, colour by group. The clean early→late gradient is gastrula → segmentation;
red `44b6` clusters late, blue `6bba` early — the confound made visible.""",
         code="""import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
s = df.sort_values("true_per_frame").reset_index(drop=True)   # sort early -> late
col = s["group"].map({"44b6": "#E15759", "6bba": "#4E79A7"})  # colour by group
fig, ax = plt.subplots(figsize=(11, 4.5))                     # canvas
ax.scatter(range(len(s)), s["true_per_frame"], c=col, s=16)   # each embryo one dot
ax.set_yscale("log"); ax.set_ylabel("TRUE cells / frame (estN / n_frames)")
ax.set_xlabel("199 embryos, sorted early → late")
fig.savefig(f"{ASSETS}/da03_stages.png", dpi=110, bbox_inches="tight", facecolor="white")
print("saved da03_stages.png")""",
         image="learning/assets/da03_stages.png\nReal true-density per embryo (log scale). early→late gradient = gastrula→segmentation; red 44b6 = late/dense, blue 6bba = early/sparse — the embryo↔stage confound."),

    dict(note="""**[Recap]** True density spans ~38→1015/frame (gastrula→segmentation), annotation
density is the inverse of biology, and stage is confounded with embryo — so validation must
stratify by stage. **Next → pt01: the conv block**, where we start building the detector."""),
]

if __name__ == "__main__":
    build_lesson(META, CELLS, Path(__file__).with_suffix(".learning"),
                 {"TRAIN": TRAIN, "ASSETS": ASSETS})
