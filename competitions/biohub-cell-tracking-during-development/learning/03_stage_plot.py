"""Step 03 figure — true per-frame density per dataset, stage bands, group color.

Shows every embryo as a dot at its TRUE cells/frame (estN / n_frames), sorted,
colored by group. Horizontal bands = developmental stages S0..S4. Makes the
group->stage confound and the annotation inversion visible at a glance.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt

DATA = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/"
            "input/biohub-cell-tracking-during-development/train")
OUT = Path(__file__).parent


def read_t(g):
    return np.asarray(zarr.open(str(g / "nodes/props/t/values"), mode="r")[:])


def read_estN(g):
    a = json.load(open(g / "zarr.json"))["attributes"]["geff"]
    return int(a.get("extra", {}).get("estimated_number_of_nodes", -1))


rows = []
for g in sorted(DATA.glob("*.geff")):
    t = read_t(g)
    nf = len(np.unique(t))
    rows.append({"group": g.name[:4], "true_cpf": read_estN(g) / nf,
                 "annot_cpf": len(t) / nf})
df = pd.DataFrame(rows).sort_values("true_cpf").reset_index(drop=True)

fig, ax = plt.subplots(figsize=(13, 6))
bands = [(38, 74, "S0"), (74, 145, "S1"), (145, 276, "S2"),
         (276, 527, "S3"), (527, 1015, "S4")]
for i, (lo, hi, name) in enumerate(bands):
    ax.axhspan(lo, hi, color="#f0f0f0" if i % 2 else "#e2e2e2", zorder=0)
    ax.text(len(df) * 0.995, np.sqrt(lo * hi), name, va="center", ha="right",
            fontsize=11, color="#555", fontweight="bold")

col = df["group"].map({"44b6": "#E15759", "6bba": "#4E79A7"})
ax.scatter(range(len(df)), df["true_cpf"], c=col, s=22, zorder=3,
           label="_", edgecolor="white", linewidth=0.3)
ax.set_yscale("log")
ax.set_ylabel("TRUE cells per frame  (estN / n_frames)", fontsize=12)
ax.set_xlabel("199 embryos, sorted early → late", fontsize=12)
ax.set_title("Real developmental density per embryo — stage from cell count\n"
             "red 44b6 = biologically LATE/dense · blue 6bba = EARLY/sparse "
             "(inverse of the label density!)", fontsize=12)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color="#E15759", label="44b6 (late / segmentation)"),
                   Patch(color="#4E79A7", label="6bba (early / gastrula)")],
          loc="upper left", fontsize=11)
ax.set_xlim(-2, len(df) + 1)
fig.tight_layout()
fig.savefig(OUT / "03_stage_plot.png", dpi=110)
print("saved 03_stage_plot.png")
