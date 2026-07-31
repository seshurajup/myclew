"""Clean, copyright-clean concept diagrams for the domain lessons.
Saves division_event.png, tracking_graph.png, metric_match.png to learning/assets/."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle

OUT = Path(__file__).parent / "assets"
OUT.parent.mkdir(exist_ok=True); OUT.mkdir(exist_ok=True)
IND = "#4f46e5"; TEAL = "#2a9d8f"; ORANGE = "#e76f51"; INK = "#1c2127"; MUTE = "#6b7480"


def node(ax, x, y, c=IND, r=0.16, label=None, lc="#fff"):
    ax.add_patch(Circle((x, y), r, fc=c, ec="none", zorder=3))
    if label:
        ax.text(x, y, label, ha="center", va="center", color=lc, fontsize=9, fontweight="bold", zorder=4)


def edge(ax, x1, y1, x2, y2, c=MUTE, lw=2.2, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                 color=c, lw=lw, linestyle=ls, shrinkA=10, shrinkB=10, zorder=2))


def frame_labels(ax, xs, names, y=-0.15):
    for x, n in zip(xs, names):
        ax.axvline(x, color="#eef0f4", lw=1, zorder=0)
        ax.text(x, y, n, ha="center", va="top", color=MUTE, fontsize=10, fontweight="bold")


# ---------- 1. division event ----------
fig, ax = plt.subplots(figsize=(7.6, 4.2)); ax.set_xlim(0, 4); ax.set_ylim(-0.4, 3.4); ax.axis("off")
xs = [0.7, 2.0, 3.3]
frame_labels(ax, xs, ["t", "t+1", "t+2"], y=-0.2)
node(ax, xs[0], 1.7, IND, label="P")                     # parent
node(ax, xs[1], 2.4, TEAL, label="c1"); node(ax, xs[1], 1.0, TEAL, label="c2")  # 2 daughters
node(ax, xs[2], 2.6, TEAL); node(ax, xs[2], 0.8, TEAL)   # grandchildren (continue)
edge(ax, xs[0], 1.7, xs[1], 2.4, ORANGE); edge(ax, xs[0], 1.7, xs[1], 1.0, ORANGE)
edge(ax, xs[1], 2.4, xs[2], 2.6); edge(ax, xs[1], 1.0, xs[2], 0.8)
ax.text(2.0, 3.2, "A division = parent → 2 children that BOTH continue", ha="center",
        fontsize=12, fontweight="bold", color=INK)
ax.text(1.35, 1.85, "split", color=ORANGE, fontsize=9, fontweight="bold", rotation=25)
ax.text(2.0, 0.15, "the metric only counts it (TP) if the divider matches GT within 7µm\nAND both children have their own successor at t+2",
        ha="center", fontsize=8.5, color=MUTE)
fig.tight_layout(); fig.savefig(OUT / "division_event.png", dpi=130, facecolor="white"); plt.close(fig)

# ---------- 2. tracking graph ----------
fig, ax = plt.subplots(figsize=(8.2, 4.2)); ax.set_xlim(0, 5); ax.set_ylim(-0.4, 3.6); ax.axis("off")
xs = [0.6, 1.7, 2.8, 3.9]
frame_labels(ax, xs, ["t=0", "t=1", "t=2", "t=3"], y=-0.2)
# track A (top, steady)
ya = [2.9, 3.0, 2.8, 2.9]
for i in range(3): edge(ax, xs[i], ya[i], xs[i+1], ya[i+1], IND)
for i in range(4): node(ax, xs[i], ya[i], IND)
# track B divides at t=1 -> two lineages
node(ax, xs[0], 1.4, IND)
edge(ax, xs[0], 1.4, xs[1], 1.7, IND); node(ax, xs[1], 1.7, IND)
edge(ax, xs[1], 1.7, xs[2], 2.1, ORANGE); edge(ax, xs[1], 1.7, xs[2], 1.1, ORANGE)  # division
node(ax, xs[2], 2.1, TEAL); node(ax, xs[2], 1.1, TEAL)
edge(ax, xs[2], 2.1, xs[3], 2.2, TEAL); edge(ax, xs[2], 1.1, xs[3], 1.0, TEAL)
node(ax, xs[3], 2.2, TEAL); node(ax, xs[3], 1.0, TEAL)
# track C (bottom, appears late)
node(ax, xs[1], 0.4, IND); edge(ax, xs[1], 0.4, xs[2], 0.4, IND); node(ax, xs[2], 0.4, IND)
edge(ax, xs[2], 0.4, xs[3], 0.35, IND); node(ax, xs[3], 0.35, IND)
ax.text(2.5, 3.45, "The tracking graph — nodes = cells, edges = same cell next frame",
        ha="center", fontsize=12, fontweight="bold", color=INK)
ax.text(2.25, 1.55, "division", color=ORANGE, fontsize=8.5, fontweight="bold")
fig.tight_layout(); fig.savefig(OUT / "tracking_graph.png", dpi=130, facecolor="white"); plt.close(fig)

# ---------- 3. metric 7um match ----------
fig, ax = plt.subplots(figsize=(7.6, 4.0)); ax.set_xlim(0, 4); ax.set_ylim(0, 3.2); ax.axis("off")
# a matched pair (within 7um)
gt1 = (1.0, 2.2); pr1 = (1.35, 2.05)
ax.add_patch(Circle(gt1, 0.55, fc="none", ec=TEAL, lw=1.6, ls=(0, (4, 3)), zorder=1))
node(ax, *gt1, TEAL, r=0.13, label="GT"); node(ax, *pr1, IND, r=0.13, label="P")
ax.text(gt1[0]-0.05, gt1[1]+0.7, "≤ 7µm → MATCH ✓", color=TEAL, fontsize=10, fontweight="bold", ha="center")
# an unmatched prediction (too far)
gt2 = (3.0, 2.2); pr2 = (3.05, 0.9)
ax.add_patch(Circle(gt2, 0.55, fc="none", ec=TEAL, lw=1.6, ls=(0, (4, 3)), zorder=1))
node(ax, *gt2, TEAL, r=0.13, label="GT"); node(ax, *pr2, "#b0b6c0", r=0.13, label="P")
ax.text(3.0, 0.45, "> 7µm → no match", color=MUTE, fontsize=10, fontweight="bold", ha="center")
ax.text(2.0, 3.0, "Scoring: a predicted cell matches a GT cell if within 7µm",
        ha="center", fontsize=12, fontweight="bold", color=INK)
ax.text(2.0, 1.5, "an edge is a true positive only if BOTH its endpoints match",
        ha="center", fontsize=9, color=MUTE, style="italic")
fig.tight_layout(); fig.savefig(OUT / "metric_match.png", dpi=130, facecolor="white"); plt.close(fig)

print("saved:", *(p.name for p in sorted(OUT.glob("*.png"))))
