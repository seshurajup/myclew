"""Generate a clean U-Net-shape diagram for pt02, using OUR real tensor shapes
(copyright-clean, matched to the lesson). Saves to learning/assets/unet_shape.png."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).parent / "assets" / "unet_shape.png"
OUT.parent.mkdir(exist_ok=True)

# (label, x, y, width, height, colour) — encoder down-left, decoder up-right
IND = "#4f46e5"; BOT = "#7c5cff"; DEC = "#2a9d8f"
enc = [("input\n1×64³", 0.5, 5.2, 1.5, 0.8, "#c9ccff"),
       ("32×64³", 0.5, 4.0, 1.5, 0.8, IND),
       ("64×32³", 2.6, 2.7, 1.4, 0.7, IND),
       ("128×16³", 4.6, 1.4, 1.3, 0.6, IND)]
dec = [("64×32³", 7.1, 2.7, 1.4, 0.7, DEC),
       ("32×64³", 9.0, 4.0, 1.5, 0.8, DEC),
       ("head\n1×64³", 9.0, 5.2, 1.5, 0.8, "#95d5c8")]

fig, ax = plt.subplots(figsize=(11, 5.2))
ax.set_xlim(0, 11.5); ax.set_ylim(0.8, 6.6); ax.axis("off")

def box(label, x, y, w, h, c, txtc="#fff"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                 fc=c, ec="none"))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", color=txtc,
            fontsize=10, fontweight="bold")

for b in enc:
    box(*b, txtc="#1c2127" if b[0].startswith("input") else "#fff")
box(*dec[0]); box(*dec[1])
box(*dec[2], txtc="#1c2127")
# bottleneck
box("128×16³\nbottleneck", 6.0, 0.95, 1.6, 0.7, BOT)

def arrow(x1, y1, x2, y2, c="#9aa3af", style="-|>", lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                 color=c, lw=lw, linestyle=ls))

# down (maxpool) path
arrow(1.25, 4.0, 3.3, 3.4); arrow(3.3, 2.7, 5.25, 2.0); arrow(5.9, 1.4, 6.6, 1.3)
# up (upsample) path
arrow(7.6, 1.35, 7.8, 2.7); arrow(8.5, 3.0, 9.75, 4.0); arrow(9.75, 4.8, 9.75, 5.2)
# skip connections (dashed, the torch.cat)
arrow(2.0, 4.35, 9.0, 4.35, c="#e76f51", style="-|>", lw=1.6, ls=(0, (4, 3)))
arrow(4.0, 3.05, 7.1, 3.05, c="#e76f51", style="-|>", lw=1.6, ls=(0, (4, 3)))

ax.text(2.4, 3.75, "MaxPool3d ↓  halve space", fontsize=9, color="#6b7480", rotation=-18)
ax.text(8.0, 3.75, "Upsample ↑  restore", fontsize=9, color="#6b7480", rotation=20)
ax.text(5.5, 4.55, "torch.cat  skip (keeps fine detail)", fontsize=9, color="#e76f51", fontweight="bold")
ax.text(5.75, 6.25, "TemporalUNet3D — the detector's shape", fontsize=14, fontweight="bold",
        color="#1c2127", ha="center")
ax.text(5.75, 0.85, "encoder: space ↓, channels ↑   ·   decoder: space ↑ back to full-res per-voxel heat-map",
        fontsize=9.5, color="#6b7480", ha="center")

fig.tight_layout()
fig.savefig(OUT, dpi=130, bbox_inches="tight", facecolor="white")
print(f"saved -> {OUT}")
