#!/usr/bin/env python3
"""xai_render — turn saved XAI saliency arrays (results/xai/*.npy) into VIEWABLE PNG heatmaps on demand.
The agent saves .npy (source of truth); this renders any of them to a Z-max-projected 'hot' heatmap so
you can SEE what the model attends to. Usage: python scripts/xai_render.py [results/xai]"""
import sys, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = sys.argv[1] if len(sys.argv) > 1 else "results/xai"
npys = sorted(glob.glob(os.path.join(d, "*.npy")))
made = []
for f in npys:
    a = np.load(f).astype("float32")
    if a.ndim == 3:
        a = a.max(0)                      # Z max-projection → 2D
    if a.ndim != 2:
        continue
    a = (a - a.min()) / (np.ptp(a) + 1e-8)
    png = f[:-4] + ".png"
    plt.figure(figsize=(3.2, 3.2)); plt.imshow(a, cmap="hot"); plt.colorbar(fraction=0.046)
    plt.title(os.path.basename(f)[:-4], fontsize=8); plt.axis("off"); plt.tight_layout()
    plt.savefig(png, dpi=100, bbox_inches="tight"); plt.close()
    made.append(png)
print(f"rendered {len(made)} PNG(s) from {len(npys)} npy in {d}")
for m in made: print(" ", m)
