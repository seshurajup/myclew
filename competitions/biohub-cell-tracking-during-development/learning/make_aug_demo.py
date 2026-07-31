"""Show the augmentations ON OUR REAL DATA — before/after on a real frame of a real
embryo, so the lessons demonstrate (not just describe) why each choice is made.
Saves aug_flips.png (aug01) and aug_brightness.png (aug02) to learning/assets/."""
from pathlib import Path
import numpy as np, zarr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TRAIN = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/"
             "input/biohub-cell-tracking-during-development/train")
OUT = Path(__file__).parent / "assets"; OUT.mkdir(exist_ok=True)
DS = "6bba_09961292"   # the densest embryo — lots of visible nuclei

z = zarr.open(str(TRAIN / f"{DS}.zarr" / "0"), mode="r")        # (T, Z, Y, X)
frame = np.asarray(z[50]).astype(np.float32)                    # a mid-movie frame
mip = frame.max(axis=0)                                         # max-project over Z -> 2-D
lo, hi = np.quantile(mip, 0.5), np.quantile(mip, 0.999)         # robust display range
disp = np.clip((mip - lo) / (hi - lo + 1e-6), 0, 1)            # normalised 0..1 for viewing


def panel(fig_path, tiles, sup):
    n = len(tiles)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6))
    for ax, (title, img) in zip(axes, tiles):
        ax.imshow(img, cmap="magma", vmin=0, vmax=1)
        ax.set_title(title, fontsize=12, color="#1c2127")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(sup, fontsize=13, fontweight="bold", color="#1c2127", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=115, bbox_inches="tight", facecolor="white")
    print("saved", fig_path)


# aug01 — flips (the real flip_augment: reverse spatial axes)
panel(OUT / "aug_flips.png",
      [("original frame", disp),
       ("flip X  (dims=[X])", disp[:, ::-1]),
       ("flip Y  (dims=[Y])", disp[::-1, :])],
      f"flip_augment on a real frame ({DS}, t=50, Z-max-projection) — a flipped nucleus is still a valid nucleus")

# aug02 — brightness jitter (the real brightness_augment: scale intensities)
panel(OUT / "aug_brightness.png",
      [("brightness ×0.7 (dim)", np.clip(disp * 0.7, 0, 1)),
       ("original frame", disp),
       ("brightness ×1.3 (bright)", np.clip(disp * 1.3, 0, 1))],
      f"brightness_augment — mimics the real 7→{int(hi)} intensity spread across embryos/stages")
