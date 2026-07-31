"""Fast distilled 3D nucleus detector (student) — Kaggle-feasible.

Works on the isotropic XY-downsampled grid (Z, Y//4, X//4) ~ (64,64,64) — same preprocessing as the
classical pipeline. A small 3D U-Net predicts a nucleus heatmap (gaussian blobs at centroids); peaks
via local-max → centroids mapped back to full XY res. Trained on Cellpose pseudo-labels (+ sparse GT).
Tiny & fast (~0.1s/frame) so the full hidden test fits the Kaggle 12h budget.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, maximum_filter

ROOT = Path(__file__).resolve().parents[1]
XY_DS = 4
SCALE = np.array([1.625, 0.40625, 0.40625])


# ---------------- model ----------------
def conv_block(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.InstanceNorm3d(co), nn.ReLU(inplace=True),
                         nn.Conv3d(co, co, 3, padding=1), nn.InstanceNorm3d(co), nn.ReLU(inplace=True))


class UNet3D(nn.Module):
    """Small 3D U-Net, base=16, 2 downsamples → light enough for Kaggle GPU."""
    def __init__(self, base=16):
        super().__init__()
        self.e1 = conv_block(1, base)
        self.e2 = conv_block(base, base * 2)
        self.e3 = conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool3d(2)
        self.u2 = nn.ConvTranspose3d(base * 4, base * 2, 2, stride=2)
        self.d2 = conv_block(base * 4, base * 2)
        self.u1 = nn.ConvTranspose3d(base * 2, base, 2, stride=2)
        self.d1 = conv_block(base * 2, base)
        self.head = nn.Conv3d(base, 1, 1)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1)); e3 = self.e3(self.pool(e2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)  # logits


# ---------------- preprocessing ----------------
def block_mean_xy(vol, f=XY_DS):
    Z, Y, X = vol.shape
    Y2, X2 = (Y // f) * f, (X // f) * f
    return vol[:, :Y2, :X2].astype(np.float32).reshape(Z, Y2 // f, f, X2 // f, f).mean(axis=(2, 4))


def normalize(g):
    lo, hi = np.percentile(g, 1), np.percentile(g, 99.8)
    return np.clip((g - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


def make_heatmap(shape_ds, centroids_ds, sigma=1.0):
    """Gaussian heatmap on the downsampled grid from centroids (in ds coords)."""
    h = np.zeros(shape_ds, np.float32)
    for z, y, x in centroids_ds:
        zi, yi, xi = int(round(z)), int(round(y)), int(round(x))
        if 0 <= zi < shape_ds[0] and 0 <= yi < shape_ds[1] and 0 <= xi < shape_ds[2]:
            h[zi, yi, xi] = 1.0
    if h.max() > 0:
        h = gaussian_filter(h, sigma=sigma)
        h /= h.max()
    return h


def to_ds_coords(centroids_full):
    """full (z,y,x) voxel -> downsampled grid coords (z same, y/x //XY_DS)."""
    c = np.asarray(centroids_full, float).copy()
    c[:, 1] /= XY_DS; c[:, 2] /= XY_DS
    return c


# ---------------- inference ----------------
def detect(model, vol, device, min_dist=2, thresh=0.3):
    """vol (Z,Y,X) -> centroids (N,3) in FULL voxel coords."""
    g = normalize(block_mean_xy(vol))
    with torch.no_grad():
        x = torch.from_numpy(g)[None, None].to(device)
        hm = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    size = 2 * min_dist + 1
    mx = maximum_filter(hm, size=size, mode="nearest")
    pk = np.argwhere((hm >= mx) & (hm > thresh)).astype(np.float32)
    if len(pk) == 0:
        return np.zeros((0, 3), np.int32)
    pk[:, 1] = pk[:, 1] * XY_DS + (XY_DS - 1) / 2
    pk[:, 2] = pk[:, 2] * XY_DS + (XY_DS - 1) / 2
    return np.rint(pk).astype(np.int32)


if __name__ == "__main__":
    # --dry-run: synthetic volume + fake centroids → 1 train step + 1 detect, validate shapes/speed
    if "--dry-run" in sys.argv:
        import time
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model = UNet3D(base=16).to(dev)
        Z, Y, X = 64, 256, 256
        vol = np.random.rand(Z, Y, X).astype(np.float32) * 100
        cents = np.column_stack([np.random.randint(0, Z, 40), np.random.randint(0, Y, 40), np.random.randint(0, X, 40)])
        g = normalize(block_mean_xy(vol)); hm = make_heatmap(g.shape, to_ds_coords(cents))
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        xb = torch.from_numpy(g)[None, None].to(dev); yb = torch.from_numpy(hm)[None, None].to(dev)
        t0 = time.time()
        for _ in range(3):
            opt.zero_grad(); loss = F.binary_cross_entropy_with_logits(model(xb), yb); loss.backward(); opt.step()
        t_train = time.time() - t0
        model.eval(); t0 = time.time(); det = detect(model, vol, dev); t_inf = time.time() - t0
        params = sum(p.numel() for p in model.parameters())
        print(f"[DRY-RUN OK] device={dev} params={params/1e6:.2f}M loss={loss.item():.4f} "
              f"3-step train {t_train:.2f}s | detect {t_inf*1000:.0f}ms/frame ({t_inf*19900/3600:.2f}h for 19900 frames) | {len(det)} centroids")
    else:
        print("use --dry-run, or import for training (trainer in experiments/train_student.py)")
