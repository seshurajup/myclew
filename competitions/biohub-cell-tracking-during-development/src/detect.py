"""Per-frame 3D cell detection (local-maxima + sub-voxel refinement).

Strategy (from public V2 sub-voxel notebook): XY block-mean by XY_DS to make the grid
~isotropic while keeping full Z, gaussian smooth, robust threshold, local-maxima peaks
(not connected components, which fuse touching cells), then refine each centroid as an
intensity-weighted centre of mass on the raw volume. Coordinates are returned in original
(z,y,x) voxel space.
"""
from __future__ import annotations
from typing import Tuple
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

try:
    from skimage.filters import threshold_otsu
except Exception:  # pragma: no cover
    threshold_otsu = None

from .config import Config


def block_mean_xy(vol: np.ndarray, factor: int) -> np.ndarray:
    Z, Y, X = vol.shape
    Y2, X2 = (Y // factor) * factor, (X // factor) * factor
    x = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return x.reshape(Z, Y2 // factor, factor, X2 // factor, factor).mean(axis=(2, 4))


def robust_threshold(sm: np.ndarray, thresh_rel: float) -> float:
    bg = float(np.median(sm))
    hi = float(np.percentile(sm, 99.8))
    rel = bg + thresh_rel * max(hi - bg, 1e-6)
    if threshold_otsu is not None:
        try:
            return max(float(threshold_otsu(sm)), rel)
        except Exception:
            return rel
    return rel


def refine_centroid(vol: np.ndarray, approx_zyx, rz: int, ryx: int) -> np.ndarray:
    Z, Y, X = vol.shape
    z, y, x = (int(round(v)) for v in approx_zyx)
    z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
    y0, y1 = max(0, y - ryx), min(Y, y + ryx + 1)
    x0, x1 = max(0, x - ryx), min(X, x + ryx + 1)
    crop = vol[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
    if crop.size == 0:
        return np.array([z, y, x], dtype=np.float64)
    bg = float(np.percentile(crop, 20.0))
    w = np.maximum(crop - bg, 0.0)
    tot = float(w.sum())
    if tot <= 1e-6:
        return np.array([z, y, x], dtype=np.float64)
    zz, yy, xx = np.indices(crop.shape)
    return np.array([
        z0 + float((zz * w).sum() / tot),
        y0 + float((yy * w).sum() / tot),
        x0 + float((xx * w).sum() / tot),
    ], dtype=np.float64)


def physical_nms(coords: np.ndarray, scores: np.ndarray, scale: np.ndarray, radius_um: float) -> np.ndarray:
    """Greedy NMS in physical space. Returns boolean keep-mask."""
    if radius_um <= 0 or len(coords) == 0:
        return np.ones(len(coords), dtype=bool)
    order = np.argsort(-scores)
    P = coords.astype(np.float64) * scale[None, :]
    keep = np.ones(len(coords), dtype=bool)
    taken = []
    for i in order:
        if not keep[i]:
            continue
        if taken:
            d = np.sqrt(((P[i] - np.array(taken)) ** 2).sum(axis=1))
            if (d < radius_um).any():
                keep[i] = False
                continue
        taken.append(P[i])
    return keep


def detect_cells(vol: np.ndarray, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Return (coords (N,3) int voxel z,y,x, scores (N,))."""
    Z, Y, X = vol.shape
    ds = block_mean_xy(vol, cfg.XY_DS)
    sm = gaussian_filter(ds, sigma=cfg.SMOOTH_SIGMA, mode="nearest")
    thr = robust_threshold(sm, cfg.THRESH_REL)

    size = 2 * cfg.MIN_PEAK_DIST + 1
    mx = maximum_filter(sm, size=(size, size, size), mode="nearest")
    mask = (sm >= mx) & (sm > thr)
    cds = np.argwhere(mask).astype(np.int32)
    if cds.size == 0:
        return np.empty((0, 3), np.int32), np.empty((0,), np.float32)
    scores = sm[cds[:, 0], cds[:, 1], cds[:, 2]].astype(np.float32)

    # map XY back to full resolution (Z kept full)
    approx = cds.astype(np.float64)
    approx[:, 1] = approx[:, 1] * cfg.XY_DS + (cfg.XY_DS - 1) / 2.0
    approx[:, 2] = approx[:, 2] * cfg.XY_DS + (cfg.XY_DS - 1) / 2.0

    if cfg.USE_SUBVOXEL:
        approx = np.array([refine_centroid(vol, p, cfg.REFINE_RZ, cfg.REFINE_RYX) for p in approx])

    coords = np.rint(approx).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, Z - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, Y - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, X - 1)

    if cfg.USE_BORDER_FILTER and cfg.BORDER_KEEP_QUANTILE > 0:
        m = cfg.BORDER_MARGIN_VOX
        near = ((coords[:, 0] < m) | (coords[:, 0] >= Z - m) |
                (coords[:, 1] < m) | (coords[:, 1] >= Y - m) |
                (coords[:, 2] < m) | (coords[:, 2] >= X - m))
        if near.any():
            cut = np.quantile(scores, cfg.BORDER_KEEP_QUANTILE)
            keep = ~(near & (scores < cut))
            coords, scores = coords[keep], scores[keep]

    if cfg.NMS_RADIUS_UM > 0:
        keep = physical_nms(coords, scores, np.asarray(cfg.SCALE), cfg.NMS_RADIUS_UM)
        coords, scores = coords[keep], scores[keep]

    return coords, scores
