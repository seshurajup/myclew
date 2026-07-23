"""volumetric_detection_pack — the heatmap DETECTION encode→tile→decode primitives distilled from two
3D-volume winners (CZII CryoET Object-ID 2nd-place particle picking; PhysioNet ECG-digitization 5th-place
heatmap keypoints). biohub's node-recall lever IS detection, yet the fleet had a UNet/ILP tracker but no
reusable, model-agnostic N-D heatmap detection toolkit. These three add exactly that — all pure numpy +
scipy.ndimage, so they run offline on CPU and are competition-agnostic (works in 2D or 3D):

  • gaussian-heatmap-encoder   — turn (x,y[,z][,class]) keypoints into an N-D, multi-class windowed-Gaussian
                                 heatmap TRAINING TARGET (4-sigma support, per-point sigma). The "nearly-lossless
                                 codec" both winners regress against instead of hard one-hot voxels.
  • volumetric-patch-inference — cover a volume larger than the model's patch with MINIMAL-overlap (or fixed-
                                 overlap) N-D tiling, then STITCH the per-patch outputs back by overlap-averaging.
                                 The sliding-window scheduler that makes big-tomogram / big-embryo inference fit.
  • heatmap-peak-decoder       — decode an N-D probability heatmap into discrete detections two ways: (a) PEAK
                                 mode = local-max pooling + confidence threshold + greedy radius-NMS (ECG recipe),
                                 (b) BLOB mode = threshold + connected-components centroid + voxel-count size filter
                                 (CZII recipe), with optional local soft-argmax (DSNT) sub-pixel refinement.

All three are the exact levers for node recall: encode the target the detector should regress, run it over a
big volume, then decode centroids — the missing half of the fleet's detection stack.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage
from .base import BaseAgent


# ================================================================ minimal / fixed-overlap N-D tiling
def patch_starts(dim: int, patch: int, overlap: int = 0):
    """Start positions of 1-D patches covering [0,dim). overlap<=0 → MINIMAL overlap (ceil tiling, the CZII
    recipe); overlap>0 → at least that many voxels of overlap between neighbours (stride=patch-overlap)."""
    if dim <= patch:
        return [0]
    if overlap and overlap > 0:
        stride = max(1, patch - overlap)
        n = int(np.ceil((dim - patch) / stride)) + 1
    else:
        n = int(np.ceil(dim / patch))
    starts = []
    for i in range(max(1, n)):
        if n > 1 and overlap <= 0:
            step = (patch - (n * patch - dim) / (n - 1))       # minimal-overlap step
            pos = int(round(i * step))
        else:
            pos = int(i * max(1, patch - max(0, overlap)))
        pos = min(pos, dim - patch)                            # clamp last patch inside
        if pos not in starts:
            starts.append(pos)
    if starts[-1] != dim - patch:
        starts.append(dim - patch)
    return sorted(set(starts))


def tile_coords(shape, patch_sizes, overlaps=None):
    """N-D grid of patch start-coordinates covering `shape`. Returns list of start-tuples."""
    shape = tuple(int(x) for x in shape); patch_sizes = tuple(int(x) for x in patch_sizes)
    if overlaps is None:
        overlaps = (0,) * len(shape)
    axes = [patch_starts(shape[d], min(patch_sizes[d], shape[d]), overlaps[d]) for d in range(len(shape))]
    grid = [()]
    for a in axes:
        grid = [g + (s,) for g in grid for s in a]
    return grid


def stitch(full_shape, patches, coords, patch_sizes):
    """Reconstruct a full array by placing each patch at its coord and AVERAGING overlaps (accumulate+count).
    patches/`full_shape` may carry a leading channel dim: full_shape is spatial only, patch may be (C,*patch)."""
    patch_sizes = tuple(int(x) for x in patch_sizes)
    lead = patches[0].shape[:patches[0].ndim - len(patch_sizes)]     # channel dims, if any
    acc = np.zeros(lead + tuple(full_shape), np.float64)
    cnt = np.zeros(tuple(full_shape), np.float64)
    for p, c in zip(patches, coords):
        sl = tuple(slice(c[d], c[d] + patch_sizes[d]) for d in range(len(patch_sizes)))
        acc[(Ellipsis,) + sl] += np.asarray(p, np.float64)
        cnt[sl] += 1.0
    cnt = np.maximum(cnt, 1e-9)
    return acc / cnt


# ================================================================ Gaussian heatmap target codec
def gaussian_heatmap(shape, points, sigma, n_classes=1, four_sigma=4.0):
    """Encode keypoints into an (n_classes, *shape) windowed-Gaussian heatmap TARGET.
    points: (N, D) or (N, D+1) with a trailing class index (0..n_classes-1). sigma: scalar or per-axis."""
    shape = tuple(int(x) for x in shape); D = len(shape)
    hm = np.zeros((n_classes,) + shape, np.float32)
    pts = np.atleast_2d(np.asarray(points, float))
    if pts.size == 0:
        return hm
    sig = np.broadcast_to(np.asarray(sigma, float).ravel(), (D,)) if np.ndim(sigma) else np.full(D, float(sigma))
    for row in pts:
        if len(row) == D + 1:
            coord = row[:D]; cls = int(row[D])
        else:
            coord = row[:D]; cls = 0
        cls = int(np.clip(cls, 0, n_classes - 1))
        lo = [max(0, int(np.floor(coord[d] - four_sigma * sig[d]))) for d in range(D)]
        hi = [min(shape[d], int(np.ceil(coord[d] + four_sigma * sig[d])) + 1) for d in range(D)]
        if any(hi[d] <= lo[d] for d in range(D)):
            continue
        grids = np.meshgrid(*[np.arange(lo[d], hi[d]) for d in range(D)], indexing="ij")
        g = np.zeros(grids[0].shape, np.float64)
        for d in range(D):
            g = g + ((grids[d] - coord[d]) ** 2) / (2.0 * sig[d] ** 2)
        patch = np.exp(-g)
        sl = tuple(slice(lo[d], hi[d]) for d in range(D))
        hm[(cls,) + sl] = np.maximum(hm[(cls,) + sl], patch.astype(np.float32))
    return hm


# ================================================================ peak / blob decoding
def _soft_argmax_local(hm, coord, radius):
    """DSNT-style sub-pixel refinement: intensity-weighted mean over a local window around an integer peak."""
    D = hm.ndim; lo = [max(0, coord[d] - radius) for d in range(D)]
    hi = [min(hm.shape[d], coord[d] + radius + 1) for d in range(D)]
    sl = tuple(slice(lo[d], hi[d]) for d in range(D)); w = hm[sl].astype(np.float64)
    s = w.sum()
    if s <= 0:
        return np.asarray(coord, float)
    grids = np.meshgrid(*[np.arange(lo[d], hi[d]) for d in range(D)], indexing="ij")
    return np.asarray([float((grids[d] * w).sum() / s) for d in range(D)])


def radius_nms(coords, confs, radius):
    """Greedy radius NMS: keep highest-conf point, suppress all within `radius` (euclidean). Returns keep idx."""
    coords = np.asarray(coords, float); confs = np.asarray(confs, float)
    order = np.argsort(-confs); keep = []
    taken = np.zeros(len(order), bool)
    pos = coords[order]
    for i in range(len(order)):
        if taken[i]:
            continue
        keep.append(int(order[i]))
        d = np.sqrt(((pos[i + 1:] - pos[i]) ** 2).sum(-1))
        taken[i + 1:][d < radius] = True
    return keep


def decode_peaks(heatmap, threshold=0.3, min_distance=2, mode="peak", size_min=0, subpixel=False):
    """Decode ONE N-D heatmap (no channel dim) into detections.
    Returns array (M, D+1): [coord..., confidence].
      peak mode  — local-max (maximum_filter) ≥ threshold, then greedy radius-NMS (radius=min_distance).
      blob mode  — binary(≥threshold) → connected components → intensity-weighted centroid, voxel-count≥size_min.
    """
    hm = np.asarray(heatmap, np.float64); D = hm.ndim
    if mode == "blob":
        lab, n = ndimage.label(hm >= threshold)
        if n == 0:
            return np.zeros((0, D + 1), float)
        out = []
        for k in range(1, n + 1):
            m = lab == k
            if int(m.sum()) < size_min:
                continue
            cen = ndimage.center_of_mass(hm, lab, k)       # intensity-weighted centroid
            conf = float(hm[m].max())
            out.append(list(cen) + [conf])
        return np.asarray(out, float) if out else np.zeros((0, D + 1), float)
    # peak mode
    size = 2 * int(min_distance) + 1
    mx = ndimage.maximum_filter(hm, size=size, mode="nearest")
    mask = (hm >= threshold) & (hm >= mx - 1e-9)
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return np.zeros((0, D + 1), float)
    confs = hm[tuple(idx.T)]
    keep = radius_nms(idx, confs, radius=max(1.0, float(min_distance)))
    out = []
    for i in keep:
        c = idx[i]
        coord = _soft_argmax_local(hm, c, int(min_distance)) if subpixel else c.astype(float)
        out.append(list(coord) + [float(confs[i])])
    return np.asarray(out, float)


def match_points(pred, gt, tol):
    """Greedy 1-1 nearest matching within `tol` → (tp, precision, recall). pred/gt are (N,D) coords."""
    pred = np.atleast_2d(np.asarray(pred, float)); gt = np.atleast_2d(np.asarray(gt, float))
    if gt.size == 0 or pred.size == 0:
        return 0, 0.0, 0.0
    used = np.zeros(len(gt), bool); tp = 0
    for p in pred:
        d = np.sqrt(((gt - p) ** 2).sum(-1)); d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True; tp += 1
    return tp, tp / max(1, len(pred)), tp / max(1, len(gt))


# ================================================================ agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


def _synth_volume(seed=0, shape=(24, 40, 40), n=12, sigma=1.6):
    rng = np.random.RandomState(seed)
    pts = np.stack([rng.uniform(4, shape[d] - 4, n) for d in range(len(shape))], 1)
    hm = gaussian_heatmap(shape, pts, sigma, n_classes=1)[0]
    hm = hm + rng.rand(*shape) * 0.05                         # background noise
    return hm, pts


class GaussianHeatmapEncoder(_B):
    name = "gaussian-heatmap-encoder"
    def run(self, q, worker):
        s = self.spec(q); shape = tuple(s.get("shape", (24, 40, 40)))
        sigma = float(s.get("sigma", 1.6)); nc = int(s.get("n_classes", 1))
        rng = np.random.RandomState(int(s.get("seed", 0)))
        pts = np.asarray(s["points"]) if "points" in s else np.stack(
            [rng.uniform(4, shape[d] - 4, int(s.get("n", 10))) for d in range(len(shape))], 1)
        hm = gaussian_heatmap(shape, pts, sigma, n_classes=nc)
        peak = float(hm.max()); occ = float((hm > 0).mean())
        msg = (f"gaussian-heatmap-encoder: encoded {len(np.atleast_2d(pts))} pts → {nc}×{shape} target, "
               f"peak={peak:.3f}, support={occ:.3%} (4σ-windowed Gaussian regression codec)")
        self.log(msg, kind="finding", recommendation="regress this soft target (BCE/MSE) instead of hard voxels")
        return self.done({"peak": peak, "support": occ, "n_points": int(len(np.atleast_2d(pts))),
                          "shape": list(shape)}, msg)


class VolumetricPatchInference(_B):
    name = "volumetric-patch-inference"
    def run(self, q, worker):
        s = self.spec(q); shape = tuple(s.get("shape", (30, 64, 64)))
        patch = tuple(s.get("patch", (16, 32, 32))); ov = s.get("overlap")
        overlaps = tuple(ov) if isinstance(ov, (list, tuple)) else ((int(ov),) * len(shape) if ov else None)
        coords = tile_coords(shape, patch, overlaps)
        rng = np.random.RandomState(int(s.get("seed", 0))); vol = rng.rand(*shape).astype(np.float32)
        patches = [vol[tuple(slice(c[d], c[d] + min(patch[d], shape[d])) for d in range(len(shape)))] for c in coords]
        recon = stitch(shape, patches, coords, [min(patch[d], shape[d]) for d in range(len(shape))])
        err = float(np.abs(recon - vol).max()); cov = float((stitch(shape, [np.ones_like(p) for p in patches],
                                                                    coords, [min(patch[d], shape[d]) for d in range(len(shape))]) > 0).mean())
        msg = (f"volumetric-patch-inference: {len(coords)} patches tile {shape} (patch {patch}); "
               f"coverage={cov:.3%}, stitch-recon-err={err:.2e} (overlap-averaged)")
        self.log(msg, kind="finding", recommendation="run the model per-patch, stitch by overlap-average, then decode")
        return self.done({"n_patches": len(coords), "coverage": cov, "recon_err": err, "coords": coords[:64]}, msg)


class HeatmapPeakDecoder(_B):
    name = "heatmap-peak-decoder"
    def run(self, q, worker):
        s = self.spec(q); mode = s.get("mode", "peak")
        thr = float(s.get("threshold", 0.3)); md = int(s.get("min_distance", 2))
        sub = bool(s.get("subpixel", True)); tol = float(s.get("match_tol", 2.0))
        if "heatmap" in s:
            hm = np.asarray(s["heatmap"], float); gt = np.atleast_2d(np.asarray(s.get("gt", []), float))
        else:
            hm, gt = _synth_volume(int(s.get("seed", 0)))
        det = decode_peaks(hm, threshold=thr, min_distance=md, mode=mode,
                           size_min=int(s.get("size_min", 0)), subpixel=sub)
        coords = det[:, :-1] if len(det) else det
        tp, prec, rec = match_points(coords, gt, tol) if gt.size else (len(det), 1.0, 1.0)
        msg = (f"heatmap-peak-decoder[{mode}]: {len(det)} detections, recall={rec:.3f} precision={prec:.3f} "
               f"@tol={tol} (local-max+radius-NMS{'+subpix' if sub else ''})")
        self.log(msg, kind="finding", recommendation="tune threshold/min_distance for the node-recall lever")
        return self.done({"n_det": int(len(det)), "recall": rec, "precision": prec, "tp": int(tp),
                          "mode": mode}, msg)


_ENC = GaussianHeatmapEncoder(); _PATCH = VolumetricPatchInference(); _DEC = HeatmapPeakDecoder()
_AGENT = _DEC                                                        # primary agent for the module


def run_encode(q, worker): return _ENC.run(q, worker)
def run_patch(q, worker): return _PATCH.run(q, worker)
def run_decode(q, worker): return _DEC.run(q, worker)
def run(q, worker): return _DEC.run(q, worker)
