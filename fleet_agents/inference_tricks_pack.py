"""inference_tricks_pack — the Kaggle-grandmaster INFERENCE/DETECTION primitives the fleet was MISSING
(found scanning 179 winner repos + the ZFTurbo canonical WBF used by the Lyft 1st-place & ~15 detection
winners). All pure numpy/torch, offline-verified, CompConfig-agnostic. These are the fuse/TTA/average
*primitives* — deliberately NOT duplicating existing agents:

  • wbf-fusion       — Weighted Boxes Fusion: cluster boxes/points from multiple models/TTA by IoU (boxes) or
                       distance (points) and CONFIDENCE-weighted-average their coords + rescale confidence.
                       POINT variant fuses cell-detection candidates from several detectors (our lever).
                       Differs from `ensemble` (averages model PROBS/logits, not spatial detections),
                       `blend-optimize` (searches blend weights for a scalar metric, no clustering),
                       and `mh-ilp` (picks ILP-optimal candidate SUBSET across time — a tracker, not a
                       per-frame coordinate fuser). WBF is greedy per-frame geometric averaging.
  • snapshot-average — average logits/probs across N model/snapshot/seed outputs (+ optional per-model
                       weights, temperature, and a RANK-average option robust to miscalibrated scales).
                       Differs from `ensemble` which is the full biohub-pipeline ensembler; this is the raw
                       reusable array reducer (softmax/rank/weight) any comp can call.
  • multi-tta        — apply a set of INVERTIBLE transforms (flip/rot90/scale) to an input, run a predict_fn,
                       invert each prediction back to canonical frame, and fuse (mean or WBF). Generic over
                       2D & 3D arrays. Differs from `aug-find`/`aug-ablation` (TRAIN-time augmentation search)
                       — this is TEST-time augmentation with inversion, a different phase entirely.
  • bn-recalibrate   — recompute BatchNorm running stats (update_bn) after weight-averaging (SWA/EMA/soup);
                       a merged model has stale BN stats → this forward-passes data to fix them. No existing
                       agent touches BN stats (checkpoint-merger/quantize merge/compress WEIGHTS only).
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ================================================================ Weighted Boxes Fusion (WBF)
def _iou_matrix(boxes, box):
    """IoU of every row in `boxes` (N,4 as x1,y1,x2,y2) against a single `box` (4,)."""
    xA = np.maximum(boxes[:, 0], box[0]); yA = np.maximum(boxes[:, 1], box[1])
    xB = np.minimum(boxes[:, 2], box[2]); yB = np.minimum(boxes[:, 3], box[3])
    inter = np.maximum(xB - xA, 0) * np.maximum(yB - yA, 0)
    aA = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    aB = (box[2] - box[0]) * (box[3] - box[1])
    return inter / (aA + aB - inter + 1e-12)


def weighted_boxes_fusion(boxes_list, scores_list, weights=None, iou_thr=0.55,
                          skip_thr=0.0, conf_type="avg"):
    """Canonical WBF (ZFTurbo) for a SINGLE class/frame. Fuse boxes from several models/TTA.

    boxes_list  : list (per model) of (Ni,4) arrays, order x1,y1,x2,y2 (any consistent scale).
    scores_list : list (per model) of (Ni,) confidences.
    weights     : per-model weight (default all 1).
    iou_thr     : IoU above which two boxes join the same cluster.
    conf_type   : 'avg' (mean conf, rescaled by cluster support) or 'max'.
    Returns fused (M,4) boxes, (M,) scores, sorted by descending score.
    """
    n_models = len(boxes_list)
    if weights is None:
        weights = np.ones(n_models)
    weights = np.asarray(weights, float)
    # flat table: [score*w, w, model_idx, x1,y1,x2,y2]  (score already weight-scaled, canonical WBF)
    rows = []
    for t in range(n_models):
        b = np.asarray(boxes_list[t], float).reshape(-1, 4)
        s = np.asarray(scores_list[t], float).reshape(-1)
        for j in range(len(b)):
            if s[j] < skip_thr:
                continue
            x1, y1, x2, y2 = b[j]
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            rows.append([s[j] * weights[t], weights[t], t, x1, y1, x2, y2])
    if not rows:
        return np.zeros((0, 4)), np.zeros((0,))
    rows = np.array(rows, float)
    rows = rows[rows[:, 0].argsort()[::-1]]                 # descending confidence

    clusters = []            # each: list of member rows
    fused = np.zeros((0, 7))  # running fused box per cluster (same 7-col layout)
    for r in rows:
        if fused.shape[0]:
            ious = _iou_matrix(fused[:, 3:], r[3:])
            k = int(np.argmax(ious)); best = ious[k]
        else:
            best = 0.0; k = -1
        if k != -1 and best > iou_thr:
            clusters[k].append(r)
            fused[k] = _fuse_cluster(clusters[k], conf_type)
        else:
            clusters.append([r.copy()])
            fused = np.vstack([fused, r.copy()])
    # rescale confidence by cluster support (min(n_models, cluster_size)/sum weights)
    out = fused.copy()
    for i, c in enumerate(clusters):
        c = np.array(c)
        if conf_type == "max":
            out[i, 0] = out[i, 0] / weights.max()
        else:
            out[i, 0] = out[i, 0] * min(n_models, len(c)) / weights.sum()
    out = out[out[:, 0].argsort()[::-1]]
    return out[:, 3:], out[:, 0]


def _fuse_cluster(members, conf_type):
    """Confidence-weighted average of a cluster's boxes → a single 7-col fused row."""
    m = np.array(members, float)
    conf = m[:, 0]                                          # already weight-scaled score
    box = np.zeros(7)
    box[3:] = (conf[:, None] * m[:, 3:]).sum(0) / (conf.sum() + 1e-12)   # coord = conf-weighted mean
    box[0] = conf.max() if conf_type == "max" else conf.sum() / len(m)  # score (pre-rescale)
    box[1] = m[:, 1].sum(); box[2] = -1
    return box


def weighted_points_fusion(points_list, scores_list, weights=None, dist_thr=5.0,
                           conf_type="avg"):
    """POINT variant of WBF — fuse point detections (e.g. cell centroids) from several detectors/TTA.

    Cluster points whose Euclidean distance to a cluster's fused centroid is < dist_thr; the fused
    coordinate is the CONFIDENCE-weighted mean, confidence rescaled by support like box-WBF. Works for
    any point dimension (2D/3D). Returns fused (M,D) points, (M,) scores sorted by descending score.
    """
    n_models = len(points_list)
    if weights is None:
        weights = np.ones(n_models)
    weights = np.asarray(weights, float)
    rows = []                                               # [score*w, w, model_idx, *coords]
    D = None
    for t in range(n_models):
        s = np.asarray(scores_list[t], float).reshape(-1)
        p = np.asarray(points_list[t], float).reshape(len(s), -1) if len(s) else np.zeros((0, 0))
        if p.size:
            D = p.shape[1]
        for j in range(len(s)):
            rows.append([s[j] * weights[t], weights[t], t, *p[j]])
    if not rows or D is None:
        return np.zeros((0, D or 1)), np.zeros((0,))
    rows = np.array(rows, float)
    rows = rows[rows[:, 0].argsort()[::-1]]

    clusters = []
    cen = np.zeros((0, D))                                  # fused centroid per cluster
    conf_pre = []                                           # pre-rescale score per cluster
    for r in rows:
        coord = r[3:]
        if cen.shape[0]:
            d = np.linalg.norm(cen - coord, axis=1)
            k = int(np.argmin(d)); best = d[k]
        else:
            best = np.inf; k = -1
        if k != -1 and best < dist_thr:
            clusters[k].append(r)
            m = np.array(clusters[k]); w = m[:, 0]
            cen[k] = (w[:, None] * m[:, 3:]).sum(0) / (w.sum() + 1e-12)
            conf_pre[k] = w.max() if conf_type == "max" else w.sum() / len(m)
        else:
            clusters.append([r.copy()])
            cen = np.vstack([cen, coord[None]])
            conf_pre.append(r[0])
    scores = []
    for i, c in enumerate(clusters):
        if conf_type == "max":
            scores.append(conf_pre[i] / weights.max())
        else:
            scores.append(conf_pre[i] * min(n_models, len(c)) / weights.sum())
    scores = np.array(scores)
    order = scores.argsort()[::-1]
    return cen[order], scores[order]


# ================================================================ snapshot / seed ensemble averaging
def _softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=axis, keepdims=True) + 1e-12)


def snapshot_average(outputs, weights=None, mode="prob", temperature=1.0, softmax_axis=-1):
    """Average N model/snapshot/seed outputs into one prediction.

    outputs : list of N arrays, all same shape.
    mode    : 'prob'  — plain weighted mean of the arrays (probs already normalised);
              'logit' — temperature-scale then softmax each, then weighted-mean the probs;
              'rank'  — average per-model RANKS (robust to miscalibrated scales; monotone, in [0,1]).
    weights : per-model weight (default equal).
    """
    arrs = [np.asarray(o, float) for o in outputs]
    n = len(arrs)
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    w = w / (w.sum() + 1e-12)
    if mode == "logit":
        probs = [_softmax(a / max(temperature, 1e-6), axis=softmax_axis) for a in arrs]
        return sum(wi * p for wi, p in zip(w, probs))
    if mode == "rank":
        ranks = []
        for a in arrs:
            flat = a.ravel()
            r = flat.argsort().argsort().astype(float)      # 0..K-1 rank
            r = r / (len(flat) - 1) if len(flat) > 1 else np.zeros_like(r)
            ranks.append(r.reshape(a.shape))
        return sum(wi * r for wi, r in zip(w, ranks))
    return sum(wi * a for wi, a in zip(w, arrs))             # 'prob'


# ================================================================ multi-transform TTA
def _flip(a, axis):
    return np.flip(a, axis=axis)


def tta_transforms_2d(flips=(None, (-2,), (-1,), (-2, -1)), rots=(0,)):
    """Build a list of (name, fwd, inv) invertible transforms over the LAST TWO spatial axes.
    fwd/inv act on an array whose last two dims are spatial (channels/leading dims preserved)."""
    tfms = []
    for f in flips:
        for k in rots:
            name = f"flip{f}_rot{k}"
            def fwd(a, f=f, k=k):
                out = a if f is None else np.flip(a, axis=list(f))
                return out if k == 0 else np.rot90(out, k=k, axes=(-2, -1))
            def inv(a, f=f, k=k):
                out = a if k == 0 else np.rot90(a, k=-k, axes=(-2, -1))
                return out if f is None else np.flip(out, axis=list(f))
            tfms.append((name, fwd, inv))
    return tfms


def multi_tta(x, predict_fn, transforms=None, fuse="mean"):
    """Test-time augmentation: for each invertible transform apply it to x, predict, invert the
    prediction back to canonical frame, then fuse. `predict_fn(arr)->arr` (same spatial shape).
    fuse='mean' averages the inverted predictions. Returns the fused prediction."""
    x = np.asarray(x, float)
    if transforms is None:
        transforms = tta_transforms_2d()
    preds = []
    for _name, fwd, inv in transforms:
        y = predict_fn(fwd(x))
        preds.append(np.asarray(inv(y), float))
    if fuse == "mean":
        return np.mean(preds, axis=0)
    return snapshot_average(preds, mode="prob")


# ================================================================ BN recalibration (SWA / averaged models)
def update_bn(loader, model, device=None, n_batches=None):
    """Recompute BatchNorm running_mean/running_var of `model` by forward-passing `loader` batches
    (the SWA/soup fix: an averaged/merged model has stale BN stats). torch-only; no grad, no labels used.

    loader : iterable yielding either a tensor batch or a (input, ...) tuple.
    Returns the number of batches consumed.
    """
    import torch
    bn_layers = [m for m in model.modules()
                 if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bn_layers:
        return 0
    momenta = {}
    for bn in bn_layers:
        bn.reset_running_stats()
        momenta[bn] = bn.momentum
        bn.momentum = None                                  # cumulative moving average
    was_training = model.training
    model.train()
    seen = 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if n_batches is not None and i >= n_batches:
                break
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            if device is not None:
                x = x.to(device)
            model(x)
            seen += 1
    for bn in bn_layers:
        bn.momentum = momenta[bn]
    model.train(was_training)
    return seen


# ================================================================ agents
class _B(BaseAgent):
    thread = "S"; kind = "finding"


class WbfFusion(_B):
    name = "wbf-fusion"
    def run(self, q, worker):
        s = self.spec(q)
        mode = s.get("mode", "box")
        need = ("points_list", "scores_list") if mode == "point" else ("boxes_list", "scores_list")
        missing = [k for k in need if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"wbf-fusion needs spec keys {missing} — none provided")
        if mode == "point":
            pts, sc = weighted_points_fusion(s["points_list"], s["scores_list"],
                                             s.get("weights"), float(s.get("dist_thr", 5.0)),
                                             s.get("conf_type", "avg"))
            msg = (f"wbf-fusion(point): fused {sum(len(np.atleast_1d(x)) for x in s['scores_list'])} "
                   f"candidate points from {len(s['points_list'])} detectors → {len(sc)} consensus points")
            self.log(msg, kind="finding", recommendation="fuse multi-detector cell centroids; raise dist_thr to merge more")
            return self.done({"points": pts.tolist(), "scores": sc.tolist()}, msg)
        boxes, sc = weighted_boxes_fusion(s["boxes_list"], s["scores_list"], s.get("weights"),
                                          float(s.get("iou_thr", 0.55)), float(s.get("skip_thr", 0.0)),
                                          s.get("conf_type", "avg"))
        msg = f"wbf-fusion(box): fused {len(s['boxes_list'])} models → {len(sc)} boxes (iou_thr={s.get('iou_thr',0.55)})"
        self.log(msg, kind="finding", recommendation="WBF beats NMS for multi-model detection ensembling (Lyft 1st)")
        return self.done({"boxes": boxes.tolist(), "scores": sc.tolist()}, msg)


class SnapshotAverage(_B):
    name = "snapshot-average"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("outputs",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"snapshot-average needs spec keys {missing} — none provided")
        out = snapshot_average(s["outputs"], s.get("weights"), s.get("mode", "prob"),
                               float(s.get("temperature", 1.0)), int(s.get("softmax_axis", -1)))
        msg = f"snapshot-average: fused {len(s['outputs'])} outputs (mode={s.get('mode','prob')})"
        self.log(msg, kind="finding", recommendation="rank-average when models are miscalibrated; logit when raw logits")
        return self.done({"averaged": np.asarray(out).tolist()}, msg)


class MultiTta(_B):
    name = "multi-tta"
    def run(self, q, worker):
        # agent path: identity-predict smoke to prove the transform round-trip (real predict_fn is in-proc only)
        s = self.spec(q)
        missing = [k for k in ("x",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"multi-tta needs spec keys {missing} — none provided")
        x = np.asarray(s["x"], float)
        tfms = tta_transforms_2d()
        out = multi_tta(x, lambda a: a, tfms, s.get("fuse", "mean"))
        err = float(np.max(np.abs(out - x)))
        msg = f"multi-tta: {len(tfms)} invertible transforms, identity round-trip max-err={err:.2e}"
        self.log(msg, kind="finding", recommendation="pass an in-proc predict_fn; invert then fuse (mean/WBF)")
        return self.done({"round_trip_err": err, "n_transforms": len(tfms)}, msg)


_WBF = WbfFusion(); _SNAP = SnapshotAverage(); _TTA = MultiTta()


def run_wbf(q, worker): return _WBF.run(q, worker)
def run_snapshot(q, worker): return _SNAP.run(q, worker)
def run_tta(q, worker): return _TTA.run(q, worker)
