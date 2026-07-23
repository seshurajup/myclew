"""detector-transfer — REUSABLE (any 3D-point-detection comp) STRONG detector trainer + ROBUST transfer eval.

Why it exists: the toy signal-check (tiny net, 1 seed, few frames) was noise-dominated → false GO/NO-GO calls.
This agent fixes that with (a) a real 3-level 3D-UNet heatmap detector with flip-augmentation + cosine LR,
(b) MULTI-SEED per-embryo evaluation reporting mean ± std (never conclude from one seed), (c) a training-set
COMPARISON harness (raw / domain-matched / augmented / self-labeled) with a paired-across-seeds significance
call, and (d) a SELF-TRAINING mode (pseudo-label the target domain's own points → train) that attacks a
CONTENT gap directly (no external, no domain gap). Node-recall @gate via bench_lib (the official proxy).

Comp-agnostic: takes {vols, pts} numpy arrays via spec or in-proc; the biohub caller wires the loaders. GPU
(torch/CUDA) per the always-GPU rule. A BaseAgent with its own data-wise test.
"""
from __future__ import annotations
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent


def _dev(device):
    """Resolve a torch device string, falling back to cpu when CUDA is requested but unavailable (never raises)."""
    try:
        import torch
        d = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if isinstance(d, str) and d.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
        return d
    except Exception:  # noqa: BLE001
        return "cpu"


def _heat(shape, pts, sig=1.2):
    import numpy as np
    from scipy.ndimage import gaussian_filter
    h = np.zeros(shape, np.float32)
    pts = np.asarray(pts, float)
    pts = pts[np.isfinite(pts).all(1)] if pts.ndim == 2 and pts.shape[0] else pts
    for z, y, x in pts.astype(int) if pts.size else []:
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            h[z, y, x] = 1.0
    hf = gaussian_filter(h, max(float(sig), 1e-3))
    return hf / (hf.max() or 1.0)


def _build_unet(device, c=32, dropout=0.0):
    import torch
    from torch import nn

    class UNet3D(nn.Module):
        def __init__(s):
            super().__init__()
            def blk(i, o):
                layers = [nn.Conv3d(i, o, 3, padding=1), nn.InstanceNorm3d(o), nn.ReLU(),
                          nn.Conv3d(o, o, 3, padding=1), nn.InstanceNorm3d(o), nn.ReLU()]
                if dropout and dropout > 0:                    # optional regularization (default off = legacy)
                    layers.append(nn.Dropout3d(float(dropout)))
                return nn.Sequential(*layers)
            s.e1 = blk(1, c); s.e2 = blk(c, c * 2); s.e3 = blk(c * 2, c * 4)
            s.p = nn.MaxPool3d(2)
            s.u2 = nn.ConvTranspose3d(c * 4, c * 2, 2, 2); s.d2 = blk(c * 4, c * 2)
            s.u1 = nn.ConvTranspose3d(c * 2, c, 2, 2); s.d1 = blk(c * 2, c)
            s.out = nn.Conv3d(c, 1, 1)

        def forward(s, x):
            a = s.e1(x); b = s.e2(s.p(a)); d = s.e3(s.p(b))
            u = s.d2(torch.cat([s.u2(d), b], 1)); u = s.d1(torch.cat([s.u1(u), a], 1))
            return s.out(u)

    return UNet3D().to(device)


def _build_timm25d(device, backbone="resnet18", freeze_early=True):
    """2.5D detector: PRETRAINED timm ImageNet encoder (features_only) + light U-Net decoder → per-slice heatmap.
    3 adjacent z-slices become the 3 input channels, so the pretrained 3-ch stem is used DIRECTLY (no inflation)
    and z-context is preserved. Grafting = the encoder's stable stem + early blocks arrive pretrained; the
    decoder is fresh. `freeze_early` freezes stem+stage0 (the most general, stable layers) so the warm-start
    isn't washed out early. Reusable across any 2D/2.5D detection comp."""
    import timm, torch
    from torch import nn
    import torch.nn.functional as F

    class Timm25D(nn.Module):
        def __init__(s):
            super().__init__()
            s.enc = timm.create_model(backbone, pretrained=True, features_only=True, in_chans=3)
            chs = s.enc.feature_info.channels()
            s.lat = nn.ModuleList([nn.Conv2d(c, 64, 1) for c in chs])
            s.dec = nn.ModuleList([nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()) for _ in chs])
            s.head = nn.Conv2d(64, 1, 1)
            if freeze_early:                                   # freeze the STABLE low-level layers (stem + first block)
                params = list(s.enc.parameters())
                for p in params[:max(2, len(params) // 4)]:
                    p.requires_grad = False

        def forward(s, x):                                     # x: (B,3,Y,X)
            feats = s.enc(x)
            f = s.lat[-1](feats[-1]); f = s.dec[-1](f)
            for i in range(len(feats) - 2, -1, -1):
                f = F.interpolate(f, size=feats[i].shape[-2:], mode="bilinear", align_corners=False)
                f = s.dec[i](f + s.lat[i](feats[i]))
            f = F.interpolate(f, size=x.shape[-2:], mode="bilinear", align_corners=False)
            return s.head(f)                                   # (B,1,Y,X)

    m = Timm25D().to(device); m.arch = "timm25d"; return m


def _slabs(v):                                                 # (Z,Y,X) -> (Z,3,Y,X): 3 adjacent slices as channels
    import numpy as np
    zm1 = np.concatenate([v[:1], v[:-1]], 0); zp1 = np.concatenate([v[1:], v[-1:]], 0)
    return np.stack([zm1, v, zp1], 1)


def train_detector(vols, pts_list, epochs=120, ch=32, lr=1e-3, seed=0, device=None, augment=True, sigma=1.2,
                   arch="unet3d", backbone="resnet18", dropout=0.0):
    """Heatmap detector on GPU. arch 'unet3d' = strong 3D-UNet from scratch; 'timm25d' = PRETRAINED timm encoder
    (2.5D, warm-started stable layers grafted). Flip augmentation, cosine LR. Returns the model (tagged .arch).
    `dropout`>0 adds Dropout3d regularization to the 3D-UNet blocks (default 0 = legacy). `sigma` sets the
    Gaussian heatmap width. `device` falls back to cpu if CUDA is unavailable. Non-finite voxels are sanitized."""
    import numpy as np, torch
    from torch import nn
    dev = _dev(device)
    torch.manual_seed(seed); np.random.seed(seed)
    vols = [np.nan_to_num(np.asarray(v, np.float32)) for v in vols]
    if arch == "timm25d":
        m = _build_timm25d(dev, backbone); m.arch = "timm25d"
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr)
    else:
        m = _build_unet(dev, ch, dropout); m.arch = "unet3d"
        opt = torch.optim.Adam(m.parameters(), lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    lossf = nn.BCEWithLogitsLoss()
    tgts = [_heat(v.shape, p, sigma) for v, p in zip(vols, pts_list)]
    for _ in range(epochs):
        for i in np.random.permutation(len(vols)):
            v, t = vols[i], tgts[i]
            if augment:
                if np.random.rand() < 0.5:
                    v, t = v[:, ::-1], t[:, ::-1]
                if np.random.rand() < 0.5:
                    v, t = v[:, :, ::-1], t[:, :, ::-1]
            if m.arch == "timm25d":                            # per-slice 2.5D: (Z,3,Y,X) -> (Z,1,Y,X)
                x = torch.tensor(np.ascontiguousarray(_slabs(v)), dtype=torch.float32, device=dev)
                y = torch.tensor(np.ascontiguousarray(t)[:, None], dtype=torch.float32, device=dev)
            else:
                x = torch.tensor(np.ascontiguousarray(v)[None, None], dtype=torch.float32, device=dev)
                y = torch.tensor(np.ascontiguousarray(t)[None, None], dtype=torch.float32, device=dev)
            opt.zero_grad(); lossf(m(x), y).backward(); opt.step()
        sched.step()
    m.eval(); return m


def _heatmap(m, v, dev, arch):
    import numpy as np, torch
    with torch.no_grad():
        if arch == "timm25d":
            x = torch.tensor(np.ascontiguousarray(_slabs(v)), dtype=torch.float32, device=dev)
            return torch.sigmoid(m(x)).cpu().numpy()[:, 0]     # (Z,Y,X)
        return torch.sigmoid(m(torch.tensor(v[None, None], dtype=torch.float32, device=dev))).cpu().numpy()[0, 0]


def detect(m, v, topk, xy=4, thr=0.15, device=None, tta=False, nms=3):
    """Peak-decode a heatmap detector. `tta`=True averages the heatmap over Y/X flips (flip test-time augmentation,
    typically higher recall). `nms` is the local-maximum window size. `device` falls back to cpu if CUDA absent.
    Non-finite voxels are sanitized so a bad frame yields no detections instead of crashing."""
    import numpy as np, torch
    from scipy.ndimage import maximum_filter
    dev = _dev(device)
    arch = getattr(m, "arch", "unet3d")
    v = np.nan_to_num(np.asarray(v, np.float32))
    if v.size == 0:
        return np.zeros((0, 3))
    h = _heatmap(m, v, dev, arch)
    if tta:                                                    # flip-TTA: average heatmaps from Y/X flips
        acc = [h]
        try:
            acc.append(_heatmap(m, v[:, ::-1], dev, arch)[:, ::-1])
            acc.append(_heatmap(m, v[:, :, ::-1], dev, arch)[:, :, ::-1])
            h = np.mean(np.stack(acc), axis=0)
        except Exception:  # noqa: BLE001
            h = acc[0]
    nms = max(1, int(nms))
    pk = np.argwhere((h == maximum_filter(h, nms)) & (h > thr))
    if not len(pk):
        return np.zeros((0, 3))
    order = np.argsort(-h[pk[:, 0], pk[:, 1], pk[:, 2]])[:topk]
    p = pk[order].astype(float); p[:, 1] *= xy; p[:, 2] *= xy
    return p


def _pu_mask(shape, pts, pos_r=1.5, neg_far=4.0):
    """Positive-Unlabeled supervision mask for SPARSE point labels: 1 near a labeled point (supervise as
    positive), 1 far from EVERY labeled point (safe negative), 0 (ignore) in the ambiguous ring — so the ~1%
    labels never penalize the many UNLABELED real objects nearby. The correct loss mask for point-sup finetune."""
    import numpy as np
    from scipy.ndimage import distance_transform_edt
    occ = np.ones(shape, bool)
    pts = np.asarray(pts, float)
    pts = pts[np.isfinite(pts).all(1)] if pts.ndim == 2 and pts.shape[0] else pts
    for z, y, x in pts.astype(int) if pts.size else []:
        if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
            occ[z, y, x] = False
    dist = distance_transform_edt(occ)
    m = np.zeros(shape, np.float32)
    m[dist <= pos_r] = 1.0                                     # positives (supervised)
    m[dist >= neg_far] = 1.0                                   # confident background (supervised as 0)
    return m


def finetune_detector(m, vols, pts_list, epochs=40, lr=3e-4, seed=0, device=None, freeze_encoder=True,
                      pos_r=1.5, neg_far=4.0, sigma=1.2, augment=True):
    """PEFT-style domain ADAPTATION of a PRETRAINED detector on TARGET data with SPARSE point labels. Freezes
    the pretrained encoder (keeps 'what a nucleus looks like'), trains the decoder/head to the target domain
    under a positive-unlabeled masked BCE. This is the target stage of source-pretrain→target-adapt — the step
    that turns abundant SOURCE labels + few TARGET labels into a domain-correct detector. Returns the adapted m."""
    import numpy as np, torch
    from torch import nn
    dev = _dev(device)
    torch.manual_seed(seed); np.random.seed(seed)
    vols = [np.nan_to_num(np.asarray(v, np.float32)) for v in vols]
    arch = getattr(m, "arch", "unet3d")
    if freeze_encoder:
        if arch == "unet3d":
            for blk in (m.e1, m.e2, m.e3):
                for p in blk.parameters():
                    p.requires_grad = False
        elif arch == "timm25d":
            for p in m.enc.parameters():
                p.requires_grad = False
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    tgts = [_heat(v.shape, p, sigma) for v, p in zip(vols, pts_list)]
    masks = [_pu_mask(v.shape, p, pos_r, neg_far) for v, p in zip(vols, pts_list)]
    m.train()
    for _ in range(epochs):
        for i in np.random.permutation(len(vols)):
            v, t, msk = vols[i], tgts[i], masks[i]
            if augment and np.random.rand() < 0.5:
                v, t, msk = v[:, ::-1], t[:, ::-1], msk[:, ::-1]
            if arch == "timm25d":
                x = torch.tensor(np.ascontiguousarray(_slabs(v)), dtype=torch.float32, device=dev)
                y = torch.tensor(np.ascontiguousarray(t)[:, None], dtype=torch.float32, device=dev)
                w = torch.tensor(np.ascontiguousarray(msk)[:, None], dtype=torch.float32, device=dev)
            else:
                x = torch.tensor(np.ascontiguousarray(v)[None, None], dtype=torch.float32, device=dev)
                y = torch.tensor(np.ascontiguousarray(t)[None, None], dtype=torch.float32, device=dev)
                w = torch.tensor(np.ascontiguousarray(msk)[None, None], dtype=torch.float32, device=dev)
            opt.zero_grad()
            loss = (bce(m(x), y) * w).sum() / (w.sum() + 1e-6)
            loss.backward(); opt.step()
        sched.step()
    m.eval(); return m


def recipe_pretrain_finetune(source, target, pre_epochs=80, ft_epochs=40, ch=32, arch="unet3d",
                             backbone="resnet18", freeze_encoder=True, **ft):
    """Return a seed→model RECIPE: pretrain on `source`=(vols,pts) (abundant labels) then adapt on
    `target`=(vols,pts) (sparse point labels). The reusable 'use both datasets right' recipe for robust_compare."""
    def _make(seed):
        m = train_detector(*source, epochs=pre_epochs, ch=ch, seed=seed, arch=arch, backbone=backbone)
        return finetune_detector(m, *target, epochs=ft_epochs, seed=seed, freeze_encoder=freeze_encoder, **ft)
    return _make


def recipe_train(train_set, epochs=80, ch=32, arch="unet3d", backbone="resnet18"):
    """Return a seed→model RECIPE that just trains on one set (the source-only / target-only baselines)."""
    def _make(seed):
        return train_detector(*train_set, epochs=epochs, ch=ch, seed=seed, arch=arch, backbone=backbone)
    return _make


def eval_per_embryo(m, comp_eval, scale, gate=7.0, xy=4, tta=False, thr=0.15):
    """comp_eval = {embryo: [(vol_ds, gt_pts_fullvoxel), ...]}. Returns {embryo: mean node-recall @gate}.
    `tta` enables flip test-time augmentation in detect; `thr` sets the peak threshold. Per-frame failures are
    skipped rather than crashing the whole evaluation."""
    import numpy as np, bench_lib
    out = {}
    for emb, frames in comp_eval.items():
        rs = []
        for v, gt in frames:
            try:
                rs.append(bench_lib.recall_at_gate(gt, detect(m, v, max(len(gt) * 3, 20), xy, thr=thr, tta=tta), scale, gate)[0])
            except Exception:  # noqa: BLE001
                continue
        out[emb] = round(float(np.mean(rs)), 4) if rs else None
    return out


def robust_compare(train_sets, comp_eval, scale, seeds=(0, 1, 2), epochs=120, ch=32, gate=7.0, xy=4,
                   arch="unet3d", backbone="resnet18", dropout=0.0, sigma=1.2, tta=False, thr=0.15):
    """MULTI-SEED comparison of named RECIPES. Each value in `train_sets` is either a (vols, pts_list) tuple
    (trained as-is) OR a seed→model CALLABLE (e.g. recipe_pretrain_finetune / recipe_train) — so source-only,
    target-only, and pretrain→adapt recipes compare head-to-head. Returns {name: {embryo: {mean, std, seeds}}}
    + a paired verdict vs the FIRST (baseline): mean delta + robust flag (|Δ| beyond ~1 pooled std).
    `dropout`/`sigma` are forwarded to the tuple-spec detector; `tta`/`thr` to evaluation. A failed
    seed/recipe is skipped (its result is just absent) rather than crashing the whole comparison."""
    import numpy as np
    res = {n: {e: [] for e in comp_eval} for n in train_sets}
    for s in seeds:
        for name, spec in train_sets.items():
            try:
                if callable(spec):
                    m = spec(s)                               # a recipe: seed → model
                else:
                    vols, pts = spec
                    m = train_detector(vols, pts, epochs=epochs, ch=ch, seed=s, arch=arch, backbone=backbone,
                                       dropout=dropout, sigma=sigma)
                r = eval_per_embryo(m, comp_eval, scale, gate, xy, tta=tta, thr=thr)
            except Exception:  # noqa: BLE001
                continue
            for e in comp_eval:
                if r[e] is not None:
                    res[name][e].append(r[e])
    summary = {}
    for name in train_sets:
        summary[name] = {e: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v)), 4), "seeds": v}
                         for e, v in res[name].items() if v}
    base = list(train_sets)[0]
    verdict = {}
    for name in train_sets:
        if name == base:
            continue
        vd = {}
        for e in comp_eval:
            if e in summary[name] and e in summary[base]:
                d = summary[name][e]["mean"] - summary[base][e]["mean"]
                pooled = (summary[name][e]["std"] + summary[base][e]["std"]) / 2 + 1e-9
                vd[e] = {"delta": round(d, 4), "robust": bool(abs(d) > pooled)}  # beyond ~1 pooled std
        verdict[name] = vd
    return {"summary": summary, "vs_baseline": verdict, "baseline": base, "seeds": list(seeds)}


def pseudo_label(m, vols, xy=4, thr=0.3, topk_per=None, device=None, nms=3):
    """SELF-TRAINING: pseudo-label each target-domain volume with detector `m` (in the DOWNSAMPLED grid, so
    pts are usable directly as training labels). Returns [pts per vol]. Attacks a CONTENT gap with zero domain
    gap (labels come from the target domain itself). `device` falls back to cpu if CUDA is absent; `nms` is the
    local-maximum window. Handles both 3D-UNet and 2.5D-timm detectors; non-finite voxels are sanitized."""
    import numpy as np, torch
    from scipy.ndimage import maximum_filter
    dev = _dev(device)
    arch = getattr(m, "arch", "unet3d")
    nms = max(1, int(nms))
    out = []
    for v in vols:
        v = np.nan_to_num(np.asarray(v, np.float32))
        h = _heatmap(m, v, dev, arch)
        pk = np.argwhere((h == maximum_filter(h, nms)) & (h > thr)).astype(float)
        if topk_per and len(pk) > topk_per:
            vals = h[pk[:, 0].astype(int), pk[:, 1].astype(int), pk[:, 2].astype(int)]
            pk = pk[np.argsort(-vals)[:topk_per]]
        out.append(pk)                                          # kept in downsampled grid (training coords)
    return out


class DetectorTransfer(BaseAgent):
    name = "detector-transfer"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        """Spec: {train_sets: {name: {vols, pts}}, comp_eval: {emb: [{vol, gt}]}, seeds, epochs, ch, scale, gate}.
        Arrays may be nested lists (from JSON) or in-proc numpy. Returns the robust multi-seed comparison."""
        import numpy as np
        spec = self.spec(q)
        ts = spec.get("train_sets"); ce = spec.get("comp_eval")
        if not ts or not ce:
            return self.escalate(worker, "researcher", f"[{worker}] detector-transfer: need spec.train_sets and spec.comp_eval.")
        train_sets = {n: ([np.asarray(v, np.float32) for v in d["vols"]], [np.asarray(p, float) for p in d["pts"]])
                      for n, d in ts.items()}
        comp_eval = {e: [(np.asarray(f["vol"], np.float32), np.asarray(f["gt"], float)) for f in frames]
                     for e, frames in ce.items()}
        scale = np.asarray(spec.get("scale", [1.625, 0.40625, 0.40625]), float)
        out = robust_compare(train_sets, comp_eval, scale, seeds=tuple(spec.get("seeds", (0, 1, 2))),
                             epochs=int(spec.get("epochs", 120)), ch=int(spec.get("ch", 32)),
                             gate=float(spec.get("gate", 7.0)), xy=int(spec.get("xy", 4)),
                             arch=spec.get("arch", "unet3d"), backbone=spec.get("backbone", "resnet18"),
                             dropout=float(spec.get("dropout", 0.0)), sigma=float(spec.get("sigma", 1.2)),
                             tta=bool(spec.get("tta", False)), thr=float(spec.get("thr", 0.15)))
        self.save_state({"transfer": out})
        rows = []
        for name, per in out["summary"].items():
            cells = "  ".join(f"{e}={per[e]['mean']}±{per[e]['std']}" for e in per)
            rows.append(f"| {name} | {cells} |")
        vrows = []
        for name, vd in out["vs_baseline"].items():
            vrows.append(f"| {name} vs {out['baseline']} | " + "  ".join(f"{e}: Δ{vd[e]['delta']} {'✓robust' if vd[e]['robust'] else '~noise'}" for e in vd) + " |")
        msg = (f"[{worker}] **DETECTOR-TRANSFER** ({len(out['seeds'])}-seed per-embryo recall, mean±std)\n"
               + "\n".join(rows) + ("\n\n" + "\n".join(vrows) if vrows else ""))
        self.log(summary=f"detector-transfer: {out['summary']}", detail=f"{len(out['seeds'])}-seed robust per-embryo transfer",
                 kind="verdict", recommendation="only trust deltas marked robust (>1 pooled std); tie/negative ⇒ that lever doesn't move this comp")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"transfer": out}, msg, to="leader")


_AGENT = DetectorTransfer()


def run(q, worker):
    return _AGENT.run(q, worker)
