"""video_pack — the VIDEO modality pack, GROUNDED in the top-5 solutions of 5 real video competitions
(deepfake-detection-challenge, dfl-bundesliga-data-shootout, nfl-player-contact-detection, nfl-impact-detection,
youtube8m-2019). Mined via the fleet's `gm-writeup-mine` agent; recurring techniques + full provenance in
docs/video_pack_grounded.md.

The fleet already covered the video techniques that are NOT video-specific:
  • per-frame-scores → action SEGMENTS (thr/min-len/merge-gap) → `temporal-segment-decoder` — referenced, NOT rebuilt
  • peak/gaussian smoothing over time                          → `audio-crop-tta` neighbor_smooth / `heatmap-peak-decoder`
  • pad-aware mean/max/attention pooling of a sequence         → `masked-sequence-pool` — referenced (this pack ADDS the
                                                                  TSM / temporal-conv / GRU aggregators it lacks)
  • overlapping-window inference TTA / multi-scale / flip      → `multi-tta` / `snapshot-average` / `wbf-fusion`
  • ROI detector (face/helmet/ball) → crop                     → Detection & Tracking pack + `geometric-spatial-augmentor`
  • sparse keyframe selection (T4 detection budget)            → `keyframe` (distinct intent)

What the fleet was genuinely MISSING (this pack), each recurring across the mined winners:
  • video-frame-sampler       — sample T frame indices from a clip: uniform / stride / dense-around-event / random-jitter;
                                variable-length safe (cyclic pad / subsample). THE video-training dataloader foundation.
  • video-temporal-aggregator — per-frame embeddings [B,T,D] → clip vector [B,D'] via mean/max/attention/1d-temporal-conv/
                                TSM-temporal-shift/GRU. Turns an image backbone into a video model.
  • video-motion-features     — frame-difference + temporal-gradient (optical-flow proxy) + brightness-constancy motion
                                magnitude + a motion-channel stacker (the deepfake/sports motion cue).

Pure torch/numpy, GPU-FIRST (every tensor op runs on CUDA when available; CPU fallback only if no CUDA). No numpy/torch
version is touched. Heavy pretrained video backbones (I3D/CSN/X-CLIP/Video-Swin/NetVLAD weights) are NOT shipped — the
aggregator builds untrained heads on your frame embeddings, never fabricating pretrained 3D-CNN weights. Data-wise
tests: test_fleet_agents/video_pack_test.py.
"""
from __future__ import annotations
from .base import BaseAgent


def _device(spec):
    import torch
    d = (spec or {}).get("device")
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


# ════════════════════════════════════════════════════════════ 1. frame / clip sampling
def sample_frame_indices(n_frames, T, mode="uniform", event=None, stride=1, start=None, seed=None,
                         jitter=True, device=None):
    """Sample T frame indices from a length-`n_frames` clip — the video-training dataloader foundation
    (grounded: deepfake 32-frame, DFL every-2nd, NFL dense-around-event {-44..0..37}, TSN random segments).

    Returns a 1-D LongTensor of length T, every value in [0, n_frames). Handles clips SHORTER than T (cyclic
    pad — repeats indices) and LONGER than T (subsample), so it never indexes out of range.

    modes:
      • "uniform" — T evenly-spaced indices across the whole clip (TSN/eval standard).
      • "stride"  — start + stride*k for k=0..T-1, wrapped cyclically (mod n_frames) so short clips still fill T.
                    `start` defaults to a centered anchor so the window is centered on the clip.
      • "dense"   — non-uniform, DENSEST around `event` (default the clip center): a quadratic offset schedule
                    reproduces the NFL "observe more frames close to the estimated frame" sampling; clamped in-range.
      • "random"  — split [0,n_frames) into T segments, pick one index per segment (uniform), optionally with
                    intra-segment `jitter`. TSN train-time sampling. Deterministic under `seed`.
    """
    import torch
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    n = int(max(1, n_frames))
    T = int(max(1, T))
    g = torch.Generator(device="cpu")
    if seed is not None:
        g.manual_seed(int(seed))

    if mode == "uniform":
        if n == 1:
            idx = torch.zeros(T, dtype=torch.long)
        else:
            idx = torch.linspace(0, n - 1, steps=T).round().long()
    elif mode == "stride":
        s0 = start if start is not None else max(0, (n - stride * (T - 1)) // 2)
        idx = (torch.arange(T, dtype=torch.long) * int(stride) + int(s0)) % n     # cyclic wrap
    elif mode == "dense":
        ev = int(event if event is not None else (n - 1) / 2.0)
        # symmetric quadratic offsets: dense near 0, sparser far (matches NFL {-44,-37,...,0,...,37})
        k = torch.arange(T, dtype=torch.float32) - (T - 1) / 2.0                  # centered rank
        span = max(1.0, (n - 1) / 2.0)
        norm = (k.abs() / max(1.0, (T - 1) / 2.0)) ** 2                           # 0..1 quadratic
        off = torch.sign(k) * (norm * span)
        idx = (ev + off).round().long().clamp(0, n - 1)
    elif mode == "random":
        # TSN random: T segments, one sample per segment (+ jitter). Cyclic-safe for short clips.
        edges = torch.linspace(0, n, steps=T + 1)
        lo = edges[:-1]
        seg = (edges[1:] - lo).clamp(min=1e-6)
        if jitter:
            r = torch.rand(T, generator=g)
        else:
            r = torch.full((T,), 0.5)
        idx = (lo + r * seg).floor().long() % n
    else:
        raise ValueError(f"video-frame-sampler: unknown mode={mode!r} (uniform|stride|dense|random)")
    return idx.clamp(0, n - 1).to(dev)


def gather_frames(clip, indices):
    """Gather sampled frames from a clip along the TIME axis (axis 0). `clip`: (n_frames, ...) tensor,
    `indices`: 1-D long tensor from `sample_frame_indices`. Returns (T, ...). Autograd/DataLoader-safe."""
    import torch
    idx = indices.to(clip.device).long()
    return clip.index_select(0, idx)


class VideoFrameSampler(BaseAgent):
    name = "video-frame-sampler"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            n = int(spec.get("n_frames", 60))
            T = int(spec.get("T", 16))
            checks = {}
            for m in ("uniform", "stride", "dense", "random"):
                idx = sample_frame_indices(n, T, mode=m, event=spec.get("event"),
                                           stride=int(spec.get("stride", 2)), seed=0, device=dev)
                checks[f"{m}_len"] = int(idx.numel()) == T
                checks[f"{m}_inrange"] = bool(((idx >= 0) & (idx < n)).all())
            # short clip (n<T) must still return T valid indices (cyclic)
            short = sample_frame_indices(5, T, mode="uniform", device=dev)
            checks["short_clip_ok"] = short.numel() == T and bool(((short >= 0) & (short < 5)).all())
            # gather helper
            clip = torch.randn(n, 3, 8, 8, device=dev)
            g = gather_frames(clip, sample_frame_indices(n, T, mode="dense", device=dev))
            checks["gather_shape"] = tuple(g.shape) == (T, 3, 8, 8)
            ok = all(checks.values())
            msg = (f"video-frame-sampler: {sum(checks.values())}/{len(checks)} ok; T={T} indices from n={n} "
                   f"frames via uniform/stride/dense/random (+cyclic short-clip pad, +gather helper) device={dev}.")
            self.log(msg, kind="finding",
                     recommendation="sample_frame_indices(n_frames,T,mode) in the dataloader; 'dense' around the event "
                                    "step (NFL), 'random' for TSN train-time, 'uniform' for eval; gather_frames to slice")
            return self.done({"checks": {k: bool(v) for k, v in checks.items()}, "T": T, "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher", f"[{worker}] video-frame-sampler checks failed: {checks}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] video-frame-sampler FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ 2. temporal aggregation of per-frame embeddings
def temporal_shift(x, fold_div=8, dims_last=True):
    """Temporal Shift Module (grounded: NFL-impact 1st, DFL camaro 3rd — TSM at block ends, near-zero overhead).
    Shifts 1/fold_div of the channels forward in time and 1/fold_div backward, leaving the rest in place — so a
    2D backbone exchanges information across frames. `x`: (B, T, D) if dims_last else (B, D, T). Shape-preserving.
    """
    import torch
    if not dims_last:
        x = x.transpose(1, 2)                                # -> (B, T, D)
    B, T, D = x.shape
    fold = max(1, D // fold_div)
    out = torch.zeros_like(x)
    out[:, :-1, :fold] = x[:, 1:, :fold]                     # shift up (future -> present)
    out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]     # shift down (past -> present)
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]                # unshifted remainder
    return out if dims_last else out.transpose(1, 2)


def temporal_aggregate(x, mode="mean", temperature=1.0):
    """Parameter-free temporal aggregation of per-frame embeddings — [B,T,D] → clip vector [B,D].
    modes (no learnable params, deterministic):
      • "mean" / "max"  — the pooling baselines (mask-aware variants live in `masked-sequence-pool`).
      • "attention"     — CONTENT/energy-weighted softmax pool: weight_t = softmax(||x_t||^2 / temperature).
                          Concentrates on high-energy frames → captures a SPARSE-in-time signal that mean dilutes.
      • "tsm"           — temporal-shift then max-pool: spreads a sparse frame to neighbours, max keeps it.
      • "std"           — temporal standard deviation (motion/dynamics summary).
    For the LEARNABLE aggregators (1D-temporal-conv, GRU, learnable attention) use `TemporalAggregator`.
    """
    import torch
    import torch.nn.functional as F
    if mode == "mean":
        return x.mean(dim=1)
    if mode == "max":
        return x.amax(dim=1)
    if mode == "std":
        return x.std(dim=1)
    if mode == "attention":
        energy = (x * x).sum(dim=-1) / max(1e-6, float(temperature))   # (B, T)
        w = F.softmax(energy, dim=1).unsqueeze(-1)                      # (B, T, 1)
        return (w * x).sum(dim=1)
    if mode == "tsm":
        return temporal_shift(x).amax(dim=1)
    raise ValueError(f"temporal_aggregate: unknown mode={mode!r} (mean|max|attention|tsm|std)")


class TemporalAggregator:
    """Learnable per-frame-embeddings → clip-vector head (grounded: DFL 1D-UNet/GRU, NFL CNN+LSTM/transformer,
    youtube8m NetVLAD/GRU). Builds the aggregator that turns an image backbone into a video model.

    modes: 'mean' | 'max' | 'attention' (learnable) | 'tconv' (1D conv over time + pool) | 'tsm' (temporal-shift
    conv + pool) | 'gru' (GRU over frames, last state). forward: (B, T, D) -> (B, out_dim). Pure torch.
    """
    def __new__(cls, dim, out_dim=None, mode="attention", hidden=None, device=None):
        import torch
        from torch import nn
        import torch.nn.functional as F
        out_dim = out_dim or dim
        hidden = hidden or dim
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

        class Head(nn.Module):
            def __init__(self):
                super().__init__()
                self.mode = mode
                if mode == "attention":
                    self.attn = nn.Linear(dim, 1)
                    self.proj = nn.Linear(dim, out_dim)
                elif mode == "tconv":
                    self.conv = nn.Conv1d(dim, hidden, kernel_size=3, padding=1)
                    self.proj = nn.Linear(hidden, out_dim)
                elif mode == "tsm":
                    self.conv = nn.Conv1d(dim, hidden, kernel_size=3, padding=1)
                    self.proj = nn.Linear(hidden, out_dim)
                elif mode == "gru":
                    self.gru = nn.GRU(dim, hidden, batch_first=True)
                    self.proj = nn.Linear(hidden, out_dim)
                elif mode in ("mean", "max"):
                    self.proj = nn.Linear(dim, out_dim)
                else:
                    raise ValueError(f"TemporalAggregator: unknown mode={mode!r}")

            def forward(self, x):                            # x: (B, T, D)
                if self.mode == "attention":
                    w = F.softmax(self.attn(x), dim=1)       # (B, T, 1)
                    pooled = (w * x).sum(dim=1)
                    return self.proj(pooled)
                if self.mode == "gru":
                    _, h = self.gru(x)
                    return self.proj(h[-1])
                if self.mode in ("tconv", "tsm"):
                    z = temporal_shift(x) if self.mode == "tsm" else x
                    z = self.conv(z.transpose(1, 2))         # (B, hidden, T)
                    pooled = z.amax(dim=-1)                   # temporal max over conv features
                    return self.proj(pooled)
                pooled = x.mean(dim=1) if self.mode == "mean" else x.amax(dim=1)
                return self.proj(pooled)
        return Head().to(dev)


class VideoTemporalAggregator(BaseAgent):
    name = "video-temporal-aggregator"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            B = int(spec.get("B", 2))
            T = int(spec.get("T", 16))
            D = int(spec.get("D", 32))
            out = int(spec.get("out_dim", D))
            x = torch.randn(B, T, D, device=dev)
            checks = {}
            # parameter-free functional aggregators
            for m in ("mean", "max", "attention", "tsm", "std"):
                v = temporal_aggregate(x, mode=m)
                checks[f"fn_{m}"] = tuple(v.shape) == (B, D) and bool(torch.isfinite(v).all())
            # learnable heads → [B, out_dim]
            for m in ("attention", "tconv", "tsm", "gru", "mean"):
                head = TemporalAggregator(D, out_dim=out, mode=m, device=dev)
                head.eval()
                with torch.no_grad():
                    y = head(x)
                checks[f"head_{m}"] = tuple(y.shape) == (B, out) and bool(torch.isfinite(y).all())
            ok = all(checks.values())
            msg = (f"video-temporal-aggregator: {sum(checks.values())}/{len(checks)} ok; [B,T,D]=[{B},{T},{D}] → "
                   f"[B,{out}] via mean/max/attention/tconv/tsm/gru. Turns an image backbone into a video model. "
                   f"device={dev}. (masked-sequence-pool covers the mask-aware mean/max/attention path.)")
            self.log(msg, kind="finding",
                     recommendation="TemporalAggregator(dim,mode='tsm'|'gru'|'tconv'|'attention') on per-frame "
                                    "embeddings; TSM/1d-conv = the DFL/NFL winner heads masked-sequence-pool lacks")
            return self.done({"checks": {k: bool(v) for k, v in checks.items()}, "out_dim": out, "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher", f"[{worker}] video-temporal-aggregator failed: {checks}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] video-temporal-aggregator FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ 3. motion features
def frame_difference(clip, absolute=True, order=1, keep_len=True):
    """Consecutive frame difference (grounded: DFL ohkawa3 5th abs-diff-from-prev, DFL kmat 2nd prev/next diff —
    'drastically improved accuracy in crowded scene'). `clip`: (T, ...) along time. order=1 → x[t]-x[t-1],
    order=2 → second difference. `keep_len` prepends zeros so the output keeps T frames (stack as a channel)."""
    import torch
    x = clip.float()
    d = x
    for _ in range(int(order)):
        d = d[1:] - d[:-1]
    if absolute:
        d = d.abs()
    if keep_len:
        pad = torch.zeros((int(order),) + tuple(d.shape[1:]), dtype=d.dtype, device=d.device)
        d = torch.cat([pad, d], dim=0)
    return d


def temporal_gradient(clip):
    """Central-difference temporal gradient (an optical-flow PROXY along time): g[t]=(x[t+1]-x[t-1])/2, with
    forward/backward diffs at the ends. `clip`: (T, ...). Keeps T frames. Non-zero exactly where the scene moves."""
    import torch
    x = clip.float()
    g = torch.zeros_like(x)
    if x.shape[0] >= 3:
        g[1:-1] = (x[2:] - x[:-2]) / 2.0
    if x.shape[0] >= 2:
        g[0] = x[1] - x[0]
        g[-1] = x[-1] - x[-2]
    return g


def flow_magnitude_proxy(clip, eps=1e-3):
    """Brightness-constancy optical-flow MAGNITUDE proxy (no RAFT/weights): |I_t| / (||∇I|| + eps), i.e. the
    temporal gradient normalized by the local spatial gradient — a cheap per-pixel motion-magnitude map (grounded:
    the RAFT/OpenCV flow used by DFL kmat 2nd & NFL-impact 1st, distilled to a training-free proxy). `clip`: (T,H,W)
    or (T,C,H,W). Returns the same spatial shape per frame with a motion magnitude, ~0 on a static clip."""
    import torch
    x = clip.float()
    it = temporal_gradient(x).abs()
    # spatial gradient magnitude on the last two dims (H, W)
    gx = torch.zeros_like(x)
    gy = torch.zeros_like(x)
    gx[..., :, 1:] = x[..., :, 1:] - x[..., :, :-1]
    gy[..., 1:, :] = x[..., 1:, :] - x[..., :-1, :]
    gmag = torch.sqrt(gx * gx + gy * gy)
    return it / (gmag + eps)


def motion_channels(clip, include=("diff", "grad")):
    """Stack motion maps as EXTRA input channels next to the raw frames (grounded: DFL/NFL frame-diff channel,
    2.5D stacking). `clip`: (T, H, W) or (T, C, H, W). Returns (T, C', H, W) with the raw frame + requested motion
    maps ('diff'=abs frame-difference, 'grad'=temporal gradient, 'flow'=flow-magnitude proxy) concatenated on C."""
    import torch
    x = clip.float()
    if x.dim() == 3:                                          # (T, H, W) -> (T, 1, H, W)
        x = x.unsqueeze(1)
    parts = [x]
    if "diff" in include:
        parts.append(frame_difference(x, absolute=True, order=1))
    if "grad" in include:
        parts.append(temporal_gradient(x).abs())
    if "flow" in include:
        parts.append(flow_magnitude_proxy(x))
    return torch.cat(parts, dim=1)


class VideoMotionFeatures(BaseAgent):
    name = "video-motion-features"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            T, H, W = int(spec.get("T", 8)), int(spec.get("H", 16)), int(spec.get("W", 16))
            # synthesize a moving bright dot to health-check motion signal
            clip = torch.zeros(T, H, W, device=dev)
            for t in range(T):
                clip[t, H // 2, min(W - 1, t)] = 1.0
            static = clip[0:1].repeat(T, 1, 1)
            fd = frame_difference(clip, absolute=True)
            tg = temporal_gradient(clip)
            fp = flow_magnitude_proxy(clip)
            mc = motion_channels(clip, include=("diff", "grad", "flow"))
            checks = {
                "diff_keeps_T": fd.shape[0] == T,
                "diff_moving_nonzero": bool(fd.sum() > 0),
                "diff_static_zero": bool(frame_difference(static, absolute=True).sum() < 1e-5),
                "grad_moving_nonzero": bool(tg.abs().sum() > 0),
                "grad_static_zero": bool(temporal_gradient(static).abs().sum() < 1e-5),
                "flow_finite": bool(torch.isfinite(fp).all()),
                "flow_static_zero": bool(flow_magnitude_proxy(static).sum() < 1e-4),
                "motion_channels_shape": mc.shape[0] == T and mc.shape[1] == 4,  # raw + diff + grad + flow
            }
            ok = all(checks.values())
            msg = (f"video-motion-features: {sum(checks.values())}/{len(checks)} ok; frame-diff + temporal-gradient "
                   f"+ flow-magnitude proxy + motion-channel stacker → non-zero on motion, ~0 static. device={dev}.")
            self.log(msg, kind="finding",
                     recommendation="motion_channels(clip) to append diff/grad/flow maps as extra CNN input channels "
                                    "(DFL/NFL motion cue); training-free — no RAFT/optical-flow weights needed")
            return self.done({"checks": {k: bool(v) for k, v in checks.items()}, "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher", f"[{worker}] video-motion-features failed: {checks}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] video-motion-features FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ handlers
_SAMP = VideoFrameSampler()
_AGG = VideoTemporalAggregator()
_MOT = VideoMotionFeatures()


def run_frame_sampler(q, worker):
    return _SAMP.run(q, worker)


def run_temporal_aggregator(q, worker):
    return _AGG.run(q, worker)


def run_motion_features(q, worker):
    return _MOT.run(q, worker)
