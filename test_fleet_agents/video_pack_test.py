"""video_pack_test — DATA-WISE, offline, deterministic (BLAS-pinned) verifier for the VIDEO pack.

Synthesizes deterministic clips / embeddings / a moving-dot video (no files, no network) and asserts the
ground-truth behaviour of each video agent's underlying function, plus that each raw handler returns a valid
(status,data,to,msg) contract on an EMPTY spec (the fleet smoke contract). Exit 0 iff all checks pass.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

import torch

from fleet_agents import video_pack as V

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_fails = []


def check(name, cond):
    print(("  ok  " if cond else "  FAIL") + "  " + name)
    if not cond:
        _fails.append(name)


# ── 1. frame sampler: T indices in range for every mode; short-clip cyclic pad ───────────────────────
N, T = 60, 16
for m in ("uniform", "stride", "dense", "random"):
    idx = V.sample_frame_indices(N, T, mode=m, event=30, stride=2, seed=1, device=DEV)
    check(f"sampler {m} returns T indices", idx.numel() == T)
    check(f"sampler {m} in [0,N)", bool(((idx >= 0) & (idx < N)).all()))
    check(f"sampler {m} long dtype", idx.dtype == torch.long)
# uniform is sorted & spans the clip
uni = V.sample_frame_indices(N, T, mode="uniform", device=DEV)
check("uniform is non-decreasing", bool((uni[1:] >= uni[:-1]).all()))
check("uniform spans clip (last near N-1)", int(uni[-1]) >= N - 2 and int(uni[0]) == 0)
# dense: indices are DENSER (smaller gaps) near the event than at the extremes
dense = V.sample_frame_indices(N, T, mode="dense", event=30, device=DEV).sort().values
gaps = (dense[1:] - dense[:-1]).float()
center_gap = gaps[len(gaps) // 2 - 1: len(gaps) // 2 + 1].mean()
edge_gap = torch.cat([gaps[:2], gaps[-2:]]).mean()
check("dense: gaps smaller near event than at edges", float(center_gap) <= float(edge_gap))
# short clip: n < T must still return T valid indices (cyclic pad, no out-of-range)
short = V.sample_frame_indices(5, T, mode="uniform", device=DEV)
check("short clip uniform returns T", short.numel() == T)
check("short clip uniform in range", bool(((short >= 0) & (short < 5)).all()))
for m in ("stride", "random", "dense"):
    s = V.sample_frame_indices(5, T, mode=m, seed=0, device=DEV)
    check(f"short clip {m} in range", s.numel() == T and bool(((s >= 0) & (s < 5)).all()))
# gather helper slices the time axis
clip = torch.randn(N, 3, 8, 8, device=DEV)
gathered = V.gather_frames(clip, V.sample_frame_indices(N, T, mode="dense", event=30, device=DEV))
check("gather_frames -> (T, ...)", tuple(gathered.shape) == (T, 3, 8, 8))
# determinism
a = V.sample_frame_indices(N, T, mode="random", seed=42, device=DEV)
b = V.sample_frame_indices(N, T, mode="random", seed=42, device=DEV)
check("random sampler deterministic under seed", bool((a == b).all()))

# ── 2. temporal aggregator: [B,T,D] -> [B,D'] for each mode; sparse signal captured better than mean ──
B, Tt, D = 4, 12, 24
x = torch.randn(B, Tt, D, device=DEV)
for m in ("mean", "max", "attention", "tsm", "std"):
    v = V.temporal_aggregate(x, mode=m)
    check(f"aggregate fn {m} -> (B,D)", tuple(v.shape) == (B, D))
    check(f"aggregate fn {m} finite", bool(torch.isfinite(v).all()))
# learnable heads -> (B, out_dim)
OUT = 16
for m in ("attention", "tconv", "tsm", "gru", "mean", "max"):
    head = V.TemporalAggregator(D, out_dim=OUT, mode=m, device=DEV)
    head.eval()
    with torch.no_grad():
        y = head(x)
    check(f"head {m} -> (B,out_dim)", tuple(y.shape) == (B, OUT))
    check(f"head {m} finite", bool(torch.isfinite(y).all()))
# temporal_shift is shape-preserving and actually moves channels across time
xs = V.temporal_shift(x)
check("temporal_shift shape-preserving", tuple(xs.shape) == (B, Tt, D))
check("temporal_shift changed the tensor", bool((xs != x).any()))

# SEPARABILITY probe: a SPARSE-in-time signal must survive attention/tsm better than naive mean.
# background clip = small noise; signal clip = same + a big spike in ONE frame at ONE dim.
torch.manual_seed(3)
bg = torch.randn(1, Tt, D, device=DEV) * 0.05
sig = bg.clone()
d0, t0 = 7, 5
sig[0, t0, d0] += 6.0                                       # sparse-in-time, one dim
def sep(mode):
    a = V.temporal_aggregate(sig, mode=mode)[0, d0]
    b = V.temporal_aggregate(bg, mode=mode)[0, d0]
    return float((a - b).abs())
sep_mean, sep_attn, sep_tsm = sep("mean"), sep("attention"), sep("tsm")
check("sparse signal: attention separates better than mean", sep_attn > sep_mean)
check("sparse signal: tsm separates better than mean", sep_tsm > sep_mean)
print(f"  separability  mean={sep_mean:.3f} attention={sep_attn:.3f} tsm={sep_tsm:.3f}")

# ── 3. motion features: non-zero where motion is, ~zero on a static clip ─────────────────────────────
Tm, H, W = 8, 16, 16
moving = torch.zeros(Tm, H, W, device=DEV)
for t in range(Tm):
    moving[t, H // 2, min(W - 1, t)] = 1.0                  # a bright dot translating each frame
static = moving[0:1].repeat(Tm, 1, 1)                       # same frame repeated → no motion
fd = V.frame_difference(moving, absolute=True)
check("frame_difference keeps T frames", fd.shape[0] == Tm)
check("frame_difference nonzero on moving clip", bool(fd.sum() > 0))
check("frame_difference ~zero on static clip", bool(V.frame_difference(static, absolute=True).sum() < 1e-5))
tg = V.temporal_gradient(moving)
check("temporal_gradient nonzero on moving clip", bool(tg.abs().sum() > 0))
check("temporal_gradient ~zero on static clip", bool(V.temporal_gradient(static).abs().sum() < 1e-5))
fp = V.flow_magnitude_proxy(moving)
check("flow_proxy finite", bool(torch.isfinite(fp).all()))
check("flow_proxy ~zero on static clip", bool(V.flow_magnitude_proxy(static).sum() < 1e-4))
# motion is localized: the moving-dot clip's frame-diff energy is concentrated, not spread everywhere
check("frame_difference localizes motion (sparse map)", bool((fd > 0).float().mean() < 0.2))
mc = V.motion_channels(moving, include=("diff", "grad", "flow"))
check("motion_channels -> (T, 1+3, H, W)", tuple(mc.shape) == (Tm, 4, H, W))
mc2 = V.motion_channels(torch.randn(Tm, 3, H, W, device=DEV), include=("diff",))
check("motion_channels on (T,C,H,W) input", mc2.shape[0] == Tm and mc2.shape[1] == 6)  # 3 raw + 3 diff

# ── 4. every raw handler returns a valid contract on EMPTY spec (fleet smoke contract) ───────────────
VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}
for h in (V.run_frame_sampler, V.run_temporal_aggregator, V.run_motion_features):
    r = h({"question": "test", "spec": {}}, "unit")
    check(f"handler {h.__name__} valid contract", isinstance(r, tuple) and len(r) == 4 and r[0] in VALID)
    check(f"handler {h.__name__} status done", r[0] == "done")

print()
if _fails:
    print("FAILURES:", _fails)
    sys.exit(1)
print("ALL VIDEO PACK CHECKS PASSED")
sys.exit(0)
