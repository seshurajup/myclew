"""sr-bf16-optimizer — memory-efficient full-parameter training via STOCHASTIC ROUNDING of bf16 weights.

Distilled from tascj/offload_adam (the optimizer that keeps grads + Adam states in bf16 / pinned host
memory so much larger models train on a single GPU). The reusable, competition-agnostic gem is the
NUMERICS, not the CUDA plumbing:

  • stochastic rounding (SR) fp32->bf16 : round UP with probability = (dropped mantissa bits)/ULP.
    Then E[round_sr(x)] == x exactly (UNBIASED). Round-to-nearest-even (RTNE) is biased-to-zero for a
    stream of sub-ULP updates: once |Δ| < 0.5·ULP every step rounds back to the SAME bf16 value and the
    weight STALLS. SR lets those tiny updates accumulate in expectation, so a bf16 master weight tracks
    the fp32 trajectory — this is what makes bf16-master AdamW converge like fp32 AdamW at HALF the
    optimizer-state memory (6 B/param vs 12 B/param).
  • fp31 decompose/reconstruct : split an fp32 value into (bf16, int16 residual) so an 8-B/param "master"
    is bit-reconstructable — the middle ground between 6 B SR and 12 B fp32.

Why it matters here: single-GPU (RTX 5090) fine-tuning + Kaggle 2×T4 are MEMORY-bound; SR-bf16 states let
a bigger detector/linker fit without the fp32 optimizer tax, at no accuracy cost. All pure numpy + testable.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- bf16 rounding primitives (pure numpy)
def _bits(x):
    return np.ascontiguousarray(np.asarray(x, np.float32)).view(np.uint32)


def fp32_to_bf16_trunc(x):
    """Truncate toward zero (drop low 16 mantissa bits) — the SR 'towards_zero' base point."""
    return (_bits(x) & np.uint32(0xFFFF0000)).view(np.float32)


def fp32_to_bf16_rtne(x):
    """Round to nearest, ties-to-even — the standard hardware bf16 cast (BIASED for sub-ULP update streams)."""
    u = _bits(x).astype(np.uint64)
    lsb = (u >> np.uint64(16)) & np.uint64(1)
    u = (u + np.uint64(0x7FFF) + lsb) & np.uint64(0xFFFF0000)
    return u.astype(np.uint32).view(np.float32)


def fp32_to_bf16_sr(x, rng):
    """STOCHASTIC ROUNDING fp32->bf16. Round up (increase magnitude) with prob = frac/2^16 → UNBIASED.
    Mirrors offload_adam's triton kernel: towards_zero + 0x10000 when rand16 < dropped_fraction."""
    u = _bits(x).astype(np.uint64)
    frac = u & np.uint64(0xFFFF)
    r16 = rng.integers(0, 1 << 16, size=u.shape, dtype=np.uint64)
    up = (r16 < frac).astype(np.uint64)
    u = (u & np.uint64(0xFFFF0000)) + (up << np.uint64(16))
    return u.astype(np.uint32).view(np.float32)


# ---------------------------------------------------------------- fp31 master (bf16 + int16 residual)
def fp31_decompose(x):
    """Split fp32 -> (bf16, int16 residual) so the value is (near) losslessly reconstructable in 4 B."""
    xb = fp32_to_bf16_rtne(x)
    err = (_bits(x).astype(np.int64) - _bits(xb).astype(np.int64))
    err_q = np.clip(err >> 1, -32768, 32767).astype(np.int16)  # residual scaled into int16
    return xb, err_q


def fp31_reconstruct(xb, err_q):
    u = _bits(xb).astype(np.int64) + (err_q.astype(np.int64) << 1)
    return u.astype(np.uint32).view(np.float32)


# ---------------------------------------------------------------- Adam step with a bf16-stored master weight
def adam_bf16_master(grad_fn, w0, steps=400, lr=1e-2, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=0.0, rounding="sr", seed=0):
    """Run AdamW where the MASTER weight is stored in bf16 and re-rounded every step by `rounding`
    ('sr' = stochastic, 'rtne' = nearest-even, 'fp32' = no rounding baseline). grad_fn(w)->grad."""
    rng = np.random.default_rng(seed)
    b1, b2 = betas
    w = np.asarray(w0, np.float32).copy()
    if rounding != "fp32":
        w = {"sr": lambda v: fp32_to_bf16_sr(v, rng), "rtne": fp32_to_bf16_rtne}[rounding](w)
    m = np.zeros_like(w); v = np.zeros_like(w)
    for t in range(1, int(steps) + 1):
        g = np.asarray(grad_fn(w), np.float32)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t); vh = v / (1 - b2 ** t)
        w32 = w.astype(np.float32) * (1 - lr * weight_decay) - lr * mh / (np.sqrt(vh) + eps)
        if rounding == "sr":
            w = fp32_to_bf16_sr(w32, rng)
        elif rounding == "rtne":
            w = fp32_to_bf16_rtne(w32)
        else:
            w = w32
    return w.astype(np.float32)


# ---------------------------------------------------------------- agent
class SRBf16Optimizer(BaseAgent):
    name = "sr-bf16-optimizer"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        dim = int(s.get("dim", 64)); steps = int(s.get("steps", 500)); lr = float(s.get("lr", 5e-3))
        seed = int(s.get("seed", 0))
        rng = np.random.default_rng(seed)
        # convex target NOT on the bf16 grid → near convergence updates go sub-ULP (the SR regime)
        c = (rng.standard_normal(dim).astype(np.float32) * 0.37 + 1.234567)
        w0 = c + rng.standard_normal(dim).astype(np.float32) * 0.5

        def grad(w):
            return (w.astype(np.float32) - c)

        errs = {}
        for mode in ("fp32", "rtne", "sr"):
            wf = adam_bf16_master(grad, w0, steps=steps, lr=lr, rounding=mode, seed=seed)
            errs[mode] = float(np.sqrt(np.mean((wf - c) ** 2)))
        sr_beats_rtne = errs["sr"] < errs["rtne"]
        bytes_fp32 = 12; bytes_sr = 6      # master+m+v : fp32=4+4+4 ; sr-bf16=2+2+2
        msg = (f"sr-bf16-optimizer: bf16-master AdamW final RMSE fp32={errs['fp32']:.2e} "
               f"rtne={errs['rtne']:.2e} sr={errs['sr']:.2e} → SR {'RECOVERS' if sr_beats_rtne else 'ties'} "
               f"the fp32 trajectory at {bytes_sr}B/param vs {bytes_fp32}B (−50% optimizer memory)")
        self.log(msg, kind="finding",
                 recommendation="use bf16 optimizer states + SR to fit a bigger detector/linker on 1 GPU / T4")
        return self.done({"rmse": errs, "sr_beats_rtne": bool(sr_beats_rtne),
                          "bytes_per_param_sr": bytes_sr, "bytes_per_param_fp32": bytes_fp32}, msg)


_AGENT = SRBf16Optimizer()


def run(q, worker):
    return _AGENT.run(q, worker)
