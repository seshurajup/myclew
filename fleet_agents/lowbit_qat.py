"""lowbit-qat — ternary / low-bit QUANTIZATION-AWARE TRAINING primitives (pure torch, no new deps).

The gap this fills (audited 2026-07-16 against the "Bonsai / Ternary-Bonsai" release + BitNet line):
  Our fleet already had PTQ-inference levers — `quantize` (INT8-W8A8 PTQ + ToMe estimate) and `compress-select`
  (ShortGPT/LaCo DEPTH pruning). Those are all POST-training / inference-time. We had NOTHING that TRAINS under
  low-bit weights: no ternary {-1,0,+1} quantizer, no k-bit fake-quant, no straight-through estimator (STE),
  no QAT Linear. That is the one missing family, and it is a clean, reusable, pure-torch primitive.

Grounding (what is real vs marketing):
  • The "Ternary-Bonsai 27B" repo (PrismML, github.com/PrismML-Eng/Bonsai-demo) turned out to be POST-TRAINING
    QUANTIZATION of off-the-shelf Qwen3 checkpoints into a group-wise low-bit FORMAT + custom llama.cpp/MLX
    kernels — NOT low-bit training. Its own 27B whitepaper says it "takes the opposite path from BitNet: it
    starts from an off-the-shelf pretrained model and moves it into a binary or ternary representation".
    The weight format is w_i = s_g · t_i, t_i ∈ {-1,0,+1}, one shared FP16 scale per group of 128 (g128) →
    b_eff ≈ 1.585 + 16/128 = 1.71 bits/weight (ternary), or 1 + 16/128 = 1.125 bits/weight (1-bit sign+scale).
    That group-wise ternary FORMAT is the genuinely reusable idea; the "training" is not disclosed there.
  • The ACTUAL source of ternary QAT-with-STE is BitNet b1.58 (Ma et al., "The Era of 1-bit LLMs: All Large
    Language Models are in 1.58 Bits", arXiv:2402.17764) and BitNet (Wang et al., arXiv:2310.11453) — absmean
    ternary weights + STE, trained from scratch. LLM-QAT (arXiv:2305.17888) is the general fake-quant+STE QAT.
    This module implements THAT (the real, reproducible primitive), plus the Bonsai group-wise effective-bits
    accounting so we can reason about footprint on the 5090 (bf16, 32GB) and Kaggle T4 (16GB).

Reusable primitives (all pure torch, GPU-safe, deps = torch only):
  • ternary_quantize(W)      — BitNet b1.58 absmean ternary {-1,0,+1} with per-tensor or per-channel scale.
  • int_fake_quant(x, bits)  — symmetric k-bit fake-quant (int4 weights / int8 activations), per-tensor/channel.
  • STETernary / STEQuant    — autograd.Function straight-through estimators (quantize forward, identity back).
  • QuantLinear              — nn.Linear with a full-precision MASTER weight + fake-quant forward (the QAT cell).
  • wrap_qat(model, ...)     — swap Linear→QuantLinear, KEEP norms/embeddings/lm_head in high precision.
  • qat_finetune(...)        — tiny STE training loop proving a quantized net still learns.
  • effective_bits(...)      — Bonsai/BitNet group-wise bits/weight accounting (1.58 + scale/group).
"""
from __future__ import annotations
from .base import BaseAgent

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------------------------------
# 1. pure quantizers (no autograd) — the math, data-wise testable in isolation
# ---------------------------------------------------------------------------------------------------------
def ternary_quantize(W: torch.Tensor, per_channel: bool = False, eps: float = 1e-8, group_size=None):
    """BitNet b1.58 absmean ternary. Returns (t, scale) with t ∈ {-1,0,+1} (same shape as W) and a scale that
    is per-tensor (scalar) or per-output-channel (per row, dim 0) when per_channel=True. Dequant = t * scale.

    Rule (Ma et al. 2402.17764): scale = mean(|W|); t = clamp(round(W / scale), -1, +1). The absmean scale is
    the value that best preserves magnitude for a symmetric ternary code; round-to-nearest maps small weights
    to the expressive 0 state (this is exactly the zero state Ternary-Bonsai credits for its +5pts over 1-bit).

    group_size (Bonsai/BitNet g128 block scaling): when set (e.g. 128) the LAST dim is chunked into groups of
    group_size and one absmean scale is computed PER GROUP; t stays same-shape as W and scale has shape
    (*W.shape[:-1], n_groups). Dequant grouped code via `dequant_ternary(t, scale, group_size)`. Default None =
    the original per-tensor / per-channel path, byte-identical.
    """
    if not torch.is_tensor(W):
        W = torch.as_tensor(W, dtype=torch.float32)
    W = W.float()
    if group_size:
        g = int(group_size); shp = W.shape; last = shp[-1]; pad = (-last) % g
        Wp = F.pad(W, (0, pad)) if pad else W
        nblk = (last + pad) // g
        Wb = Wp.reshape(*shp[:-1], nblk, g)
        scale = Wb.abs().mean(dim=-1, keepdim=True).clamp_min(eps)       # (*, nblk, 1)
        t = torch.clamp(torch.round(Wb / scale), -1.0, 1.0)
        t = t.reshape(*shp[:-1], nblk * g)[..., :last].contiguous()
        return t, scale.squeeze(-1)                                     # scale (*, nblk)
    if per_channel and W.dim() >= 2:
        scale = W.abs().mean(dim=tuple(range(1, W.dim())), keepdim=True).clamp_min(eps)
    else:
        scale = W.abs().mean().clamp_min(eps)
    t = torch.clamp(torch.round(W / scale), -1.0, 1.0)
    return t, scale


def int_fake_quant(x: torch.Tensor, bits: int = 4, per_channel: bool = False,
                   signed: bool = True, eps: float = 1e-8, group_size=None):
    """Symmetric k-bit fake-quant (dequantized back to float). Returns (x_q, scale). Use bits=4 for int4
    weights, bits=8 for int8 activations. per_channel scales per row (dim 0) — the standard weight granularity.

    qmax = 2^(bits-1)-1 (signed) or 2^bits-1 (unsigned); scale = amax/qmax; x_q = round(x/scale)·scale.

    group_size (Bonsai/BitNet g128 block scaling): when set the LAST dim is chunked into groups of group_size
    and one absmax scale is computed PER GROUP; returns the dequantized x_q (same shape) and scale of shape
    (*x.shape[:-1], n_groups). Default None = the original per-tensor / per-channel path, byte-identical.
    """
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x.float()
    bits = int(bits)
    qmax = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    qmax = max(qmax, 1)
    qmin = -qmax if signed else 0
    if group_size:
        g = int(group_size); shp = x.shape; last = shp[-1]; pad = (-last) % g
        xp = F.pad(x, (0, pad)) if pad else x
        nblk = (last + pad) // g
        xb = xp.reshape(*shp[:-1], nblk, g)
        amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
        scale = amax / qmax
        q = torch.clamp(torch.round(xb / scale), qmin, qmax) * scale
        q = q.reshape(*shp[:-1], nblk * g)[..., :last].contiguous()
        return q, scale.squeeze(-1)                                     # scale (*, nblk)
    if per_channel and x.dim() >= 2:
        amax = x.abs().amax(dim=tuple(range(1, x.dim())), keepdim=True).clamp_min(eps)
    else:
        amax = x.abs().amax().clamp_min(eps)
    scale = amax / qmax
    q = torch.clamp(torch.round(x / scale), qmin, qmax)
    return q * scale, scale


# ---------------------------------------------------------------------------------------------------------
# MX (Microscaling) formats — Kimi-K3's training precision: MXFP4 weights + MXFP8 activations.
# OCP Microscaling Formats spec v1.0: a BLOCK of `block_size` (=32) elements shares ONE 8-bit power-of-two
# (E8M0) scale; each element is a low-bit float (FP4 E2M1 / FP8 E4M3 / FP8 E5M2). The shared power-of-two
# scale gives a wide dynamic range at ~4-8 bits/element — this is why K3 can TRAIN in fp4/fp8 where flat
# int4/int8 (one linear scale, no per-block exponent) loses too much range on heavy-tailed weights/acts.
# ---------------------------------------------------------------------------------------------------------
_MX_FORMATS = {   # (exp_bits, mant_bits, elem_emax, max_normal)
    "e2m1": (2, 1, 2, 6.0),           # FP4  — MXFP4 weights
    "e4m3": (4, 3, 8, 448.0),         # FP8  — MXFP8 activations (default)
    "e5m2": (5, 2, 15, 57344.0),      # FP8  — wider range, less precision
}


def _mx_elem_grid(fmt: str):
    """Sorted positive representable magnitudes of an MX element float format (incl. 0 and subnormals)."""
    eb, mb, _, _ = _MX_FORMATS[fmt]
    bias = (1 << (eb - 1)) - 1
    vals = {0.0}
    # subnormals: exponent field 0 → value = m/2^mb * 2^(1-bias)
    for m in range(1 << mb):
        vals.add(m / (1 << mb) * (2.0 ** (1 - bias)))
    # normals: exponent field e in [1, 2^eb-1] (top may be inf/nan in IEEE but MX FP8 uses it for finite maxes)
    for e in range(1, (1 << eb)):
        for m in range(1 << mb):
            vals.add((1.0 + m / (1 << mb)) * (2.0 ** (e - bias)))
    g = sorted(v for v in vals if v <= _MX_FORMATS[fmt][3] + 1e-9)
    return torch.tensor(g, dtype=torch.float32)


def _quant_to_grid(x: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Round each |x| to the nearest value in a sorted positive `grid`, keeping sign. Pure torch."""
    sign = torch.sign(x); ax = x.abs()
    idx = torch.searchsorted(grid, ax.clamp(max=float(grid[-1])))
    idx = idx.clamp(max=grid.numel() - 1)
    lo = grid[(idx - 1).clamp(min=0)]; hi = grid[idx]
    q = torch.where((ax - lo).abs() <= (hi - ax).abs(), lo, hi)
    return sign * q


def mxfp_quantize(x: torch.Tensor, fmt: str = "e2m1", block_size: int = 32, eps: float = 1e-12):
    """Microscaling (MX) fake-quant along the LAST dim: blocks of `block_size` share one E8M0 (power-of-two)
    scale, each element quantized to the `fmt` float grid. Returns (x_q dequantized, shared_scales).
    fmt='e2m1'→MXFP4 (weights), 'e4m3'/'e5m2'→MXFP8 (activations). Byte-shape-preserving fake-quant for QAT.

    Shared scale (OCP): X = 2^(floor(log2(blockmax)) - elem_emax) so the block's largest element lands near
    the top of the element range; then elem = round_to_grid(x / X); x_q = elem * X."""
    if fmt not in _MX_FORMATS:
        raise ValueError(f"unknown MX format {fmt!r}; use one of {list(_MX_FORMATS)}")
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x.float()
    _, _, elem_emax, _ = _MX_FORMATS[fmt]
    grid = _mx_elem_grid(fmt).to(x.device)
    shp = x.shape; last = shp[-1]; g = int(block_size); pad = (-last) % g
    xp = F.pad(x, (0, pad)) if pad else x
    nblk = (last + pad) // g
    xb = xp.reshape(*shp[:-1], nblk, g)
    blockmax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    shared_exp = torch.floor(torch.log2(blockmax)) - elem_emax     # E8M0 power-of-two exponent
    scale = torch.exp2(shared_exp)
    q = _quant_to_grid(xb / scale, grid) * scale
    q = q.reshape(*shp[:-1], nblk * g)[..., :last].contiguous()
    return q, scale.squeeze(-1)


def gpu_compute_capability():
    """(major, minor) CUDA compute capability of the current GPU, or None. 5090=(12,0) Blackwell sm_120;
    T4=(7,5) sm_75; A100=(8,0). Used to pick the natively-supported low-bit format."""
    try:
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:  # noqa: BLE001
        pass
    return None


def preferred_4bit_scheme(cap=None):
    """Capability-GATED default 4-bit scheme (adopt NVFP4 everywhere it's NATIVE, fall back otherwise):
      • sm_120 (Blackwell / RTX 5090)         → 'nvfp4'   — native FP4 tensor cores (1.5× faster, best acc).
      • sm_100 (B200)                          → 'nvfp4'.
      • sm_89 (Ada / 4090) / sm_90 (H100)      → 'mxfp4'   — FP4 emulated via block-scaling (no native FP4 conv).
      • sm_75 (Kaggle 2×T4) / older            → 'int8'    — NO FP4 support at all; int8 is the real lever there.
      • no CUDA                                → 'int8'.
    Pass cap=(major,minor) to override (e.g. plan for a target device). This is what makes 'adopt NVFP4
    everywhere possible' safe: it's the default ON Blackwell and degrades correctly on T4 submissions."""
    cap = cap or gpu_compute_capability()
    if cap is None:
        return "int8"
    major, minor = cap
    if major >= 10:                     # sm_100 (B200), sm_120 (5090) — native NVFP4
        return "nvfp4"
    if (major, minor) >= (8, 9):        # Ada/Hopper — FP4 by emulation
        return "mxfp4"
    return "int8"                       # Turing/T4 and older — no FP4


def nvfp4_quantize(x: torch.Tensor, block_size: int = 16, eps: float = 1e-12):
    """NVIDIA NVFP4 (Blackwell-native 4-bit, the format behind Unsloth's Gemma-4 NVFP4 quants). Differs from
    MXFP4 in three ways that matter on sm_120 (RTX 5090): (1) 16-element micro-blocks (MXFP4 uses 32),
    (2) each block's shared scale is an FP8 E4M3 float — a FINE, non-power-of-2 scale — not MXFP4's E8M0
    power-of-two, and (3) a second PER-TENSOR FP32 global scale sits above the block scales (two-level scaling).
    The finer block scale + global scale is why NVFP4 keeps more accuracy than MXFP4 at the same 4 bits, and
    Blackwell tensor cores run it natively (1.5× faster inference, Gemma-4-12B in ~11GB).

    Recipe: global = amax(|x|)/ (fp4_max · e4m3_max) → FP32 tensor scale; per-16-block scale = amax(|block|/global)
    / fp4_max, itself quantized to the E4M3 grid; elem = round_to_E2M1(x / (global·block_scale)). Returns
    (x_q dequantized, block_scales, global_scale). Fake-quant (shape-preserving) for QAT / accuracy estimation."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    x = x.float()
    fp4_grid = _mx_elem_grid("e2m1").to(x.device); fp4_max = float(fp4_grid[-1])          # 6.0
    e4m3_grid = _mx_elem_grid("e4m3").to(x.device); e4m3_max = float(e4m3_grid[-1])        # 448.0
    g = int(block_size); shp = x.shape; last = shp[-1]; pad = (-last) % g
    xp = F.pad(x, (0, pad)) if pad else x
    nblk = (last + pad) // g
    xb = xp.reshape(*shp[:-1], nblk, g)
    # per-tensor FP32 global scale so block scales land in the E4M3 range
    glob = (x.abs().amax() / (fp4_max * e4m3_max)).clamp_min(eps)
    blk_amax = xb.abs().amax(dim=-1, keepdim=True).clamp_min(eps)
    raw_bscale = blk_amax / (fp4_max * glob)                                   # target per-block scale
    bscale = _quant_to_grid(raw_bscale, e4m3_grid).clamp_min(eps)             # quantized to E4M3 (real NVFP4)
    scale = glob * bscale
    q = _quant_to_grid(xb / scale, fp4_grid) * scale
    q = q.reshape(*shp[:-1], nblk * g)[..., :last].contiguous()
    return q, bscale.squeeze(-1), glob


def nvfp4_effective_bits(block_size: int = 16, e4m3_bits: int = 8) -> float:
    """NVFP4 storage bits/weight: 4-bit element + one E4M3 (8b) block scale amortized over `block_size` + a
    negligible per-tensor FP32 global. @b16: 4 + 8/16 = 4.5 bits/w."""
    return round(4 + e4m3_bits / max(int(block_size), 1), 4)


def ste_nvfp4(x: torch.Tensor, block_size: int = 16) -> torch.Tensor:
    """NVFP4 fake-quant with a straight-through gradient (detach trick) — the QAT cell for training/finetuning
    an NVFP4 model on Blackwell (RTX 5090 sm_120)."""
    xq, _, _ = nvfp4_quantize(x, block_size=block_size)
    return x + (xq.to(x.dtype) - x).detach()


def mx_effective_bits(fmt: str = "e2m1", block_size: int = 32, scale_bits: int = 8) -> float:
    """MX storage bits/element: elem bits (1+exp+mant) + shared E8M0 scale amortized over the block.
    MXFP4 e2m1 @ b32: 4 + 8/32 = 4.25 bits/w. MXFP8 e4m3 @ b32: 8 + 8/32 = 8.25 bits/act."""
    eb, mb, _, _ = _MX_FORMATS[fmt]
    elem = 1 + eb + mb
    return round(elem + scale_bits / max(int(block_size), 1), 4)


def effective_bits(scheme: str = "ternary", group_size: int = 128, scale_bits: int = 16) -> float:
    """Bonsai/BitNet group-wise storage accounting (bits per weight), one FP16 scale per group.
    ternary → log2(3)+scale_bits/group ≈ 1.585 + 16/128 = 1.71; onebit → 1 + 16/128 = 1.125; intN → N.
    """
    import math
    s = str(scheme).lower()
    if s == "ternary":
        return round(math.log2(3) + scale_bits / max(group_size, 1), 4)
    if s in ("onebit", "1bit", "binary", "sign"):
        return round(1.0 + scale_bits / max(group_size, 1), 4)
    if s.startswith("int"):
        try:
            return float(int(s[3:]))
        except ValueError:
            return 4.0
    return round(1.585 + scale_bits / max(group_size, 1), 4)


def gemma4_format_bits(fmt: str = "q4_0", block_size: int = 32) -> float:
    """Bits-per-weight for the two Gemma-4 QAT weight formats (arXiv 2607.02770, §2.5), so we can
    estimate a quantized checkpoint's footprint like Table 3 without a model.
      • 'q4_0'   : blockwise 4-bit weights + one fp16 scale per block → 4 + 16/block_size.
      • 'mobile' : per-channel low-bitwidth weights, a mix of int2 and int4 (report says int2 AND int4);
                   modelled as the 3-bit average + a per-block fp16 scale. Activations are int8 (not a
                   weight cost). Pass fmt='int8'/'bf16'/'fp16' for the raw-precision baselines.
    """
    f = str(fmt).lower().replace("_", "").replace("-", "")
    if f in ("bf16", "fp16", "f16"):
        return 16.0
    if f in ("fp32", "f32"):
        return 32.0
    if f == "int8":
        return 8.0
    if f in ("q40", "q4"):
        return round(4.0 + 16.0 / max(1, block_size), 4)
    if f == "mobile":                       # int2/int4 mix ≈ 3-bit avg + fp16 block scale
        return round(3.0 + 16.0 / max(1, block_size), 4)
    return round(4.0 + 16.0 / max(1, block_size), 4)


def gemma4_quant_footprint(n_params, fmt: str = "q4_0", block_size: int = 32) -> dict:
    """Quantized weight footprint in GB for `n_params` weights under a Gemma-4 format, plus the
    compression ratio vs the bf16 raw checkpoint (the Table 3 comparison). n_params may be a raw count
    or a count in billions (values < 1e6 are treated as billions)."""
    n = float(n_params)
    if n < 1e6:                              # given in billions
        n *= 1e9
    bits = gemma4_format_bits(fmt, block_size)
    gb = n * bits / 8.0 / 1e9
    bf16_gb = n * 16.0 / 8.0 / 1e9
    return {"format": fmt, "bits_per_weight": bits, "gb": round(gb, 4),
            "bf16_gb": round(bf16_gb, 4), "compression_vs_bf16": round(bf16_gb / gb, 3) if gb > 0 else 0.0}


# ---------------------------------------------------------------------------------------------------------
# 2. straight-through estimators — quantize on the forward, pass the gradient through on the backward
# ---------------------------------------------------------------------------------------------------------
class STETernary(torch.autograd.Function):
    """Ternary fake-quant with a straight-through gradient. forward = t·scale (BitNet absmean ternary),
    backward = identity (dL/dW = dL/dW_q). This is what lets a ternary net TRAIN: the FP master weight
    receives real gradients even though the forward pass only ever sees {-scale,0,+scale}."""
    @staticmethod
    def forward(ctx, W, per_channel):
        t, scale = ternary_quantize(W, bool(per_channel))
        return t * scale

    @staticmethod
    def backward(ctx, g):
        return g, None


class STEQuant(torch.autograd.Function):
    """k-bit fake-quant with a straight-through gradient (identity backward)."""
    @staticmethod
    def forward(ctx, x, bits, per_channel, signed):
        xq, _ = int_fake_quant(x, int(bits), bool(per_channel), bool(signed))
        return xq

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


def ste_ternary(W: torch.Tensor, per_channel: bool = False) -> torch.Tensor:
    return STETernary.apply(W, per_channel)


def ste_quant(x: torch.Tensor, bits: int = 4, per_channel: bool = False, signed: bool = True) -> torch.Tensor:
    return STEQuant.apply(x, bits, per_channel, signed)


_MXFP_SCHEME = {"mxfp4": "e2m1", "mxfp8": "e4m3", "mxfp8e5m2": "e5m2"}


def ste_mxfp(x: torch.Tensor, fmt: str = "e2m1", block_size: int = 32) -> torch.Tensor:
    """MX (microscaling) fake-quant with a straight-through gradient (identity backward), via the detach
    trick: forward returns the MX-quantized value, backward passes the gradient straight to the FP master.
    This is the K3 QAT cell — MXFP4 weights (fmt='e2m1') / MXFP8 activations (fmt='e4m3')."""
    xq, _ = mxfp_quantize(x, fmt=fmt, block_size=block_size)
    return x + (xq.to(x.dtype) - x).detach()


# ---------------------------------------------------------------------------------------------------------
# 3. QuantLinear — a QAT cell: full-precision master weight, fake-quant forward
# ---------------------------------------------------------------------------------------------------------
class QuantLinear(nn.Module):
    """Drop-in for nn.Linear that keeps a full-precision master weight and quantizes it (and optionally the
    activation) on the forward pass through an STE. scheme='ternary' → BitNet b1.58; scheme='int4' → 4-bit
    fake-quant. a_bits>0 quantizes the input activation to a_bits (per-tensor, e.g. 8) — the W-and-A path."""

    def __init__(self, in_features, out_features, bias=True, scheme="ternary", w_bits=4,
                 a_bits=0, per_channel=True, device=None, dtype=None, group_size=None, act_bits=None,
                 act_mx=None, act_mx_block=32):
        super().__init__()
        self.act_mx = act_mx                                   # K3 MXFP8 activations, e.g. 'e4m3' (None=off)
        self.act_mx_block = int(act_mx_block)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.scheme = str(scheme)
        self.w_bits = int(w_bits)
        self.a_bits = int(a_bits)
        self.per_channel = bool(per_channel)
        self.group_size = int(group_size) if group_size else None
        self.act_bits = int(act_bits) if act_bits else None    # BitNet-style per-token activation quant (W+A)
        self.quant_lambda = 1.0                                # gradual-quant ramp knob (0=fp master, 1=full quant)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def quant_weight(self) -> torch.Tensor:
        lam = getattr(self, "quant_lambda", 1.0)
        gs = getattr(self, "group_size", None)
        if self.scheme == "nvfp4":                             # NVIDIA NVFP4 (Blackwell-native, Gemma-4 quants)
            wq = ste_nvfp4(self.weight, block_size=gs or 16)
            return self.weight + float(lam) * (wq - self.weight).detach() if lam < 1.0 else wq
        if self.scheme in _MXFP_SCHEME:                        # Kimi-K3 MXFP weights (block-microscaled)
            fmt = _MXFP_SCHEME[self.scheme]; bs = gs or 32
            wq = ste_mxfp(self.weight, fmt=fmt, block_size=bs)
            return self.weight + float(lam) * (wq - self.weight).detach() if lam < 1.0 else wq
        if gs is None and lam >= 1.0:                          # EXISTING path, byte-identical
            if self.scheme == "ternary":
                return ste_ternary(self.weight, self.per_channel)
            return ste_quant(self.weight, self.w_bits, self.per_channel, signed=True)
        # extended path: group-wise scaling and/or gradual ramp, STE via detach-trick (identity backward).
        w = self.weight
        if self.scheme == "ternary":
            t, scale = ternary_quantize(w, self.per_channel, group_size=gs)
            wq = dequant_ternary(t, scale, gs)
        else:
            wq, _ = int_fake_quant(w, self.w_bits, self.per_channel, signed=True, group_size=gs)
        wq = wq.to(w.dtype)
        return w + float(lam) * (wq - w).detach()             # lam=1 → wq (STE); lam=0 → fp master

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if getattr(self, "act_mx", None):                     # Kimi-K3 MXFP8 activations (block-microscaled)
            x = ste_mxfp(x, fmt=self.act_mx, block_size=getattr(self, "act_mx_block", 32))
        elif self.act_bits:                                   # BitNet-style dynamic per-token activation quant
            x = act_fake_quant(x, bits=self.act_bits, per_token=True)
        elif self.a_bits > 0:
            x = ste_quant(x, self.a_bits, per_channel=False, signed=True)   # activations per-tensor (legacy)
        return F.linear(x, self.quant_weight(), self.bias)

    @classmethod
    def from_linear(cls, lin: nn.Linear, scheme="ternary", w_bits=4, a_bits=0, per_channel=True,
                    group_size=None, act_bits=None):
        q = cls(lin.in_features, lin.out_features, bias=lin.bias is not None, scheme=scheme,
                w_bits=w_bits, a_bits=a_bits, per_channel=per_channel,
                device=lin.weight.device, dtype=lin.weight.dtype, group_size=group_size, act_bits=act_bits)
        with torch.no_grad():
            q.weight.copy_(lin.weight)
            if lin.bias is not None:
                q.bias.copy_(lin.bias)
        return q

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"scheme={self.scheme}, w_bits={self.w_bits}, a_bits={self.a_bits}, "
                f"act_bits={self.act_bits}, group_size={self.group_size}, per_channel={self.per_channel}")


# ---------------------------------------------------------------------------------------------------------
# 4. wrap_qat — swap Linear→QuantLinear, keep norms/embeddings/head in higher precision
# ---------------------------------------------------------------------------------------------------------
DEFAULT_SKIP = ("norm", "ln", "layernorm", "rmsnorm", "embed", "lm_head", "head", "wte", "wpe")


def wrap_qat(model: nn.Module, bits: int = 4, scheme: str = "ternary", a_bits: int = 0,
             per_channel: bool = True, skip=DEFAULT_SKIP, group_size=None, act_bits=None):
    """In-place swap every nn.Linear whose qualified name does NOT contain a `skip` token into a QuantLinear
    (master weight copied). Embeddings/norms/lm_head stay full precision (Bonsai & BitNet both keep these in
    higher precision for stability). Returns (model, n_swapped).

    group_size — Bonsai/BitNet g128 group-wise weight scaling for the swapped layers (None = per-tensor/channel).
    act_bits   — BitNet-style dynamic per-token activation quant (None = weight-only). Both default None = unchanged."""
    skip = tuple(s.lower() for s in (skip or ()))
    n_swapped = 0

    def _skip(qual: str) -> bool:
        q = qual.lower()
        return any(s in q for s in skip)

    def _recurse(module: nn.Module, prefix: str = ""):
        nonlocal n_swapped
        for name, child in list(module.named_children()):
            qual = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not _skip(qual):
                setattr(module, name, QuantLinear.from_linear(
                    child, scheme=scheme, w_bits=bits, a_bits=a_bits, per_channel=per_channel,
                    group_size=group_size, act_bits=act_bits))
                n_swapped += 1
            else:
                _recurse(child, qual)

    _recurse(model)
    return model, n_swapped


# ---------------------------------------------------------------------------------------------------------
# 5. qat_finetune — tiny STE training loop (proves a quantized net still learns)
# ---------------------------------------------------------------------------------------------------------
def qat_finetune(model: nn.Module, data: torch.Tensor, target: torch.Tensor, steps: int = 150,
                 lr: float = 5e-3, loss_fn=None):
    """Minimal Adam loop. Returns the list of per-step losses. Because QuantLinear uses an STE, the FP master
    weights get real gradients and the loss falls even though the forward only sees quantized weights."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    lf = loss_fn or nn.MSELoss()
    losses = []
    model.train()
    for _ in range(int(steps)):
        opt.zero_grad()
        out = model(data)
        loss = lf(out, target)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses


# =========================================================================================================
# APPEND-ONLY EXTENSIONS (2026-07-16) — finishing low-bit TRAINING into an end-to-end capability.
# Everything above is unchanged in behaviour (new kwargs default to the original path). Grounded in
# BitNet b1.58 (arXiv:2402.17764) group-wise ternary + LLM-QAT (arXiv:2305.17888) W&A fake-quant, plus the
# 4-/8-bit block-quantized optimizer-state idea (8-bit Adam, arXiv:2110.02861) — pure torch, no new deps.
# =========================================================================================================

def dequant_ternary(t: torch.Tensor, scale: torch.Tensor, group_size=None) -> torch.Tensor:
    """Dequantize a ternary code. Per-tensor/channel: t*scale. Group-wise (scale shape (*, n_groups)):
    reblock the last dim, multiply per group, unblock. Inverse of ternary_quantize(..., group_size)."""
    if group_size is None:
        return t * scale
    g = int(group_size); shp = t.shape; last = shp[-1]; pad = (-last) % g
    tt = F.pad(t, (0, pad)) if pad else t
    nblk = (last + pad) // g
    tb = tt.reshape(*shp[:-1], nblk, g)
    deq = (tb * scale.unsqueeze(-1)).reshape(*shp[:-1], nblk * g)[..., :last]
    return deq.contiguous()


# ---------------------------------------------------------------------------------------------------------
# 5b. ACTIVATION quantization — true low-bit COMPUTE (W&A), dynamic per-token, STE via the detach trick
# ---------------------------------------------------------------------------------------------------------
def act_fake_quant(x: torch.Tensor, bits: int = 8, per_token: bool = True, eps: float = 1e-8) -> torch.Tensor:
    """Dynamic symmetric activation fake-quant with a straight-through gradient (LLM-QAT / BitNet W-ternary/A8).
    per_token=True → one absmax scale PER ROW (the last dim is the feature dim, each token scaled on the fly);
    per_token=False → per-tensor. Returns the fake-quantized activation (same shape); backward = identity, so
    it composes into training. Pure detach-STE (x + (q - x).detach()) — no autograd.Function needed."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32)
    xf = x.float()
    bits = int(bits)
    qmax = max((1 << (bits - 1)) - 1, 1)
    if per_token and xf.dim() >= 1:
        amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(eps)       # per-token (per-row over features)
    else:
        amax = xf.abs().amax().clamp_min(eps)
    scale = amax / qmax
    q = torch.clamp(torch.round(xf / scale), -qmax, qmax) * scale
    return (xf + (q - xf).detach()).to(x.dtype)                         # STE: forward=q, backward=identity


# ---------------------------------------------------------------------------------------------------------
# 5c. LowBitAdam — block-quantized optimizer states (the OTHER "4-bit training": a memory lever, 8-bit Adam)
# ---------------------------------------------------------------------------------------------------------
class LowBitAdam(torch.optim.Optimizer):
    """Adam whose exp_avg / exp_avg_sq moments are STORED block-quantized (per-block absmax int8/int4) and
    dequantized on use, then requantized after the update. Halves (int8) or quarters (int4) optimizer-state
    memory vs fp32 Adam — the memory lever that lets bigger detectors/heads fine-tune on the 5090 / T4.
    Pure torch; grounded in 8-bit Adam (Dettmers et al., arXiv:2110.02861). state_bits∈{8,4}, block=128."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0,
                 state_bits=8, block=128):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.state_bits = int(state_bits)
        self.block = int(block)

    def _q(self, x: torch.Tensor, nonneg: bool = False) -> dict:
        """Block-quantize a moment tensor (per-block absmax → int8/int4). nonneg=True stores the SQRT of a
        non-negative tensor (exp_avg_sq): sqrt compresses the wide dynamic range so linear low-bit quant keeps
        small second-moment values → the denominator stays well-conditioned (no divide-by-~0 blow-up)."""
        b = self.block
        r = x.reshape(-1)
        if nonneg:
            r = r.clamp_min(0).sqrt()
        n = r.numel()
        pad = (-n) % b
        if pad:
            r = torch.cat([r, r.new_zeros(pad)])
        blk = r.reshape(-1, b)
        qmax = max((1 << (self.state_bits - 1)) - 1, 1)
        scale = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / qmax
        code = torch.clamp(torch.round(blk / scale), -qmax, qmax).to(torch.int8)
        return {"code": code, "scale": scale.squeeze(1).to(torch.float32), "n": n,
                "shape": tuple(x.shape), "nonneg": bool(nonneg)}

    def _dq(self, s: dict) -> torch.Tensor:
        deq = (s["code"].float() * s["scale"].unsqueeze(1)).reshape(-1)[: s["n"]]
        if s.get("nonneg"):
            deq = deq.clamp_min(0).pow(2)
        return deq.reshape(s["shape"])

    def state_bytes(self) -> int:
        """Real stored bytes of the (packed) optimizer state: state_bits per moment element + fp32 per-block scale."""
        import math
        total = 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                for key in ("exp_avg_q", "exp_avg_sq_q"):
                    s = st.get(key)
                    if s is not None:
                        total += math.ceil(s["code"].numel() * self.state_bits / 8)   # packed moment codes
                        total += s["scale"].numel() * 4                               # fp32 per-block scale
        return total

    def fp32_state_bytes(self) -> int:
        """Bytes an fp32 Adam would use for the same states (2 moments × 4 bytes × numel)."""
        total = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    total += 2 * p.numel() * 4
        return total

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            b1, b2 = group["betas"]; lr = group["lr"]; eps = group["eps"]; wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("LowBitAdam does not support sparse gradients")
                st = self.state[p]
                if not st:
                    st["step"] = 0
                    st["exp_avg_q"] = self._q(torch.zeros_like(p, dtype=torch.float32))
                    st["exp_avg_sq_q"] = self._q(torch.zeros_like(p, dtype=torch.float32), nonneg=True)
                st["step"] += 1
                m = self._dq(st["exp_avg_q"]).to(p.device)
                v = self._dq(st["exp_avg_sq_q"]).to(p.device)
                gf = g.float()
                if wd:
                    gf = gf.add(p.float(), alpha=wd)
                m.mul_(b1).add_(gf, alpha=1 - b1)
                v.mul_(b2).addcmul_(gf, gf, value=1 - b2)
                bc1 = 1 - b1 ** st["step"]; bc2 = 1 - b2 ** st["step"]
                denom = (v.sqrt() / (bc2 ** 0.5)).add_(eps)
                p.addcdiv_(m, denom, value=-lr / bc1)
                st["exp_avg_q"] = self._q(m)
                st["exp_avg_sq_q"] = self._q(v, nonneg=True)
        return loss


# ---------------------------------------------------------------------------------------------------------
# 5d. lowbit_finetune — the complete hardware-aware, gradual QAT fine-tune loop
# ---------------------------------------------------------------------------------------------------------
def lowbit_finetune(model: nn.Module, batches, *, scheme="ternary", weight_bits=None, act_bits=None,
                    group_size=128, keep_fp=("norm", "embed", "lm_head"), gradual=True, warmup_frac=0.3,
                    optimizer="lowbit", loss_fn=None, hardware_aware=True, epochs=1, lr=5e-3):
    """End-to-end low-bit QAT fine-tune. Wraps the model (skipping keep_fp layers) into QuantLinear cells with
    FP master weights, then trains under STE. GRADUAL: a lambda ramps 0→1 over warmup_frac of the steps,
    interpolating fp↔quantized weights in the forward so training isn't shocked, then full quant. HARDWARE-AWARE:
    reads hardware_tune.load_config() and autocasts the master-weight compute in its amp_dtype (bf16 on the 5090).
    optimizer='lowbit' → LowBitAdam (block-quantized states), else AdamW. Grounded in BitNet b1.58 + LLM-QAT.

    batches: iterable of (x, y). Returns {loss_curve, initial_loss, final_loss, effective_bits, amp_dtype_used,
    quantized_layers, scheme, group_size, optimizer}."""
    batches = list(batches)
    scheme = str(scheme)
    wbits = int(weight_bits) if weight_bits else 4
    model, n_swapped = wrap_qat(model, bits=wbits, scheme=scheme, per_channel=True, skip=tuple(keep_fp),
                                group_size=group_size, act_bits=act_bits)
    try:
        dev = next(model.parameters()).device
    except StopIteration:
        dev = torch.device("cpu")

    # hardware-aware amp dtype (bf16 on the 5090); autocast only when CUDA + a low-precision amp_dtype is set.
    amp_dtype_used = "fp32"; amp_dtype = None; cfg = {}
    if hardware_aware:
        try:
            from . import hardware_tune
            cfg = hardware_tune.load_config() or {}
        except Exception:  # noqa: BLE001
            cfg = {}
    use_cuda = torch.cuda.is_available() and dev.type == "cuda"
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(cfg.get("amp_dtype"))
    if use_cuda and amp_dtype is not None:
        amp_dtype_used = cfg.get("amp_dtype")

    params = [p for p in model.parameters() if p.requires_grad]
    if str(optimizer) == "lowbit":
        opt = LowBitAdam(params, lr=lr, state_bits=8, block=max(int(group_size or 128), 8))
    else:
        opt = torch.optim.AdamW(params, lr=lr)
    lf = loss_fn or nn.MSELoss()

    qlins = [m for m in model.modules() if isinstance(m, QuantLinear)]
    total = max(1, int(epochs) * len(batches))
    warm = max(1, int(warmup_frac * total))
    losses = []; step = 0
    model.train()
    for _ep in range(int(epochs)):
        for x, y in batches:
            lam = 1.0 if not gradual else min(1.0, step / warm)
            for m in qlins:
                m.quant_lambda = float(lam)
            opt.zero_grad()
            if use_cuda and amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    out = model(x); loss = lf(out, y)
            else:
                out = model(x); loss = lf(out, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach())); step += 1

    ebits = effective_bits("ternary" if scheme == "ternary" else f"int{wbits}", int(group_size or 128))
    return {"loss_curve": losses, "initial_loss": losses[0], "final_loss": losses[-1],
            "effective_bits": ebits, "amp_dtype_used": amp_dtype_used, "quantized_layers": n_swapped,
            "scheme": scheme, "group_size": int(group_size or 0), "optimizer": str(optimizer)}


# ---------------------------------------------------------------------------------------------------------
# 5e. REAL packing + memory — make the win actual (not simulated): trit/nibble packing, KV-cache int4
# ---------------------------------------------------------------------------------------------------------
def pack_ternary(q: torch.Tensor, group_size=None) -> dict:
    """Pack a ternary code {-1,0,+1} at ~1.6 bits/weight — 5 trits per byte (3^5=243<256). Exact, reversible.
    Returns {packed:uint8, n, shape, group_size, bpw}. Inverse: unpack_ternary."""
    orig = tuple(q.shape)
    c = (q.reshape(-1).round().to(torch.long) + 1)                     # {-1,0,1} -> {0,1,2}
    n = c.numel(); pad = (-n) % 5
    if pad:
        c = torch.cat([c, c.new_zeros(pad)])
    c5 = c.reshape(-1, 5)
    w = torch.tensor([1, 3, 9, 27, 81], device=c.device, dtype=torch.long)
    packed = (c5 * w).sum(dim=1).to(torch.uint8)                       # base-3, 5 trits/byte
    return {"packed": packed, "n": n, "shape": orig, "group_size": group_size, "bpw": 1.6}


def unpack_ternary(d: dict) -> torch.Tensor:
    x = d["packed"].to(torch.long).clone()
    trits = []
    for _ in range(5):
        trits.append(x % 3); x = x // 3
    c = torch.stack(trits, dim=1).reshape(-1)[: d["n"]].to(torch.float32) - 1.0
    return c.reshape(d["shape"])


def pack_int4(codes: torch.Tensor, group_size=None) -> dict:
    """Pack signed 4-bit integer CODES (values in [-8,7]) two nibbles per byte. Exact, reversible.
    Returns {packed:uint8, n, shape, group_size}. Inverse: unpack_int4."""
    orig = tuple(codes.shape)
    c = torch.clamp(codes.reshape(-1).round().to(torch.long), -8, 7) + 8      # -> {0..15}
    n = c.numel(); pad = n % 2
    if pad:
        c = torch.cat([c, c.new_zeros(1)])
    c2 = c.reshape(-1, 2)
    packed = (c2[:, 0] | (c2[:, 1] << 4)).to(torch.uint8)
    return {"packed": packed, "n": n, "shape": orig, "group_size": group_size}


def unpack_int4(d: dict) -> torch.Tensor:
    p = d["packed"].to(torch.long)
    lo = p & 0xF; hi = (p >> 4) & 0xF
    c = torch.stack([lo, hi], dim=1).reshape(-1)[: d["n"]].to(torch.float32) - 8.0
    return c.reshape(d["shape"])


def _count_weight_elems(model_or_tensor) -> int:
    if torch.is_tensor(model_or_tensor):
        return int(model_or_tensor.numel())
    if isinstance(model_or_tensor, nn.Module):
        tot = 0
        for m in model_or_tensor.modules():
            if isinstance(m, (nn.Linear, QuantLinear)):
                tot += int(m.weight.numel())
        return tot
    # a shape tuple/list
    try:
        import math
        return int(math.prod(model_or_tensor))
    except Exception:  # noqa: BLE001
        return 0


def effective_memory_bytes(model_or_tensor, scheme="ternary", group_size=128) -> dict:
    """REAL packed footprint (incl. per-group fp16 scales) vs fp16/fp32 baselines → compression ratio.
    ternary = 5 trits/byte (~1.6bpw), int4 = 2/byte, int8 = 1/byte, onebit = 8/byte. Accepts a tensor,
    an nn.Module (sums Linear/QuantLinear weights), or a shape."""
    import math
    N = _count_weight_elems(model_or_tensor)
    s = str(scheme).lower()
    if s == "ternary":
        packed = math.ceil(N / 5)
    elif s in ("int4", "4bit"):
        packed = math.ceil(N / 2)
    elif s in ("int8", "8bit"):
        packed = N
    elif s in ("onebit", "1bit", "binary", "sign"):
        packed = math.ceil(N / 8)
    else:
        packed = math.ceil(N / 5)
    nblocks = math.ceil(N / max(int(group_size), 1)) if N else 0
    scale_bytes = nblocks * 2                                          # one fp16 scale per group
    total = packed + scale_bytes
    fp16 = N * 2; fp32 = N * 4
    return {"elements": N, "packed_bytes": packed, "scale_bytes": scale_bytes, "total_bytes": total,
            "fp16_bytes": fp16, "fp32_bytes": fp32,
            "ratio_vs_fp16": round(fp16 / total, 2) if total else 0.0,
            "ratio_vs_fp32": round(fp32 / total, 2) if total else 0.0,
            "scheme": s, "group_size": int(group_size)}


def quantize_kv(k: torch.Tensor, v: torch.Tensor, bits: int = 4):
    """Per-token/per-head absmax int4 KV-cache quant (Bonsai's actual '4-bit' = a 4-bit KV cache). Scales over
    the last (head) dim so each token/head is scaled independently. Returns (kq, vq) dicts. Inverse: dequantize_kv."""
    bits = int(bits); qmax = max((1 << (bits - 1)) - 1, 1)

    def _q(x):
        xf = x.float()
        scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / qmax
        code = torch.clamp(torch.round(xf / scale), -qmax, qmax).to(torch.int8)
        return {"code": code, "scale": scale, "bits": bits}

    return _q(k), _q(v)


def dequantize_kv(kq: dict, vq: dict):
    def _d(s):
        return s["code"].float() * s["scale"]
    return _d(kq), _d(vq)


# ---------------------------------------------------------------------------------------------------------
# 6. the agent
# ---------------------------------------------------------------------------------------------------------
class LowbitQAT(BaseAgent):
    name = "lowbit-qat"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        mode = str(spec.get("mode", "primitives"))
        if mode == "finetune":
            return self._run_finetune(spec, worker)
        if mode == "memory":
            return self._run_memory(spec, worker)
        scheme = str(spec.get("scheme", "ternary"))
        bits = int(spec.get("bits", 4))
        a_bits = int(spec.get("a_bits", 0))
        group_size = int(spec.get("group_size", 128))
        steps = int(spec.get("steps", 150))
        seed = int(spec.get("seed", 0))
        dev = "cuda" if (torch.cuda.is_available() and spec.get("gpu", True)) else "cpu"
        torch.manual_seed(seed)

        # a tiny MLP, wrapped for QAT, trained on a synthetic regression to show STE lets it learn
        model = nn.Sequential(
            nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1)
        ).to(dev)
        n_lin = sum(isinstance(m, nn.Linear) for m in model.modules())
        model, n_swapped = wrap_qat(model, bits=bits, scheme=scheme, a_bits=a_bits, per_channel=True)
        model = model.to(dev)

        X = torch.randn(512, 16, device=dev)
        Wt = torch.randn(16, 1, device=dev)
        y = (X @ Wt) + 0.1 * torch.randn(512, 1, device=dev)
        losses = qat_finetune(model, X, y, steps=steps, lr=5e-3)
        l0, l1 = losses[0], losses[-1]
        learned = l1 < 0.7 * l0

        beff = effective_bits(scheme, group_size)
        comp = round(16.0 / beff, 2)   # vs bf16/fp16 storage
        msg = (f"lowbit-qat: {scheme} QAT ({'w%d/a%d' % (bits, a_bits) if scheme != 'ternary' else '~1.58b'}) — "
               f"swapped {n_swapped}/{n_lin} Linears (norms/embeds/head kept fp); STE trained tiny MLP "
               f"{l0:.3f}→{l1:.3f} ({'LEARNS' if learned else 'stalled'}). "
               f"effective {beff} bits/weight (g{group_size}) → ~{comp}× smaller than bf16.")
        self.log(msg, kind="finding",
                 recommendation=("use lowbit-qat to FINE-TUNE a detector/head under ternary/int4 weights (STE, "
                                 "pure torch) so it fits the 5090 (bf16 32GB) or T4 (16GB) at ~"
                                 f"{comp}× smaller — the QAT lever we lacked (we only had PTQ + depth-prune)."))
        return self.done({"scheme": scheme, "bits": bits, "a_bits": a_bits, "n_swapped": n_swapped,
                          "n_linear": n_lin, "loss_start": l0, "loss_end": l1, "learned": learned,
                          "effective_bits": beff, "compression_vs_bf16": comp, "device": dev}, msg)


    def _run_finetune(self, spec, worker):
        """PROVE the full loop end-to-end on a small built-in synthetic task."""
        scheme = str(spec.get("scheme", "ternary"))
        group_size = int(spec.get("group_size", 128))
        act_bits = spec.get("act_bits", None)
        optimizer = str(spec.get("optimizer", "lowbit"))
        epochs = int(spec.get("epochs", 3))
        seed = int(spec.get("seed", 0))
        dev = "cuda" if (torch.cuda.is_available() and spec.get("gpu", True)) else "cpu"
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                              nn.Linear(64, 1)).to(dev)
        Wt = torch.randn(16, 1, device=dev)
        batches = []
        for _ in range(8):
            X = torch.randn(128, 16, device=dev)
            Y = X @ Wt + 0.1 * torch.randn(128, 1, device=dev)
            batches.append((X, Y))
        res = lowbit_finetune(model, batches, scheme=scheme, group_size=group_size, act_bits=act_bits,
                              optimizer=optimizer, epochs=epochs, hardware_aware=True)
        fell = res["final_loss"] < res["initial_loss"]
        msg = (f"lowbit-qat[finetune]: {scheme} g{group_size}"
               f"{'/A%s' % act_bits if act_bits else ''} QAT via {optimizer} — "
               f"loss {res['initial_loss']:.4f}→{res['final_loss']:.4f} "
               f"({'FELL' if fell else 'flat'}), {res['effective_bits']} bits/weight, "
               f"{res['quantized_layers']} layers quantized, amp={res['amp_dtype_used']} (hardware-aware).")
        self.log(msg, kind="finding",
                 recommendation="lowbit_finetune() is the finished end-to-end low-bit QAT lever — gradual "
                                "quant + hardware-aware bf16 autocast + LowBitAdam states; fine-tune detectors "
                                "under ternary/int4 weights on the 5090 / T4.")
        return self.done({"mode": "finetune", **res, "loss_fell": fell, "device": dev}, msg)

    def _run_memory(self, spec, worker):
        """Report the REAL packed compression for a given shape/scheme."""
        import math
        shape = spec.get("shape", [4096, 4096])
        scheme = str(spec.get("scheme", "ternary"))
        group_size = int(spec.get("group_size", 128))
        N = int(math.prod(shape))
        mem = effective_memory_bytes(tuple(shape), scheme=scheme, group_size=group_size)
        beff = effective_bits(scheme if scheme == "ternary" else f"int{4 if scheme in ('int4','4bit') else 8}",
                              group_size)
        msg = (f"lowbit-qat[memory]: {scheme} g{group_size} on {tuple(shape)} ({N:,} weights) — "
               f"packed {mem['total_bytes']:,}B vs fp16 {mem['fp16_bytes']:,}B → "
               f"{mem['ratio_vs_fp16']}× smaller (vs fp32 {mem['ratio_vs_fp32']}×); {beff} bits/weight.")
        self.log(msg, kind="finding",
                 recommendation="effective_memory_bytes() gives the REAL packed footprint (trit/nibble packing + "
                                "group scales) — size a detector/head for the 5090 (32GB) or T4 (16GB).")
        return self.done({"mode": "memory", **mem, "effective_bits": beff, "shape": list(shape)}, msg)


_AGENT = LowbitQAT()


def run(q, worker):
    return _AGENT.run(q, worker)
