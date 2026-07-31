"""precision_backends — the ONE canonical way this repo touches low-precision hardware on the RTX 5090
(sm_120 / Blackwell). All training/inference code MUST build its Linears and attention through here so the
whole fleet uniformly exploits the installed FP8/FP4 stack instead of each agent re-deciding.

Installed & verified in the `kaggle_nlp` conda env (torch 2.10.0+cu128):
  • transformer-engine 2.15.0  — FP8 GEMMs on Blackwell tensor cores  (make_linear backend "te")
  • flash-attn 2.8.3.post1      — IO-aware exact attention              (attention backend "flash")
  • nvidia-modelopt 0.45.0      — FP8/FP4 post-training quantization    (quantize_for_infer)

Preference order is auto-detected and overridable with env HW_BACKEND=te|native|torch (linears) and
HW_ATTENTION=flash|sdpa (attention). Everything degrades safely: on a T4 / when a lib is missing, this
falls back to the native fp8 path (Fp8Linear) or plain torch, so the same code runs everywhere.
"""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- capability detection (import-time, cheap)
def _blackwell() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(0)
    return major >= 9  # Hopper(9.0)/Blackwell(10.0,12.0) have native FP8 tensor cores

try:
    import transformer_engine.pytorch as _te          # noqa
    from transformer_engine.common.recipe import DelayedScaling as _DS, Format as _Fmt
    HAVE_TE = True
except Exception:
    HAVE_TE = False

try:
    from flash_attn import flash_attn_func as _flash_attn_func  # noqa
    HAVE_FLASH = True
except Exception:
    HAVE_FLASH = False

try:
    import modelopt.torch.quantization as _mtq         # noqa
    HAVE_MODELOPT = True
except Exception:
    HAVE_MODELOPT = False

# native fp8 fallback (no external lib) — the repo's own torch._scaled_mm Linear
try:
    from .fp8_cell_detector import Fp8Linear as _Fp8Linear
    HAVE_NATIVE_FP8 = True
except Exception:
    _Fp8Linear = None
    HAVE_NATIVE_FP8 = False


def linear_backend() -> str:
    """Which Linear backend will be used: 'te' | 'native' | 'torch'. Honours env HW_BACKEND."""
    forced = os.environ.get("HW_BACKEND", "").strip().lower()
    if forced in ("te", "native", "torch"):
        if forced == "te" and not (HAVE_TE and _blackwell()):
            return "native" if HAVE_NATIVE_FP8 else "torch"
        return forced
    if HAVE_TE and _blackwell():
        return "te"
    if HAVE_NATIVE_FP8:
        return "native"
    return "torch"


def attention_backend() -> str:
    """'flash' | 'sdpa'. Honours env HW_ATTENTION."""
    forced = os.environ.get("HW_ATTENTION", "").strip().lower()
    if forced in ("flash", "sdpa"):
        return "flash" if (forced == "flash" and HAVE_FLASH) else "sdpa"
    return "flash" if HAVE_FLASH else "sdpa"


# ---------------------------------------------------------------- Linear factory
def make_linear(in_f: int, out_f: int, bias: bool = True) -> nn.Module:
    """Build a Linear on the best available backend: TransformerEngine FP8 › native Fp8Linear › nn.Linear.
    TE Linears only run in FP8 inside `fp8_autocast()` (use autocast() below); outside it they run bf16.
    Dims should be 16-aligned for the FP8 GEMM path (both TE and native) or they silently run higher-precision."""
    be = linear_backend()
    if be == "te":
        return _te.Linear(in_f, out_f, bias=bias)
    if be == "native":
        return _Fp8Linear(in_f, out_f, bias=bias)
    return nn.Linear(in_f, out_f, bias=bias)


def autocast(enabled: bool = True):
    """FP8 autocast context. Wrap the forward pass with this so TE Linears actually use FP8 tensor cores.
    No-op (nullcontext) when TE isn't the active backend, so it's always safe to wrap."""
    import contextlib
    if enabled and linear_backend() == "te" and HAVE_TE:
        return _te.fp8_autocast(enabled=True, fp8_recipe=_DS(fp8_format=_Fmt.HYBRID))
    return contextlib.nullcontext()


# ---------------------------------------------------------------- attention
def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False,
              dropout_p: float = 0.0) -> torch.Tensor:
    """Exact attention on flash-attn when available, else SDPA. Convention: q,k,v are (B, S, H, D)
    (flash-attn's layout). Returns (B, S, H, D). flash-attn needs fp16/bf16 & CUDA; otherwise falls to SDPA."""
    if attention_backend() == "flash" and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
        return _flash_attn_func(q, k, v, dropout_p=dropout_p, causal=causal)
    # SDPA wants (B, H, S, D)
    qs, ks, vs = (t.transpose(1, 2) for t in (q, k, v))
    o = F.scaled_dot_product_attention(qs, ks, vs, dropout_p=dropout_p, is_causal=causal)
    return o.transpose(1, 2)


# ---------------------------------------------------------------- inference quantization (modelopt)
def quantize_for_infer(model: nn.Module, forward_loop=None, fmt: str = "fp8") -> nn.Module:
    """Post-training quantize a model for inference with nvidia-modelopt. fmt: 'fp8' | 'fp4'(nvfp4) | 'int8'.
    forward_loop(model) should run a few calibration batches. Returns the (in-place) quantized model; if
    modelopt is unavailable it returns the model untouched so callers never break."""
    if not HAVE_MODELOPT:
        return model
    cfg = {"fp8": _mtq.FP8_DEFAULT_CFG, "int8": _mtq.INT8_DEFAULT_CFG}.get(fmt)
    if cfg is None and fmt in ("fp4", "nvfp4"):
        cfg = getattr(_mtq, "NVFP4_DEFAULT_CFG", None)
    if cfg is None:
        return model
    loop = forward_loop or (lambda m: None)
    return _mtq.quantize(model, cfg, forward_loop=loop)


def summary() -> dict:
    return {"blackwell": _blackwell(), "linear_backend": linear_backend(),
            "attention_backend": attention_backend(), "have_te": HAVE_TE,
            "have_flash": HAVE_FLASH, "have_modelopt": HAVE_MODELOPT,
            "have_native_fp8": HAVE_NATIVE_FP8}


if __name__ == "__main__":
    import json
    print(json.dumps(summary(), indent=2))
