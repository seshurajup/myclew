# fp8 on the RTX 5090 (Blackwell sm_120) — box-verified compatibility matrix

Everything here is **measured on this box** (torch 2.8.0+cu128, RTX 5090, cc 12.0), not docs.
Reproduce with the probes in the run log; numbers are ms for a 4096³ GEMM unless noted.

## TL;DR — I was wrong that "fp8 is slow"; I was right that "my detector got slower"
fp8 tensor cores on the 5090 are **fast**. The loss came from my eager wrapper, not fp8:

| path (full training step, fwd+grad_x+grad_w, 4096³) | ms | vs bf16 |
|---|---|---|
| bf16 | 2.31 | 1.00× |
| **fp8 naive** (what I shipped — per-GEMM quantize + redundant `.contiguous()` copies) | 2.5× **slower** on the small detector; ~0.9× at 4096 | ❌ |
| fp8 quant-once (each tensor quantized once/step, correct layouts) | 1.85 | 1.25× |
| fp8 cached-W layout | 1.71 | 1.35× |
| **fp8 + `torch.compile`** (fuses the quantize into neighbors) | **1.26** | **1.84×** |
| raw `_scaled_mm` alone (pre-quantized) | 0.36 | 2.13× |
| raw MXFP8 block-scaled | 0.26 | **2.92×** |

**The three things I got wrong (not fp8):**
1. Quantized inside every GEMM (6×/step) and forced a `.t().contiguous()` copy even when the layout was already right → gave the win back. Fix: quantize each tensor **once**, keep needed layouts.
2. Ran **eager**. fp8 needs `torch.compile` to fuse the amax/scale/cast into surrounding ops — that alone is 1.25×→**1.84×**.
3. Applied it to a model whose matmuls are **tiny** (M=B·T=128, K=256). fp8 **never** wins below ~1–2k dims regardless of implementation. For the small cell-detector, **bf16 is the correct choice** — the win only exists for large from-scratch training.

## What trains in fp8 on the 5090 — ARCHITECTURE MATRIX

| building block | fp8 on 5090? | how / caveat |
|---|---|---|
| **Linear / MLP / FFN** | ✅ YES | `torch._scaled_mm`, the fp8 workhorse. Needs `torch.compile` to be fast. |
| **Attention qkv/proj (the GEMMs)** | ✅ YES | same as Linear. |
| **Attention scores/context (softmax·V)** | ⚠️ bf16 | `scaled_dot_product_attention` on fp8 tensors → `NotImplementedError`. SDPA runs bf16; fp8 flash-attn needs a custom/TE kernel we don't have. |
| **Transformer encoder/decoder, ViT, LLM** | ✅ YES | matmul-dominated → the target arch for fp8. |
| **Conv2d / Conv3d** | ❌ NO | `conv3d(fp8)` → `getCudnnDataTypeFromScalarType() not supported`. **No fp8 conv kernel exists.** UNet3D is stuck at bf16 — this is why our production detector cannot go fp8. |
| **LayerNorm / RMSNorm / GELU / softmax** | ⚠️ bf16 | pointwise/reduction ops stay bf16; negligible FLOPs, but they cap the fp8 fraction (~89% for our detector). |
| **BatchMatMul (`bmm`/`baddbmm`)** | ❌ NO (direct) | `bmm(fp8)` → `not implemented`. Reshape to 2D and use `_scaled_mm`, or compile. |
| **MoE grouped GEMM** | ❌ NOT on 5090 | `torch._scaled_grouped_mm` requires **cc == 9.0 (Hopper)**; on sm_120 it raises. MoE-fp8 must fall back to looped `_scaled_mm` per expert. **Real surprise avoided.** |

## fp8 FORMATS available on the 5090 (all dtypes present in torch 2.8)

| format | dtype | use | verified |
|---|---|---|---|
| E4M3 | `float8_e4m3fn` | forward/weights/activations (max ±448) | ✅ GEMM OK |
| E5M2 | `float8_e5m2` | **gradients** (wider exp, max ±57344) | ✅ as lhs; ❌ **E5M2×E5M2 not supported** — one operand must be E4M3 |
| E4M3/E5M2 fnuz | `*_fnuz` | ROCm-style; present but not the CUDA path | n/a |
| E8M0 | `float8_e8m0fnu` | **MXFP8 block-scale factors** (per-32 block) | ✅ block-scaled GEMM OK, **2.92×** |
| FP4 E2M1 (packed) | `float4_e2m1fn_x2` | **NVFP4** (2 values/byte) | ✅ `_scaled_mm` with e4m3 block-scales OK — Blackwell-only, inference-grade |

## SCALING granularities (all work via `_scaled_mm`)
- **per-tensor** (scalar scale): simplest, 2.13×. Risk: one outlier shrinks the whole scale.
- **per-row / rowwise** (`scale_a=(M,1)`, `scale_b=(1,N)`): 2.13×, more robust, near-free.
- **MXFP8 block** (e8m0 scales, 1 per 32 elems): **2.92× — fastest**, best numerics. Preferred for real fp8 training on Blackwell.
- **NVFP4** (fp4 + e4m3 block scales): max compression/speed, inference/PTQ territory, not for from-scratch training accuracy-critically.

## HARD constraints that will bite (verified)
- **K (contraction dim) must be a multiple of 16.** K=255/248/240/16/8 all → shape error via `.t()` layout. Pad to /16.
- **Layout: lhs row-major, rhs column-major.** Any other combo → `Only multiplication of row-major and column-major matrices is supported by cuBLASLt`. Plan your `.t()`/`.contiguous()` so exactly one operand is transposed.
- **E5M2×E5M2 forbidden** — gradient GEMMs must pair E5M2 activation-grad with E4M3 weight/input.
- **No fp8 for conv, SDPA-core, bmm, grouped-mm (on 5090).** Keep those bf16.

## DECISION RULE for tomorrow (what select_train_precision SHOULD encode)
1. **conv-dominated (UNet3D, our production detector)** → **bf16**. No fp8 conv kernel. Non-negotiable.
2. **matmul-dominated (transformer/ViT/LLM) AND matmul dims ≥ ~1–2k AND `torch.compile` on** → **fp8** (MXFP8 block-scale), expect ~1.8× real, up to 2.9× on the GEMMs. Gradients E5M2.
3. **matmul-dominated but SMALL matmuls (our patch-token detector, M≤~256)** → **bf16**. fp8 overhead > gain; this is the trap I fell into. Precision policy must gate on matmul SIZE, not just conv-vs-linear.
4. **MoE on 5090** → no grouped fp8 kernel; loop `_scaled_mm` per expert or stay bf16.
5. Always run fp8 under `torch.compile`; eager fp8 leaves ~30–45% on the table.

## Refinement flagged for `hardware_tune.select_train_precision`
It currently picks fp8 purely on conv-vs-linear param split. That made it force fp8 on the small cell-detector, which is **slower**. It should additionally require (a) representative matmul dim ≥ ~1024 and (b) `torch.compile` enabled before returning fp8 — otherwise bf16. (Not yet changed; noted here so tomorrow's run doesn't repeat the mistake.)
