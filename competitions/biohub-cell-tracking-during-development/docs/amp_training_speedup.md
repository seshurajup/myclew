# AMP (bf16) fast-path wired into the official biohub trainer

**Date:** 2026-07-20 · **Scope:** training-speed tooling, NOT a score lever.

## What changed

`research/official_repo/scripts/train_unet_transformer.py` historically ran the
UNet+transformer detector trainer in **pure FP32** (the batch `imgs` arrive as fp16 for
transfer bandwidth but are upcast to fp32 in the loop; the model itself is fp32). The
MEASURED `hardware_tune` speed config (`docs/hardware_config.json`: bf16 ~1.84x, TF32,
channels_last, torch.compile) existed and was used in **toy trainings only** — it was
**never wired into the official biohub trainer**. This closes that gap.

Everything is **opt-in and gated**. Default (flags unset) = byte-identical FP32.

| Env flag | Effect | Default |
|---|---|---|
| `CELLMOT_AMP=1` | TF32 matmul/cudnn + `set_float32_matmul_precision('high')` + **bf16 autocast** around forward+loss (no GradScaler — bf16 doesn't need one; master weights stay fp32) + forces the **math SDPA backend** (see gotcha) | OFF |
| `CELLMOT_CHANNELS_LAST=1` | `channels_last_3d` memory format on the model (sub-lever; only under AMP, single-GPU) | OFF |
| `CELLMOT_COMPILE=1` | `torch.compile(model.unet)` (sub-lever; try/except → eager fallback) | OFF |

Driven by `hardware_tune.load_config()` (read by file path to avoid the heavy
`fleet_agents` package `__init__`; falls back to reading `docs/hardware_config.json`,
then to bf16 defaults — so AMP works in any env). `amp_dtype: "fp16"` in the config would
select fp16 instead, but bf16 is the measured/recommended dtype here.

## Measured on RTX 5090 (torch 2.8.0+cu128), same seed/data, realistic downsampled
## volume (Z=64,Y=64,X=64):

| Config | s/iter (end-to-end `train_epoch`) | speedup | peak mem |
|---|---|---|---|
| FP32 (baseline) | 0.417 | 1.00x | 9.18 GB |
| bf16 | 0.242 | **1.72x** | 6.71 GB |
| bf16 + channels_last | 0.199 | **2.09x** | 6.48 GB |

Pure UNet fwd+bwd microbench (compute-bound ceiling, B=16):

| Config | s/iter | speedup |
|---|---|---|
| FP32 | 0.767 | 1.00x |
| bf16 | 0.414 | 1.85x (≈ hardware_tune's measured 1.84x matmul) |
| bf16 + torch.compile | 0.258 | **2.98x** |

**Convergence match:** bf16 loss curve tracks FP32 with no divergence
(detection loss FP32 0.0216→0.0035 vs bf16 0.0212→0.0037 over 6 mini-epochs; edge loss
identical trend). **GPU-bound:** live util 87.5%, CPU load 0.05 → the speedup is real
compute acceleration, not data-bound.

## Gotcha found & fixed (would have crashed real AMP training)

The UNet's `_TemporalAttention` runs MHA with **seq-len 2 over a huge `B·Z·Y·X` batch**.
Under bf16 autocast the fused **flash/mem-efficient SDPA** kernels are auto-selected and
**fail** at realistic sizes (`B·S ≈ 2M` → `CUDA error: invalid configuration argument`,
grid-dim overflow), whereas FP32 used the math backend. Fix: under `CELLMOT_AMP` we
`enable_flash_sdp(False)` + `enable_mem_efficient_sdp(False)` to force the math backend.
Seq-len 2 gets zero benefit from fused attention anyway (the bf16 win is entirely in the
3D convs), so this costs nothing.

`compute_loss` uses `F.binary_cross_entropy`, which is **autocast-unsafe** (raises); it is
now run inside a fp32 autocast-disabled block under AMP (no-op when AMP is OFF, so
byte-identical).

## Honest framing

This is a **training-speed tooling fix** (~1.7x bf16 / ~2.1x +channels_last / ~3x
+compile on future biohub training), **NOT a score lever**. biohub is recall-saturated
(~0.880 structural ceiling); faster training does not beat it. It matters because it makes
every future train — self-train, arch experiments, and other comps that reuse this trainer
— ~2x (up to ~3x) cheaper. channels_last and torch.compile both worked on the custom 3D
convs (no error) and are kept as opt-in sub-levers.
