# SpatialDINO — RTX 5090 fp8 smoke-test (MEASURED)

Honest, measured smoke-test of **SpatialDINO** (kirchhausenlab, 3D DINOv2 ViT for light-sheet
fluorescence) as a candidate fp8-fast transformer foundation for the biohub cell detector.
Date: 2026-07-20. GPU: RTX 5090 (sm_120). venv: `research/cellmot_venv` (torch 2.8.0+cu128).
Repro script: `scratchpad/smoke.py` (results JSON alongside). **No fine-tuning; a few measured steps only.**

## TL;DR verdict

SpatialDINO **loads, is genuinely transformer-heavy (98.95% Linear), and select_train_precision
returns `fp8`.** But the measured fp8 win is **conditional on token count**: on the transformer stack
(fwd+bwd, torch.compile ON) fp8 is **1.42× FASTER at N=8192 tokens** but **0.76× (31% SLOWER) at
N=1024 tokens**. Reason: embed_dim is only **384** (small GEMM inner dim), so fp8's quantize overhead
is only amortized when the token (M) dimension is large — i.e. only on **large input volumes**. Also,
the full end-to-end model **does not compile under fp8 as-is**: the single CLS token makes the token
count odd (1025/8193), and fp8 `_scaled_mm` requires M divisible by 16 (fixable by padding the
sequence, not done here). **So: fp8-viable in principle, but only a ~1.4× win and only at scale, plus
one engineering fix — not the 2.1× raw-matmul number, and a net loss on small patches.**

## 1. Weights load

- Source: `s3://spatialdino/models/spatial_dino/step=249999/backbone.pth` (latest of 250 checkpoints
  in the bucket; downloaded over HTTPS `--no-sign-request` equivalent — **no awscli needed**, bucket
  is public-readable). MIT license.
- Size: **86,068,439 bytes (82 MB)**. `torch.load(weights_only=True)` succeeds → `OrderedDict`, 174
  tensors.
- Instantiated the real repo `Encoder` class (`spatialdino.models.layers.encoder`) as `vits8`
  (embed 384 / depth 12 / heads 6 / patch 8³ / in_chans 1 / no pos-embed) and
  `load_state_dict(strict=False)` → **0 missing, 0 unexpected keys** (perfect match).
  (Repo cloned to `research/spatialdino_repo`; xformers/omegaconf/torch_pca stubbed in-process — no
  installs — to dodge the heavy package `__init__`; attention uses the SDPA/eager fallback path.)

## 2. Architecture + fp8-able fraction (from the real modules)

| Property | Value |
|---|---|
| Total params | **21.50 M** (ViT-Small) |
| Linear / attention params | **21,275,136 = 98.95%** ← real fp8-able fraction |
| Conv params | **196,992 = 0.92%** (one 3D patch-embed conv, `384×1×8×8×8`) |
| Norm/bias/tokens | ~0.13% |
| embed_dim / depth / heads | 384 / 12 / 6 |
| patch size / MLP hidden | 8×8×8 / 1536 |
| **max_linear_dim** | **1536** |

The doc's "~90%+ fp8-able" claim is **confirmed and exceeded — 98.95%** of params are Linear/attention;
the only conv is the 0.92% patch-embed tokenizer (which has no fp8 kernel and stays bf16). This is a
near-pure transformer.

## 3. select_train_precision

`fleet_agents.hardware_tune.select_train_precision(model)` →
```
{'amp_dtype': 'fp8', 'fallback': 'bf16', 'arch': 'transformer',
 'reason': 'matmul/transformer-heavy → fp8 (max_linear_dim=1536 ≥ 1024; conv=196992 linear=21275136)'}
```
Confirmed **fp8** (transformer-dominant params, max_linear_dim 1536 clears the ≥1024 size gate).

## 4. MEASURED fp8 vs bf16 (RTX 5090, torch.compile ON)

**Decisive number — transformer blocks (the 98.95% fp8-able stack), full fwd+bwd, compiled**, on
zebrafish-realistic token counts (N divisible by 16 so fp8 `_scaled_mm` runs):

| Input (tokens) | bf16 s/iter | fp8 s/iter | **fp8 speedup** | bf16 VRAM | fp8 VRAM |
|---|---|---|---|---|---|
| N=1024 (~32×128×128 patch) | 0.00778 | 0.01020 | **0.76× (SLOWER)** | 0.48 GB | 0.44 GB |
| N=8192 (~64×256×256 volume) | 0.06113 | 0.04313 | **1.42× (FASTER)** | 1.64 GB | 1.34 GB |

Full end-to-end model (fwd+bwd), bf16 baseline: 0.0088 s/iter @1026 tok, 0.0578 s/iter @8194 tok.
**Full-model fp8 fails to compile** — `RuntimeError: Expected self.size(1) to be divisible by 16, got
1025/8193`: the CLS token makes the sequence length odd. Fixable by padding tokens to a multiple of
16; not done here (bounded smoke-test). Full-model fwd-only fp8 = 0.94× (consistent: small-K, not
amortized).

**Interpretation (honest):** fp8 is a real win for SpatialDINO **only at large token counts**
(≥ ~8k tokens → 1.42×). At small patch sizes it is a **net loss** (0.76×) because embed_dim=384 gives
small GEMM inner dims; the crossover sits between 1024 and 8192 tokens. The win (1.42×) is well below
the 2.1× raw-`_scaled_mm` matmul number precisely because the model's matmuls (K=384–1536) are modest.
fp8 also modestly cuts VRAM (~18%).

## 5. Feature sanity

Forward on a **real 44b6 zebrafish frame** (`44b6_0113de3b.zarr`, 64×256×256 volume, center crop):
patch tokens `[1, 1024, 384]`, **all finite, mean 0.021, std 0.594, 0% zeros** → non-degenerate,
healthy features. Model runs on real comp data out of the box.

## Bottom-line verdict

SpatialDINO is a **legitimately transformer-heavy (98.95% Linear), fp8-able 3D ViT that loads cleanly,
matches the checkpoint exactly, and produces sane features on real zebrafish volumes.** As an
**fp8-fast foundation on the 5090 it is viable but qualified**:
- ✅ fp8 gives a **measured 1.42× fwd+bwd speedup + ~18% less VRAM** on the transformer stack — but
  **only at large token counts** (large input volumes).
- ⚠️ At small patches (N≈1024) fp8 is a **31% net loss** (embed_dim 384 → small matmuls; the exact
  trap the hardware_tune size-gate warns about, which the 1536 max-dim gate does *not* catch because
  the *token* dimension, not the weight dim, is what's too small).
- ⚠️ Full end-to-end fp8 needs a **one-line fix** (pad sequence length to a multiple of 16 to clear
  the odd CLS token) before it will compile.
- Not blocked by env/ABI (zero installs; repo code available; weights public MIT).

So: **fp8-fast for us IF we train on large token batches and pad the CLS sequence — expected ceiling
~1.4×, not 2×.** This is a throughput lever at scale only, and (per `pretrained_transformer_detectors.md`)
still carries the unresolved content-gap (organelles vs embryo nuclei) and the missing detection head —
so it remains an R&D track, not a score lever.

## Env intact (post-run)

`torch 2.8.0+cu128 · numpy 2.4.6 · cv2 4.13.0 · cv2 op OK · CUDA True (RTX 5090)` — unchanged. No pip
installs were performed (download via curl, repo via git clone, all optional deps stubbed in-process).

## Artifacts
- Weights: `research/spatialdino/backbone.pth` (82 MB)
- Repo: `research/spatialdino_repo/` (kirchhausenlab/spatialdino)
- Script: `scratchpad/smoke.py`
