# FP8 Ecosystem for RTX 5090 (consumer Blackwell, sm_120) — grounded catalog

Research sweep 2026-07-20. Goal: what can we actually USE to make training/inference
faster in fp8 on our box, with honest "works on sm_120?" status per tool. Every claim
is sourced. This is research only — nothing installed (ABI safety).

## The one fact that governs everything

Consumer Blackwell (RTX 5090 = GB202 = **sm_120 / cc 12.0**; also RTX PRO 6000; DGX
Spark/GB10 = sm_121) is **binary-incompatible with both Hopper (sm_90) and datacenter
Blackwell (sm_100/B200, cc 10.0) tensor-core kernels**. `wgmma` (Hopper) and `tcgen05`/
2-CTA-cluster MMAs (sm_100) are not available on sm_120; the narrow-precision (FP8/FP4
block-scaled) MMAs are **arch-conditional PTX** that must be compiled for `sm_120a`
(or `sm_120f`, CUDA 13+), not plain `sm_120`. So sm_120 has the fp8/fp4 *hardware*, but
every library needs **sm_120-specific kernels** — inheriting sm_90/sm_100 code silently
fails ("no kernel image", "cvt PTX instructions are architecture-specific", or garbage
output). This is why so many projects run on H100/H800/B200 yet stall on a 5090.

Source: https://github.com/deepseek-ai/DeepGEMM/issues/317 ·
https://github.com/NVIDIA/cutlass/issues/3096 ·
NVIDIA RTX Blackwell arch PDF: https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf

Corollary: **"Blackwell supported" almost always means sm_100 (datacenter). Do not
transfer that claim to sm_120 without a per-cc check.**

---

## 1. Ranked table — tools

Ranked by how usable they are for us on the 5090 (sm_120) today.

| # | Tool | fp8 train? | fp8 infer? | Linear / conv / MoE | Works on sm_120 consumer Blackwell? | Source |
|---|------|-----------|-----------|---------------------|-------------------------------------|--------|
| 1 | **torch `_scaled_mm`** (torch 2.8 cu128) | yes (prim) | yes (prim) | Linear/GEMM only | **YES** — gate is cc≥8.9/9.0; sm_120 passes. Our measured 2.13× vs bf16 | https://github.com/comfyanonymous/ComfyUI/issues/4531 |
| 2 | **torchao float8** (pytorch/ao) | **yes** (Linear) | yes (Linear) | Linear only, **no conv** | **YES tensorwise/rowwise** via `_scaled_mm` on prebuilt cu128 wheels (from-source rowwise build reported broken, open #156613); MXFP8/NVFP4 = names 5090 as target but **unverified** on sm_120 | https://github.com/pytorch/ao/blob/main/torchao/float8/README.md · https://github.com/pytorch/ao/releases · https://github.com/pytorch/pytorch/issues/156613 |
| 3 | **NVIDIA cutlass** | yes (GEMM prim) | yes (GEMM prim) | GEMM: fp8/mxfp8/nvfp4 dense+sparse+**grouped(MoE)**; **no conv** | **YES, first-class** — dedicated `sm_120a` "geforce" examples 79/80/87; fp8 grouped-GEMM gap closed in **4.6.0 (2026-07-01)**. Must compile `sm_120a`/`sm_120f` | https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md · https://github.com/NVIDIA/cutlass/tree/main/examples/79_blackwell_geforce_gemm |
| 4 | **NVIDIA TensorRT** | no (infer-only) | **yes (fp8/fp4)** | conv **and** transformer | **YES** — support matrix lists cc 12.0; fp8+fp4 precisions. Also T4 int8 (see #12) | https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html · https://developer.nvidia.com/blog/nvidia-tensorrt-unlocks-fp4-image-generation-for-nvidia-blackwell-geforce-rtx-50-series-gpus/ |
| 5 | **TransformerEngine** | **yes** | yes | Linear/attn/**GroupedLinear(MoE)**; **no conv** | **PARTIAL** — standard fp8 + `Float8BlockScaling` work on sm_120 (proper cu128 build); **MXFP8 blocked** ("not supported on 12.0+"); **NVFP4 buggy** (open #3217/#3219/#3062/#2255) | https://github.com/NVIDIA/TransformerEngine/issues/2668 · https://github.com/NVIDIA/TransformerEngine/issues/2255 |
| 6 | **TensorRT-LLM** | no | yes (fp8/fp4) | Linear + MoE | **YES since v0.20.0rc3** (removed GeForce-Blackwell exclusion); NVFP4 MoE on 5090 still needs runtime patches | https://github.com/NVIDIA/TensorRT-LLM/issues/5018 · https://github.com/NVIDIA/TensorRT-LLM/discussions/8334 |
| 7 | **TensorRT Model Optimizer (modelopt)** | QAT + PTQ | yes | **conv + transformer** (int8/fp8/int4/nvfp4) | int8/fp8 PTQ produce anywhere; **NVFP4 needs Blackwell + TRT≥10.8**; native sm_120 nvfp4 kernel exec = **unknown, verify** | https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/llm_ptq/README.md |
| 8 | **vLLM** | no | **yes (fp8 W + KV cache)** | Linear ✅; MoE nvfp4 selector buggy | **YES (recent)** — block-fp8 for Blackwell added via PR #22131; sm_120 fp8 GEMM ships in later releases; NVFP4-MoE falls back to Marlin (selector bug #31085) | https://github.com/vllm-project/vllm/issues/21648 · https://github.com/vllm-project/vllm/issues/31085 |
| 9 | **cudnn-frontend** | yes (graphs) | yes | **conv (2D fp8 sample)** + matmul + attn | 2D fp8 conv: **likely yes** (Hopper-or-newer guard) but sm_120 kernel presence **unverified**; **fp8 conv3d: unknown, likely UNSUPPORTED** (only 2D sample exists) | https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/convolution/fp8_fprop.cpp |
| 10 | **torchtitan** | yes (via torchao) | — | Linear (LLM) | **UNKNOWN** — needs 2+ GPUs + compile; all benchmarks H100/B200; no sm_120 notes. Inherits torchao | https://github.com/pytorch/torchtitan/blob/main/docs/float8.md |
| 11 | **Megatron-LM** | yes (via TE) | yes | Linear/MoE | **PARTIAL** — inherits TE; `--fp8-recipe mxfp8` hits TE's "not supported on 12.0+"; block-scaled fp8 likely OK | https://github.com/NVIDIA/Megatron-LM |
| 12 | **NeMo** | yes (via TE) | yes | Linear/MoE | **UNKNOWN** — TE-bounded; no 5090 pretraining confirmation | https://github.com/NVIDIA/TransformerEngine |
| 13 | **bitsandbytes** | no (QLoRA) | int8 / nf4 (storage 4-bit, dequant to bf16) | Linear | **PARTIAL** — build from source with sm_120 arch; **no hardware fp8** (nf4/fp4 are storage-only). T4 int8 ✅ | https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1937 |
| 14 | **DeepGEMM** | yes (GEMM lib) | yes | Linear/MoE GEMM | **NO** — README requires SM90 or SM100; sm_120 impl files don't exist (open #236, #317); also wants **CUDA 12.9+** for Blackwell (we have 12.8) | https://github.com/deepseek-ai/DeepGEMM/issues/236 · https://github.com/deepseek-ai/DeepGEMM/issues/317 |
| 15 | **SGLang (fp8)** | no | yes | Linear/MoE | **NO / lagging** — "FP8 blockwise on Blackwell sm120 not supported yet" | https://github.com/sgl-project/sglang/issues/9233 |
| 16 | **Marlin / Machete** (vLLM kernels) | no | weight-only (W4A16/W8A16) | Linear | **Machete NO** (Hopper wgmma); **Marlin = works on sm_120 as weight-only fallback** (functional, not true HW-fp8) | https://github.com/vllm-project/vllm/issues/35439 · https://github.com/vllm-project/vllm/issues/43906 |
| 17 | **torch `_scaled_grouped_mm`** (grouped fp8 MoE) | — | — | MoE grouped | **NO — DEAD on consumer Blackwell.** Gated cc==9.0; SM100+ proposal open and omits sm_120; CUTLASS lacked sm_120 grouped specialization (now in cutlass 4.6, but torch not wired) | https://github.com/pytorch/pytorch/issues/155434 · https://github.com/pytorch/pytorch/issues/156238 · https://github.com/vllm-project/vllm/issues/43507 |
| 18 | **nanotron** | roadmap (not shipped) | — | — | **N/A** — fp8 training not in current tree | https://github.com/huggingface/nanotron |
| 19 | **llm.c** | experimental (PRs) | — | Linear | **UNKNOWN, likely NO** — Hopper-tuned cuBLASLt, unmerged | https://github.com/karpathy/llm.c/pull/678 |
| — | **HF `kernels` loader + `kernels-community/deep-gemm`** | (loader) | (loader) | GEMM only, **no conv** | **NO fp8 GEMM on sm_120** — deep-gemm ships only sm_90/sm_100 impls; `quantization` (vLLM cutlass scaled_mm) has an sm_120 source path but the HF build's sm_120 gencode is **unverified**. See §1b | https://huggingface.co/kernels-community/deep-gemm · https://github.com/huggingface/kernels |

---

## 1b. HF `kernels` + `kernels-community` (pre-compiled, no local compile) — HIGH-PRIORITY

**Net answer: NO.** HF `kernels` cannot give us a fused fp8 GEMM that runs on sm_120 /
CUDA 12.8 for Gemma training today — *without* torchao/TE. Evidence:

**The loader (`huggingface/kernels`):** `get_kernel("kernels-community/<x>")` detects your
exact Python/torch/CUDA and **downloads the matching pre-compiled binary from the Hub at
runtime** (needs internet on first use; caches to HF cache). Offline is possible via
`get_local_kernel()` / `get_locked_kernel(local_files_only=True)` / lockfiles only if you
pre-download the exact-matching variant. Transformers integration:
`@use_kernel_forward_from_hub` + `kernelize()`, surfaced as `use_kernels=True` /
`attn_implementation="kernels-community/flash-attn2"` — still resolves from the Hub unless
pre-cached. This is the ABI-safe attraction: **no local compile, no torchao/TE install.**
Sources: https://github.com/huggingface/kernels · https://huggingface.co/blog/hello-hf-kernels · https://huggingface.co/docs/kernels/main/en/api/kernels

**CRITICAL build-variant caveat:** variant folders are named
`torch<ver>-cxx11-cu<ver>-<CPUarch>-linux` (e.g. `torch28-cxx11-cu128-x86_64-linux`).
**The GPU compute capability (sm_120) is NOT in the folder name** — only CUDA toolkit +
CPU arch. sm_120 gencode is baked into the `.so` via the source repo's `build.toml`
`cuda-capabilities`. So a `cu128` folder confirms CUDA 12.8 availability but **cannot**
confirm sm_120 — that needs reading `build.toml`, `cuobjdump -lelf <so> | grep sm_120`,
or running on the 5090.

- **`kernels-community/deep-gemm` (the fp8 GEMM, "3–6× vs Triton"): NO on sm_120.**
  cu128 x86_64 builds exist, but DeepGEMM's code targets **only SM90 + SM100**; sm_120
  impls don't exist upstream (`sm120_*.cuh` missing → open #236, #317, "cannot complete a
  forward pass on any consumer/workstation Blackwell"). It also wants **CUDA 12.9+** for
  Blackwell (we're on 12.8), and its SM100 path uses tcgen05 / 228 KiB shared-mem
  instructions absent on sm_120 (99 KiB/SM) — so it's not portable, needs bespoke kernels.
  https://huggingface.co/kernels-community/deep-gemm · https://github.com/deepseek-ai/DeepGEMM/issues/317
- **`kernels-community/quantization` (vLLM cutlass fp8/int8 `scaled_mm`): UNKNOWN — NEEDS
  TEST.** Upstream vLLM added `scaled_mm_c3x_sm120` (PR #22131), but the HF snapshot's
  cu128 `.so` sm_120 gencode is not visible from folder names, and it's an
  inference-oriented scaled matmul, not a training fused-fp8-GEMM with a Gemma layer map.
  Verify: `cuobjdump -lelf <cached .so> | grep sm_120`, or run a small fp8 `scaled_mm` on
  the 5090 (watch for "no compiled kernel for capability 120"). https://github.com/vllm-project/vllm/pull/22131
- **Others** (`finegrained-fp8`, `flash-attn2`, `flash-attn3`, `activation`, `rotary`):
  attention/activation/rope/quant ops, **not fp8 GEMMs**, so they don't answer the fused
  fp8 GEMM question; and their sm_120 gencode is itself unverified (upstream flash-attn
  default wheels historically lacked sm_120 cubins). https://huggingface.co/kernels-community
- **Convolution (conv3d) at fp8/int8: CONFIRMED ABSENT** — `kernels-community` has zero
  convolution kernels of any precision (attention/GEMM/quant/activation/rope/norm only).
  So HF kernels does nothing for our UNet. https://huggingface.co/kernels-community
- **Offline Kaggle T4 note:** even if a kernel were pre-cached for offline use, T4 = sm_75
  has no fp8 hardware — every fp8 kernel here is 5090-only regardless of the loader.

**Bottom line:** the ABI-safe "no-compile" appeal is real, but the one fp8 GEMM HF hosts
(deep-gemm) is a hard NO on sm_120, and the only other candidate (quantization) is
unverified inference-oriented scaled_mm — not a drop-in Gemma-training fused fp8 GEMM. For
sm_120 fp8 GEMM in training, the working paths remain **torchao / TE / vLLM-built-for-sm_120**
— exactly what we wanted to avoid. If we still want to try HF kernels, the single decisive
test is `cuobjdump`/run of `kernels-community/quantization` on the 5090.

---

## 2. Ranked table — papers

| Paper | conv / transformer | fp8 train? | fp8 infer? | Headline | Blackwell / HW note | Source |
|-------|--------------------|-----------|-----------|----------|---------------------|--------|
| FP8 Formats for Deep Learning | **general (CNN+RNN+Transformer)** | yes | yes (PTQ) | fp8 matches 16-bit up to 175B, no HP changes; defines E4M3/E5M2 | Hopper-era | https://arxiv.org/abs/2209.05433 |
| FP8-LM | transformer | yes | — | GPT-175B: 75% faster than bf16 Megatron; fp8 grads+optim+comm | H100 | https://arxiv.org/abs/2310.18313 |
| DeepSeek-V3 tech report | transformer/MoE | yes | yes | 671B fp8 training; 1×128 act / 128×128 wt tiles; FP32 accum every 128 MMA | **H800 Hopper**; accum trick is a Hopper workaround Blackwell fixes in HW | https://arxiv.org/abs/2412.19437 |
| Scaling FP8 to trillion-token LLMs | transformer | yes | — | fp8 to 2T tokens, +34% throughput; Smooth-SwiGLU fixes late-train blowup | **Intel Gaudi2** (not NVIDIA) | https://arxiv.org/abs/2409.12517 |
| Microscaling (MX) formats | general | yes | yes | Defines MXFP8/6/4 + E8M0 block scale (block=32); sub-8-bit training | Native on **Blackwell** tensor cores | https://arxiv.org/abs/2310.10537 |
| NVIDIA Recipes for MXFP8 pretraining | transformer | yes (MXFP8) | — | MXFP8 = bf16 accuracy IF E4M3 for all tensors incl. act-grads + correct RN-even scale rounding | **Blackwell**-targeted recipe | https://arxiv.org/abs/2506.08027 |
| Pretraining LLMs with NVFP4 | transformer | yes (fp4) | — | 12B/10T tokens in NVFP4 ≈ fp8 baseline; ~2× math vs fp8 | **Blackwell** (GB200/GB300) native fp4 | https://arxiv.org/abs/2509.25149 |
| Optimizing LLM Training with FP4 (MS) | transformer | yes (fp4) | — | First fp4 training ≈ bf16; differentiable quant + outlier clamp | fp4 **simulated**; targets future Blackwell HW | https://arxiv.org/abs/2501.17116 |
| **PTQ for 3D Medical Image Segmentation (MedPTQ)** | **conv / 3D-CNN** | no | **int8 (PTQ)** | U-Net/nnU-Net/SwinUNETR: size ↓2.4–3.9×, latency ↓2.0–2.7×, Dice preserved (0.822→0.822) | **RTX 4090 consumer + TensorRT** — directly applicable to us | https://arxiv.org/abs/2501.17343 |

Papers gap: **no paper does fp8 *training* of a conv3d UNet.** Closest 3D-CNN work is
int8 *inference* PTQ (MedPTQ). fp8/mxfp8 conv3d training is only extrapolated from the
general fp8 spec (which did validate CNNs) + the LLM MXFP8/NVFP4 recipes.

---

## 3. TOP 3 concretely-usable options for us

### (a) Transformer / Gemma-4 training on the 5090
**Install torchao (`pip install torchao`, already-built cu128 wheel) → wrap `nn.Linear`
with `convert_to_float8_training` (tensorwise recipe) under `torch.compile`.**
This is the only fp8-training path that is confirmed-runnable on sm_120 without bespoke
kernels — it dispatches to `torch._scaled_mm` (cc≥8.9 gate, sm_120 passes). Tensorwise
is the fastest recipe; rowwise is more accurate but the from-source build was reported
broken on sm_120 (use the prebuilt wheel). **Must** use `torch.compile` and large matmuls
or it loses (our rule below). Avoid MXFP8/NVFP4 recipes for now (unverified/buggy on
sm_120). Runner-up: TransformerEngine `Float8BlockScaling` recipe (confirmed on sm_120)
— but TE adds a heavier dependency and its MXFP8/NVFP4 paths are blocked/buggy on sm_120.
**HF `kernels` is NOT a shortcut here:** its hosted fp8 GEMM (`kernels-community/deep-gemm`)
does not run on sm_120, and the only other candidate (`kernels-community/quantization`) is
unverified/inference-oriented — so the ABI-safe no-compile route does not replace torchao.
Source: https://github.com/pytorch/ao/blob/main/torchao/float8/README.md · https://huggingface.co/kernels-community/deep-gemm

### (b) UNet conv3d detector on the 5090
**No proven fp8 conv3d path exists — do NOT design around fp8 conv today.**
TE and CUTLASS have no convolution at all. cuDNN has fp8 conv but only a **2D** sample,
guarded "Hopper or newer"; whether it has real sm_120 kernels and any conv3d support is
unverified and likely unsupported. The realistic near-term win for the UNet is **int8
inference PTQ via TensorRT / modelopt** (MedPTQ proves 2–2.7× latency, zero Dice loss on
3D UNets on consumer NVIDIA + TensorRT) — but note this is inference, and our competition
inference is on T4 anyway (see c). For 5090 *training* speedups on the UNet, fp8 is a dead
end; stick to bf16 + `torch.compile` + channels_last. If we want to chase fp8 conv3d, the
one thing to *test directly* is a cuDNN graph with a 3D conv node at fp8 I/O on the 5090
and check the heuristic returns a real engine (not NOT_SUPPORTED).
Source: https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/convolution/fp8_fprop.cpp · https://arxiv.org/abs/2501.17343

### (c) T4 competition inference (int8)
**Build a TensorRT INT8 engine (via modelopt PTQ / ONNX QDQ → TensorRT) — fully
first-class on T4 (Turing sm_75).** This is the canonical, safe accelerator for the
2×T4 Kaggle path. No fp8/fp4 on T4 ever (no hardware) — int8/int4/fp16 only. MedPTQ
demonstrates the exact ONNX-QDQ → real-TensorRT-INT8 flow on 3D UNets with preserved
Dice. **Smoke-test numerically:** there are scattered field reports of int8 corruption
on sm_75 in specific (non-TensorRT) stacks — verify engine outputs on a real T4 before
trusting. Source: https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html

---

## 4. Honest gaps — what does NOT work on consumer Blackwell sm_120

- **Grouped fp8 MoE is dead on sm_120.** `torch._scaled_grouped_mm` gates cc==9.0
  (Hopper); the SM100+ proposal is open and omits sm_120; vLLM's CUTLASS grouped-fp8 MoE
  fell back to Triton because CUTLASS lacked a sm_120 specialization. CUTLASS 4.6.0
  (2026-07-01) finally added the sm_120/121 tensor/token-scaled fp8 grouped-GEMM collective,
  but PyTorch is not wired to it yet — so via torch it's still dead; via raw CUTLASS 4.6+
  it's now possible but unproven. https://github.com/pytorch/pytorch/issues/156238 ·
  https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md
- **MXFP8 training on sm_120 is blocked in TransformerEngine** — runtime assertion
  "MXFP8 (for all gemm layouts) is not supported on 12.0+ architectures yet." So Megatron/
  NeMo `--fp8-recipe mxfp8` fails on a 5090; only `Float8BlockScaling` works.
  https://github.com/NVIDIA/TransformerEngine/issues/2668
- **NVFP4 in TransformerEngine is buggy on sm_120** — multiple open issues (no kernel
  image, shared-mem cap exceeded, "cvt PTX architecture-specific"). Kernels were hardcoded
  to datacenter sm_100 TMEM/UMMA paths. https://github.com/NVIDIA/TransformerEngine/issues/2255
- **DeepGEMM / DeepSeek-V3 infra does not run on sm_120** — README requires SM90 or
  SM100; the sm_120 impl `.cuh` files don't exist; also wants CUDA 12.9+ for Blackwell
  (we have 12.8). This is also true of the **HF `kernels` hosted `kernels-community/deep-gemm`**
  — same upstream, so the "no-compile from the Hub" route inherits the NO. https://github.com/deepseek-ai/DeepGEMM/issues/236 · https://github.com/deepseek-ai/DeepGEMM/issues/317
- **HF `kernels` has no usable fp8 GEMM for sm_120 training** — deep-gemm is NO; the only
  other fp8 candidate (`kernels-community/quantization`, vLLM cutlass `scaled_mm`) has an
  sm_120 source path but unverified HF-build gencode and is inference-oriented, not a
  training fused GEMM. No convolution kernel of any precision exists in the org (UNet gets
  nothing). https://huggingface.co/kernels-community
- **fp8 convolution has no proven sm_120 path** — TE/CUTLASS have no conv; cuDNN's only
  fp8 conv sample is 2D and sm_120-kernel-presence is unverified; **no fp8 conv3d anywhere.**
  This is our UNet blocker. https://github.com/NVIDIA/cudnn-frontend
- **torchao MXFP8/NVFP4 on sm_120 is unverified** — torchao v0.12 names the 5090 as an MX
  target, but all published benchmarks are B200 (sm_100), and the underlying block-scaled
  kernels historically skip sm_120. Needs a runtime smoke-test. https://github.com/pytorch/ao/releases
- **Machete fp8/W4A8 is dead on Blackwell** (Hopper wgmma); Marlin works only as a
  weight-only (W4A16/W8A16) fallback, leaving the fp8/fp4 tensor cores idle.
  https://github.com/vllm-project/vllm/issues/35439
- **SGLang block-fp8 not supported on sm_120** yet (lags vLLM).
  https://github.com/sgl-project/sglang/issues/9233
- **bitsandbytes** stable wheels lag sm_120 (build from source); its "fp4/nf4" is storage-
  only 4-bit dequant, NOT hardware fp8/fp4. https://github.com/bitsandbytes-foundation/bitsandbytes/issues/1937
- **T4 (Turing sm_75) has no fp8/fp4 hardware at all** — int8/int4/fp16 only. Every fp8
  path above is 5090-only; the competition inference target can never use fp8.

### "Unknown — needs verification" (with what to check)
- **torchao MXFP8/NVFP4 Linear on sm_120** — run a `torch._scaled_mm` MX smoke test on the
  card; confirm it dispatches vs silently falls back.
- **cuDNN fp8 conv2d and any conv3d on sm_120** — build `fp8_fprop.cpp` on the 5090; run a
  3D conv node at fp8 I/O and check heuristics return a real engine (not NOT_SUPPORTED).
- **modelopt native nvfp4 kernel exec on sm_120** — run a tiny NVFP4 PTQ on the 5090; check
  the hf_ptq support matrix.
- **NeMo / llm.c fp8 on sm_120** — TE-bounded (NeMo) / Hopper-tuned unmerged (llm.c); run a
  tiny fp8 config and watch for the TE MXFP8 assertion / cuBLASLt kernel selection.
- **HF `kernels-community/quantization` sm_120 gencode** — `get_kernel(...)` on the 5090,
  then `cuobjdump -lelf <cached .so> | grep sm_120`, or run a small fp8 `scaled_mm` and
  watch for "no compiled kernel for capability 120". This is the single decisive HF-kernels
  test if we want to pursue the no-compile route.

---

## 5. Cross-check every "it's fast" claim against our measured reality

Our measured box numbers (facts): raw `torch._scaled_mm` fp8 = **2.13×** vs bf16;
fp8+`torch.compile` in-model = **1.84×**; MXFP8 block-scale = **2.92×**. **fp8 needs
compile + large matmuls to win.** Every repo speed claim below is flagged for re-measure
per our rule (measured on their HW, not our sm_120).

| Claim (source) | Their HW | Re-measure on our 5090? |
|----------------|----------|-------------------------|
| FP8-LM "75% faster than bf16" | H100 | **YES** — Hopper TE path; sm_120 kernel coverage differs; also fp8-comm/optim not just GEMM |
| DeepSeek-V3 fp8 throughput | H800 | **YES + N/A** — DeepGEMM doesn't run on sm_120 at all |
| MXFP8 recipes "= bf16 accuracy, 1.28× (torchtitan)" | B200 | **YES** — MXFP8 blocked in TE on sm_120; our own MXFP8 2.92× was a raw-kernel microbench, not in-model — re-measure in-model |
| NVFP4 "~2× math vs fp8" | GB200/GB300 | **YES** — NVFP4 buggy/unverified on sm_120; also a raw-throughput ceiling, not end-to-end |
| MedPTQ "2.0–2.7× latency" (int8 3D UNet) | RTX 4090 + TRT | **YES** — closest to us (consumer + TRT) but 4090≠5090 and it's inference PTQ; our T4 int8 will differ again |
| torchao tensorwise "fastest recipe" | H100 | **PARTIAL** — matches our 2.13× raw / 1.84× compiled shape; still verify tensorwise vs rowwise on sm_120 |

**Universal caveats from our own data that apply to every claim above:**
1. **No compile, no win** — in-model fp8 only reached 1.84× *with* `torch.compile`; eager
   is far worse. Any repo quoting eager fp8 speedups won't hold for us.
2. **Small-matmul trap** — fp8 only wins on large GEMMs (the 2.13× was a large matmul).
   Gemma-4 attention/MLP need to be wide enough; the conv3d UNet has no large-GEMM fp8
   path at all.
3. **Raw-kernel ≠ end-to-end** — our 2.92× MXFP8 and the papers' fp4 "2×" are matmul-only
   ceilings; real training is bounded by non-GEMM ops, memory, and (on sm_120) kernel
   coverage. Re-measure end-to-end before believing any headline.

---

## TL;DR for our two workloads

- **Gemma-4 fp8 training on 5090:** use **torchao float8 tensorwise + torch.compile**
  (only confirmed sm_120 path). Expect ~1.8× on wide layers, re-measured. Avoid MXFP8/
  NVFP4/grouped-MoE-fp8 on sm_120 (blocked/buggy/dead).
- **conv3d UNet on 5090:** **no fp8 path.** Stay bf16+compile+channels_last for training;
  for inference use **int8 TensorRT PTQ** (proven on 3D UNets, MedPTQ). Competition
  inference is T4 → **TensorRT INT8** is the only accelerator (no fp8 hardware on Turing).
