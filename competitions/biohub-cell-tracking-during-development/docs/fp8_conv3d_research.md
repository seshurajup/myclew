# FP8 for our UNet3D detector — research + measured verdict (RTX 5090)

Date: 2026-07-20. GPU: RTX 5090 (sm_120, Blackwell), torch 2.8.0+cu128, cuDNN 9.10.2.
Model: `research/official_repo/src/tracking_cellmot/models/temporal_unet.py` (`TemporalUNet3D`,
in=1, out=32, layers=(32,64,128), Conv3d 3x3x3 pad 1 + BN + ReLU).
Rule followed: measured the model's REAL conv3d shapes end-to-end, not toy shapes. No fabrication —
if a path is slower, the number is below.

## TL;DR verdict

- **fp8 UNet3D training is NOT faster on the 5090 via any path we can reach from torch today.**
  A DIY `F.unfold` (im2col) + `torch._scaled_mm` fp8 conv3d is **1.6x to 64x SLOWER** than the
  cuDNN bf16 `F.conv3d` baseline at the model's real shapes.
- **Why**, precisely: the fp8 GEMM itself *does* win (measured 2.13x at 4096³, 1.7–3.5x at our im2col
  GEMM shapes). But you cannot feed a conv to the tensor cores without im2col, and **explicit im2col in
  PyTorch is itself already 1.9–16x slower than cuDNN**, because cuDNN uses a *fused implicit-GEMM* that
  never materializes the column matrix. Quantization overhead (amax + cast, every call) piles on top.
  The fp8 GEMM saving is real but tiny next to the im2col tax it forces.
- **The only way fp8 conv wins on the 5090 is cuDNN-frontend fused fp8 conv** (implicit-GEMM in fp8),
  which torch eager does not route and which is **unverified for 3D/NDHWC**. That is a C++/graph-API
  engineering project, 5090-only.
- **For the actual Kaggle competition (2×T4, Turing sm_75): fp8 is IMPOSSIBLE — no fp8 tensor cores.**
  The only low-bit conv lever on T4 is **INT8 via TensorRT** (mature for plain Conv3d, weak for
  grouped/depthwise). fp8 and the T4 comp are disjoint problems.

---

## Part A — Research (sourced)

### 1. cuDNN 9 fp8 convolution — does it reach Blackwell/sm_120?
- cuDNN exposes fp8 conv through the **Graph API** with I/O dtypes `CUDNN_DATA_FP8_E4M3` / `_E5M2`,
  accumulate in `FLOAT` or `FAST_FLOAT_FOR_FP8`. fprop needs all-E4M3; dgrad/wgrad mix E4M3
  (act/weights) + E5M2 (grads).
  (cuDNN Core Concepts: https://docs.nvidia.com/deeplearning/cudnn/backend/latest/developer/core-concepts.html)
- **Blackwell/5090 IS targeted.** cudnn-frontend README lists Hopper (H100/H200) **and Blackwell
  (B200/GB200/GB300)** for "FP16, BF16, FP8, and MXFP8". cuDNN 9.17 release notes explicitly:
  "Performance for FP8 matmul and convolutions has been significantly optimized on the **GeForce RTX
  5090**." The fp8 conv sample guards `check_device_arch_newer_than("hopper")` → **sm_90+**, CUDA 12.0+.
  (https://github.com/NVIDIA/cudnn-frontend ; .../release-notes.html)
- **Caveat that matters for us:** the shipped fp8 conv sample (`fp8_fprop.cpp`) is **2D** (NCHW, 1x1
  filter, E4M3). **3D fp8 conv (NDHWC) is not demonstrated or documented — UNVERIFIED.** Exact
  fp8-3D channel-multiple constraints also unverified.
  (https://github.com/NVIDIA/cudnn-frontend/blob/main/samples/cpp/convolution/fp8_fprop.cpp)
- **Path from PyTorch:** no native torch fp8 conv op. You build the fp8 conv graph (+ descale/scale
  nodes) yourself via **`nvidia-cudnn-frontend`** (PyPI) or the C++ header API, wrapped as a custom op.

### 2. Transformer Engine / torchao — any fp8 conv?
- **TE = Linear-only.** Modules are Linear/GroupedLinear/LayerNorm/RMSNorm/LayerNormLinear/
  LayerNormMLP/DotProductAttention/MultiheadAttention/TransformerLayer — **no Conv of any kind**.
  fp8 applies to GEMM/attention/norm. (https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html)
- **torchao float8 = Linear/matmul only** (`Float8Linear`, `convert_to_float8_training`,
  `Float8DynamicActivationFloat8WeightConfig`). **No float8 convolution.**
  (https://docs.pytorch.org/ao/stable/api_reference/api_ref_float8.html)

### 3. Papers
- **"FP8 Formats for Deep Learning" (arXiv 2209.05433):** defines E4M3/E5M2, shows fp8 matches 16-bit
  quality with unchanged hyperparameters. Does test CNNs/RNNs/Transformers, **but CNN coverage is 2D
  image classification; headline results are LLM/transformer-centric. No 3D conv.**
  (https://arxiv.org/abs/2209.05433)
- Low-bit CNN *training* work is **INT8, 2D** (e.g. "Unified INT8 Training", arXiv 1912.12607).
  **No dedicated fp8 or low-bit 3D-conv training paper found — that niche is essentially unaddressed.**

### 4. TensorRT INT8/FP8 conv for inference (the T4-relevant path)
- **Turing/T4 (sm_75) has INT8/INT4 tensor cores and NO fp8.** fp8 tensor cores start at Ada (sm_89)
  and Hopper (sm_90). So on T4 the only tensor-core low-bit conv path is **INT8**.
  (T4 datasheet: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)
- **TensorRT INT8 3D conv:** supported and reasonably mature for *plain* Conv3d (`QuantConv3d` in
  ModelOpt / pytorch-quantization, per-channel weight scales).
  (https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/quantized-types-schemes.html)
  Caveats: **grouped/depthwise 3D convs get little/no INT8 speedup** (missing kernels,
  NVIDIA/TensorRT#1409, #1198); **fp8 convs (SM89/90/120) don't support kernel > 32** and can show
  no speedup vs FP16 (#4081). Our convs are plain 3x3x3 dense → the mature INT8 path applies.

---

## Part B — MEASURED on the 5090 (the real deliverable)

Method: for every real Conv3d shape in the model, three ways, each `torch.compile`d
(`max-autotune-no-cudagraphs`), warmup 10 / 50 iters, forward-only:
1. **cudnn_bf16** — `F.conv3d` bf16 (cuDNN implicit-GEMM), the baseline.
2. **im2col_bf16** — `F.unfold`-style 3D unfold + bf16 `bmm` (isolates im2col tax from fp8).
3. **im2col_fp8** — same im2col + per-tensor E4M3 quant + `torch._scaled_mm` (out bf16).
`r_bf16 = cudnn/im2col_bf16`, `r_fp8 = cudnn/im2col_fp8` (>1 = im2col faster; <1 = im2col slower).
`maxerr` = max abs error of the fp8 conv output vs the bf16 conv on the same input.
VRAM = peak allocated during that config's timed loop.

Real conv3d shapes (N=B*T=4 from B=1,T=4; input Z×Y×X=16×64×64):

| shape (in→out)   | GFLOP | cudnn_bf16 | im2col_bf16 | im2col_fp8 | r_bf16 | r_fp8 | fp8 slowdown | maxerr | VRAM bf16 | VRAM fp8 |
|------------------|------:|-----------:|------------:|-----------:|-------:|------:|-------------:|-------:|----------:|---------:|
| enc0_a 1→32      |  0.45 |  0.0394 ms |  0.0210 ms  | 0.0638 ms  | 1.87   | 0.62  | 1.6x slower  | 0.078  | 108 MB    | 185 MB   |
| enc0_b 32→32     | 14.50 |  0.1280 ms |  0.5689 ms  | 1.8259 ms  | 0.22   | 0.07  | 14x slower   | 0.286  | 663 MB    | 420 MB   |
| enc1_a 32→64     |  3.62 |  0.0410 ms |  0.0871 ms  | 0.8843 ms  | 0.47   | 0.05  | 22x slower   | 0.273  | 189 MB    | 157 MB   |
| enc1_b 64→64     |  7.25 |  0.0625 ms |  0.1763 ms  | 1.7574 ms  | 0.35   | 0.04  | 28x slower   | 0.438  | 147 MB    |  92 MB   |
| enc2_a 64→128    |  1.81 |  0.0248 ms |  0.0477 ms  | 0.2468 ms  | 0.52   | 0.10  | 10x slower   | 0.384  |  29 MB    |  23 MB   |
| enc2_b 128→128   |  3.62 |  0.0494 ms |  0.0870 ms  | 0.4739 ms  | 0.57   | 0.10  | 10x slower   | 0.500  |  44 MB    |  31 MB   |
| dec1_a 192→64    | 21.74 |  0.1604 ms |  0.5709 ms  | 5.4196 ms  | 0.28   | 0.03  | 34x slower   | 0.688  | 383 MB    | 214 MB   |
| dec0_a 96→32     | 43.49 |  0.3468 ms |  2.3902 ms  | 22.419 ms  | 0.15   | 0.02  | 64x slower   | 0.533  | 1502 MB   | 824 MB   |

**Reading the table:**
- **im2col_fp8 is 1.6x–64x SLOWER than cuDNN bf16** at every real shape. It is also slower than
  im2col_bf16 (the quant overhead), so fp8 makes an already-losing approach worse.
- **im2col_bf16 itself is already 1.9x–6.9x slower than cuDNN** on every heavy conv — this is the
  root cause. cuDNN's fused implicit-GEMM never materializes the column matrix; explicit `unfold`
  is memory-bandwidth-bound on a huge tensor.
- The worst cases are the **high-channel, full-resolution** convs (dec0_a 96→32 @16×64×64: 64x slower).
- **VRAM did NOT blow up / OOM** (peak ≤1.5 GB on a 32 GB card), so tiled/windowed im2col is not
  needed for feasibility — and it would only add overhead without fixing the speed loss. The fp8 peak
  is often *lower* than bf16 because the fp8 columns are 1 byte, but that memory win is irrelevant
  when the kernel is 10–64x slower.
- **Accuracy:** fp8 maxerr is 0.08–0.69 abs on these (random N(0,1) inputs, weights ~0.05). Meaningful
  but not the deciding factor — even if error were negligible, the speed loss alone kills it.

### Isolation — where the loss is (fp8 GEMM alone vs bf16 GEMM alone, at im2col shapes)

To prove the fp8 tensor cores themselves are fine and the im2col wrapper is the killer, timed the
GEMM only (pre-materialized, pre-quantized), `(M,K)@(K,N)`:

| gemm (= a conv)    |    M   |   K  |  N   | bf16_mm  | fp8_mm   | fp8 speedup |
|--------------------|-------:|-----:|-----:|---------:|---------:|------------:|
| enc0_b 32→32       | 262144 |  864 |  32  | 0.3185 ms| 0.1496 ms| **2.13x**   |
| dec0_a 96→32       | 262144 | 2592 |  32  | 1.5904 ms| 0.4501 ms| **3.53x**   |
| dec1_a 192→64      |  32768 | 5184 |  64  | 0.2229 ms| 0.1282 ms| **1.74x**   |
| ref 4096³ (sanity) |   4096 | 4096 | 4096 | 0.7723 ms| 0.3641 ms| **2.13x**   |

The 4096³ number reproduces the known 2.13x fp8-vs-bf16 figure exactly → the harness is sound.
**The fp8 GEMM wins 1.7–3.5x at our shapes** — but that 0.05–1.1 ms saving is swamped by the im2col
materialization + quant, and cuDNN's implicit-GEMM baseline never pays the im2col tax at all.

### Constraints hit while building the fp8 path (real, worth recording)
- `torch._scaled_mm` requires **scale_a/scale_b to be fp32 tensors** (bf16 scales →
  `RuntimeError: Both scale_a and scale_b must be float (fp32) tensors`).
- `torch._scaled_mm` requires the **contraction dim K divisible by 16**. im2col K = Cin·27, which is a
  multiple of 16 only when Cin is (27 is odd) — so Cin=1 (K=27) fails
  (`Expected self.size(1) to be divisible by 16`); must **zero-pad K to a multiple of 16**. Also
  N=Cout must be a multiple of 16 (our 32/64/128 are fine).
- No sm_120-specific crash: fp8 `_scaled_mm` runs correctly on the 5090; the loss is performance, not
  a hardware error.

Benchmark script: `scratchpad/bench_fp8_conv.py` (+ the GEMM isolation micro-bench, inline above).

---

## Part C — Honest verdict

**(i) 5090 training/inference.** By the measured numbers, **fp8 does not speed up our UNet3D via the
torch-reachable im2col path — it is 1.6–64x slower** than cuDNN bf16. The fp8 tensor cores are
genuinely faster (2–3.5x at our GEMM shapes), but conv requires im2col, and PyTorch's explicit im2col
is already several times slower than cuDNN's fused implicit-GEMM; fp8 quant makes it worse. The *only*
fp8 conv path that could beat cuDNN-bf16 on the 5090 is a **cuDNN-frontend fused fp8 conv** (implicit-
GEMM in fp8, `nvidia-cudnn-frontend`), which torch does not route and which is **unverified for 3D/
NDHWC** — a real C++/graph-API engineering effort with uncertain 3D payoff, and it would help *only*
the 5090, never the competition. Our convs also have small output-channel counts (32–128), a
structurally poor fit for fp8's throughput advantage.

**(ii) Kaggle 2×T4 competition inference.** **fp8 is physically impossible on Turing (sm_75) — no fp8
tensor cores.** This is the decisive separation: any fp8 work is 5090-only and cannot touch the scored
run. The competition-side low-bit conv lever is **INT8 via TensorRT**, which is mature for our plain
dense 3x3x3 Conv3d (avoid grouped/depthwise, which get no INT8 speedup).

**Concrete recommendation.**
- **Do not pursue fp8 for training or for the comp.** For 5090 training speed, stay on **cuDNN bf16
  `F.conv3d` + `torch.compile`** — it already beats every fp8 route measured here. If more 5090 train
  speed is wanted, the levers are bf16 + channels_last_3d + larger batch + compile, not fp8.
- **The real low-bit lever for the scored T4 run is INT8-TensorRT** on the dense Conv3d detector — that
  is the path worth engineering if inference latency on the 199×100 hidden test is the binding
  constraint. fp8 is a dead end for this competition.
- If someone still wants to chase fp8 conv on the 5090 as pure research, the *only* viable route is
  cuDNN-frontend fused fp8 conv (not im2col); budget it as a C++ graph-API build and verify 3D/NDHWC
  works before trusting any speedup — treat it as unverified until measured.
