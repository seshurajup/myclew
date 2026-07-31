# Bonsai / Ternary-Bonsai — low-bit LLM assessment & what we incorporated

**Date:** 2026-07-16  **Target:** "ternary Bonsai 27B" — https://github.com/PrismML-Eng/Bonsai-demo
**Whitepapers read (in-repo PDFs):** `bonsai-27b-whitepaper.pdf`, `1-bit-bonsai-8b-whitepaper.pdf`, `ternary-bonsai-8b-whitepaper.pdf`.

## TL;DR verdict
Bonsai is **post-training quantization (PTQ) of off-the-shelf Qwen3 checkpoints into a group-wise low-bit
*format* + custom llama.cpp/MLX kernels** — **not** low-bit *training*. The repo is a **deployment/inference
demo** (download GGUF/MLX, run llama-server). There is **no QAT, no STE, no from-scratch low-bit training, and
no "4-bit training"** anywhere in the code or the three whitepapers. The "4-bit" references are a **4-bit KV
cache** (`BONSAI_KV4`, inference) and **4-bit comparison baselines** — not weights being trained at 4 bits.

So the premise "they train at 4-bit" is **not substantiated**. The genuinely reusable idea is the **group-wise
ternary weight format**, whose real, reproducible training method comes from **BitNet b1.58**, not Bonsai. We
built that primitive (`lowbit-qat`), grounded in BitNet — a clean fleet gap-filler.

## 1. The quantization scheme (grounded, quoted)
- **Ternary-Bonsai:** weights in `{-1, 0, +1}` with one **shared FP16 scale per group of 128** (g128).
  `w_i = s_g · t_i,  t_i ∈ {-1,0,+1}`. Effective storage `b_eff ≈ log2(3) + 16/128 = 1.585 + 0.125 = 1.71`
  bits/weight → idealized `16/1.71 ≈ 9.4×` vs FP16. Applied to embeddings, attention projections, MLP
  projections, **and** the LM head; **norms + scale metadata stay higher precision**.
- **1-bit Bonsai:** **binary sign** `{-1,+1}` (`w_i = s_g·(2b_i−1)`, `b_i∈{0,1}`), one FP16 scale per g128 →
  `1 + 16/128 = 1.125` bits/weight (`14.2×`). GGUF type `Q1_0_g128`; MLX packs scale+bias → 1.25 bpw.
  Note this "1-bit" is **binary**, which is *more* aggressive than BitNet b1.58 (ternary).
- **Activations:** kept in the runtime's native precision (FP16/decode path); sign/scale are **decoded inline
  inside the matmul kernel** (not expanded to FP16). This is a **weight-only** low-bit scheme (W-only, not W&A).
- **Granularity:** per-**group** (128) scaling, i.e. between per-tensor and per-channel.
- Ternary GGUF ships as `Q2_0` (group-128, this demo's fork) / `Q2_0_g64` (mainline, 2.25 bpw) — packed into
  a 2-bit container for fast kernels.

## 2. How they "train" at low bit — they don't
The 27B whitepaper is explicit (Section 3): *"Bonsai takes the opposite path from BitNet: it starts from an
off-the-shelf pretrained model and moves it into a binary or ternary representation, so the specific model
practitioners already rely on is preserved rather than replaced."* The 8B papers repeat: *"the architecture is
unchanged: the novelty lies entirely in the weight representation / deployment stack."* Base models = **Qwen3-8B
/4B/1.7B** (and Qwen3.6-27B hybrid-attention for 27B). "End-to-end 1-bit" in their wording means **all layers
quantized (no higher-precision escape-hatch tensors)** — *not* trained end-to-end.

**Skeptic flag (important):** the papers **never disclose the algorithm** that converts FP16→ternary while
keeping ~95% of quality (no rounding rule, no calibration set, no distillation/healing described). Pure
round-to-nearest PTQ to 1.58 bits normally collapses quality; real BitNet b1.58 needed **QAT from scratch**.
Either an undisclosed calibration/healing step exists, or the benchmark numbers (run by PrismML on their own
EvalScope harness) are optimistic. Treat the quality claims as **unverified**.

## 3. Training recipe — n/a (PTQ). What IS engineered
Custom **inline-dequant matmul kernels** per backend: llama.cpp CUDA + Metal, an MLX fork, an mlx-swift fork
(all linked from the repo). Plus a 4-bit KV-cache quantizer, a DSpark speculative-decoding drafter, and a
262K-token context. Norms/embeddings/scales kept higher precision. **No optimizer/gradient/activation bit-width
choices** because nothing is trained.

## 4. Claimed results & tradeoffs (their numbers)
- Ternary-Bonsai 8B: **75.5** avg at **1.75 GB** (9.36× smaller than 16.38 GB FP16 Qwen3-8B @ 79.3) → "retains
  >95% quality at ~1/9 memory"; +5.0 pts over 1-bit Bonsai 8B (70.5) for +0.6 GB (the ternary **zero state**).
- 27B: Ternary 80.49 (95% of FP16) @ ~5.9 GB; 1-bit 76.11 (90%) @ ~3.9 GB.
- Throughput/energy: on Apple/CUDA, decode is memory-bandwidth-bound → 1.7–8.4× faster token-gen, 3–6× lower
  energy/token vs FP16 (bandwidth argument, sound in principle).

## 5. Novelty vs prior art (be a skeptic)
- **Not novel:** ternary `{-1,0,+1}` + absmean scaling = **BitNet b1.58** (Ma et al., arXiv:2402.17764);
  1-bit sign+scale Transformers = **BitNet** (Wang et al., arXiv:2310.11453). Group-wise low-bit weight
  formats with an FP scale per group are standard GGUF/PTQ practice (Q2_K, IQ-quants, AWQ/GPTQ groups).
- **The real (modest) contribution:** an **engineering/packaging** one — very-low-bit **PTQ of a strong
  modern checkpoint (Qwen3)** into deployable `Q1_0_g128` / `Q2_0` formats **with working cross-backend
  kernels** (llama.cpp/MLX/mlx-swift) and an honest "true average bits/weight" accounting (their Table 1
  calling out that "4-bit" builds are really ~5.2 bpw). The claim that PTQ (no retrain) reaches sub-2-bit at
  95% quality — if real — would be the interesting part, but the **method is undisclosed and unverified**.
- Distinct from GPTQ/AWQ/QuIP#/EfficientQAT/PEQA in *format+kernel*, not in any new *learning* method.

## 6. What our fleet already had vs the gap
- **Had (all PTQ / inference-time):** `quantize` (INT8-W8A8 PTQ + ToMe token-merge estimate for T4 ViT),
  `compress-select` (ShortGPT block-influence DEPTH prune / LaCo layer-collapse), `binary-size-compressor`,
  `code-compress-optimizer`, `nnue-trainer` (has a small QAT note).
- **Gap:** no **weight quantizer** (ternary or int4), no **fake-quant**, no **straight-through estimator**, no
  **QAT Linear** — i.e. no way to **train/fine-tune under low-bit weights**. That is exactly the reusable,
  pure-torch primitive worth having (and the one thing Bonsai's *format* points at, even though Bonsai itself
  doesn't train).

## 7. What we incorporated — `fleet_agents/lowbit_qat.py` (agent `lowbit-qat`)
Pure torch (deps = torch only; ABI numpy 2.4.6 + torch 2.8.0+cu128 untouched), GPU-aware:
- `ternary_quantize(W, per_channel)` — BitNet b1.58 absmean ternary `{-1,0,+1}` (per-tensor / per-channel scale).
- `int_fake_quant(x, bits, per_channel, signed)` — symmetric k-bit fake-quant (int4 weights / int8 activations).
- `STETernary` / `STEQuant` (autograd.Function) + `ste_ternary` / `ste_quant` — straight-through estimators.
- `QuantLinear` — FP32 **master weight** + fake-quant forward (the QAT cell); `from_linear` copies weights.
- `wrap_qat(model, bits, scheme, a_bits, per_channel, skip)` — swaps only `nn.Linear`, keeps
  norms/embeddings/lm_head in high precision (Bonsai & BitNet both do this).
- `qat_finetune(...)` — STE training loop; `effective_bits(scheme, group_size)` — Bonsai/BitNet bits/weight.

**Why it helps us:** a real **QAT fine-tune lever** — quantize a detector/head to ternary or int4 with STE and
keep training, so bigger models fit the **RTX 5090 (bf16, 32 GB)** and especially the **Kaggle T4 (16 GB)** at
~9.4× (ternary) / ~4× (int4) smaller, complementing our PTQ (`quantize`) and depth-prune (`compress-select`)
levers with the one family we lacked. (The agent's demo reports the same **9.36×** ternary figure Bonsai cites.)

## 8. Verification
- `test_fleet_agents/lowbit_qat_test.py` — **26/26 PASS** (BLAS-pinned, `sys.exit(1)` on fail): ternary values
  ∈ {-1,0,+1}·scale + sign preserved + absmean scale exact; int4/int8 round-trip in tolerance & monotone in
  bits + on-grid; STE grads finite/non-zero (ternary + int); `wrap_qat` swaps only Linears (lm_head/embed/norm
  kept) and both ternary & int4 wrapped MLPs train (loss falls >40%); effective-bits accounting.
- Fleet: `import fleet_agents` clean; **HANDLERS 253 → 254** (+1); no SEED/HANDLER dups; `coverage-audit`
  **UNCLASSIFIED = []**, new **"Compression/Quantization"** pack = `{lowbit-qat}`; agent run → `done`, `learned=True`.

## 9. Low-bit TRAINING — FINISHED (end-to-end capability, 2026-07-16)
Section 7 shipped weight-only fake-quant *primitives*. We have now **completed `lowbit_qat.py` into an
end-to-end low-bit TRAINING capability** (pure torch, ABI untouched: numpy 2.4.6 / torch 2.8.0+cu128;
GPU/CUDA fast path, CPU fallback). All additions are **append-only** — the original functions/signatures and
the 26 original checks are unchanged. Grounded in **BitNet b1.58** (arXiv:2402.17764) + **LLM-QAT**
(arXiv:2305.17888) + **8-bit Adam** (Dettmers et al., arXiv:2110.02861) — **not** Bonsai's undisclosed PTQ.

**What is now complete (all measured green):**
- **Group-wise quantization** (Bonsai/BitNet **g128** block scaling): `ternary_quantize(..., group_size=128)`
  and `int_fake_quant(..., group_size=128)` chunk the last dim into groups and scale **per group** (absmean for
  ternary, absmax for int) — the genuine Bonsai weight format. Default `group_size=None` is byte-identical to
  the original per-tensor/channel path. `dequant_ternary(t, scale, g)` inverts the grouped code.
- **Activation quantization (W&A, true low-bit compute — not weight-only):** `act_fake_quant(x, bits, per_token)`
  is a dynamic **per-token** int8/int4 activation fake-quant with an STE (detach-trick, identity backward).
  `QuantLinear(act_bits=8)` / `wrap_qat(act_bits=8)` give the **BitNet W-ternary/A8** path (loss falls 12.49→1.50).
- **Low-bit optimizer states (the *other* "4-bit training" — a memory lever):** `LowBitAdam(state_bits=8,
  block=128)` stores `exp_avg`/`exp_avg_sq` **block-quantized** (per-block absmax int8/int4) and dequantizes on
  use. The second moment is stored in **sqrt-domain** so linear low-bit quant keeps small values (denominator
  stays conditioned — no blow-up). **Measured:** trains a tiny net 6.59→0.002 (100% drop) with stored state
  **2376 B vs fp32 Adam 7176 B (~3.0× smaller;** 8-bit codes are 1 B vs 4 B/param + tiny per-block scales;
  int4 states shrink again ~2×).
- **Complete hardware-aware, gradual QAT fine-tune loop:** `lowbit_finetune(model, batches, *, scheme,
  weight_bits, act_bits, group_size=128, keep_fp=(...), gradual=True, warmup_frac=0.3, optimizer, loss_fn,
  hardware_aware=True, epochs)` — wraps the model (FP master weights, skips norms/embeds/head), **linearly ramps
  a lambda 0→1** over `warmup_frac` interpolating fp↔quantized weights in the forward (no training shock), then
  full quant; when `hardware_aware`, reads `hardware_tune.load_config()` and **autocasts the master-weight
  compute in its `amp_dtype` (bf16 on the 5090)** — verified `amp_dtype_used="bf16"` on CUDA. Returns
  `{loss_curve, final_loss, effective_bits, amp_dtype_used, quantized_layers, ...}`. **Measured:** ternary
  16.33→11.90 (CPU), 8.62→6.44 (CUDA/bf16), effective **1.71 bits/weight**.
- **REAL packing + memory (the win is actual, not simulated):** `pack_ternary`/`unpack_ternary`
  (**5 trits/byte, 3⁵=243<256 → ~1.6 bpw**), `pack_int4`/`unpack_int4` (2 nibbles/byte) — **round-trip EXACT**.
  `effective_memory_bytes(...)` reports packed bytes **incl. per-group fp16 scales** vs fp16/fp32:
  **measured ternary 9.28× / int4 3.88× vs fp16** (18.55× / 7.75× vs fp32). `quantize_kv`/`dequantize_kv` =
  per-token/per-head **absmax int4 KV-cache** (Bonsai's actual "4-bit") — round-trips within tol (err ≈0.25σ).
- **Agent modes** over the board: `spec['mode']` ∈ `{'primitives'`(default, unchanged)`, 'finetune'`(runs the
  full loop on a synthetic task → `done` with a falling loss)`, 'memory'`(reports packed compression for a
  shape/scheme)`}`. Empty spec → `done` (byte-identical to before).

**Memory math / why it matters.** Weights: ternary at ~1.6 bpw + g128 fp16 scales ≈ **9.3× vs fp16** (a
100 M-param head: 200 MB fp16 → ~21.5 MB); int4 ≈ **3.9×** (→ ~51 MB). Optimizer: fp32 Adam is **8 B/param**
of state; **8-bit block states ≈ 2.1 B/param (~3–4× less)**, int4 less again — often the real OOM wall when
fine-tuning. Together this lets us **fine-tune bigger detectors/heads on the RTX 5090 (bf16, 32 GB)** and,
critically, **fit + run on the Kaggle T4 (16 GB)** where fp16 weights + fp32 Adam would not fit — the QAT
*training* lever we lacked (we previously had only PTQ `quantize` + depth-prune `compress-select`).

**Verification (BLAS-pinned, exit 0):** `test_fleet_agents/lowbit_qat_test.py` **26 → 51 checks, all PASS**
(new: group-wise on-grid/sign/differing-scales for ternary & int; act round-trip + STE grad; W+A training;
LowBitAdam trains >30% + packed state < fp32; `lowbit_finetune` final<initial, effective_bits<4, amp reflects
hw config on CUDA; pack/unpack ternary+int4 EXACT; `effective_memory_bytes` >4× ternary vs fp16; KV int4
round-trip; agent modes finetune/memory/empty). `import fleet_agents` clean; **`lowbit-qat` remains a single
handler — these are extensions, not a new handler** (no `__init__.py` change).

## Links
- Repo: https://github.com/PrismML-Eng/Bonsai-demo — whitepapers in-repo (27B / 1-bit-8B / ternary-8B PDFs).
- HF collections: prism-ml/Bonsai-27B, prism-ml/Bonsai (1-bit), prism-ml/ternary-bonsai.
- **BitNet b1.58** (the real ternary QAT+STE source): Ma et al., arXiv:2402.17764.
- **BitNet**: Wang et al., arXiv:2310.11453.  **LLM-QAT** (fake-quant+STE QAT): arXiv:2305.17888.
- Base models: Qwen3 technical report, arXiv:2505.09388.

## 10. MEASURED ON THE RTX 5090 (2026-07-19) — proof, not literature
We stopped citing and RAN every method on one toy task: a char-level Transformer LM (`TinyGPT` d96·h6·L2)
predicting an **order-2 Markov source** (irreducible entropy floor ≈ **4.64 ppl**). Scripts:
`experiments/lowbit_qat_toy_train.py` (→ `docs/lowbit_train_proof.json`, `docs/lowbit_method_bench.json`)
and `experiments/lowbit_qat_ptq_5090.py` (→ `docs/lowbit_ptq_bench_5090.json`). Learning lesson: **lbq07**.

- **Low-bit TRAINING converges (our `lowbit-qat` agent):** ternary STE **4.867** vs FP bf16 **4.841** (+0.5%,
  **9.28×** smaller, trit-pack round-trip exact); int4 STE 4.836 (≈FP). Real curve: val ppl **31.0 → 4.86**.
- **W4 PTQ (no retrain) is near-lossless — real algorithms, measured:** GPTQ (Hessian+Cholesky OBQ) **4.778**
  (best), HQQ (real `hqq` lib) 4.787, FP4 E2M1 4.798, AWQ 4.809, int8 4.84, NF4 4.872, our int4 4.906.
- **Sub-2-bit PTQ collapses, QAT recovers (the key law, reproduced):** ternary-RTN **12.44**, int2-RTN
  **18.06** (no retrain) — but ternary-STE 4.87, int2-STE 4.96, int3-STE 4.83 (QAT). → **QAT is mandatory
  below ~2 bits**; this is why the fleet needs `lowbit-qat`, not only PTQ.
- **FP4/NVFP4 honesty:** we measured E2M1 *quality* only; NVFP4 tensor-core *throughput* needs Blackwell +
  TransformerEngine (not installed) and **zero** benefit on a Turing T4 → not a T4 lever.
- **Verdict:** b1.58 ternary = best QAT default for max shrink; int3/int4 STE if a bit can be spared; **GPTQ
  or AWQ = best no-retrain W4** for shrinking a trained model for the offline T4. **Fleet gap candidate:** a
  calibration W4 **PTQ agent (GPTQ/AWQ) with Turing kernels** (build when a comp needs it); NVFP4-training
  not worth building for us. Env kept intact (numpy 2.2.6 / torch 2.8.0+cu128; hqq installed `--no-deps`).
