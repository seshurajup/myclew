# fp8 Training Proof — RTX 5090 (sm_120)

**Verdict: PASS.** fp8 training is real and **faster than bf16** on our RTX 5090, end-to-end
(forward + loss + backward + optimizer.step), with `torch.compile` ON, on LARGE matmuls — measured,
not fabricated. Both a native `torch._scaled_mm` path and the torchao float8 path clear the ≥1.3×
bar and converge identically to bf16.

Date: 2026-07-20 · torch 2.8.0+cu128 · GPU RTX 5090, cap (12,0) · `research/cellmot_venv`
Reusable script: `fleet_agents/fp8_train_proof.py`

## Why large matmuls (the earlier failure explained)
Our earlier fp8 cell-detector ran at **0.40× (SLOWER)** because its patch-token GEMMs were tiny
(M=128, K=256). fp8 has a fixed quantize/dequantize cost that only pays off on big GEMMs.
Confirmed raw (`torch._scaled_mm` vs bf16 `@`, this GPU):

| GEMM (M,K,N)      | bf16    | fp8     | ratio |
|-------------------|---------|---------|-------|
| 128, 256, 256     | 0.005ms | 0.009ms | **0.52×** (slower) |
| 4096, 4096, 4096  | 0.764ms | 0.361ms | **2.12×** |
| 8192, 8192, 8192  | 5.33ms  | 2.94ms  | **1.82×** |

So the proof model must be a REAL transformer with big dims. It is.

## The model (a genuine decoder-only transformer LM)
d_model **3072**, ffn **12288** (4×), depth **8**, heads **16**, seq_len **512**, batch **12**,
vocab **32000**. ~1.10B Linear params. Standard decoder blocks: causal SDPA + GELU MLP.
Softmax/LayerNorm/embeddings/attention score-context stay bf16 — **only the big Linear GEMMs go
fp8** (qkv, o_proj, MLP up/down, and the 3072×32000 LM head). Synthetic token data (this is a
speed+convergence proof, not an accuracy benchmark). Same model/seed/shapes for fp8 and bf16;
`torch.compile` ON for both.

## Graph-break status (critical)
`torch._dynamo.explain` on the fp8 model under compile: **graph_breaks = 0, graphs = 1.** The custom
`Fp8LinearFn` autograd.Function does NOT graph-break — dynamo traces through it and the absmax
quantize fuses with the GEMM. This is exactly why fp8 wins here and did not in the tiny-GEMM detector.

## Measured results (full train step, compile ON, identical config)

### PATH A — native `Fp8Linear` (`torch._scaled_mm`, no install), 200 timing / 300 convergence steps
| metric        | fp8 (native) | bf16     | ratio |
|---------------|--------------|----------|-------|
| **s/iter**    | **183.82ms** | 245.79ms | **1.34×** |
| peak VRAM     | **14.74GB**  | 16.59GB  | 0.89× |
| loss start→end| 10.529 → 5.6e-6 | 10.538 → 5.0e-6 | tracks bf16, no NaN |

### PATH B — torchao float8 tensorwise (`convert_to_float8_training`), 150 timing / 200 conv steps
| metric        | fp8 (torchao)| bf16     | ratio |
|---------------|--------------|----------|-------|
| **s/iter**    | **186.88ms** | 248.65ms | **1.33×** |
| peak VRAM     | **12.46GB**  | 14.31GB  | 0.87× |
| loss start→end| 10.529 → 1.5e-5 | 10.538 → 1.4e-5 | tracks bf16, no NaN |

Both paths: fp8 s/iter < bf16 s/iter by ~1.33–1.34×, fp8 loss decreases monotonically and matches
bf16 to numerical noise, and fp8 uses ~11–13% less VRAM.

## Verdict
**SOLID PROOF — PASS** for "fp8 training works and is faster on our 5090":
- fp8 s/iter < bf16 (1.34× native, 1.33× torchao) ✓ ≥1.3× bar
- fp8 converges ≈ bf16 (no divergence/NaN) ✓
- torch.compile ON, compile fuses (0 graph breaks) ✓
- large matmuls ✓

### Honest caveats (the ceiling)
- The win is **Linear-only** and requires **large GEMMs** (≥~1024 dims, M≥~4096) **plus
  torch.compile**. Non-fp8 work (SDPA, LayerNorm, the fp32 CE over 32000 vocab, optimizer) is fixed
  overhead that dilutes the ratio below the raw 1.8–2.1× GEMM speedup.
- At **d_model=2048** the same model gives only **1.27× (FAIL the 1.3× bar)** — the matmuls are too
  small a fraction of the step. 3072 is where it clears comfortably. Tiny GEMMs (our old detector,
  M=128/K=256) are **slower** in fp8 — do NOT fp8 a conv/patch-token model.
- No fp8 conv kernel exists — conv-dominated models (our UNet3D detector) cannot be fp8-trained.
- torchao 0.17.0 prebuilt wheel: its cpp/cuda extensions require torch≥2.11 and are skipped on our
  torch 2.8, but the **float8 training path is pure-Python over `torch._scaled_mm`** and works fully.

## ABI safety (torchao install)
Installed `torchao==0.17.0` via prebuilt cu128-compatible wheel with `--no-deps`. Env verified
intact **before and after**:
`torch 2.8.0+cu128 · numpy 2.4.6 · cv2 4.13.0 · torchao 0.17.0` — torch/numpy/cv2 unchanged, nothing
clobbered.

## Reproduce
```
cd /home/seshu/kaggle/2026/biohub-cell-tracking-during-development
OMP_NUM_THREADS=1 research/cellmot_venv/bin/python fleet_agents/fp8_train_proof.py \
  --path native  --steps 200 --conv-steps 300 --d-model 3072 --ffn 12288 --batch 12 --seq 512
OMP_NUM_THREADS=1 research/cellmot_venv/bin/python fleet_agents/fp8_train_proof.py \
  --path torchao --steps 150 --conv-steps 200 --d-model 3072 --ffn 12288 --batch 12 --seq 512
```
