# fp8 LLM framework — PHASE 2 (MTP + QAT training knobs, offline inference companion)

**Track:** LLM / Gemma-scale transformer trainer. **NOT biohub cell-tracking** — the biohub UNet is bf16-conv;
this is the reusable matmul-heavy decoder LM path. The LM's "CV" is val-loss / perplexity, never edge_jaccard.

Phase 2 adds, all **OFF by default so the proven phase-1 fp8 path is byte-identical** (dense dry-run still reports
`fp8`, `compile=True`, `cfg_hash=ae35cbbc583c282f` — the same hash the phase-1 checkpoints carry, so they resume):

- **A.1 — MTP head** (`model.mtp_heads`, `train.mtp_coef`)
- **A.2 — QAT mode** (`precision.qat`) — an *alternative* low-bit mode to fp8
- **B — offline inference companion** `experiments/infer_fp8_transformer.py` (kv-cache + speculative-decode + moe-cost)

All measured on RTX 5090 / `research/cellmot_venv` (torch 2.8.0+cu128, numpy 2.4.6, cv2 4.13.0 — **ABI verified intact
before and after**). Proof LM is tiny + short-trained: these numbers verify **MECHANICS + measured deltas, not SOTA**.

---

## New YAML knobs (`config/fp8_transformer.yml`)

```yaml
model:
  mtp_heads: 0          # A.1: >0 → N extra heads predict t+2..t+N+1 (Gemma-4/DeepSeek MTP). 0 = phase-1 identical.
precision:
  qat: off              # A.2: off | int8 | int4 | ternary. Wraps Linears with lowbit_qat QuantLinear INSTEAD of fp8.
train:
  mtp_coef: 0.3         # A.1: total loss = next-token CE + mtp_coef · Σ(extra-head CE)
infer:                  # B: offline inference companion
  checkpoint: ""        # "" → <out_dir>/<name>_best.pt (discovered by PATH, offline)
  seed_bytes: 128       # real-corpus prompt length
  gen_tokens: 192       # auto-capped so seed+gen ≤ model seq_len
  spec_gamma: 0         # 0 → use all MTP heads as the drafter
  greedy: true          # greedy → cached vs uncached output MUST be byte-identical (a cache-correctness proof)
  budget_s: 120         # whole run timed against this; prints "FITS BUDGET"
  out_dir: experiments/fp8_infer
```

CLI overrides: `--mtp-heads N`, `--qat int4|int8|ternary` (trainer); `--checkpoint PATH`, `--gen-tokens N` (infer).
Every checkpoint now also stores an `arch` dict so the inference companion rebuilds the exact model offline.

**Modes never stack:** fp8 = FAST-TRAIN (native `_scaled_mm`, compiled). QAT = SHRINK-TO-DEPLOY (fake-quant + STE,
eager). MoE / QAT / MTP each force eager. Config `off`/`on` are YAML booleans → normalized via a positive whitelist
(`int8|int4|ternary`) so `qat: off` can never mis-fire (a bug caught + fixed during the build).

---

## PART A — trainer proof (real log lines)

Small dedicated configs `config/fp8_mtp_test.yml` / `config/fp8_qat_test.yml` (d_model 512, depth 3, seq 128, real
byte corpus), 3 epochs fresh.

### A.1 MTP (`mtp_heads=2`) — trains, main loss ↓, aux losses logged + finite
```
[mtp] mtp_heads=2 mtp_coef=0.3 → 2 extra heads predict t+2..t+3 (Gemma-4/DeepSeek MTP)
[epoch 1/3] loss=2.7174 | ... | mtp_aux=[3.1553,3.3225] finite=True | val_loss=2.6562 ppl=14.24
[epoch 2/3] loss=2.5885 | ... | mtp_aux=[3.0689,3.2613] finite=True | val_loss=2.5449 ppl=12.74
[epoch 3/3] loss=2.5071 | ... | mtp_aux=[2.9956,3.1563] finite=True | val_loss=2.4609 ppl=11.72
```
Main CE 2.717→2.507, val 2.656→2.461, both MTP head aux losses finite and **decreasing**. Verdict: **WORKS.**
(Head construction + target-shift are inline — `mtp_speculative_pack` is decode arithmetic, not head modules;
it IS reused at inference below.)

### A.2 QAT (`qat=int4`) — trains + converges via lowbit_qat
```
[qat] mode=int4 → lowbit_qat.wrap_qat scheme=int4 w_bits=4 | 12 Linears fake-quantized (STE master weights);
      norms/embeds/head kept high-precision. QAT=shrink-to-deploy (ALTERNATIVE to fp8=fast-train).
[fp8-train] precision policy → qat-int4 ...
[epoch 1/3] loss=2.5754 | ... | val_loss=2.6484 ppl=14.13
[epoch 3/3] loss=2.4347 | ... | val_loss=2.4434 ppl=11.51
```
12 Linears wrapped by `fleet_agents/lowbit_qat.wrap_qat` (int4, STE); loss 2.575→2.435, val 2.648→2.443.
Verdict: **WORKS — int4 QAT converges.**

---

## PART B — offline inference companion (`experiments/infer_fp8_transformer.py`)

```
OMP_NUM_THREADS=1 research/cellmot_venv/bin/python experiments/infer_fp8_transformer.py --config config/fp8_mtp_test.yml
```
Offline path-discovery → rebuild from saved `arch` (or reconstruct shapes for legacy ckpts) → load weights → time
against budget → write `experiments/fp8_infer/<name>_{generated_sample.txt,infer_report.json}` → emit :7788 events.

### KV-CACHE (reuses `kv_cache_pack` for cache memory; runtime incremental cache inline)
| checkpoint | no-cache tok/s | cache tok/s | speedup | greedy cache==no-cache | kv_cache_pack size |
|---|---|---|---|---|---|
| MTP (10M, d512, seq128) | 727 | 754 | **1.04×** | **True (100% match)** | 0.79 MB |
| MoE (341M, d1536, seq256) | 143 | 222 | **1.55×** | **True (100% match)** | 6.29 MB |

The identical greedy output **proves the incremental cache is correct**; speedup grows with model size (bigger =
more recompute the cache saves). Verdict: **WORKS.**

### SPECULATIVE DECODE (reuses `mtp_speculative_pack` for the speedup arithmetic)
MTP checkpoint (γ=2 self-draft heads, Medusa-style draft+verify):
```
[spec-decode] γ=2 measured α=0.042 → mtp_speculative_pack: 1.04 tok/verify, theoretical 1.03× (c=0.0065); best γ=1→1.03×
[spec-decode] wall-clock: spec 706 tok/s vs cache 754 tok/s = 0.94× (24 target passes for 24 tokens)
```
Mechanism **verified correct** (drafts proposed by MTP heads, verified by the target in one pass, longest correct
prefix accepted + a bonus token). **Honest — modest on tiny model:** measured acceptance α=0.042 is low because the
3-epoch model's greedy rollout is degenerate (`" (s the the the the"`), and wall-clock is 0.94× (the 2-pass impl +
near-zero acceptance loses to an already-cheap cache baseline at this scale). `mtp_speculative_pack` quantifies the
real lever at measured α (1.03× here; scales up with α on a well-trained drafter). Checkpoints without MTP heads
report **N/A and skip** (honest).

### MOE-INFERENCE-COST (reuses `moe_inference_pack.moe_cost`)
MoE checkpoint (4 experts, top-2, 4 layers):
```
[moe-cost] moe_inference_pack: 190.1M active / 341.1M total (compute 55.7% of dense-total → 1.8× cheaper compute)
```
Verdict: **WORKS.** Dense checkpoints report **N/A** (honest).

### Kaggle-offline discipline
- Paths discovered, no internet. Whole run **timed vs `budget_s`**: e.g. MTP run `TOTAL 1.02s / budget 120s → FITS BUDGET`.
- Artifacts written; `infer_report.json` carries tps / cache-speedup / spec α+speedup / moe-cost / fits_budget.
- **:7788 events** land on the LIVE board `tools/researchpapers/.research-mvp-data/runtime/thread.jsonl`
  (`infer_start`/`kv_cache`/`spec_decode`/`moe_cost`/`infer_done`; `.runtime/` was the stale phase-1 path — mirror
  now points at the live board and de-dupes against `post_thread`).

---

## Honest verdicts

| item | verdict |
|---|---|
| A.1 MTP head trains, aux losses logged + finite | **WORKS** |
| A.2 int4 QAT trains + converges (lowbit_qat) | **WORKS** |
| B kv-cache tokens/s + cache-correctness | **WORKS** (1.04×–1.55×, greedy output identical) |
| B speculative decode (MTP drafter) | **MECHANISM WORKS; modest on tiny model** (α=0.042, theo 1.03×, wall 0.94×) |
| B moe-inference-cost | **WORKS** (190M/341M, 55.7%) / N/A when dense |
| dense/default path (mtp=0, qat=off) unchanged | **byte-identical** (fp8, compile=True, cfg_hash `ae35…`) |
| ABI (torch 2.8.0+cu128 / numpy 2.4.6 / cv2 4.13.0) | **intact** (no installs) |

**Reuse notes (honest fits):** `kv_cache_pack` (memory calculator) and `mtp_speculative_pack` (decode arithmetic)
have no runtime module for a concrete byte-LM, so the incremental KV cache and the draft/verify loop are implemented
**inline** and cite the pack they mirror; both packs ARE called for their arithmetic (cache bytes; expected-tokens /
speedup / optimal-γ). `moe_inference_pack.moe_cost` and `lowbit_qat.wrap_qat` applied **directly**.
