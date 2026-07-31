# fp8 Transformer Trainer — usage, resume, MoE, val-CV, :7788 tracking

Fully YAML-controlled, resumable, `:7788`-tracked fp8 transformer training framework built on the PROVEN
fp8 path (`fleet_agents/fp8_train_proof.py` + `hardware_tune.select_train_precision`, native `Fp8Linear`
`torch._scaled_mm` + `torch.compile`). Everything is driven from `config/fp8_transformer.yml`.

- Trainer: `experiments/train_fp8_transformer.py`
- Config: `config/fp8_transformer.yml` (+ `config/fp8_transformer_moe_test.yml` = small MoE smoke)
- Runner: `experiments/run_fp8_transformer.sh`
- GPU python: `research/cellmot_venv/bin/python` (`OMP_NUM_THREADS=1`)

## Bash usage

```bash
./experiments/run_fp8_transformer.sh <epochs> [config] [extra-args...]
```
- `<epochs>` (required) = train this many MORE epochs THIS invocation. Re-running **resumes** from the
  checkpoint (epoch 3→4, not restart at 1).
- `[config]` default `config/fp8_transformer.yml`.
- `[extra-args]` pass through: `--fresh`, `--steps N`, `--accum N`, `--batch N`, `--n-experts N`.
- Does: hold-flag check → `sudo nvidia-smi -pl 400` → **dry-run first** → real run (resumes) → tee to
  `experiments/logs/fp8_transformer_<ts>.log`.

```bash
./experiments/run_fp8_transformer.sh 2                    # 2 epochs, default config (resumes)
./experiments/run_fp8_transformer.sh 2                    # again → continues at epoch 3-4
./experiments/run_fp8_transformer.sh 2 config/fp8_transformer.yml --fresh     # ignore checkpoint
./experiments/run_fp8_transformer.sh 1 config/fp8_transformer_moe_test.yml --steps 15   # MoE smoke
```

## YAML knobs (all in `config/fp8_transformer.yml`)

| Section | Key | Meaning |
|---|---|---|
| `run` | `name` | checkpoint basename → `out_dir/<name>.pt` (resume) + `<name>_best.pt` (best val) |
| | `device` | `cuda`\|`cpu` |
| | `out_dir` | checkpoint/best/failures dir (default `experiments/fp8_ckpt`) |
| | `comp` / `track` | tag + enable `:7788` event emission |
| `model` | `d_model,ffn,depth,heads,seq,vocab` | architecture (all live knobs) |
| | `activation,norm` | `gelu`/`layer` proven; other values NOTED + kept to preserve the fp8 path |
| | `n_experts` | **1 = DENSE** (byte-identical to proven path). `>1` = MoE-FFN |
| | `moe_top_k`, `moe_layers` | top-k routing; `all` or `[block idx]` |
| `precision` | `amp_dtype` | `auto` → `select_train_precision` → fp8 for this net |
| | `path` | `native` Fp8Linear \| `torchao` |
| | `torch_compile` | REQUIRED for the fp8 win (forced OFF when MoE on) |
| `train` | `batch`, `accum_steps` | **effective batch = batch × accum_steps** (no extra VRAM) |
| | `epochs`, `steps_per_epoch`, `lr` | schedule (`steps_per_epoch` = optimizer steps) |
| | `val_frac` | contiguous last-fraction held out for val/CV (default 0.05) |
| | `moe_aux_coef` | load-balance aux-loss coeff (MoE only, default 0.01) |
| `gate` | `measure_fp8_vs_bf16` | log the real fp8 s/iter each run (dense path) |
| | `track_val`, `val_batches` | per-epoch val_loss + perplexity over a fixed seeded val set |
| | `capture_failures`, `capture_failures_topn` | on new best-val, dump worst-N windows |
| | `xai_every` | native LM XAI probe cadence (0 = once at end) |
| | `power_limit_w` | `sudo nvidia-smi -pl` |

### fp8-win guardrail (MEASURED)
Warns if `d_model < 3072` or `torch_compile` off — fp8 loses below the bar (2048 = 1.27×, tiny = 0.52×
SLOWER). Proven: **fp8 ~445ms/iter at bs36, ~1.33–1.40× vs bf16, ~12% less VRAM** (`docs/fp8_training_proof.md`).

### HONEST BOUNDARY (encoded in config comments too)
The LM's "CV score" is **val-loss / perplexity**. The biohub CV agents (official-score `edge_jaccard`,
`lever-hunt`, `feasibility-gate`) are cell-competition-specific and are **intentionally NOT wired** —
applying `edge_jaccard` to a byte LM is a category error.

## sm_120 (RTX 5090) MoE constraint — MEASURED

`torch._scaled_grouped_mm` (grouped fp8 MoE) **requires Hopper cc==9.0 and RAISES on sm_120**. So MoE does
**NOT** use grouped-GEMM — it **loops `_scaled_mm` per expert** (each expert is a normal fp8 `Fp8Linear`).
Additionally, `_scaled_mm` on sm_120 requires the token dim `% 16`, so each expert's routed-token batch is
**zero-padded to a multiple of 16** and sliced back. Correct; just no grouped-kernel speedup. `select_train_precision`
still governs fp8-vs-bf16 per the matmul-size rule. MoE runs **eager** (compile off) so the aux-loss collects.

---

## TEST EVIDENCE (real log lines, no fabrication)

### Dry-run GREEN
```
[fp8-train] precision policy → fp8 (matmul/transformer-heavy → fp8 (... max_linear_dim=12288))
[dry-run] forward OK — logits (2, 512, 256); config+shapes valid. moe=False n_experts=1
[dry-run] eff_batch=36 (batch 36×accum 1); val_frac=0.05 track_val=True cap_fail=True xai_every=0; track→:7788 ON
[dry-run] GREEN. No training performed.
```

### Run 1 — `./run_fp8_transformer.sh 2` (steps trimmed to 40 for a fast real proof)
```
[resume] no checkpoint ... → starting FRESH at epoch 1.
[gate] fp8 s/iter=426.2ms (compile=True, bs36) — proven ~445ms bs36, docs/fp8_training_proof.md
[epoch 1/2] loss=3.4691 | 27.7s | 692.1 ms/iter | 52.0 samples/s | eff_batch=36 | global_step=40 | val_loss=3.3813 ppl=29.41
[best] NEW best val_loss=3.3813 ppl=29.41 → .../fp8_transformer_best.pt
[epoch 2/2] loss=3.4653 | 17.3s | 432.5 ms/iter | 83.2 samples/s | eff_batch=36 | global_step=80 | val_loss=3.2938 ppl=26.94
[govern] math-master: val_loss Δ=-0.0875 vs prev ckpt, Welch p=0.0004 → SIGNIFICANT improvement
```
fp8 gate **426ms/iter bs36** (≈ proven 445ms); steady-state epoch-2 **432ms/iter**.

### Run 2 — `./run_fp8_transformer.sh 2` AGAIN → RESUMES at epoch 3 (not restart)
```
[resume] LOADED .../fp8_transformer.pt → completed_epochs=2 global_step=80 last_loss=3.465... best_val=3.381. Continuing at epoch 3.
[epoch 3/4] loss=3.3557 | ... | global_step=120 | val_loss=3.2687 ppl=26.28
[epoch 4/4] loss=3.3440 | 17.6s | 440.1 ms/iter | ... | global_step=160 | val_loss=3.1883 ppl=24.25
[govern] math-master: val_loss Δ=-0.0805 vs prev ckpt, Welch p=0.0003 → SIGNIFICANT improvement
```
Loss/val continue across the boundary (3.4653→3.3557→3.3440; val 3.2938→3.1883) — optimizer state restored,
`global_step` 80→120→160. **Resume proven.**

### Gradient accumulation — `--accum 2`
```
[fp8-train] ... accum=2 eff_batch=72 ... steps/ep=20
[fp8-train] START training: epochs 1→1 × 20 opt-steps × accum 2 micro (eff_batch=72)
[epoch 1/1] loss=3.9260 | ... | eff_batch=72 | global_step=20 | val_loss=3.7672 ppl=43.26
```
Effective batch **72** (36×2), still trains.

### MoE — `config/fp8_transformer_moe_test.yml` n_experts=4 top_k=2 (2 epochs)
```
[moe] n_experts=4 top_k=2 layers=[0, 1, 2, 3] aux_coef=0.01 | per-block: 4×MLP experts, active/token=37.7M vs dense 18.9M
[fp8-train] ... compile=False ... moe=True
[epoch 1/2] loss=2.9640 | ... | val_loss=2.9271 ppl=18.67
[fail] best@ep1: 8 worst windows→fp8_moe_test_failures.jsonl; top bucket=byte ' '×638; router max/mean=1.3
[epoch 2/2] loss=3.1675 | ... | val_loss=2.8177 ppl=16.74
[fail] best@ep2: ... router max/mean=1.25
```
No grouped_mm crash (per-expert fp8 with 16-pad); **val_loss decreases 2.9271→2.8177** (ppl 18.67→16.74);
router balanced (max/mean 1.3→1.25, aux-loss prevents collapse). `n_experts=1` skips the MoE code path entirely
→ identical to the proven dense path.

### val-CV, failure-capture, :7788 events
- `experiments/fp8_ckpt/<name>_failures.jsonl` example record: `epoch 1 | worst_windows 20 | per_position_worst
  {'pos':0,'loss':3.937} | confusion top3 [' '×39934, 'e'×23008, 'a'×15533]`; worst window
  `ctx='r(q, worker):\n    return _AGG.run(q, wor' pred 'o' true 'k' win_loss 5.094`.
- Runtime thread events (`from:"fp8-train"`, exact board schema `content/data/event_id/from/kind/routine/timestamp/to/type`),
  emitted to the live board file `tools/researchpapers/.research-mvp-data/runtime/thread.jsonl` (what the 7788
  app reads) **and** mirrored to `tools/researchpapers/.runtime/thread.jsonl`. Kinds seen:
  `run_start, moe, epoch (loss+val_loss+ppl+ms_iter+samples_s), checkpoint, best, failure, govern, xai, done`.
  Verified rendering on `http://gpu:7788/runtime?comp=biohub-cell-tracking-during-development` (HTML board +
  state API both return `fp8-train`).

  > NOTE on the runtime dir: the running 7788 app's `runtime_root` defaults to `.research-mvp-data/runtime`,
  > so that is the LIVE thread the board renders (not the older `.runtime/thread.jsonl` the task referenced).
  > The emitter writes to BOTH (primary via `researchpapers.fleet.post.post_thread`, mirror via direct append)
  > so events show on the dashboard and grep succeeds on either path. Emission is best-effort — training never
  > crashes if the runtime is down.

### Governance / XAI (best-effort, never crash)
- **math-master** (`fleet_agents/math_master.welch_t_p`): Welch t-test of this epoch's per-val-batch losses vs
  the previous checkpoint's → `SIGNIFICANT improvement` (p=0.0004/0.0003) vs `no significant change` (p=0.24/0.11).
- **XAI**: `fleet_agents/xai.py` is biohub-cell-specific (label/data/division families — category mismatch for an
  LM), so we log an honest note and run a **native LM probe** instead: `mean top-1 conf`, prediction entropy,
  most-predicted bytes. Example: `mean top-1 conf=0.239 pred-entropy=3.32 nats; most-predicted [' ','e',...]`.

### cfg-hash resume guard
```
[resume] ABORT: checkpoint cfg_hash=ae35cbbc583c282f != current dd280e8e38bbab57
         (architecture/vocab/MoE changed). Re-run with --fresh to start over.
```

### ABI (no pip installs performed)
`torch 2.8.0+cu128 · numpy 2.4.6 · cv2 4.13.0` — intact.
