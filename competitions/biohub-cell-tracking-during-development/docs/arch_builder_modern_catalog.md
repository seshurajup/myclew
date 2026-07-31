# arch-builder modern-technique catalog

`fleet_agents/arch_builder.py` now carries a GROUNDED, queryable catalog of modern architecture components
and training recipes that the builder can COMPOSE and PROPOSE. Same idea `arc-idioms` did for ONNX golf: a
menu the builder proposes from — except every entry's CONSTRAINT is a MEASURED number from this session's
artifacts, not generic hype.

## API (pure, deterministic, data-wise tested)

- `arch_builder.catalog(category=None)` — the full technique list (or one of `architecture / quantization /
  training / gate`). Each entry: `name, category, what, when, constraint, plugs_in, fleet_agent, source,
  measured, match`.
- `arch_builder.propose(target_profile=None)` — given a target, returns `{recommended, excluded,
  substitutions, gate}`. `target_profile` keys (all optional; defaults = the biohub competition target =
  Kaggle **T4-offline, sparse-label**): `hardware` (t4/turing/5090/blackwell/cpu), `data_regime`
  (sparse_label/weak_label/heterogeneous/multi_stage/dense/homogeneous), `bit_budget` (bits/weight),
  `context` (short/long), `multimodal`, `pretrained_backbone`, `task`.
- Fleet handler `arch-catalog` (`catalog_query`) exposes both via `fleet_dispatch`. Read-only, no training.
- `arch-builder`, `arch-search`, `detector-arch-search` now append a grounded `propose()` note to their
  posted verdicts, so an arch/config search is never arch-blind about how to SHIP the winner.

## The measured constraints it codifies

These are the load-bearing verdicts the builder must never contradict:

- **int8, not FP4, on T4.** `propose({hardware:'t4'})` EXCLUDES `fp4-nvfp4` (Blackwell-only; ZERO throughput
  benefit on Turing — `docs/lowbit_ptq_bench_5090.json` `fp4_hardware_honesty`) and recommends
  `int8-w8a8` (measured near-lossless: int8 PTQ 4.842 ppl; GPTQ int4 4.778 = best no-retrain W4).
- **ternary needs QAT.** `bit_budget < 2` REQUIRES `qat-bitnet-ternary`. Measured: ternary PTQ RTN collapses
  to 12.4–18.1 ppl; ternary STE QAT recovers to 4.867 (+0.5% vs FP) at 1.71 bpw → 9.28× smaller
  (`docs/lowbit_train_proof.json`, `docs/lowbit_method_bench.json`). int2 PTQ 18.06 vs int2 QAT 4.958.
- **trust-region for weak labels.** `data_regime='sparse_label'` → `trust-region-self-train` (Direct-OPD
  weak-to-strong under a firm anchor: freeze-backbone + low-LR / KL-leash). Cites the MEASURED retraction —
  the external +0.035 "gain" was a red herring (dense GT ≠ competition sparse GT), caught by per-embryo LOEO
  (`biohub_autonomous_run_20260714`).
- **hardware-tune default + LOEO gate always.** Every proposal ships with the measured 5090 train config
  (bf16 1.83×, batch_scale 2.11, tf32/compile/channels_last/muon — `docs/hardware_config.json`) and the
  mini-first + keep-if-improves embryo-disjoint 2-CV gate (multi-seed, never 1-seed).

## Catalog entries

Architecture: `moe-conditional-compute` (Gemma-4; memory=total-params is the constraint → pair with QAT),
`encoder-free-multimodal`, `kv-cache-efficiency` (5:1 local:global + values-as-keys → ≤37.5% KV cut),
`component-graft` (reuse early encoder blocks under our fast head).
Quantization: `int8-w8a8`, `qat-bitnet-ternary`, `fp4-nvfp4` (hard-negative on T4).
Training: `hardware-tune-config`, `trust-region-self-train`, `speculative-draft-verify` (MTP optimal γ),
`gm-training-tricks` (EMA/SWA/mixup/focal/SAM/ArcFace — focal was MEASURED to hurt the biohub linker, killed).
Gate: `mini-first-loeo-gate`.

Each entry references the fleet agent that already implements it (`moe-inference-cost`, `kv-cache-longctx`,
`component-graft`, `quantize`, `lowbit-qat`, `hardware-tune`, `pseudo-label`/`detector-transfer`,
`mtp-speculative-decode`, `train-tricks`, `feasibility-gate`) — integrate by reference, do not duplicate.

## Test

`test_fleet_agents/arch_catalog_test.py` (7 checks) asserts propose() composes the measured constraints:
FP4-excluded-on-T4/int8-preferred, sub-2-bit-requires-QAT, sparse-label→trust-region, heterogeneous→MoE,
gate+hardware-tune always, catalog shape/fields, handler returns a proposal. Auto-discovered by `run_all.py`.
