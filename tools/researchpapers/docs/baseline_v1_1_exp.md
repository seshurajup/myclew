# baseline_v1 — (1,2,2) higher-resolution detector

> Status: **QUEUE-READY.** Dry-run PASS (all 3 configs) + real 1-iter GPU smoke PASS.
> Researcher-owned design note. Result note (trainer): `docs/baseline_v1_1_exp_result.md`.

## 1. Version goal
Beat the reproduced golden-12 baseline **0.8708** (official metric = adj edge-Jaccard + 0.1·div-Jaccard,
7 µm node match) by retraining the detector at **`(1,2,2)`** finer XY resolution (≈0.81 µm XY vs
pilkwang's ≈1.63 µm) for better localization → higher edge-precision → higher `adj_edge`.
Absorb augmentation testing into v1 (the standalone `config/aug_ablation/` stalled, no `RESULTS.txt`).

## 2. Experiments (A/B, golden-12 fold 0)
Two clean A/Bs sharing one budget:

| id | config | downsample | batch | augs | isolates |
|---|---|---|---|---|---|
| **v1_1** | `baseline/experiments_v1/v1_1_ctrl_1x4x4.yaml` | 1,4,4 | 16 | brightness+flip | control / internal ref (~0.8708) |
| **v1_2** | `baseline/experiments_v1/v1_2_hr_baseaug.yaml` | **1,2,2** | 4 | brightness+flip | **resolution** (v1_2 vs v1_1) |
| **v1_3** | `baseline/experiments_v1/v1_3_hr_richaug.yaml` | 1,2,2 | 4 | +rot90_xy/gamma/contrast | **augmentation** (v1_3 vs v1_2) |

Shared: `lr=1e-4`, `epochs=30`, `max_iters=300` (bounded epoch, tractable at hi-res),
`early_stop_patience=8`, `window_size=2`, `pool_kernel_um=5.0` (µm, grid-independent),
`det_loss_weight=10.0`, `det_neg_weight=0.01`, split=golden-12 fold 0.

## 3. Trainer route (decided, with justification)
Wired to **`research/official_repo/scripts/train_unet_transformer.py`** — the detector trainer that
**all 12 existing `config/*.yml` already use**. This IS pilkwang's UNet+transformer detector pipeline
plus the infra the leader's requirements need:
- **config-driven augmentations** (`CELLMOT_AUGMENT`) → required for v1_3 (rot90_xy/gamma/contrast).
- **built-in MLflow** (`MLFLOW_*`) → required for "log every run".
- **`CELLMOT_CACHE_DIR`** fast fp16 cache **with strided-zarr fallback** → honors the "no cache
  dependency" intent (works with the cache removed) while giving ~3× faster hi-res epochs when present.

> **Grounded deviation from "pilkwang_support_pack copy":** that copy has **no** `CELLMOT_AUGMENT`,
> **no** MLflow, **no** cache, and its `augmentations.py` has **no** rot90/gamma/contrast (grep-verified).
> The leader's v1_3 (richer augs) + MLflow-every-run are **impossible** there and native in official_repo.
> Same detector science; only logging/aug/cache infra differs. Flagged to leader.

## 4. Cache note
`research/cache/ds1x2x2` was **built** (CPU preprocessing, `scripts/build_cache.py --downsample 1,2,2`,
199 files, **41.7 GB, 39 s**) during the earlier human+leader "build the ds1x2x2 cache" window. The
later "don't build the 40 GB cache" message arrived after it was already done. It is harmless and only
used if `CELLMOT_CACHE_DIR` is set; configs set it for speed but the trainer falls back to strided-zarr
if it's removed. Leader/human may keep, ignore, or delete it — no correctness impact. `ds1x4x4` (9.8 GB)
already existed and is used by v1_1.

## 5. Memory (the 4× risk, resolved)
`(1,2,2)` = 4× voxels of `(1,4,4)`. Real 1-iter smoke ran cleanly at batch 2 with the 2.08 M-param
UNet on the 32 GB RTX 5090; configs use **batch 4** for (1,2,2). The trainer has **no grad-accum flag**
(confirmed) — raising effective batch would require modifying reference code, which we avoid; if batch 4
OOMs at full scale, drop to batch 2 (a one-line config edit), not a code change.

## 6. Measurement plan (IMPROVE_PLAYBOOK discipline)
- Primary: **official metric on golden-12** (`research/official_repo/scripts/evaluate.py`), full pipeline.
- A finer detector is **density-CHANGING** → golden-12 partly blind (Rule 1). Also report **detector
  recall proxy + predicted-count vs density cap**; **flag (1,2,2) detection gains as NEEDS-LB (human submits).**
- The `adj_edge`/edge-precision gain is the CV-judgeable part → lead with it.
- **DO NOT submit to Kaggle.** **Every number from a real run.** Log every run to MLflow
  (`http://localhost:5000`, exp `kaggle-biohub-cell-tracking`) — native in the trainer.

## 7. Package (queue-ready paths)
- code: `src/baseline/__init__.py`, `src/baseline/train.py` (observable launcher: STARTUP banner +
  YAML→env/argv + tee to `output/baseline_v1/<id>/train.log`; GPU-safe dry-run).
- configs: `baseline/experiments_v1/{v1_1_ctrl_1x4x4,v1_2_hr_baseaug,v1_3_hr_richaug}.yaml`.
- entrypoint: `baseline/run_baseline.py --config <yaml> [--dry-run] [--fold 0]` → `python -m src.baseline.train`.
- formal runner: `baseline/run_experiments_v1.sh [--dry-run]` (fans out to all 3 configs).
- outputs: `output/baseline_v1/<id>/` (train.log; checkpoints in `official_repo/weights/<method>/split_0/`).

## 8. Validation evidence (real, run 2026-07-04)
- **Reproduction:** golden-12 FULL post-proc micro adjJ = **0.8708** (`learning/ensemble_work`), div_tp=0.
- **Dry-run (all 3):** PASS — paths/splits(187/12)/cache(199)/aug names validated, trainer import-check
  PASS with `CUDA_VISIBLE_DEVICES=""` (zero GPU), runner rc=0.
- **Real 1-iter GPU smoke (v1_2 stack, 1,2,2):** PASS — cache loaded, model built (2.08 M params, CUDA),
  startup + per-epoch log emitted (`Epoch 0/1 | edge=0.0019 | det=0.1328 | train=1.4s test=4.0s`),
  checkpoint saved, no OOM.

## 9. Observability (satisfied)
- `src/baseline/train.py` STARTUP banner + the trainer's own `Starting training for N epochs...` startup log.
- Trainer per-epoch stdout: `Epoch e/N | edge | det | test_loss | acc | recall | best | train/test s` (≥1/epoch),
  mirrored to MLflow.

## 10. Expected outcome & next
- v1_1 control anchors our-retrain reference at (1,4,4); v1_2 vs v1_1 = the resolution delta on golden-12
  official (CV-judgeable part) + recall proxy (needs-LB). v1_3 vs v1_2 keeps only augs that raise held-out recall.
- If (1,2,2) helps: v2 = division-aware training (localization now finer) + best augs, and prepare a
  submission-ready notebook for the **human** to submit (never us).
