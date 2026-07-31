# Stage 1 — dumb end-to-end LOEO baseline (researcher prep, dry-run GREEN)

**Journey Stage 1** (leader green-lit 2026-07-05). ONE baseline, no fan-out (variants = Stage 2).
Pipeline = **learned U-Net detector → greedy link → official_scorer**, trained AND scored on the
Stage-0 FROZEN embryo-disjoint LOEO split. exp#4 fine-tune stays PARKED.

## Config
`config/loeo_detector.yml` — U-Net + transformer edge head, downsample `1,4,4`, 12 epochs, `max_iters 150`.
Two changes made during Stage-1 prep:
1. `splits:` repointed to the FROZEN split `learning/ensemble_work/finetune/fleet_loeo_mini.json`
   (train **and** score on the same split — CV-harness parity, per leader's caveat decision).
   `open_dataset` resolves the split's suffix-less ids → `<id>.zarr` (image) + `<id>.geff` (tracks);
   the warm cache `research/cache/ds1x4x4/<id>.npy` is keyed by stem, so no repathing needed.
2. `batch_size: 16 → 8`. **bs=16 OOMs the dense `44b6` fold** (fold1, `max_nodes≈33`, larger frames):
   ~30 GiB peak on the 32 GiB GPU, crash in `loss.backward()`. bs=8 fits **both** folds (~14 GiB).

## Verification (all GREEN)
| check | result |
| :-- | :-- |
| (c) leak-assert `fleet_agents.cv_contract` on frozen split | **PASS (exit 0)** — fold0 8/71, fold1 8/128, embryo-disjoint |
| (a) dry-run fold 0 (`--epochs 1 --max-iters 2 --single-gpu`) | **GREEN** — `Fold 0: 8 train, 71 test`, epoch+ckpt, cache hit |
| (a) dry-run fold 1 @ bs=16 | **OOM** (dense fold) → refit |
| (a) dry-run fold 1 @ bs=8 | **GREEN** — `Fold 1: 8 train, 128 test`, epoch+ckpt (train 1.7s / test 132.5s) |
| (b) logging | **GREEN** — startup log (`Starting training for N epochs…`, device, params) + per-epoch line (`Epoch e/N \| edge= \| det= \| …`) already present; no patch |
| (d) metric anatomy `fleet_agents.metric_anatomy` | built + smoke-tested; `--verify-pilk` reproduces `official_score 0.8527` |

`acc/recall = 0` in the dry-runs is expected (2 iters) — a wiring test, not a training result.

## Full pipeline (GPU stages = trainer @ :7799; CPU scoring = researcher)
Per fold `N ∈ {0,1}` (weights land at `research/official_repo/weights/loeo_detector/split_N/`):
```
# 1. TRAIN (GPU)
research/cellmot_venv/bin/python scripts/train_from_config.py config/loeo_detector.yml --split N --single-gpu
# 2. PREDICT (GPU) — greedy link is the DEFAULT (no --use-ilp); emits <ds>.geff per test dataset
research/cellmot_venv/bin/python research/official_repo/scripts/predict_unet_transformer.py \
    --splits learning/ensemble_work/finetune/fleet_loeo_mini.json --split N \
    --data-dir input/biohub-cell-tracking-during-development/train \
    --weights research/official_repo/weights/loeo_detector/split_N/edge_predictor_best.pth --single-gpu
# 3. SCORE + 4. DECOMPOSE (CPU) — REQUIRES PYTHONPATH so the fleet_agents package __init__ resolves `researchpapers`:
export PYTHONPATH=$PWD/tools/researchpapers:$PWD
research/cellmot_venv/bin/python -m fleet_agents.official_scorer --split learning/ensemble_work/finetune/fleet_loeo_mini.json --fold N --pred-dir <pred_out>
research/cellmot_venv/bin/python -m fleet_agents.metric_anatomy  --split learning/ensemble_work/finetune/fleet_loeo_mini.json --fold N --pred-dir <pred_out>
```

## Metric anatomy (item d) — where the baseline loses
`fleet_agents/metric_anatomy.py` reuses the official loaders + `estN`, so the headline is identical to
the scorer. Buckets, micro-aggregated over a fold's test datasets:
`R_node` (detection recall) · `R_edge` (edge recall) · `Q_link = R_edge/R_node²` (link quality given
detection) · `edge_P` · `count_ratio = t_pred/estN` + realized penalty · `div_J`. Relation
`R_edge ≈ R_node²·Q_link` separates detection failure from linking failure. Smoke (golden-12 pilk,
SECONDARY): R_node **0.99** (detection saturated), R_edge 0.918, Q_link 0.938, count_ratio **1.24**
(over-prediction), div_J 0 — consistent with Thread-1 (loss is linking + over-prediction, not detection).
Run it on the LOEO baseline preds to read the PRIMARY anatomy.
