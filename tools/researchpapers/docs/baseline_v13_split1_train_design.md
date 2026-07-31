# baseline_v13 split_1 training package — design (PARKED, dry-run GREEN)

**Date:** 2026-07-07 · **Author:** researcher · **Status:** queue-ready + GPU-free dry-run GREEN.
Do NOT launch until leader greenlights (gated behind fold0 mtl10 passing its gate).

## Purpose
split_1 = the MISSING fold for a true 2-fold embryo-disjoint LOEO. Train pilk on 44b6 (fold1.train, 71
datasets) → predict 6bba (fold1.test, 128, held-out). Pairs with the existing split_0 (train 6bba /
test 44b6) to complete the 2-fold. Confirms the locked mtl10/gap5.5 win on the fold we have NO honest
weights for yet.

## REPRODUCIBILITY VERDICT: reproducible, NO BLOCKER
(Full infra map done.) All ingredients on disk:
- **Trainer (single JOINT run):** `research/official_repo/scripts/train_unet_transformer.py` trains the
  U-Net detector + transformer edge predictor jointly → one `edge_predictor_best.pth`. (The separate
  full-frame center detector is optional/absent, not needed.)
- **Config interface:** `start_train.sh <cfg>` → `scripts/train_from_config.py` (YAML → env + CLI).
- **Corrected recipe documented:** cloned from `config/aug_ablation/loeo_aug_uncap60.yml` — the fix for the
  `loeo_129ep` overfit (no-aug + max_iters=150 cap → val 0.921 / cross-embryo 44b6 0.61). Fixes: aug ON
  (crop_scale density-up + flip_xy), `max_iters` OMITTED → uncapped, ~60ep, ds 1,4,4 (pilk canonical per
  support-pack; NOT official_repo's 1,2,2 variant).
- **Split + data:** `fleet_loeo_mini.json` fold1 = train 44b6(71)/test 6bba(128) CONFIRMED; all 71 44b6
  `.zarr`+`.geff` present.
- **Only gap:** no pre-authored split_1 config at the corrected recipe (existing `loeo_detector/split_1`,
  `loeo_detector_aug_f1/split_1` are 12-epoch quick-signal, undertrained). Closed by the config below.

## Package (all under `tools/researchpapers/baseline/`)
- `experiments_v13/v13_split1_train.yml` — corrected-recipe config, `split: 1`.
- `dryrun_split1_train.py` — GPU-free validator (config/paths/split/data/aug/uncapped + reconstructs the
  exact trainer + predict CLI). **GREEN.**
- `run_train_split1_v13.sh` — parked runner: train → predict 6bba → score fold1 with locked mtl10/gap5.5
  (via `run_experiments_v13_loeo_confirm.py --folds 1`).

## Launch (ONLY on greenlight)
`bash tools/researchpapers/baseline/run_train_split1_v13.sh`  (weights →
`research/official_repo/weights/v13_split1_uncap60/split_1/`). GPU quick-check: append
`--epochs 1 --max-iters 1 --single-gpu` to the printed train cmd.

## Honest caveats for the leader
- The "correct" recipe is a documented JUDGMENT CALL (aug + uncapped, ~50–60ep), not a canonical epoch
  count — pilk's public checkpoints were 106/129/159ep. Cross-embryo transfer quality is the real unknown;
  a fresh split_1 may not match split_0's cross-embryo strength on the first recipe.
- This is a real GPU TRAINING run (~60 uncapped epochs), materially costlier than the fold0 inference-only
  job — hence gated behind fold0 confirming mtl10 first.
- Synergy (in `baseline_v13_gpu_job_spec.md`): the fold0 job's 44b6 preds (26 GT divisions) double as
  div-head fork-classifier training data — no extra inference.
