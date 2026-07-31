# aug_journey — clean isolated augmentation ablation

**Goal (north star):** close the LOEO 44b6 collapse (adjJ_44b6≈0.02) toward 6bba (≈0.44) WITHOUT crashing
6bba, by finding which augmentation(s) help. Leader owns the ordered plan; researcher authors + runs in order.

**Why re-authored from scratch:** the prior `config/aug_ablation/*.yml` were CONFOUNDED — `base.yml` baked in
brightness+flip and every other config inherited it, so no single-aug effect was isolable. The `contrast.yml`
run that the fleet auto-submitted is likewise confounded (see `docs/aug_journey_contrast_CONFOUNDED_superseded.md`)
and must NOT be journaled as a clean row.

## Design (one change per config)
Every config is **identical** except the one `augment:` entry. Substrate = the human's stage-matched screen
`learning/ensemble_work/finetune/splits_screen_matched.json` (fold 0; both embryos 12/12 train, 6/6 test),
`bs8`, `epochs 10`, `max_iters 150`, `seed 1234`. **Judge BY EMBRYO** via `baseline/score_v1.py --split-file
splits_screen_matched.json --fold 0` → `adjJ_44b6` (money) and `adjJ_6bba` (guardrail) SEPARATELY, never the
blend. Close calls: re-judge on `splits_screen_matched_k6.json` (6-fold) for a robust read.

## Manifest & order
| order | config | augment (isolated) | class |
| :-- | :-- | :-- | :-- |
| Phase 0 | `00_no_aug.yml` | `[]` (TRUE no-aug reference) | reference |
| Phase 1a | `10_crop_scale.yml` | crop_scale `s∈[0.55,1.8]` | DENSITY (run first) |
| Phase 1a | `11_translate_static.yml` | translate_static `≤20% FOV` | DENSITY |
| Phase 1b | `20_flip_xy.yml`, `21_rot90_yx.yml` | flip / rot90 | geometric |
| Phase 1c | `30..35` | brightness/contrast/gamma/bias_field/blur/noise | photometric |
| Phase 2 | `mix_*.yml` | greedy forward-select from best single | on-the-fly |

## Grounding (scale ranges) + citation
6bba↔44b6 median density gap = **3.37×** (44b6 327 vs 6bba 97 cells/frame; `e57_timeline_eda.md` /
`learning/03_true_density_stage.csv`; augfinder `crop_scale=PASS`, "embryos span 3.5× density").
- **crop_scale** = isometric in-plane zoom `s∈[0.55,1.8]` = ±~3.3× density BOTH ways (`1/√3.37=0.545 … √3.37=1.84`).
  In-plane (Y,X) only; Z (1.625µm) untouched → anisotropy preserved. `s<1` shrinks→denser, `s>1` enlarges→sparser.
- **translate_static** = static in-plane shift ≤20% FOV, SAME across the window (inter-frame motion preserved).
- **Cite:** Cellpose (Stringer et al., Nature Methods 2021 — diameter/scale-normalization) + StarDist3D
  (Weigert et al., WACV 2020 — anisotropic voxels). StarDist3D substitutes the nnU-Net prior (not in-repo).

## Code added (vendored `research/official_repo/scripts/augmentations.py` + launchers) — CPU shape-validated
- `translate_augment` (new) + registry aliases `crop_scale`/`flip_xy`/`rot90_yx`/`translate_static` matching the
  augfinder physics-menu names; `scale_augment` gained directional `smin`/`smax`.
- `--seed` wired (`train_unet_transformer.py` argparse + `train()` + `train_from_config.py` flag_map) for a
  reproducible A/B (weight init + DataLoader order; note: the in-loop aug RNG is not seeded — second-order noise,
  mitigated by `_k6` for close calls).
- **`train_from_config.py`: `if cfg.get("augment") is not None`** (was truthiness) so `augment: []` is a TRUE
  no-aug reference instead of silently falling back to DEFAULT_AUGMENTATIONS (brightness) — this would have
  corrupted `00_no_aug`.

## Pipeline per config (GPU stages = trainer :7788/:7799; CPU judge = researcher)
`train_from_config.py <cfg> --split 0 --single-gpu` → `predict_unet_transformer.py --splits
splits_screen_matched.json --split 0 --weights weights/<method>/split_0/edge_predictor_best.pth` (greedy link,
default) → `score_v1.py --method <name> --split-file splits_screen_matched.json --fold 0` (adjJ by embryo).
Status: authored + CPU-validated; **GPU dry-run pending** (fleet still occupying GPU with old confounded configs).
NO Kaggle — local CV only.
