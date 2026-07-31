# Insights — rogii-wellbore-geology-prediction

_Auto-generated from the experiment journal + findings (18 experiments, 24 findings)._

## Best experiments (by CV)

| Experiment | CV | Description |
|---|---|---|
| blend_AB | 10.4308 | blend_AB |
| blend_AB_1539 | 10.4308 | Track blend experiment (blend_AB): CV RMSE 10.4308 vs baseline None, goal 9.0 |
| blend_AB_1728 | 10.4308 | Track blend experiment (blend_AB): CV RMSE 10.4308 vs baseline None, goal 6.39 |
| blend_AB_1901 | 10.4353 | Track blend experiment (blend_AB): CV RMSE 10.4353 vs baseline None, goal 6.39 |
| trackB_particle_filter | 10.9103 | trackB_particle_filter |
| trackB_particle_filter_1518 | 10.9103 | Track B experiment (trackB_particle_filter): CV RMSE 10.9103 vs baseline 12.02,  |
| trackB_particle_filter_1901 | 10.9867 | Track B experiment (trackB_particle_filter): CV RMSE 10.9867 vs baseline 12.02,  |
| exp03_trackB_pf_multiscale | 11.1273 | exp03_trackB_pf_multiscale |

## Key findings & decisions

- **[engine]** FIELD-GROUPED CV = 11.928   (PF 11.13 → GBM stack 11.93)
- **[engine]** fold 4 — field RMSE 13.605
- **[engine]** fold 3 — field RMSE 14.736
- **[engine]** fold 2 — field RMSE 12.237
- **[engine]** fold 1 — field RMSE 11.747
- **[engine]** fold 0 — field RMSE 8.878
- **[engine]** GPU switch: XGBoost device=cuda (LightGBM build lacked GPU)
- **[engine]** training LightGBM (GPU) meta-stack — field-grouped 5-fold
- **[cv-builder]** field-grouped folds merged: 3783989 rows, 16 features
- **[engine]** loading honest_feat.parquet + field folds
- **[notebook]** 2×T4 all-GPU 8h max-ensemble harness spec'd; self-calibrating budget controller
- **[uncertainty]** conformal_prediction wired → per-well intervals (Working-Note + guarded-fallback trigger)
- **[blend]** blend_optimize wired for method+seed weights (PF/NCC/DTW/beam/GBM)
- **[research]** honest method locked: affine GR cal + windowed NCC + guarded prefix self-verify + GBM stack
- **[engine]** picking up geology_honest.py — porting Pilkwang affine-cal + multi-scale NCC + selector
- **[cv-builder]** field-grouped CV online — 12 spatial fields, 5 leave-field-out folds
- **[infra]** Submission = 2xT4 all-GPU 8h max-ensemble with self-calibrating budget controller → _Fill 8h with seeds/methods AFTER core CV is good._
- **[validation]** Added field-grouped (leave-field-out) CV as the private proxy → _Adopt a lever only if it improves field-grouped CV by >0.12._
- **[strategy]** North star = PRIVATE gold; public LB is a leakage board (ignore for rank) → _Optimize field-grouped (leave-field-out) CV; two finals field-robust; never public-LB-tune._
- **[analysis]** Naive spatial neighbor transfer is dip-limited (~16-30 ft), NOT a cheap win → _Use offset-well prior only as guarded/dip-corrected feature, not raw donor._
- **[analysis]** Guarded fallback ALONE cannot reach 6 (oracle ceiling 8.9) → _Replace pointwise-GR PF likelihood with affine-cal + multi-scale NCC core._
- **[analysis]** Failure wells are THE lever: PF median 5.80 (Deotte-level) but pooled 11.1 → _Guarded prefix self-verification + better matcher (windowed NCC) instead of conf-gating._
- **[analysis]** Trust-CV comp: LB ≈ CV + 0.30 (yu4u); our const-cont CV=15.91 ≡ community 15.9 → _Average 5 folds x 5 seeds; adopt only gains >0.12; target CV<6 -> LB<6.3._
- **[research]** Honest ceiling is ~6 RMSE, not ~10 — Chris Deotte at 6.122 no-leak → _Adopt Pilkwang affine-cal + multi-scale NCC + selector; NOT DTW/UNet (underperform)._
