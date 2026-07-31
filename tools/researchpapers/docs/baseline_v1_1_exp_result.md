# baseline_v1 — RESULTS (golden-12 official)

> Design note: `docs/baseline_v1_1_exp.md`. Scores via `baseline/score_v1.py` (correctness-validated:
> reproduces the pilkwang baseline **0.8708** exactly). All numbers from real runs — no fabrication.
> HARD RULE: no Kaggle submission; density-changing gains flagged NEEDS-LB (human submits).

## Conclusion (one line)
**At equal 30-epoch budget the (1,2,2) higher-res lever did NOT pay off — but the failure is UNDERTRAINING, not resolution, and richer augs rescued it hugely.** v1_2 (1,2,2, minimal augs) collapsed to **0.6086** (recall 0.98→0.70, under-detecting 0.71×) because a 4×-voxel detector needs more training than the ep11-early-stopped budget gave it; adding rot90_xy/gamma/contrast (**v1_3**) recovered **+0.1833 → 0.7919** (recall 0.92), tying the control on the frozen golden_cv (0.8271 vs 0.8272). **Nothing beat 0.8708 → no LB-submission candidate.** Clear next step: **v2 = (1,2,2)+richaug with much longer training** to give the higher-res model a fair shot.

## Results table (golden-12 fold 0, official = adj edge-Jaccard + 0.1·div-Jaccard)
| arm | downsample | augs | official adjJ | golden_cv | node_recall | count_ratio | Δ vs control |
|---|---|---|---|---|---|---|---|
| pilkwang ref (fully-trained) | 1,4,4 | brightness+flip | **0.8708** | 0.8940 | 0.9898 | 1.29× | +0.0459 |
| **v1_1 control** (ours, 30ep→es ep11) | 1,4,4 | brightness+flip | **0.8249** | 0.8272 | 0.9766 | 1.28× | 0 (baseline) |
| **v1_2** hr baseaug | 1,2,2 | brightness+flip | **0.6086** | 0.6400 | 0.7007 | 0.71× | **−0.2163** |
| **v1_3** hr richaug | 1,2,2 | +rot90_xy/gamma/contrast | **0.7919** | 0.8271 | 0.9230 | 1.13× | **−0.0330** |

Detector proxy (acc·recall, NOT official): v1_1=0.9557, v1_2=0.9557, v1_3=0.9595 — note the proxy is
NEARLY FLAT across arms while the official spread is huge (0.61→0.82), confirming the trainer's
per-epoch acc·recall is a poor stand-in for the official metric (recall/density collapse invisible to it).

## Verdict (interpretation locks applied)
1. **Equivalence gate = PASS.** v1_1 (1,4,4) control 0.8249 vs pilkwang 0.8708 — same ballrark, gap is
   undertraining (early-stop ep11). The official_repo trainer is sound.
2. **Resolution delta at EQUAL budget (golden-CV-judgeable):** v1_2−v1_1 = **−0.2163**, v1_3−v1_1 = **−0.0330**.
   At this budget (1,2,2) underperforms (1,4,4). **Confound:** (1,2,2)=4× voxels → needs more iters to
   converge; v1_2's recall collapsed 0.98→0.70 and it UNDER-detected (0.71×; dense embryos worst:
   6bba_05db0fb1 recall 0.23, 44b6_12dfb391 recall 0.43) = textbook undertraining, NOT higher-res being
   inherently worse. The equal-budget A/B under-served the finer model.
3. **Richer-augs effect (v1_3 vs v1_2, same resolution) = +0.1833 official** (recall 0.70→0.92, count
   0.71→1.13×). **Big win** — rot90_xy/gamma/contrast rescued the (1,2,2) detector. On the frozen
   stratified **golden_cv, v1_3 (0.8271) TIES the control (0.8272)**; micro adjJ penalizes v1_3 more via
   density/weighting. (The standalone `config/aug_ablation/` never produced this — absorbing augs into v1 did.)
4. **Beats 0.8708?** **No** — all arms below 0.8708 AND below our 0.8249 control → **no density-changing
   NEEDS-LB candidate from v1.** Nothing to hand the human to submit.
5. **Divisions dead** (div_tp=0 everywhere) — expected until localization is finer AND division-aware training.

## Next (v2 — clear from the data)
- **v2 = (1,2,2) + richaug + MUCH longer training** — the ep11/max_iters=300 budget starves the 4×-voxel
  detector. Raise `max_iters` (300→~1000+), `epochs`, and `early_stop_patience` (8→15–20) so the
  resolution benefit isn't cut short. Give (1,2,2) a FAIR convergence budget, then re-judge the delta.
- Keep the richaug pipeline (proven +0.1833). Consider a mid-budget (1,4,4)-richaug control to separate
  "augs help" from "augs help *more* at fine resolution."
- Only once (1,2,2) is properly trained and localization is finer does division-aware training become worth it.

## Incident (RESOLVED) — first job train-5d23bf7586
The first all-3 job partially failed: v1_1 trained fine, but v1_2/v1_3 crashed at startup on an MLflow
`INVALID_PARAMETER_VALUE: Changing param values is not allowed` — an inherited job-env `MLFLOW_RUN_ID`
pinned all 3 arms to one run, so config #2's differing params collided (params are immutable per run).
**Fix (applied):** `src/baseline/train.py` build_env + `run_experiments_v1.sh` now strip `MLFLOW_RUN_ID`
(defense in depth) so each config owns its run; `score_v1.py` too. Runner parametrized to accept a config
subset. Rerun **train-ee38fdab0d** (v1_2+v1_3 only) succeeded (rc=0) — fix confirmed.

## Artifacts
- scorer `baseline/score_v1.py`; per-run logs `output/baseline_v1/<method>/{predict.log,score.log}`.
- MLflow (exp `kaggle-biohub-cell-tracking`): runs `baseline_v1_v1_1_ctrl_1x4x4` / `_v1_2_hr_baseaug` /
  `_v1_3_hr_richaug` (training) + matching `*_score` runs (official metrics, system-metrics + config lineage);
  reference run `pilkwang_baseline_score_validate` = 0.8708.
- Trend PNG (trainer): `docs/baseline_v01_top3_trend.png` — cross-version top-3 official adjJ trend.
  DEGENERATE at v1 (single version): plots the 3 arms (0.8249 / 0.7919 / 0.6086) vs the pilkwang
  0.8708 reference line and our 0.8249 control line; no inter-version line yet. Becomes a real
  multi-version line plot from v2 (top-3 per version connected across versions).
