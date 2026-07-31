# baseline_v2 — FAIR converged (1,2,2)-vs-(1,4,4) resolution A/B (richaug everywhere)

> Status: **QUEUE-READY** (design + dry-run PASS). Design note. Result note (later): `docs/baseline_v2_1_exp_result.md`.
> Follows `docs/baseline_v1_1_exp_result.md`: v1 showed (1,2,2) LOST at equal budget but from
> UNDERTRAINING (recall 0.98→0.70), and richaug = +0.1833 (proven win). v2 gives BOTH resolutions a
> FAIR CONVERGED budget with richaug folded in everywhere, so the resolution A/B is finally real.

## 1. Goal
Answer the v1 open question honestly: **once properly trained, does the (1,2,2) higher-res detector
beat (1,4,4)?** Both arms use the v1-proven rich augmentation and a long converged budget; the only
difference between v2_1 and v2_2 is downsample → a fair resolution A/B.

## 2. Experiments (2 core + 1 optional; golden-12 fold 0)
| id | config | ds | augs | batch | budget | role |
|---|---|---|---|---|---|---|
| **v2_1** | `experiments_v2/v2_1_ctrl_1x4x4_rich.yaml` | 1,4,4 | richaug | 16 | 1000it×30ep, pat12, cos+wu2 | converged control; "richaug alone beat 0.8708?" |
| **v2_2** | `experiments_v2/v2_2_hr_1x2x2_rich.yaml` | **1,2,2** | richaug | 4 | 1000it×30ep, pat12, cos+wu2 | **THE fair resolution test (v2_2 > v2_1?)** |
| v2_3 (opt) | `experiments_v2/v2_3_hr_1x2x2_rich_conv.yaml` | 1,2,2 | richaug | 6 | 1200it×35ep, pat15, cos+wu4 | convergence insurance (batch6/warmup4) |

- **richaug** (v1-proven +0.1833): brightness + flip(Z-off) + rot90_xy + gamma(p.5) + contrast(p.5).
- **Long budget:** max_iters 300→1000 (v2_3: 1200), patience 8→12 (v2_3: 15), cosine LR + warmup
  (PINNED in-config for reproducibility — v1 got cosine only from the job env).
- **Per-config cache:** ds1x4x4 for (1,4,4), ds1x2x2 for (1,2,2) — the per-config `cache_dir` overrides
  any job-env `CELLMOT_CACHE_DIR` (verified), so caches never cross-contaminate.
- **Batch memory (measured on 32 GB):** (1,2,2) batch 8 **OOMs**, batch 6 fits, batch 4 proven → v2_2=4,
  v2_3=6 (convergence aid). (1,4,4) batch 16 native.

## 3. Measurement (unchanged discipline)
- Official golden-12 (`baseline/predict_and_score_v2.sh <method>` → predict + `score_v1.py`).
- **Primary A/B (golden-CV-judgeable):** v2_2 − v2_1 official + golden_cv at equal converged budget.
- v2_1 vs 0.8708 answers "does richaug alone beat pilkwang" (richaug is density-CHANGING vs his →
  if it clears 0.8708, that's **NEEDS-LB**, human submits — flag, don't conflate with the resolution delta).
- Report official adjJ + micro + golden_cv + node_recall + count_ratio per arm; div_tp (expect 0 still).
- All runs → MLflow (system-metrics + config lineage; `MLFLOW_RUN_ID` stripped so each arm owns its run).

## 4. Package (queue-ready)
- configs: `baseline/experiments_v2/{v2_1_ctrl_1x4x4_rich,v2_2_hr_1x2x2_rich,v2_3_hr_1x2x2_rich_conv}.yaml`
- runner (one canonical, subset-selectable): `baseline/run_experiments_v2.sh [names...] [--dry-run]`
  → `run_baseline.py` → `src.baseline.train` → official trainer.
- scoring: `baseline/predict_and_score_v2.sh <method>` (post-training).
- outputs: `output/baseline_v2/<method>/`; checkpoints `official_repo/weights/<method>/split_0/`.

## 5. Wall-clock ETA (from real v1 timing: (1,2,2) b4=311s/ep, (1,4,4) b16=399s/ep @300it)
Scaled to 1000 iters, early-stop (pat12) likely ~20-25 epochs:
| arm | min/epoch (~1000it) | ETA (early-stop) |
|---|---|---|
| v2_1 (1,4,4 b16) | ~21.6 | **~7-9 h** |
| v2_2 (1,2,2 b4) | ~15.5 | **~5-7 h** |
| v2_3 (1,2,2 b6, 1200it) | ~29 | ~10-14 h |
- **v2_1 + v2_2 (core A/B): ~12-16 h.** All 3: ~22-30 h. **Recommend v2_1+v2_2 first**, v2_3 only if
  v2_2's curve looks under-converged.

## 6. Expected outcome
- v2_1 (1,4,4 richaug, converged) should clearly beat our v1 control 0.8249, plausibly approach/► 0.8708.
- v2_2 vs v2_1 = the real resolution verdict. If v2_2 > v2_1: (1,2,2) lever confirmed → v3 pushes it
  (division-aware training now that localization is finer). If v2_2 ≈/< v2_1 even converged: the finer
  detector doesn't help this pipeline → drop the resolution lever, chase other levers.
- Dry-run: `bash baseline/run_experiments_v2.sh --dry-run` (GPU-safe) — validated.
