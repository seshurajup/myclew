# yusuketogashi "clean, no-metric-hack" 0.909 recipe — LOCAL held-out reproduction

Reproduced the public notebook `yusuketogashi/biohub-clean-approach-no-metric-hacking`
(hidden-test LB = 0.909) locally on 12 held-out training datasets, using the exact tuned
recipe env and the pilkwang support-pack `split_0` edge_predictor weights
(md5 `fd822d8723cb3d1fa3139751308fa39e`).

Pipeline: `research/kernels/yaroslav_v4_ilp/pipeline.py` (the advanced inline ILP pipeline
that reads the `BIOHUB_*` recipe flags) → `submission.csv` → `csv_to_geffs.py` →
patched official metric. Post-processing (gap-close, safe-div, motion-relink, short-track
filter) IS applied — it lives in the pipeline's `filter_output_graph` stage that writes the
submission, not in the raw predicted `.geff`.

## 1. Per-dataset (patched official metric, 7 µm match, score = edge_J + 0.1·div_J)

| dataset | embryo | split | edge_J | div_J | score |
|---|---|---|---|---|---|
| 44b6_0113de3b | 44b6 | held-out | 0.9216 | 0.0 | 0.9216 |
| 44b6_0b24845f | 44b6 | held-out | 0.9038 | 0.0 | 0.9038 |
| 44b6_0c582fdc | 44b6 | held-out | 0.9718 | 0.0 | 0.9718 |
| 44b6_0db75fae | 44b6 | held-out | 0.9608 | 0.0 | 0.9608 |
| 44b6_12dfb391 | 44b6 | held-out | 0.8933 | 0.0 | 0.8933 |
| 44b6_144b256d | 44b6 | held-out | 1.0000 | 0.0 | 1.0000 |
| 6bba_05b6850b | 6bba | in-domain | 0.9789 | 0.0 | 0.9789 |
| 6bba_05db0fb1 | 6bba | in-domain | 0.8385 | 0.0 | 0.8385 |
| 6bba_062c8d37 | 6bba | in-domain | 0.9844 | 0.0 | 0.9844 |
| 6bba_07477033 | 6bba | in-domain | 0.9593 | 0.0 | 0.9593 |
| 6bba_07e24132 | 6bba | in-domain | 0.8849 | 0.0 | 0.8849 |
| 6bba_085bf656 | 6bba | in-domain | 0.9957 | 0.0 | 0.9957 |

## 2. Per-embryo micro (aggregated TP/FP/FN over the 12)

| group | edge_J | div_J | score | div TP / FP / FN |
|---|---|---|---|---|
| 44b6 (held-out, 6 ds) | 0.9174 | 0.0 | 0.9174 | 0 / 9 / 1 |
| 6bba (in-domain, 6 ds) | 0.9396 | 0.0 | 0.9396 | 0 / 8 / 7 |
| all 12 (micro) | 0.9352 | 0.0 | 0.9352 | 0 / 17 / 8 |

Note: this run was bounded to these 12 datasets only (no full-71 / full-199 run, per the task
constraint), so the full held-out 44b6 number was intentionally not computed.

## 3. Honest interpretation

The local held-out micro score on the 12 picks is **0.9352** (44b6 held-out = **0.9174**,
6bba in-domain = **0.9396**). This is **not** the 0.909 hidden-test LB and cannot be made
identical to it: 0.909 is scored on the organizers' hidden test (both embryos, ~full training
size) and is not locally recomputable. Our numbers are on a different 12-dataset held-out
slice of the training set, and they run higher partly because these particular datasets are
individually strong and because division contributes nothing here (see below).

The tuned recipe **did improve** over the base pilkwang `split_0` baseline on 44b6. Baseline
(det=default, no tuned post-proc) scored 0.882 on held-out 44b6 (edge_J 0.874, div_J 0.077);
the tuned recipe lifts held-out 44b6 to edge_J **0.9174** (score 0.9174) — roughly **+0.03–0.04**,
driven entirely by edge_J. The caveat: the 0.882 baseline was over the broader held-out 44b6
set while this run is 6 datasets, so it is directionally — not byte-for-byte — comparable.

Division is effectively dead on these picks: **div_J = 0.0 everywhere, 0 true-division TP**
(44b6: 1 GT division, missed; 6bba: 7 GT divisions, all missed, plus false positives). The
recipe's safe-division machinery produced no correct divisions on this sparse founder-lineage
GT, so 100% of the score comes from edge_jaccard. This matches the known "division lever
exhausted" finding under the patched metric.

## 4. Commands + anti-hack gate

Predict + tuned post-proc (recipe env in `run_pipeline.sh`, GPU RTX 5090, OMP_NUM_THREADS=1):

```
BIOHUB_TEST_DIR=<12-zarr dir> BIOHUB_MODEL_ARTIFACTS=research/pilkwang_support_pack \
BIOHUB_ALLOW_ARTIFACT_FALLBACK=1 BIOHUB_DET_THRESHOLD=0.9690 \
BIOHUB_OUTPUT_FILTER_SHORT_TRACKS=1 BIOHUB_MOTION_RELINK_LEARNED_BONUS=1.0 \
BIOHUB_ILP_APPEARANCE_WEIGHT=0.0 BIOHUB_ILP_DISAPPEARANCE_WEIGHT=1.5 \
BIOHUB_GAP_CLOSE_MAX_GAP=2 BIOHUB_GAP_CLOSE_UM=5.8 BIOHUB_GAP_DENSITY_ADAPTIVE=1 \
BIOHUB_GAP_DENSITY_REFERENCE_UM=6.5 BIOHUB_GAP_DENSITY_GAIN=0.040 \
BIOHUB_GAP_DENSITY_MAX_STEP_DELTA_UM=0.125 BIOHUB_GAP_DENSITY_NEIGHBORS=3 \
BIOHUB_OUTPUT_MIN_TRACK_LEN=6 BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS=1 \
BIOHUB_OUTPUT_GAP2_RECOVERY=0 BIOHUB_SAFE_DIV_MAX_UM=4.66 BIOHUB_SAFE_DIV_SISTER_MAX_UM=8.5 \
BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM=7.65 BIOHUB_SAFE_DIV_FRAME_FRAC_CAP=0.0076 \
BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP=0.00375 BIOHUB_USE_DEEPCENTER_VETO=0 \
BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE=0 \
  research/cellmot_venv/bin/python pipeline_clean.py     # → submission.csv
```

Convert + score:

```
research/official_repo/.venv/bin/python research/official_repo/scripts/csv_to_geffs.py \
  --csv submission.csv --out-dir pred_geff

env CELLMOT_PRED_DIR=pred_geff \
    CELLMOT_DATASETS=44b6_0113de3b,44b6_0b24845f,44b6_0c582fdc,44b6_0db75fae,44b6_12dfb391,44b6_144b256d,6bba_05b6850b,6bba_05db0fb1,6bba_062c8d37,6bba_07477033,6bba_07e24132,6bba_085bf656 \
    CELLMOT_OS_OUT=score_out \
    research/official_repo/.venv/bin/python research/official_score/score_preds.py
```

**Anti-hack gate — PASS (clean).** Checked all 303,781 predicted nodes: t range [0, 99]
(no t<0), z [0,63], y [0,253], x [0,255] — all within the image volume; 0 negative coords,
0 off-volume nodes, no off-volume hub / fake-fork. Genuine tracking output only.

Pipeline adaptations from the Kaggle version (paths/robustness only, no scoring change):
stripped the notebook cover-image cell so `from __future__` is first; `copytree(symlinks=True,
ignore_dangling_symlinks=True)` to tolerate a broken symlink in the support-pack repo; wiped
the stale `predictions/` the support-pack ships so only the fresh 12 are scored. Predict ran
in 2.75 min on the RTX 5090.
