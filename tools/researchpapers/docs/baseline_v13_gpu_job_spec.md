# baseline_v13 — DEFERRED GPU job spec (do NOT run until greenlit)

**Date:** 2026-07-07 · **Author:** researcher · **Status:** SPEC ONLY — gated behind the cached
`min_track_len × gap` + `consensus_prune` verdict. Leader hands to trainer on greenlight.

Two jobs that CANNOT run on the cached golden-12 preds (proven) and need GPU. Both are single-change
A/Bs vs the pilk base, scored on the **frozen fold0/1 LOEO** (`learning/ensemble_work/finetune/fleet_loeo_mini.json`;
fold0 test=71 `44b6_*`, fold1 test=128 `6bba_*`) — NOT golden-12.

Shared assets:
- Detector + edge-predictor weights: `research/pilkwang_support_pack/weights/unet_transformer/split_0/{checkpoint_last.pth, edge_predictor_best.pth}` (v2 copy under `learning/public_pull/data/pilkwang_support_pack_v2/weights/...`).
- Predict entry: `research/official_repo/scripts/predict_unet_transformer.py --weights <edge_predictor_best.pth> --det-threshold <t> --pool-kernel-um <k> --splits <split> --split <fold> --data-dir input/biohub-cell-tracking-during-development/train`.
- Score: `fleet_agents.official_scorer` + `fleet_agents.metric_anatomy` (full metric = adj_edge_jaccard + 0.1·division_jaccard). Run from repo root with `PYTHONPATH=tools/researchpapers:.`.

---

## JOB 1 — EXP-A: det_threshold re-detection sweep (GPU, inference-only)
**Why GPU:** `det_threshold` gates prediction-time peak detection; it is a NO-OP on cached geffs
(nodes already fixed — det 0.988/0.99/0.992 → identical 0.8315). Must re-run the U-Net + peak detect.
- **ONE change vs pilk base:** `--det-threshold` swept `{0.99, 0.995, 0.997}` (raising it trims nodes →
  relieves the over-prediction penalty `min(1, estN/predN)`); optionally × `--pool-kernel-um {5.0,6.0}`.
- **Recipe applied after predict:** the shipped pilk chain (`pilk_post.filter_output_graph`) → the
  best `min_track_len` from the cached sweep. (Scaffold exists: `baseline/postproc/det_grid_sweep.sh`
  — but it points at OVERFIT `loeo_129ep` weights; RE-POINT `W` to the pilk `edge_predictor_best.pth` above.)
- **Score:** fold0 + fold1 LOEO, full metric. Report Δ vs the cached-sweep winner.
- **Honest expectation:** the public 0.890 log raises det to 0.997 for ~+0.006 BEFORE postproc, but
  boristown postproc then makes det ~neutral (they settle at 0.99). Likely small; run only if the cached
  ceiling stalls below ~0.874.

---

## JOB 2 — Stage-6 division head: learned FORK-CLASSIFIER (GPU inference + tiny CPU train)
**Frame:** demote FALSE forks (over-split ILP divisions), NOT add-daughter. Grounded:
add-daughter AUCPR **0.000** (DEAD — 1.04M candidates, 40 positives, density-honest); fork-prune
AUCPR **0.163** (REAL-but-weak, only 7 train positive forks, data-starved). Competitor fork intel:
`yusuketogashi/lb893-baseline`+`khj1222/biohub-yusuke-lb893-fork`, `beicicc/biohub-exp036-vmerckle-altunet-divseed`.

- **(a) div-rich TRAIN data source:** `tools/researchpapers/eda/thread2/division_rich_minisplit.json`
  (12 datasets, **36 GT divisions** — vs golden-12's ~8). GT geffs present in
  `input/biohub-cell-tracking-during-development/train/<ds>.geff`, but **pilk predictions are NOT cached**
  for these → **GPU inference required** to produce predicted-node fork candidates. Feature/label builder
  already exists: `experiments/divisions/fork_pruning_classifier.py` (metric-visible forks only; features
  `persist_min/persist_max/persist_ratio/local_density/frame_gap`; label=1 iff a GT-tracked division).
  To lift AUCPR past 0.163, expand the training fork pool by bulk-predicting more of the 199 train
  datasets (more true forks — the only real lever) — still GPU inference.
  - **★ SYNERGY — piggyback on the v13 fold0 preds (NO extra GPU):** the fold0 LOEO confirmation job
    already generates native pilk predictions for **71 datasets of the 44b6 embryo, which carries 26 GT
    divisions**. Those preds double as div-rich fork-classifier TRAINING data — run
    `fork_pruning_classifier.py` feature-mining directly on the fold0 pred dir once it lands, no separate
    div-rich inference pass needed. This collapses JOB 2's costliest prerequisite (div-rich GPU inference)
    into work already done for the mtl10 confirmation. Mine features the moment fold0 preds are on disk.
- **(b) SCORE:** full metric **including the division term** on frozen fold0/1 LOEO. Golden-12 is
  division-BLIND (~8 GT divs) and **cannot gate this** — must use the LOEO folds. Report div_J (div_tp/fp/fn
  via `metric_anatomy`) AND adj_edge (fork demotion must not regress linking).
- **(c) ONE change vs pilk base:** after ILP+postproc, apply the fork-classifier to each metric-visible
  fork; at precision≥0.9 drop the weaker daughter edge (demote fork → single successor). Nothing else changes.
- **(d) HONEST CEILING — go in eyes-open:** realistic div_J ~0.05→0.10 ⇒ **~+0.005 official, NOT +0.1**
  (the +0.1 is the metric WEIGHT; the achievable jaccard gain is GT-sparsity-bound: only ~14–36 GT divisions
  on eval). TP/FP overlap means we likely can't keep all TP forks while cutting FP. **This is a +0.005
  bet on a data-starved head — worth GPU only if the cached ceiling is exhausted and nothing cheaper remains.**

---

## SEQUENCING
1. Cached `min_track_len × gap` (fleet combo_search, patched) + `consensus_prune` complement — THIS HOUR, CPU.
2. Leader reads the cached verdict → greenlights JOB 1 and/or JOB 2 only if the cached ceiling justifies GPU.
3. Trainer launches the greenlit job(s) on the frozen fold0/1 LOEO; researcher scores full metric + reports Δ.
