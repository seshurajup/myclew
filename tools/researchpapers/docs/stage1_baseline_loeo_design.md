# Stage 1 — Baseline LOEO + Metric Decomposition (Design)

**Status: PENDING LEADER APPROVAL (2026-07-06 16:45 UTC)**

## Goal

Establish a dumb end-to-end baseline on the FROZEN embryo-disjoint leave-one-out (LOEO) fold, using pilkwang's FIXED detections, scored by the official metric, and decomposed into component buckets (R_node, R_edge, Q_link, count_penalty) to seed the GRANDMASTER JOURNEY 9-stage process.

## CV Contract (FROZEN, §1 of `stage0_cv_contract.md`)

- **Split:** `learning/ensemble_work/finetune/fleet_loeo_mini.json`
  - Fold 0: train = 128 embryos (6bba full), test = 71 embryos (44b6 full)
  - Fold 1: train = 71 embryos (44b6 full), test = 128 embryos (6bba full)
  - Embryo-disjoint (matches Kaggle hidden-test axis)

- **Scorer:** `fleet_agents.official_scorer --split <split.json> --fold <N> --pred-dir <geffs>`
  - Official score = adj_edge_jaccard + 0.1 · division_jaccard
  - Gate: `fleet_agents.cv_contract` (leak-assert, exit 0=PASS / 1=FAIL)

## Baseline: pilkwang FIXED detections

- **Model:** pilkwang (public_repo golden-12 weight set, cell-transformer backbone)
- **Inputs:** full `train/` dataset (all 199 embryos)
- **Outputs:** pilkwang geffs for all fold-0 test and fold-1 test datasets
- **Node/Edge:** FIXED (no linker/detector changes; only decomposition)
- **Division signal:** free (pilkwang reports 0 TP / ~30 FP divisions on golden-12; measure on LOEO folds)

## Metrics to extract

Per official scorer output + component decomposition:

1. **adj_edge_jaccard** (primary metric, weight 1.0)
2. **division_jaccard** (weight 0.1, but free from linker)
3. **R_node** = node recall (Hungarian ≤ 7 µm gate)
4. **R_edge** = edge recall (both endpoints must match)
5. **P_edge** = edge precision (linker-dependent)
6. **Q_link** = linker quality (fraction of realizable edges linked)
7. **count_penalty** = 1 − 0.1 · |T_pred − T_true| / T_true (calibration)

## Expected outputs

- `output/stage1_baseline_loeo/fold0/official_score.json` — fold-0 test (44b6 embryos)
- `output/stage1_baseline_loeo/fold1/official_score.json` — fold-1 test (6bba embryos)
- `docs/stage1_baseline_loeo_result.md` — per-embryo table + interpretation
- `docs/stage1_baseline_loeo_decomp.md` — component-metric breakdown (R_node/R_edge/Q_link/count)

## Relationship to Thread-2

Thread-2 (exp#0-3, L5-L10) was a **golden-12-only decision-point audit** on pilkwang's FIXED detections. Stage 1 validates that audit at scale (all 199 embryos, 2-fold LOEO) and decomposes the score into buckets that inform future lever sizing (Stages 2-5: linker, detection, division, augmentation).

## Note on beat_ceiling & orchestrate

beat_ceiling (0.8735 winning_config) is a parallel optimization phase targeting the public LB. Stage 1 is orthogonal — it's the research CV-space foundation. No overlap; both proceed in parallel.

## Next steps

1. ✓ Design locked (this doc)
2. Pending: leader approval to proceed
3. Researcher: run pilkwang inference on both folds (if not cached)
4. Researcher: score with official_scorer
5. Researcher: extract + decompose metrics
6. Researcher: write `result.md` with findings + next-stage recommendations
