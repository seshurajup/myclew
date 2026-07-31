# Stage 2 Design: Post-Processing Variants (baseline_v11)

## Goal

Close 0.018 gap: **0.8527 (golden-12 pilot) → 0.8708 (pilkwang reference)** via post-processing variants.

**Context:** Stage 1 froze pilkwang FIXED detections (R_node=0.9896, no detector work). Thread-2 closed learned-linker path (point-substrate mismatch, L10). Stage 2 pivots to RoI-aware post-processing that respects frozen detection geometry.

---

## Experiments (baseline_v11)

### Dataset Coverage
- **Evaluation Set:** Full LOEO fold CV (199 datasets: fold0 71 44b6 test, fold1 128 6bba test)
- **Baseline Predictions:** pilkwang frozen unet_transformer (split_0) on all 199 datasets
- **Ground Truth:** official_scorer + metric_anatomy decomposition

### Experiment Design

Each variant targets a specific failure mode identified in Stage 1:

#### exp#1: Edge-Length Threshold + Gap-Recovery
**Target:** Over-prediction penalty (count_ratio=1.2370, 23.7% over-predict).

Post-proc strategy:
- Threshold edges by length: discard edges > L_max (calibrate via golden-12)
- Gap-recovery: re-link broken tracks within local neighborhood (motion-aware)
- Recalibrate count_ratio toward 1.0 without sacrificing R_edge

**Config:** `baseline/experiments_v11/exp1_edge_threshold_gap_recovery.yaml`
**Expected delta:** +0.002 to +0.005 (count penalty relief)

#### exp#2: Centroid Refinement + Local Smoothing
**Target:** Linking precision (edge_P=0.9408, FP edges still present).

Post-proc strategy:
- Refine node centroids via cluster statistics (sub-voxel precision if data permits)
- Smooth trajectories via motion model (velocity consistency check)
- Remove kinematically impossible edges (accel bounds)

**Config:** `baseline/experiments_v11/exp2_centroid_refine_smooth.yaml`
**Expected delta:** +0.003 to +0.008 (edge precision + linking stability)

#### exp#3: Division-Aware Edge Filtering
**Target:** Division consistency + false-positive suppression.

Post-proc strategy:
- Identify division events (parent → 2 daughters)
- Filter edges that violate division geometry (e.g., parent-to-non-daughter)
- Suppress FP edges near division nodes

**Config:** `baseline/experiments_v11/exp3_division_filter.yaml`
**Expected delta:** +0.001 to +0.003 (edge precision, low weight given div_J=0.0 on golden-12)

---

## Implementation

### Phase 1: Full LOEO Predictions
**Owner:** researcher (this task)

1. Load pilkwang frozen unet_transformer weights
2. Generate predictions for all 199 LOEO datasets (fold0 + fold1 test)
3. Save as final-format .geff in `output/stage2/loeo_predictions/`
4. Validate schema + row counts via dry-run

**Expected time:** ~2-4 hours (GPU inference, parallel if possible)
**Output:** `output/stage2/loeo_predictions/{dataset}.geff` (199 files)

### Phase 2: Post-Proc Variant Design & Validation
**Owner:** trainer (after researcher hands off)

1. For each experiment (exp#1-3):
   - Implement post-proc logic in Python (reusable module)
   - Validate on golden-12 (smoke-test)
   - Run on full LOEO fold CV
   - Score via official_scorer + metric_anatomy
   - Log metrics to MLflow
   
2. Submission to train_service (:7799):
   - One formal runner script: `baseline/run_experiments_v11.sh`
   - Calls `python baseline/run_baseline.py --config experiments_v11/exp{1,2,3}.yaml` for each variant
   - Parallel or sequential (depends on GPU availability)

### Phase 3: Results Synthesis
**Owner:** trainer (after all runs complete)

1. Summarize Stage 2 results in `docs/stage2_results.md`
2. Compare all variants against golden-12 baseline (0.8527)
3. Pick winning variant (largest delta toward 0.8708)
4. Generate trend PNG: `docs/stage1_to_stage2_top3_trend.png`

---

## Success Criteria

- **Credibility:** Any variant that closes ≥50% of gap (0.8527 + 0.009 = 0.8617) is credible
- **Minimum delta:** +0.005 (threshold for Stage 3 progression)
- **Full LOEO validation:** metric_anatomy shows consistent R_node/R_edge/Q_link across folds
- **No regressions:** No variant undershoots golden-12 (0.8527)

---

## Risk & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Full LOEO GEFFS gen fails | Cannot validate Stage 2 | Pre-validate pilkwang weights; dry-run on 1-2 datasets |
| Post-proc introduces bugs | Metric regression | Unit tests on golden-12 first; debug before full run |
| GPU OOM on full LOEO | Long pipeline time | Batch predictions (fold0 + fold1 separately if needed) |
| Post-proc delta < +0.005 | Insufficient for Stage 3 | Iterate on variant design (exp#4-5 if budget allows) |

---

## Next Gate

✅ **Stage 1 complete:** pilkwang baseline 0.8527 (golden-12), anatomy locked, L11 lesson documented.

**Handoff to Stage 2:**
1. Researcher: Generate full LOEO predictions (199 GEFFS)
2. Researcher: Design doc (this file)
3. Trainer: Implement + test post-proc variants (exp#1-3)
4. Trainer: Submit to :7799 for full LOEO validation

**Timeline:** Stage 2 eval expected 4-6 hours (GPU + post-proc turnaround).

---

*GRANDMASTER JOURNEY: Stage 1 → Stage 2 (post-proc pivot). Stages 3-9 TBD post-validation.*
