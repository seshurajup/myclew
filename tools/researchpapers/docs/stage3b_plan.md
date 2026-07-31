# Stage 3b: Full LOEO Validation (baseline_v13)

## Overview

**Pivot from Stage 3a (v12 golden-12 sweep) to Stage 3b (v13 full LOEO validation).**

**Rationale:** Golden-12 (12 datasets) insufficient for statistical delta gate. Full LOEO (199) is real validation scale for Stage 3 lock-in.

---

## Stage 3b Execution Plan

### Phase 1: LOEO Predictions Generation

**Objective:** Complete pilkwang split_0 predictions for all 199 LOEO datasets.

**Current state:**
- Fold 0 (44b6, 71 datasets): Cached in loeo_moreep_relink/split_0 → already linked to output/stage2/loeo_predictions/
- Fold 1 (6bba, 128 datasets): NOT GENERATED — need GPU inference

**Actions:**
1. Verify cached fold0 is complete (71 predictions)
2. Generate fold1 predictions using pilkwang frozen model
   - Path: output/stage3/fold1_predictions/ (new)
   - Method: GPU inference (pilkwang split_0)
   - ETA: 2-4h GPU runtime
3. Merge fold0 + fold1 → output/stage3/loeo_predictions_full/

### Phase 2: baseline_v13 Experiments

**Objective:** Apply two variants to full LOEO, score across both folds.

#### Experiment Config

| Exp | Name | Focus | Params |
|-----|------|-------|--------|
| v13-1 | baseline | Control (no transformation) | — |
| v13-2 | combined_opt | Best-guess optimization | distance=7.5um, velocity_pred=true |

#### Config Structure

**baseline/experiments_v13/v13_1_baseline.yaml:**
```yaml
name: stage3_v13_1_baseline
description: Control baseline (Stage 1 pilkwang frozen detections, no post-proc transformation)
input:
  predictions_dir: output/stage3/loeo_predictions_full
postproc:
  edge_length_max_um: 12.0
  gap_recovery_enabled: false  # No transformation
output:
  predictions_dir: output/stage3/v13_1_baseline
evaluation:
  split_file: learning/ensemble_work/finetune/fleet_loeo_mini.json
  eval_set: full_loeo
  use_official_scorer: true
  use_metric_anatomy: true
```

**baseline/experiments_v13/v13_2_combined_opt.yaml:**
```yaml
name: stage3_v13_2_combined_opt
description: Combined optimization (7.5um distance + velocity prediction)
input:
  predictions_dir: output/stage3/loeo_predictions_full
postproc:
  edge_length_max_um: 12.0
  gap_recovery_enabled: true
  gap_recovery_max_distance_um: 7.5
  gap_recovery_max_gap_frames: 1
  gap_recovery_velocity_weight: 0.8
  gap_recovery_velocity_prediction: true
output:
  predictions_dir: output/stage3/v13_2_combined_opt
evaluation:
  split_file: learning/ensemble_work/finetune/fleet_loeo_mini.json
  eval_set: full_loeo
  use_official_scorer: true
  use_metric_anatomy: true
```

### Phase 3: Scoring & Gate

**Objective:** Validate v13 variants on full LOEO (199 datasets, 2-fold CV).

**Scoring Process:**
```
For each v13 variant:
  1. Apply transformation to output/stage3/loeo_predictions_full/ (if enabled)
  2. Score on fold0 (71) via official_scorer
  3. Score on fold1 (128) via official_scorer
  4. Compute average (fold0 + fold1) / 2
  5. Compute delta vs 0.8527 baseline
  6. Log metrics → output/stage3/v13_results.json
```

**Gate Decision:**
- ✅ **PASS:** If best variant delta > +0.005 on full LOEO → Lock in Stage 3b results, report for final submission
- ❌ **FAIL:** If all deltas ≤ +0.005 → Close Stage 3, evaluate alternative levers (new architecture, e.g., learned linker, division classifier)

---

## Timeline & Dependencies

| Phase | Task | ETA | Blocker |
|-------|------|-----|---------|
| Phase 1a | Verify fold0 cached (71) | 5min | None |
| Phase 1b | Generate fold1 (128 GPU) | 2-4h | GPU availability |
| Phase 1c | Merge fold0+fold1 full set | 5min | Phase 1b complete |
| Phase 2 | Create v13 configs | 15min | None |
| Phase 3 | Score v13 on full LOEO | 30min | Phase 1c + Phase 2 |
| Gate | Report + decision | 5min | Phase 3 |

**Total Stage 3b ETA:** 2.5-4.5 hours (dominated by Phase 1b GPU inference)  
**Expected completion:** ~22:00-23:00 UTC

---

## Infrastructure

**GPU:** RTX 5090 (32GB VRAM, 11.8GB free available)  
**CPU:** Sufficient for scoring pipeline  
**Storage:** output/stage3/ (predictions + results)

---

## Success Criteria

### V13-1 (baseline control)
- Reproduce Stage 1 baseline (0.8527±0.01) on full LOEO
- Confirms data consistency across fold0+fold1

### V13-2 (combined_opt)
- If delta > +0.005: Credible improvement, lock in for submission
- If delta ≤ +0.005: No meaningful gain, close Stage 3

---

## Next Steps (Researcher)

1. ✓ Skip v12 (golden-12 insufficient for gate)
2. Create baseline_v13/ configs (2 experiments)
3. Verify fold0 cached completeness
4. Start fold1 generation (GPU inference)
5. Create run_experiments_v13.sh runner
6. Execute v13 scoring on full LOEO
7. Report results + gate decision

---

*GRANDMASTER JOURNEY: Stage 1 → Stage 2 (design) → Stage 3 (v13 full LOEO validation) → Stage 4+ (TBD post-decision)*
