# Stage 3 Design: Post-Proc Refinement (baseline_v12-v13)

## Goal

Refine exp#1 (EdgeThresholdGapRecovery) gap-recovery parameters to maximize delta toward 0.8708 reference.

**Target:** Achieve delta > +0.005 on golden-12 pilot, gate to full LOEO validation.

**Context:** Stage 1 baseline (0.8527) + Stage 2 Phase 2C cached fold0 (71 predictions) ready for refinement.

---

## Stage 3 Structure

### Baseline V12: Gap-Recovery Parameter Sweep
**Focus:** exp#1 EdgeThresholdGapRecovery optimization

#### Experiment Design

| Exp | Name | gap_recovery_max_distance_um | max_gap_frames | velocity_prediction | Description |
|-----|------|-----|-----|-----|-----|
| v12-1 | baseline | 6.0 | 1 | false | Stage 2 baseline (control) |
| v12-2 | distance_5 | 5.0 | 1 | false | Tighter distance threshold |
| v12-3 | distance_7 | 7.0 | 1 | false | Relaxed distance threshold |
| v12-4 | distance_10 | 10.0 | 1 | false | Very relaxed threshold |
| v12-5 | frames_2 | 6.0 | 2 | false | Allow 2-frame gaps |
| v12-6 | velocity_pred | 6.0 | 1 | true | Velocity-aware re-linking |
| v12-7 | combined_opt | 7.5 | 1 | true | Combined distance + velocity |

#### Rationale per Experiment

**v12-2 (distance_5):** Tighter threshold tests if over-aggressive gap-recovery in baseline hurts precision.

**v12-3 (distance_7), v12-4 (distance_10):** Sweep finds sweet spot for distance threshold. Hypothesis: current 6.0um may be too tight for sparse embryos.

**v12-5 (frames_2):** Test if allowing 2-frame gaps (e.g., t→t+2) captures biologically plausible re-links. May help dense regions.

**v12-6 (velocity_pred):** Velocity-aware linking prioritizes re-links consistent with motion. Requires velocity history computation.

**v12-7 (combined_opt):** Best-guess combination if individual sweeps show additive gains.

#### Validation Pipeline (Golden-12 Pilot)

```
For each experiment:
  1. Load golden-12 pilkwang predictions (cached)
  2. Apply exp#1 transformation with variant params
  3. Score via official_scorer + metric_anatomy
  4. Log: official_score, adj_edge_jaccard, R_edge, Q_link, count_ratio
  5. Compute delta vs baseline (0.8527)
```

**Scoring:** Use Phase 2A pipeline (fleet_agents.official_scorer + fleet_agents.metric_anatomy on golden-12 test set).

#### Expected Outcomes

- **Credible deltas:** Any variant with delta > +0.002 on golden-12 is worth attention
- **Target delta:** > +0.005 (high confidence)
- **Top variant:** Promote best v12-X to full LOEO validation (Stage 3 Phase 2)

---

### Baseline V13: Refinement + Validation
**Focus:** Top v12-X variant extended to full LOEO (if delta > +0.005 on golden-12)

#### Prerequisites
- Best v12-X experiment identified from golden-12 sweep
- Delta > +0.005 on golden-12 confirmed
- Leader approval for full LOEO generation

#### Experiments (v13)

| Exp | Name | Scope | Goal |
|-----|------|-------|------|
| v13-1 | fold0_validate | Fold 0 (cached, 71) | Confirm v12-best on full fold0 |
| v13-2 | fold1_generate | Fold 1 (new, 128) | Generate full LOEO predictions for fold1 |
| v13-3 | full_loeo_validate | Full LOEO (199) | Final CV validation on all datasets |

#### Process

1. **v13-1 (fold0_validate):** Apply best v12-X variant to ALL fold0 cached predictions (71), score via official_scorer.
   - Target: Confirm delta on full fold0 matches golden-12 trend
   
2. **v13-2 (fold1_generate):** If v13-1 delta > +0.005, generate full fold1 predictions (128 6bba datasets) using pilkwang frozen model.
   - Infrastructure: Restart Phase 2B GPU inference with corrected environment + splits file
   - Output: 128 fold1 .geff files
   
3. **v13-3 (full_loeo_validate):** Apply best v12-X variant to fold1 predictions, score both folds, compute CV average.
   - Final gate: If avg delta > +0.005 across both folds, Stage 3 complete + Stage 4 unlock

---

## Implementation Details

### Golden-12 Validation Loop (v12)

```python
# Pseudo-code for Stage 3 validation
from baseline.postproc import apply_variant, EdgeThresholdGapRecovery
from fleet_agents.official_scorer import score_datasets, GOLDEN12
from fleet_agents.metric_anatomy import anatomy

baseline_score = 0.8527
results = []

for params in [distance_5, distance_7, distance_10, frames_2, velocity_pred, combined_opt]:
    config = {"postproc": params}
    
    # Apply variant to golden-12
    nodes, edges = io.read_geff(pilk_cache / "44b6_0113de3b.geff")
    transform = EdgeThresholdGapRecovery(params)
    nodes_t, edges_t = transform(nodes, edges)
    # ... repeat for all 12 golden-12 datasets
    
    # Score
    agg_score, _ = score_datasets(GOLDEN12, output_dir)
    agg_anatomy, _ = anatomy(GOLDEN12, output_dir)
    
    delta = agg_score["score"] - baseline_score
    results.append({
        "params": params,
        "score": agg_score["score"],
        "delta": delta,
        "anatomy": agg_anatomy,
    })

# Find best
best = max(results, key=lambda r: r["delta"])
print(f"Best: {best['params']} (delta {best['delta']:+.4f})")
```

### Config Structure (baseline_v12)

Each v12-X experiment gets a YAML config:

```yaml
# baseline/experiments_v12/v12_2_distance_5.yaml
name: stage3_v12_2_distance_5
description: Gap-recovery distance sweep (5um threshold)

input:
  predictions_dir: output/stage2/loeo_predictions

postproc:
  edge_length_max_um: 12.0
  gap_recovery_enabled: true
  gap_recovery_max_distance_um: 5.0      # <- sweep parameter
  gap_recovery_max_gap_frames: 1
  gap_recovery_velocity_weight: 0.5

output:
  predictions_dir: output/stage3/v12_2_distance_5

evaluation:
  split_file: learning/ensemble_work/finetune/fleet_loeo_mini.json
  eval_set: golden_12  # <- golden-12 pilot
  use_official_scorer: true
  use_metric_anatomy: true

tracking:
  mlflow_enabled: true
  run_name: stage3_v12_2_distance_5_golden12
```

### File Paths

```
docs/stage3_design.md                          <- this file
baseline/experiments_v12/
  ├── v12_1_baseline.yaml
  ├── v12_2_distance_5.yaml
  ├── v12_3_distance_7.yaml
  ├── v12_4_distance_10.yaml
  ├── v12_5_frames_2.yaml
  ├── v12_6_velocity_pred.yaml
  └── v12_7_combined_opt.yaml

baseline/run_experiments_v12.sh               <- runner for v12 sweep

output/stage3/
  ├── v12_1_baseline/
  ├── v12_2_distance_5/
  ├── ... (one per variant)
  └── stage3_sweep_summary.json               <- results table

docs/stage3_results.md                        <- results doc (gate decision)
```

---

## Success Criteria

### Phase 1 (v12 Golden-12 Sweep)
✅ **PASS:** Best v12-X shows delta > +0.005 on golden-12 → Proceed to Phase 2 (full LOEO)  
❌ **FAIL:** All deltas < +0.005 on golden-12 → Stop refinement, evaluate alternative levers

### Phase 2 (v13 LOEO Validation)
✅ **PASS:** Best v13-X shows delta > +0.005 on full LOEO (199 datasets) → Stage 4 unlock  
❌ **FAIL:** Delta < +0.005 on full LOEO → Complete Stage 3, stop post-proc refinement

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Golden-12 sweep non-monotonic | Tuning traps | Use physics-informed priors (velocity, distance bounds) |
| Fold1 generation fails | Cannot validate full LOEO | Have contingency: use loeo_129ep fold1 if available |
| Parameter optimization overfit | Delta doesn't generalize to full data | Validate each v12 on full fold0 before promoting to v13 |
| GPU OOM during fold1 gen | Inference halts | Batch fold1 into smaller chunks (32-64 datasets per job) |

---

## Timeline

**V12 (Golden-12 Sweep):** 1-2 hours (CPU, Phase 2A pipeline)
- Design configs: 30min
- Run 7 variants on golden-12: 1.5h
- Analyze results + gate decision: 15min

**V13 (LOEO Validation, if Phase 1 passes):** 4-6 hours total
- v13-1 (fold0): 30min (CPU, already cached)
- v13-2 (fold1 gen): 3-4h (GPU inference)
- v13-3 (validate): 30min (CPU)

**Total Stage 3 timeline:** 2-8 hours (depending on v12 results and Phase 1 gate)

---

## Next Steps (Researcher)

1. Create baseline_v12 YAML configs (7 variants)
2. Update baseline/run_experiments_v12.sh runner
3. Design Phase 2A golden-12 sweep + result aggregation
4. Create docs/stage3_results.md template (ready before running)
5. Run v12 sweep, analyze results
6. Report delta > +0.005 gate decision
7. If gate PASS: Prepare fold1 generation (GPU inference)
8. If gate FAIL: Summarize insights + close Stage 3

---

## Integration with Previous Stages

- **Stage 1:** Baseline 0.8527 (golden-12 pilkwang frozen detections) ← used as control
- **Stage 2 Phase 2A:** Post-proc validation pipeline (golden-12 scoring) ← reused for v12 sweep
- **Stage 2 Phase 2B/2C:** Cached fold0 predictions (loeo_moreep_relink, 71 datasets) ← used for v13-1 validation
- **Stage 3:** Refinement sweep (v12-v13) ← extends Phase 2C results
- **Stage 4 (future):** Full LOEO lock-in + final submission (if Stage 3 gate passes)

---

*GRANDMASTER JOURNEY: Stage 1 → Stage 2 (post-proc design) → Stage 3 (refinement) → Stages 4-9 (TBD post-Stage3 validation).*
