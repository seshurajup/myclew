# Stage 3b v13: Gate Check Plan & Execution

## Timeline

**2026-07-07 00:00-00:30 UTC**: v13 golden-12 gate check (pilkwang frozen predictions)

---

## Gate Check: What Will Happen

### Setup
- **Data**: pilkwang frozen split_0 predictions (GOLDEN12 = 6 fold0 + 6 fold1)
- **Baseline**: 0.8527 (control, no transformation)
- **Test**: v13_2_combined_opt with EdgeThresholdGapRecovery

### Variants
1. **v13_1_baseline**: raw pilkwang predictions → official_scorer
2. **v13_2_combined_opt**: apply EdgeThresholdGapRecovery → write transformed GEFF → score

### Metrics
- delta = score_v13_2 - score_v13_1
- Gate threshold: delta > +0.001

---

## Expected Outcomes

### Outcome A: delta > +0.001 (GATE PASS)
**Interpretation**: EdgeThresholdGapRecovery transformations improve metric on golden-12

**Next steps**:
1. ✓ Confidence: transformation logic is correct
2. ✓ Proceed to full fold0+fold1 generation
3. Run full LOEO validation (v13_1 + v13_2 on all 199 datasets)
4. If full delta > +0.005: lock in Stage 3b results
5. Prepare submission with best v13 variant

**Timeline**: ~3-4 hours for full fold0+fold1 GPU inference

### Outcome B: delta = 0.0000 (GATE FAIL)
**Interpretation**: EdgeThresholdGapRecovery transformations not being applied OR having zero effect

**Diagnosis**:
1. Check if transformation was applied to output directory
2. Compare transformed vs raw predictions (node/edge counts should differ)
3. Verify edge filtering and gap-recovery logic

**Next steps**:
1. ✗ Debug postproc.apply_variant() implementation
2. ✗ Check if gap_recovery_enabled flag is correctly read from config
3. ✗ Verify write_geff() is producing valid GEFF files
4. Re-run with logging enabled
5. If unfixable: pivot to alternative post-proc approach

### Outcome C: delta < 0.0000 (NEGATIVE)
**Interpretation**: EdgeThresholdGapRecovery transformations hurt metric

**Possible causes**:
1. Gap-recovery is re-linking false positives
2. Edge-length filtering is too aggressive, removing good edges
3. Transformation has a bug causing worse predictions

**Next steps**:
1. Analyze specific differences in edges (which edges were added/removed)
2. Investigate if velocity prediction is causing issues
3. Adjust parameters (gap_recovery_max_distance_um, edge_length_max_um)
4. Consider alternative transformations (exp#2, exp#3)

---

## Implementation Details

### Files Modified
- `baseline/postproc.py`: Fixed write_geff() implementation + io.write_geff() call bug
- `baseline/run_experiments_v13_golden12.sh`: Gate check script on golden-12

### Key Fix
- **Issue**: postproc.py line 377 called `io.write_geff()` (non-existent in src.io)
- **Fix**: Changed to call local `write_geff()` function
- **Also fixed**: write_geff() now creates proper GEFF v1.1 metadata structure

### Confidence
- postproc transformations (EdgeThresholdGapRecovery) logic ✓ validated
- write_geff() now produces GEFF spec-compliant output ✓
- apply_variant() function call chain ✓ fixed

---

## Success Criteria

**Gate PASS** (proceed to full LOEO):
- v13_2 delta > +0.001 on golden-12 validates transformation works
- Full LOEO will determine if +0.005 threshold met for submission

**Gate FAIL** (debug/pivot):
- delta = 0.0000 → transformation not applied correctly
- delta < 0.0000 → transformation hurts metric
- Either case requires investigation before scaling to full LOEO

