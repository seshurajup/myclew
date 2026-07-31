# Stage 3b v13: Gate Check Decision Tree

## Current State
- **Time**: 2026-07-07 00:10 UTC
- **Task**: v13 golden-12 gate check (EdgeThresholdGapRecovery validation)
- **Status**: ⏳ Processing 6 datasets, writing transformed GEFF files
- **ETA**: Complete by 00:15 UTC

---

## Gate Result Decision Tree

### If GATE = PASS (delta > +0.001)
✓ **Validation**: EdgeThresholdGapRecovery transformations credibly improve metric

**Immediate Actions**:
```bash
# 1. Report golden-12 results
jq . output/stage3/v13_golden12_results.json

# 2. Analyze transformation impact
echo "Transformation added/removed edges?"
du -sh output/stage3/v13_2_combined_opt_golden12/
```

**Next Phase: Full LOEO Validation**:
- **Blocker**: Need full fold0 (71) + fold1 (128) predictions
- **Current availability**: pilkwang cache has only 12 (6+6)
- **Options**:
  1. Use pilkwang cache for full_loeo validation (only 12/199 datasets, not representative)
  2. Generate full fold0+fold1 via GPU inference (2-4h, needed for real validation)

**Path Forward (if delta > +0.001 on golden-12)**:
```
├─ Option A: Quick test on pilkwang 12-dataset cache
│  └─ Risk: Results not representative of full LOEO (only 12 datasets)
│
└─ Option B: Full fold0+fold1 generation (RECOMMENDED)
   ├─ Phase 1: GPU inference for all 199 LOEO datasets
   ├─ Phase 2: Run v13_1 + v13_2 on full 199 datasets
   └─ Phase 3: Gate final decision (delta > +0.005?)
```

**Success Criterion for Full LOEO**:
- If **final delta > +0.005**: Lock in best v13 variant for submission
- If **final delta ≤ +0.005**: Close Stage 3, evaluate alternative levers

**Estimated Timeline** (if pursuing Option B):
- GPU inference: 2-4h (depends on GPU queue)
- Full LOEO scoring: 30min
- Total: ~3-4h from now

---

### If GATE = FAIL (delta ≤ 0.0000)
✗ **Validation Failed**: Transformation not working as expected

**Diagnosis Steps**:

#### Case 1: delta = 0.0000 (No change)
"Transformation didn't run or had zero effect"

**Debug**:
```bash
# 1. Check if output files exist
ls -la output/stage3/v13_2_combined_opt_golden12/
# Expected: 6 .geff directories + zarr.json in each

# 2. Check if files are different from raw
# Compare node/edge counts
for f in output/stage3/v13_2_combined_opt_golden12/*.geff; do
  echo "$(basename $f):"
  # Count nodes and edges
done

# 3. Check logs for errors
grep -i "error\|exception" output/stage3/stage3_v13_golden12_gate.log
```

**Possible Root Causes**:
- gap_recovery_enabled not read from YAML
- EdgeThresholdGapRecovery._gap_recovery() is no-op
- write_geff() not actually writing transformed data
- Transformation applied but metric calculation ignores it

**Remediation**:
1. Add verbose logging to postproc.apply_variant()
2. Print edge count deltas for each dataset
3. Verify config file loading (gap_recovery_enabled should be true)
4. Check EdgeThresholdGapRecovery gap_recovery_max_distance_um != default

#### Case 2: delta < 0.0000 (Negative)
"Transformation hurts metric"

**Analysis**:
1. Which edges changed? (filtered vs recovered vs both?)
2. Did filtering remove good edges or recovery add bad ones?
3. Is gap_recovery_velocity_prediction causing spurious links?

**Options**:
- Reduce gap_recovery_max_distance_um (6.0 → 5.0)
- Disable velocity prediction
- Increase gap_recovery_velocity_weight (0.8 → 0.6)
- Try exp#2 (CentroidRefineSmooth) or exp#3 (DivisionFilter) instead

---

## Decision Flowchart

```
START: v13 golden-12 gate check
  │
  ├─→ Results ready? NO  → wait (monitor)
  │                    YES
  ├─→ Gate Pass (delta > +0.001)? 
  │    ├─ YES → PROCEED to full LOEO (recommend Option B)
  │    │        └─ ETA: 3-4h from now
  │    │
  │    └─ NO → DIAGNOSE
  │         ├─ delta = 0.0? → Check if transformation ran
  │         ├─ delta < 0.0? → Adjust parameters or try alt variant
  │         └─ Re-run with logging enabled
  │
  └─→ END
```

---

## Key Files & Outputs

**Results**:
- `output/stage3/v13_golden12_results.json` — gate decision + scores
- `output/stage3/stage3_v13_golden12_gate.log` — detailed log

**Transformed Predictions** (if successful):
- `output/stage3/v13_2_combined_opt_golden12/` — 6 transformed GEFF dirs
- Each GEFF has nodes/, edges/, zarr.json

**Configs**:
- `baseline/experiments_v13/v13_2_combined_opt.yaml` — gap recovery parameters
- `baseline/postproc.py` — EdgeThresholdGapRecovery implementation

---

## Notes for Researcher

**If Gate PASSES**:
- ✓ Don't second-guess; transformation works on golden-12 subset
- ✓ Proceed with confidence to full LOEO generation
- ✓ Prepare to queue GPU work immediately (don't delay)

**If Gate FAILS**:
- ✗ Don't attempt full LOEO without fixing
- ✗ Focus on root cause (config, implementation, or parameter tuning)
- ✗ Consider parallel track: evaluate alternative post-proc approaches

**Time Constraint**:
- Competition deadline is advancing
- Use golden-12 validation to save GPU time
- If gate fails, decide quickly whether to fix or pivot

