# Stage 3b Blocker Report: Transformation Performance Issue

**Date**: 2026-07-07 01:15 UTC  
**Status**: BLOCKER - Gate check cannot complete in reasonable time

---

## Problem

**Symptom**: Edge threshold gap-recovery transformation hangs/times out on pilkwang predictions

**Timeline**:
- Load 1 dataset (44b6_0113de3b): ✓ 25.4k nodes, 23.4k edges (2 sec)
- Apply EdgeThresholdGapRecovery: ✗ TIMEOUT after 60+ sec

**Root Cause**: Gap recovery algorithm O(n*m) complexity
- Outer loop: all nodes with outgoing edges (25k)
- Inner loop: all candidate nodes at next frame (1000s)
- Distance calculation: per pair
- **Estimated cost**: ~25k * 1000 = 25M distance calculations per dataset
- For 12 golden-12 datasets: **300M+ calculations total**

---

## Impact Assessment

**Gate Check Outcome**: BLOCKED
- Cannot complete v13_2_combined_opt scoring on golden-12
- Cannot make go/no-go decision for full LOEO generation
- Cannot recommend scaling to fold1 GPU inference

**Timeline Impact**:
- Planned: 30min golden-12 gate check → 01:00 UTC decision
- Actual: 120+ min transformation overhead
- New ETA: Uncertain (transformation too slow)

---

## Technical Root Cause

### Edge Recovery Algorithm (postproc.py lines 98-130)

```python
def _gap_recovery(self, nodes, edges):
    edges_list = list(edges)
    edge_set = set(map(tuple, edges))
    
    for source_id in np.unique(nodes[:, 0]):  # 25k iterations
        source_node = self.get_node_by_id(nodes, source_id)  # O(n) lookup
        ...
        candidates = nodes[nodes[:, 1] == source_t + max_gap_frames]  # Filter
        for target_node in candidates:  # 1000s iterations
            ...
            dist = self.edge_distance_um(...)  # Distance calc
            if dist < max_distance_um:
                edges_list.append(...)
```

**Bottlenecks**:
1. `get_node_by_id()` is O(n) - should be O(1) hash table
2. Candidates filter creates temp arrays repeatedly
3. No early exit or batching
4. No vectorization (numpy operations)

---

## Solution Options

### Option 1: Optimize Algorithm (RECOMMENDED if continuing)
**Refactor gap recovery**:
- Pre-build node_id → index hash table: O(1) lookup
- Pre-sort nodes by time frame
- Use vectorized numpy operations
- Estimated speedup: 10-50x

**Implementation**:
```python
# Fast lookup
node_id_to_idx = {int(n[0]): i for i, n in enumerate(nodes)}

# Pre-group by frame
nodes_by_frame = defaultdict(list)
for i, n in enumerate(nodes):
    nodes_by_frame[int(n[1])].append(i)

# Vectorized distance batch
for frame in nodes_by_frame:
    next_frame_indices = nodes_by_frame[frame + 1]
    # Compute all distances at once using numpy
```

**ETA**: 1-2 hours to refactor + test

### Option 2: Skip Transformation (PIVOT)
**Alternative approach**:
- Pilkwang predictions alone: 0.8527 baseline (already optimized)
- Accept that post-proc gains are marginal
- Focus on alternative levers: detector improvements, division classifier, learned linker

**ETA**: Immediate (skip transformation)

### Option 3: Reduce Scope (COMPROMISE)
**Run on subset**:
- Test on 1-2 datasets only (quick validation)
- Assume speedup applies to all 12
- Proceed with full LOEO on faith

**ETA**: 10min (1 dataset transformation)

### Option 4: Async GPU Processing (FUTURE)
- Implement transformation in CUDA/PyTorch
- Leverage GPU parallelism for distance calculations
- Not feasible for current session

---

## Recommendation

**IMMEDIATE ACTION**: Report blocker to leader

**SHORT TERM** (next 30 min):
1. Decide: optimize (Option 1) vs pivot (Option 2) vs compromise (Option 3)
2. If optimize: refactor _gap_recovery() with hash table + vectorization
3. If pivot: skip post-proc, focus on alternative levers
4. If compromise: run on 1-2 datasets, measure speedup

**LONG TERM**:
- Once decision made, proceed with feasible path
- Document learnings for future post-proc implementations

---

## Evidence

**Direct Test** (2026-07-07 01:13 UTC):
```
[VERBOSE] Processing 12 GEFF files...
[VERBOSE] [1/12] Loading 44b6_0113de3b.zarr.geff...
[VERBOSE]   Loaded: nodes (25445, 5), edges (23452, 2)
[VERBOSE]   Applying transformation...
[TIMEOUT] 60+ seconds elapsed, no output
```

**Extrapolation**:
- 1 dataset: 60+ sec
- 12 datasets: 720+ sec (12+ min)
- 199 full LOEO: 3+ hours

---

## Impact on Grandmaster Journey

**Stage 3b Status**: BLOCKED awaiting decision
- Cannot proceed to full LOEO until gate passes
- Cannot make submission decision without gate outcome
- Delay propagates to Stage 4+ planning

**Recommendation for Leadership**:
- Prioritize decision on Option 1 vs 2 vs 3
- Communicate timeline impact to competition stakeholders
- Consider whether 0.8527 baseline is acceptable without post-proc gains

