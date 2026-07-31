# Temporal Affinity Fields for 3D Cell Lineage (hengck23 idea — "Your Affinity Field Tells Your Fate")

**Idea:** instead of geometric/learned pairwise linking, predict a **local vector (flow) field** per voxel
that tells the ILP/graph optimizer *how cells should move* — a learned motion prior for linking.
**GT supervision:** build flow GT from **optical flow + sparse annotation tracks** (semi-supervised).

**Feasibility (link displacement stats, scale (1,4,4), 63,751 links):**
- dz: median 0, |p90|=1, |p95|=2, |p99|=4, tails to ±37 (rare fast z-moves)
- dy: median 0, |p90|=1.25, |p95|=1.75, |p99|=3
- dx: median 0, |p90|=1.5, |p95|=2, |p99|=3.75
→ **most motion is sub-2-voxel**; a short-range affinity field (±4 vox) covers p99, with a rare-large tail.

**Why it fits our metric:** linking quality = edge-Jaccard (our dominant term after node recall). A flow
prior that predicts the p50–p95 displacement would sharpen `motion_relink` and reduce edge FP/FN.
**Cheap first step (no training):** use the empirical per-axis displacement distribution above as a
motion PRIOR in the linker cost (bias links toward the p50 vector), score on golden-12 — a post-proc
test of the idea before training a flow net.
**Full step (train):** small 3D flow head on frame pairs, GT = optical flow blended with sparse tracks.

---

## hengck23 update — the winning formula is DENSE TRAINING TRACKS, not just the model

The real bottleneck is supervision: **<1% of competition links are labelled.** hengck23's recipe:

1. **3D points are easy** — cellpose / DoG / our cached peaks already give locations.
2. **Short tracks (2–3 frame) are easy** — ultrack, rule-based heuristics, or open-source trackers.
   Run **≥5 trackers and keep the CONSENSUS** links → far more confident short tracks than any single one.
3. Feed those dense short tracks to a flow/affinity model; then **ILP / min-cost graph-cut** stitches the
   long submission tracks. The public rule-based post-processor is itself a link *generator* for step 2.
4. **External zebrafish lineage data** (real + synthetic dense tracks, no images needed — step 2 only
   needs *locations*) is fair game and abundant.

### What we actually have downloaded (huge)
- `input/zebrahub/tracks/ZSNS00{1,3,4,5}*.csv` — **~25.3M labelled nodes** with `track_id` + time ⇒ full
  lineage links; `ZSNS001_tail` also has `ParentTrackID` ⇒ **division events**.
- `research/zebrahub/zeb_only_train/*.geff` — 507 lineage graphs (ZSNS + our 44b6 embryos).
- ⇒ ~1000× more supervision than the competition's <1% — enough to train a flow field **and a division head**.

### Why this targets OUR open lever
Our pipeline's `div_J = 0` (measured). A division is a distinctive affinity pattern: **one parent → two
outgoing flow vectors.** A flow/affinity model trained on the external dense tracks (which contain real
divisions via `ParentTrackID`) is the principled way to predict true divisions and finally move `div_J`
off zero — the one term that separates our 0.8803 from the public 0.897.

### Agents that implement this recipe (fleet) — all spec-driven / reusable
- `ext-label-stats` — inventory external labels: link/division counts + displacement prior (feasibility).
  Result: **698K links, 129K divisions, 96.1% link completeness** across ZSNS001/003/004/005.
- `flow-gt-build` — materialise per-node `(dz,dy,dx)` flow GT + `is_division` from the external tracks.
  Result: `results/flow_gt/flow_node_gt.parquet` — **24.3M flow vectors + 129,050 divisions**.
- `gnn-probe` — does a GNN help, or is pairwise geometry enough? (spec: tracks_glob, radius_um, k_neigh…).
- `flow-field-train` / `gnn-link-train` (next) — train the flow/affinity + division head / message-passing GNN.
- `affinity-link` (next) — flow-gated ILP linking: use the predicted field to score candidate links.

## Is a GNN worth it? — MEASURED (gnn-probe)
The problem is a spatiotemporal graph, so a GNN is the natural model — but our *pairwise* learned edge head
was already saturated (Δ0.000 vs geometry). A GNN's extra power is **neighbourhood message-passing**.
Probe on 139,365 external candidate edges (held-out AUC):

| model | edge AUC |
|---|--:|
| pairwise geometry (disp + dist) | 0.820 |
| **+ neighbourhood-flow context** (the GNN signal) | **0.888** |

**Δ +0.068** — the local-flow-coherence signal a GNN would learn is real and geometry misses it. The gain
is concentrated in hard/dense/division cases (why the *pairwise* head looked saturated on easy links).
⇒ **A message-passing GNN edge+division classifier is justified.** Next: `gnn-link-train`.

## Recipe coverage map (hengck23's winning formula → our agents)
| # | recipe piece | agent | status |
|---|---|---|---|
| 1 | 3D points (cellpose/DoG/peaks) | detectors (`arch-probe`, peak cache) | ✅ have |
| 2 | short tracks (2–3 frame) via trackers | `tracker-consensus` | ✅ built |
| 3 | 5 trackers + **consistency** → more links | `tracker-consensus` (5 linkers, ≥K agree) | ✅ **77K links/2 embryos** |
| 4 | rule-based post-proc as link source | `tracker-consensus` (runs on post-proc detections) | ✅ covered |
| 5 | external zebrafish dense tracks | `ext-label-stats`, `flow-gt-build` | ✅ **24.3M + 129K div** |
| 6 | flow/affinity field; is a GNN worth it? | `gnn-probe` | ✅ **+0.068 AUC → yes** |
| 7 | ILP / min-cost graph-cut → long tracks | `pilk_post` ILP | ✅ have (0.8803) |
| 8 | **train** the flow/GNN edge+division head | `gnn-link-train` / `flow-field-train` | ⏳ next (GPU) |

Steps 1–7 are covered by reusable, spec-driven fleet agents. The only remaining piece is the trainer (8)
that consumes the assembled supervision (external 24.3M GT + in-domain consensus pseudo-labels) to learn
the edge+division head — the direct assault on `div_J = 0` and the gap to public 0.897.
