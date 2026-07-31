# Stage 0 — CV harness & scoring contract (FROZEN)

**Status: LOCKED (2026-07-05).** This is the day-one contract every future experiment obeys. Do not
change the split or the metric mid-competition. Higher CV = better.

## 1. Frozen folds (embryo-disjoint LOEO)

- **File:** `learning/ensemble_work/finetune/fleet_loeo_mini.json` (list of `{"train":[...],"test":[...]}`).
- **RE-LOCKED 2026-07-05 → FULL-embryo LOEO 128/71** (upgraded from the original mini-8, PRE-FIRST-SCORE so
  no result is invalidated; full LOEO is strictly more Kaggle-faithful than an 8-frame subsample, and the
  real S1 weights already trained on it). `sha256 =
  61747a2e9f42dd7742182987a75c43562217eeeb78fabf3c7824723874496b53`; file is `chmod 444` + backed up at
  `fleet_loeo_mini.FROZEN_128x71.json`. **If the sha changes, the frozen split was clobbered — restore from backup.**
- **Built by:** `fleet_agents.cv.handle(spec={"k":2})` → `src.cv.kfold_embryo`, run in `research/cellmot_venv`.
- **Folds:**

| fold | train | test | axis |
| :-- | :-- | :-- | :-- |
| 0 | 128 × `6bba` (full embryo) | 71 × `44b6` (full held-out embryo) | leave-embryo-out |
| 1 | 71 × `44b6` (full embryo) | 128 × `6bba` (full held-out embryo) | leave-embryo-out |

- Pool = 199 datasets; the two embryos (`44b6`, `6bba`) are the leak axis. The hidden Kaggle test is
  embryo-disjoint, so this split matches test generation.
- **Churn-guard:** `fleet_agents/cv.py::handle` now REFUSES to overwrite `fleet_loeo_mini.json` unless
  `spec.allow_overwrite_frozen=true`. Root cause of the earlier churn: a fleet `cv-build` re-run defaulted
  `mini_per_fold=0` (full) + `out=fleet_loeo_mini.json` → silent full-regen clobber (mtime 16:39→17:25).
  Regenerate any secondary/screen split to a DIFFERENT `out`.
- **SECONDARY fast-screen split (NOT primary):** `fleet_loeo_mini8_secondary.json` (mini-8 train / full test,
  cv_contract-clean) for cheap triage only. Never gate a result on it — the primary gate is the 128/71 above.
- **ID-hygiene fix:** the pool build now normalizes file suffixes and drops phantom ids
  (`fleet_agents/cv.py` strips `.zarr`/`.geff` and keeps only ids that resolve to a real train geff).
  This removed 5 malformed `.zarr`-suffixed duplicate ids (e.g. `44b6_12dfb391.zarr`) that were in the
  earlier split and would have scored as missing datasets.

## 2. THE scorer — full official metric only

**Command (single source of truth):**
```
research/cellmot_venv/bin/python -m fleet_agents.official_scorer \
    --split learning/ensemble_work/finetune/fleet_loeo_mini.json --fold <N> --pred-dir <preds>
```
- `official_score = adj_edge_jaccard + 0.1 · division_jaccard` (`src.metric.official_score`).
- Node match = 1-to-1 Hungarian ≤ 7 µm (scaled by `SCALE=(1.625,0.40625,0.40625)`); an edge is TP only
  if **both** endpoints match a GT edge; over-prediction penalised via `t_true = estimated_number_of_nodes`.
- **NEVER** score a component (node recall / edge-F1 / a model's own loss). Every EXP row logs THIS number.
- Predictions are final-format geffs: one `<ds>.geff` per test dataset with `nodes[node_id,t,z,y,x]` +
  `edges[source_id,target_id]`. The pipeline logs the result to MLflow; `fleet_agents/scorer.py` surfaces
  the trajectory.
- **Wiring smoke-test (verified):** `--verify-pilk` scores pilkwang golden-12 → `official_score 0.8527`
  (= adj_edge_jaccard, div_j 0.0). This is the pilkwang **raw-ILP-solution** number (matches Thread-2's
  0.853 baseline); the **0.8708** anchor is pilkwang's FULL post-proc chain. The scorer is correct — it
  faithfully scores whatever prediction graph it is given. golden-12 here is SECONDARY (see §4).

## 3. Leak-assert — every run must pass

**Command (exit 0 = pass, exit 1 = fail):**
```
research/cellmot_venv/bin/python -m fleet_agents.cv_contract learning/ensemble_work/finetune/fleet_loeo_mini.json
```
Checks, per fold: (1) **no embryo in both train and test**; (2) every id resolves to a real train geff
(catches phantom/`.zarr` ids); (3) no duplicate id within train or test; (4) `train ∩ test = ∅`.
**Verified:** PASS (exit 0) on the frozen split; FAIL (exit 1) on a synthetic split with a leaked embryo
+ a `.zarr` phantom + a dup. Wire this as a precondition of every scored run.

## 4. golden-12 is LEAKY → SECONDARY only

golden-12 mixes the same two embryos that form the CV test folds, so it is **not** embryo-held-out
relative to any model trained on the mini folds — it is **leaky**. Use it only as SECONDARY / sanity /
Stage-5 evidence, **never** as the primary gate. The primary gate is the LOEO `official_score` above.

**Ledger re-anchor:** all Thread-1 / Thread-2 numbers were golden-12 = **SECONDARY**:
pilkwang 0.8708, exp#0 (post-proc, negative), exp#1 (Trackastra 0.6465), exp#1b (iso_z 0.6876), and
ledger lessons L1–L10. Keep them as **decision-point evidence** (they correctly ranked the buckets and
killed dead ends); do **not** let them stand in as primary CV. Backfilled into
`docs/experiment_ledger.md` as SECONDARY rows (`trn_set=golden12`).

## 5. Carried-forward inputs for later stages

- **Trackastra coordinate scaling (for any future learning-based linking, Stage 5):** use **`iso_z`
  (z × 4.0, xy kept in voxel range)** — verified in exp#1b as the correct anisotropy handling
  (`SCALE.z/SCALE.x ≈ 4`). Physical-µm scaling (`um`) HURTS (shrinks xy below the model's
  `spatial_pos_cutoff=256`). Trackastra ctc reads centroids in voxel units with no anisotropy correction,
  so the scaling must be applied to `WRFeatures.coords` before `build_windows`
  (`eda/thread2/exp1_trackastra_link.py::track_scaled`).
- **Stage-5 linker baseline to beat:** pilkwang's **geometric ILP** linker is the linker to beat.
  Off-the-shelf learned linking (pretrained Trackastra ctc) on our **point-blob** substrate is NOT
  competitive even de-confounded — exp#4 fine-tune was pre-registered **NO-GO** (iso_z only partial: dense
  tail 0.55 vs 0.87 anchor). A learned linker would need **real instance masks**, not synthetic point
  blobs, to be worth the GPU. Link gate ≈ 8.5 µm for 6bba (not 7.0) per the fleet linking note.

## 6. Advancing to Stage 1

When this contract is locked (folds frozen + scorer wired + leak-assert green + doc written), run
`fleet_agents.journey.status` to confirm the next stage, then the leader green-lights **Stage 1** =
a dumb end-to-end baseline on the LOEO fold, scored by §2 and decomposed into R_node/R_edge/Q_link/count
buckets. Pending `EXP_00–06` rows are Stage 3/4 and stay pending until we get there.
