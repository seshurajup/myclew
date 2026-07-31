# baseline_v13 — Cheap post-proc sweep (Stage 3b pivot) — DESIGN + FEASIBILITY MAP

**Date:** 2026-07-07 · **Author:** researcher · **Status:** plan-confirm + feasibility findings

## Context
Leader pivoted Stage 3b: **SHELVE `EdgeThresholdGapRecovery`** (O(n·m) hang, 60s+/dataset — see
`docs/STAGE3B_BLOCKER_REPORT.md`; would need a scipy cKDTree/broadcast rewrite as a separate task).
Instead run a CHEAP, GPU-free post-proc sweep on the pilkwang CACHED golden-12 predictions, screened
on the LB-faithful golden-12 CV gate with the OFFICIAL metric, delta vs the **0.8735 anchor**
(= pilk full postproc + `min_track_len4`, our bankable best; reproduced by
`experiments/pipeline/winning_config.py`).

Requested 3 variants: **EXP-A** det_thresh {0.99,0.995,0.997} · **EXP-B** min_track_len {4,5,6} ·
**EXP-C** fork-based division PRUNING.

## KEY FINDING — the sweep is ALREADY RUNNING autonomously (fleet `combo_search`)
`fleet_agents/combo_search.py` is a running fleet agent doing exactly this: coordinate descent over
`det_threshold × gap_close_um × min_track_len × safe_div_max_um` on golden-12, scored via
`scripts/score_postproc_golden12.py` (env-driven `pilk_post.filter_output_graph`, official metric),
state in `config/_auto/combo_search_state.json`, escalates to human at 0.885. **We should steer this,
not hand-build a duplicate.** Current axes: `det {0.988,0.99,0.992}`, `gap {5.5,6.0,6.8}`,
`min_track_len {4,6,12}`, `safe_div {4.7,5.5}`.

## FEASIBILITY MAP (on the CACHED golden-12 `.zarr.geff`, the only GPU-free surface)

| Variant | Cheap on cache? | Finding |
|---|---|---|
| **EXP-A det_thresh** | ❌ NO | `BIOHUB_DET_THRESHOLD` is read at `pilk_post.py:11` but **referenced nowhere in the post-proc path** — it only gates prediction-time peak detection (`predict_unet_transformer.py --det-threshold`). On cached geffs the nodes are already fixed, so it is a **silent no-op**. Empirically confirmed: det 0.988 vs 0.992 on the 4-subset → **identical 0.8315**. ⇒ combo_search's det axis wastes 1/3 of its grid and will falsely "conclude" det doesn't matter. TRUE det_thresh needs GPU re-detection: `baseline/postproc/det_grid_sweep.sh` (needs weights). |
| **EXP-B min_track_len** | ✅ YES | Bankable lever. `min4` = 0.8735 anchor. `min_track_len_prune.py --min-track-len N` (union-find component prune, keeps dividing components). Competitor intel (drkongvis v23–v27, boristown) swept min6/min12/min13/min14 — combo_search only tries {4,6,12}; **widen to {4,5,6,8,10,12,14}** to find the golden-12 optimum. |
| **EXP-C fork prune** | ⚠️ NO cheap EVAL | golden-12 is **div-BLIND** (only ~8 GT divisions) AND the division-rich minisplit datasets (`eda/thread2/division_rich_minisplit.json`, 36 divs) are **NOT in the pilk cache** — so fork-pruning has **no cheap surface to show a div_J gain**; on golden-12 it can only remove real division edges (adj_edge loss). Implementable via `BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER=1` (+ `DIV_DROP_TO_SINGLE_IF_BAD`) or `baseline/postproc.py::DivisionFilter` (`exp3_division_filter`), but its real test needs **GPU inference on div-rich datasets**. NOTE combo_search's `safe_div` axis is boristown ADD-daughter, which is **DEAD** in our metric (div_tp=0) — not fork-pruning. |

## PLAN (what I recommend + built)
1. **Steer combo_search** (patch `AXES`): drop the dead `BIOHUB_DET_THRESHOLD` axis; widen
   `BIOHUB_OUTPUT_MIN_TRACK_LEN` → `{4,5,6,8,10,12,14}`; keep `gap_close`; replace the dead
   add-daughter `safe_div` axis with the real fork-demotion knob `BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER`.
2. **Complementary lever not in combo_search:** `consensus_prune` (beicicc lb884 edge-precision prune)
   stacked on the best `min_track_len` — the genuine cached analog of "raise det_threshold to relieve
   the over-prediction penalty." Driver: `baseline/run_experiments_v13_postproc_sweep.py`
   (anchor → min_track_len sweep → +consensus_prune; by-embryo + micro official; delta vs 0.8735).
3. **EXP-A/EXP-C defer to a GPU job** (det re-detect + div-rich inference) — out of scope for the
   cheap sweep; documented here so they aren't mistaken for cheap wins.

**Promote rule:** any config > 0.8735 by +0.001 on golden-12 → promote candidate for fold0/fold1 LOEO.
