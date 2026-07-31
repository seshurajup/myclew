# baseline_v13 — Cheap post-proc sweep RESULTS (golden-12, GPU-free)

**Date:** 2026-07-07 · **Author:** researcher · Driver: `baseline/run_experiments_v13_postproc_sweep.py`
Results JSON: `output/baseline_v13/v13_postproc_sweep_results.json`

## Headline
**`min_track_len=10` → golden-12 official 0.8818, +0.0083 over the 0.8735 anchor** (pilk full
postproc + min_track_len4). 3× the prior bankable win. golden-12 CV is rank-preserving & ~+0.02
conservative vs LB → **implied LB ~0.90** (above the 0.891 public cluster). CHEAP, CPU-only, inference-free.

## min_track_len sweep (stacked on the pilk full-postproc anchor)
| min_track_len | official | Δ vs 0.8735 | note |
|---:|---:|---:|---|
| anchor (pilk_post) | 0.8691 | — | 396,683 edges relinked (full-12 confirmed) |
| 4 | 0.8735 | +0.0000 | reproduces the bankable anchor EXACTLY (harness validated) |
| 5 | 0.8762 | +0.0027 | |
| 6 | 0.8781 | +0.0046 | boristown min6 |
| 8 | 0.8815 | +0.0080 | |
| **10** | **0.8818** | **+0.0083** | **PEAK** |
| 12 | 0.8782 | +0.0047 | drkongvis min12 |
| 14 | 0.8710 | −0.0025 | over-pruning (removes real tracks) |

Clean inverted-U. Plateau at 8–10 (0.8815–0.8818), peak at 10. Mechanism = over-prediction penalty
`min(1, estN/predN)`: pruning short spurious tracks lowers predN toward estN (penalty → 1) until, past
~12, real tracks start dropping and adj_edge (numerator) falls. Confirms the metric's over-prediction
lever is the dominant cheap knob. Competitor intel (min6/min12/min14) pointed the right direction but the
golden-12 optimum is **~8–10, not 12–14**.

## consensus_prune complement (EXP-A' — cached det_thresh analog)
`mtl10 + consensus_prune = 0.8817` (−0.0001 vs mtl10 alone). **SUBSUMED / honest negative** — the
edge-precision prune adds nothing once min_track_len10 has already trimmed the over-prediction. Drop it.

## Caveats
- Absolute scores use the `write_geff` round-trip anchor (0.8691), ~0.0017 below the in-memory 0.8708
  anchor; all rows share this base so the **deltas are internally consistent** (in-memory equivalent of
  min10 ≈ 0.8835). The +0.0083 min10-over-min4 gain is the trustworthy number.
- golden-12 is the cheap GATE, not final. **Promote min_track_len {8,10} to fold0/1 LOEO** for confirmation
  (leader promote rule: anchor+0.001 → LOEO). min10 is the clear pick; 8 as a robustness check.
- The patched fleet `combo_search` (min_track_len × gap) will independently cross-check this optimum and
  probe gap-close interactions (its subset screen + in-memory base give different absolutes, same shape).

## mtl × gap cross-check (in-memory pilk_post, BOTH env flags set)
Discovered a SECOND silent no-op: `pilk_post.filter_short_track_components` (pilk_post.py:703) early-returns
unless `BIOHUB_OUTPUT_FILTER_SHORT_TRACKS=1`. `score_postproc_golden12.py`/`combo_search` set the *length*
but not the *flag* → the whole min_track_len axis was dead there. Patched into combo_search BASE. Grid
(full-12, flag on):

| | gap 5.5 | gap 6.0 | gap 6.8 |
|---|---:|---:|---:|
| mtl 8 | 0.8826 | 0.8818 | 0.8792 |
| mtl 10 | **0.8846** | 0.8837 | 0.8809 |

(sanity mtl4/gap6.0 = 0.8759 in-memory = +0.0024 above the 0.8735 round-trip anchor = write_geff cost.)
**Best = mtl10/gap5.5 = 0.8846**, gap monotonic 5.5>6.0>6.8 (tighter gap-close = less spurious gap edges =
over-prediction relief). gap 5.0/5.2 probe run to confirm 5.5 is a real peak, not a range edge, before the
expensive LOEO lock.

## DENSITY DIAGNOSTIC — per-embryo mtl optima DIVERGE (both held-out)
| mtl | adj_44b6 (DENSE) | adj_6bba (LOW-density) |
|---:|---:|---:|
| 4 | 0.8409 | 0.8817 |
| 8 | 0.8472 | 0.8900 |
| **10** | 0.8488 | **0.8900** (peak) |
| 12 | 0.8505 | 0.8851 ↓ |
| 14 | 0.8524 ↑ (still climbing) | 0.8756 ↓↓ |

**44b6 (dense) wants mtl ≥ 14 (monotonic up); 6bba (low-density) peaks at 8–10 then over-prunes.** A single
global mtl is a compromise. **Implications:** (a) mtl10 is well-matched to 6bba (fold1's test embryo) — it does
NOT over-prune the low-density case; over-pruning only starts at 12–14. Good news for the deferred fold1. (b) On
fold0 (44b6, dense) mtl10 UNDERSHOOTS — 44b6 alone would score higher at mtl12–14; the global mtl10 is a
6bba-protective choice. (c) **Per-density mtl is a real future lever** (dense→high mtl, sparse→~10) and changes
the split_1 calculus — worth revisiting if the global mtl10 leaves fold0 headroom on the LOEO.

## Recommendation
1. **Promote `min_track_len=10` (primary) + `=8` (robustness), gap locked from the probe** to a **FOLD0-ONLY
   INTERIM** LOEO confirmation (split_0, 44b6 held-out, 71 datasets). split_1/fold1 GPU training DEFERRED behind
   mtl10 holding on full fold0 (leader call). Report fold0 explicitly as INTERIM, not yet 2-fold LOEO-primary.
2. Drop `consensus_prune` (subsumed). Drop the standalone det/fork cheap attempts (infeasible — see
   `docs/baseline_v13_gpu_job_spec.md`).
3. New bankable cheap ceiling on golden-12 = **~0.8846 @ mtl10/gap5.5** (was 0.8735 @ min4/gap6.0).
