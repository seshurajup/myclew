# baseline_v14 — PER-DENSITY min_track_len RESULTS (golden-12 PoC)

**Date:** 2026-07-07 · Driver: `baseline/run_experiments_v14_perdensity_mtl.py` ·
Results JSON: `output/baseline_v14/v14_perdensity_results.json`

## Verdict: HONEST NEGATIVE — mechanistic 2-bin does NOT beat global mtl10 (−0.0019, bar was +0.001)

| config | official | adj_44b6 | adj_6bba | Δ vs global mtl10 |
|---|---:|---:|---:|---:|
| global_mtl10 | **0.8832** | 0.8518 | 0.8910 | — |
| global_mtl14 | 0.8726 | 0.8556 | 0.8768 | −0.0106 |
| **perdensity_2bin** | 0.8813 | **0.8548** | 0.8878 | **−0.0019** |

Proxy: `learning/03_true_density_stage.csv` `stage`; mapping {S0,S1,S2}→mtl10, {S3,S4}→mtl14; gap=5.5.
Golden-12 split 6/6, crossing embryos (6bba_05db0fb1 S4→high, 44b6_0db75fae S2→low). Mechanistic, NOT golden-fit.

## Why it failed (real finding)
- The 2-bin **did** lift 44b6 (0.8518→0.8548, +0.0030) — density-conditioning genuinely helps DENSE data
  (dense 44b6 wants aggressive pruning, as v13 predicted).
- But it **loses on micro** because the highest-weight dataset in golden-12, **6bba_05db0fb1 (estN=69,800,
  stage S4 = genuinely dense)**, over-prunes at mtl14 (adj_6bba 0.8910→0.8878). 6bba's track-length
  statistics make it hate mtl>10 **even when dense** — so TRUE density does NOT predict the mtl optimum
  across embryos. The v13 per-embryo divergence (44b6→14, 6bba→10) was **embryo-track-structure-driven,
  not purely density-driven.** The over-prediction penalty is micro-weighted by estN, so the one dense-6bba
  outlier (largest weight) sinks the whole 2-bin.
- macro (per-fold mean) view: 2-bin (0.8548+0.8878)/2 = 0.8713 vs global mtl10 0.8714 — a dead tie. No win
  either way.

## Recommendation
1. **Do NOT promote per-density mtl.** Global **mtl10 / gap5.5** stands as the locked config.
2. An embryo-aware rule (44b6→14, 6bba→10) WOULD net positive on golden-12 but is exactly the embryo-fit
   the guardrail forbids (2-embryo overfit, won't generalize) — correctly rejected.
3. Real future lever is NOT a density proxy but the per-dataset **track-length distribution** (what fraction
   of tracks are genuinely short) — but that is data-dependent and not a clean cheap win. Parked.
4. The v13 fold0 global-mtl10 confirmation remains the priority; v14 adds no config to it.
