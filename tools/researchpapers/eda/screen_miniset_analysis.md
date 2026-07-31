# EDA — Screening mini-set matched to the golden-12 / Kaggle CV split

CPU-only analysis (no GPU/training). Source: `learning/03_true_density_stage.csv` (199 embryos:
group, n_frames, estN, estN_per_frame density, stage), golden-12 = `splits_ft.json` fold-0 test.

## 1. golden-12 test composition (the distribution a screen must mirror)
| axis | golden-12 (n=12) |
|---|---|
| group | **6 × 44b6 / 6 × 6bba** (50/50) |
| stage | S0=3 S1=1 S2=2 S3=2 **S4=4** (late-skewed) |
| density (cells/frame) | min 48 · q25 80 · **median 304** · q75 604 · max 820 · mean 358 |
| per-group density | 44b6: [172,394,495,587,654,820] · 6bba: [48,61,64,85,215,698] |

**golden-12 is DENSER than the full pool** (median 304 vs pool 182, mean 358 vs 253) and group-balanced —
so a screen on a *pool-representative* set would be biased sparse/6bba-heavy and mis-predict golden-12.

## 2. The existing `splits_screen_mini.json` FAILS — two ways
| check | existing mini | verdict |
|---|---|---|
| leak-free vs golden-12 | **all 12 golden-12 embryos appear in its folds** | ❌ 100% LEAK (screen⇒judge trains on test) |
| group match | fold0 val = 44b6-only (9), fold1 val = 6bba-only (10) | ❌ single-group, not 6/6 balanced |
| density match | fold means 320 / 259 | ~ok but confounded by the group split |

→ **Must be rebuilt.** Screening on it and then confirming on golden-12 would be self-leaking.

## 3. New matched set — `splits_screen_matched.json` (built here)
Construction (seed 20260705, deterministic): exclude all 12 golden-12 embryos; for each golden-12
embryo pick the **same-group nearest-density** pool embryo → val mirrors the joint (group,density)
distribution; second-nearest → a disjoint fold1 val; train = 24 density-spanning pool embryos
(12/12 group), disjoint from both vals.

| set | n | group | density min/median/max | mean |
|---|---|---|---|---|
| golden-12 TARGET | 12 | 6/6 | 48 / **304** / 820 | 358 |
| **fold0 VAL** | 12 | 6/6 | 49 / **303** / 786 | 349 |
| **fold1 VAL** | 12 | 6/6 | 49 / **314** / 759 | 350 |
| TRAIN (shared) | 24 | 12/12 | 38 / 240 / 1015 | 301 |

Density-quantile match vs golden-12 (near-identical):
| q | golden-12 | fold0 val | fold1 val |
|---|---|---|---|
| q10 | 61 | 61 | 61 |
| q25 | 80 | 79 | 80 |
| q50 | 304 | 303 | 314 |
| q75 | 604 | 595 | 606 |
| q90 | 694 | 654 | 666 |

## 4. Leak-freeness (verified)
- val0 ∩ golden-12 = **0**; val1 ∩ golden-12 = **0**; train ∩ golden-12 = **0**.
- val0 ∩ val1 = 0; train ∩ (val0 ∪ val1) = 0.
- Splits are **embryo-disjoint** → **zero shared annotated cells** (cells belong to one embryo).

## 5. Conclusion
`splits_screen_matched.json` **supersedes** `splits_screen_mini.json`: it is leak-free vs golden-12,
group-balanced 6/6, and density-matched to golden-12 (median 304, full 48–820 spread) → a screen on
its val is a **faithful, unbiased predictor of golden-12/LB**. 2 folds give a small-fold variance
estimate. Authoritative spec: `docs/screen_miniset_v2.md`.
