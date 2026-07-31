# Plan-DA — Heavy domain-invariant aug for cross-embryo GENERALIZATION (design, 2026-07-10)

**Status:** designed + elastic implemented/unit-tested, configs dryrun-GREEN, **GPU-PARKED (do NOT launch —
gated on human GO / the 156 LB).** The SOLE forward lever past 0.897.

## Why this is the only lever left
The honest OOD gap (**~0.73 embryo-disjoint LOEO vs 0.897 in-distribution**) is the real unsolved problem
and likely decides the PRIVATE LB (unseen test embryos). Every LEARNED lever OVERFIT the 1-embryo LOEO — the
recurring wall: convergence (+0.017 fold0 → **−0.023 fold1**, mean −0.003), the windowed temporal linker
(fit the train embryo, no transfer). Non-learned levers are exhausted (we run the SOTA pilk_post==lb897
pipeline; [[honest-loeo-linker-bound-headroom]]). The detector **keys on the training embryo's appearance**
(44b6 late/dense/large nuclei vs 6bba early/sparse/distinct; per-embryo staining/optics). Domain-invariant
aug is the principled counter: **manufacture the 44b6↔6bba appearance gap SYNTHETICALLY** so the detector
can't overfit one embryo's look → generalizes to the unseen one.

## The augs (grounded in the 44b6↔6bba gap)
Baseline = the GENTLE E50 aug (loeo_conv_baseline_f0). Plan-DA AMPLIFIES the axes that separate the embryos:
- **PHOTOMETRIC (staining/optics — the biggest embryo gap), HEAVY:** brightness 0.5/0.08→**0.7/0.15**,
  contrast 0.5/0.15→**0.7/0.35**, gamma 0.3/0.25→**0.6/0.5**, bias_field 0.3/0.2→**0.6/0.4**. These break the
  detector's reliance on the training embryo's intensity signature.
- **ELASTIC (NEW — nucleus-SHAPE / local-geometry invariance):** `elastic` p0.5 alpha2.5 ctrl4. In-plane
  smooth random displacement (Z untouched, anisotropy-safe), warps image + GT coords consistently.
  Manufactures the nucleus-morphology variation between early-6bba and late-44b6.
- **DENSITY (crowding gap):** scale 0.3/0.12→**0.5/0.25** (bridge dense↔sparse), crop min_frac 0.65→0.55,
  cutout n1→n2 max_frac0.2→0.3.
- **SENSOR (PSF/read-noise):** blur 0.2/0.6→**0.4/1.0**, noise 0.2/0.02→**0.5/0.05**.
- Geometric flip+rot90 always on (free).

### `elastic_augment` — implemented + unit-tested (net-new)
`research/official_repo/scripts/augmentations.py`: in-plane smooth field (low-res random control points
upsampled), image via `grid_sample(output[p]=input[p+d(p)])`, coords via `q−d(q)` (consistent to 1st order).
CPU unit-test: image feature + its GT coord move **together** — dist 0.72px @alpha2, 1.64px @alpha2.5 (both
sub-voxel-ish, well within the 7µm=~4.3vox match gate); alpha≥4 degrades tracking (2.8px) so alpha is
**capped ≤3**. Registered as `"elastic"`; default OFF (only used when a config lists it).

## Exact A/B (one variable = augmentation strength)
- **Treatment:** `config/loeo_domaug_f{0,1}.yml` (heavy aug above).
- **Control:** `loeo_conv_baseline_f{0,1}` (GENTLE E50 aug) = the 150it floor (fold0 0.7152 / fold1 0.7322,
  mean **0.7237**).
- Everything else IDENTICAL (same detector, 150it/12ep, leak-clean val-holdout, canonical scorer, same SCIP
  pipeline). Screen→confirm: fold0 first; promote only on 2-fold MEAN > 0.7237.

## The OVERFIT-REDUCTION readout (the real signal, not just the score)
The hypothesis is *generalization*, so the primary readout is whether cross-embryo overfit SHRINKS:
1. **Train-val vs test gap:** baseline peaks on the train-embryo val early then the unseen-embryo test decays
   (overfit). Heavy aug should make the val curve peak LATER / decay LESS.
2. **Fold asymmetry:** the "fold0-win-inverts-on-fold1" pattern (convergence +0.017/−0.023) should SHRINK —
   if aug generalizes, both folds move together, not oppositely.
3. **2-fold MEAN vs 0.7237:** the bottom line. Even a small mean lift = the FIRST lever that generalizes
   (unlike convergence/linker), and it's the one most likely to lift the PRIVATE LB (OOD test).
Report all three, not just the score.

## Est GPU wall-time
Baseline 150it/12ep ≈ ~40 min/fold. Heavy aug adds per-iter cost (elastic grid_sample + more photometric
ops) ≈ +20–40% → **~50–55 min/fold**, **~2 h for the 2-fold screen** (+ possible NaN-skip overhead).

## Dev risk (flagged)
1. **fp16 STABILITY (the #1 risk):** the memory notes the full aug stack at high strength overflowed the
   fp16 UNet forward (det=NaN) — which is WHY the E50 baseline is gentle. The trainer has a NaN-batch-skip +
   input safety-net, but heavy aug → more skips → wasted iters. **The 1-iter GPU smoke MUST report the
   NaN-skip rate**; if high, back off magnitudes or enable AMP/fp32 for the detector head.
2. **Elastic coord noise:** alpha≤3 keeps coord-tracking error <2px (<3.3µm << 7µm gate) → sub-gate label
   noise, acceptable; alpha>3 is capped out.
3. **Over-aug underfit:** too-heavy aug can prevent the detector from learning at 150it. If the train loss
   won't descend, reduce probabilities (curriculum) — watch the smoke's train-loss trajectory.

## Deliverables / plan
1. Design (done) + `elastic_augment` implemented+unit-tested + `config/loeo_domaug_f{0,1}.yml` dryrun-GREEN
   (both parse the 12-aug block incl elastic). 2. On human GO: 1-iter GPU smoke (assert train-loss descends +
   report NaN-skip rate) → route fold0@150 → overfit-reduction readout → if the mean lifts + gap narrows,
   fold1 → 2-fold verdict. Promote gate: 2-fold MEAN > 0.7237. **NOT LAUNCHED.**

## Provenance
Augs: `research/official_repo/scripts/augmentations.py` (elastic new). Overfit wall:
[[gap-decomposition-detector-is-lever]], [[honest-loeo-linker-bound-headroom]]. Floor: loeo_conv_baseline_f{0,1}
(EXP_161=0.7152 / EXP_163=0.7322).
