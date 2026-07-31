# baseline_v14 — PER-DENSITY min_track_len (design + mechanism)

**Date:** 2026-07-07 · **Author:** researcher · Runs parallel to the v13 fold0 GPU confirmation.

## Motivation
v13 per-embryo diagnostic: 44b6 wants mtl≥14, 6bba peaks 8–10 (opposite optima). A density-conditioned
mtl should beat global mtl10 on both. **But the win must be MECHANISTIC, not embryo-fit.**

## KEY CORRECTION — density ≠ embryo label
The per-embryo aggregate was misleading. True per-dataset density (`estN_per_frame`, `stage` in
`learning/03_true_density_stage.csv`) does NOT track embryo:
- **6bba spans S0 (estN/f=48) → S4 (698)** — mostly low, but one S4 high-density dataset.
- **44b6 spans S2 (172) → S4 (820)** — mostly high, but one S2 mid dataset.

(Note: `cpf_median` from `01_cells_per_frame_per_dataset.csv` is ANNOTATION density from sparse GT
(~1–18/frame) and is NOT true density — e.g. 44b6_0113de3b cpf_median=1 but estN/f=495. Use `stage`/estN.)

So binning by embryo would (a) over-fit the 2 embryos and (b) mis-assign the crossover datasets. Bin by
the mechanistic density proxy instead.

## Mechanism → mapping
High true-density ⇒ many spurious short tracks from over-detection ⇒ wants aggressive pruning (high mtl).
Low-density ⇒ few cells ⇒ aggressive pruning removes REAL short tracks ⇒ wants gentle mtl.

**2-bin mapping (from the 199-row `stage`, NOT fit to golden-12 scores):**
- `stage ∈ {S0,S1,S2}` → **mtl 10** (low/mid density)
- `stage ∈ {S3,S4}` → **mtl 14** (high density)
- gap fixed at 5.5.

Golden-12 lands 6/6 and CROSSES embryos (6bba_05db0fb1 S4 → high bin; 44b6_0db75fae S2 → low bin),
proving the split is density-driven, not embryo-driven.

## PoC test
Driver `baseline/run_experiments_v14_perdensity_mtl.py`: build gap5.5 anchor → prune all at mtl10 and at
mtl14 → 2-bin = select low datasets from the mtl10 output + high datasets from the mtl14 output. Score
golden-12 micro + per-embryo, compare vs global mtl10. Round-trip scale (deltas clean; in-memory ~+0.0014).

**Promote rule:** if the mechanistic 2-bin beats global mtl10 by +0.001 on golden-12, it becomes the
fold0/1 LOEO config alongside the global-mtl10 confirmation. golden-12 is PoC only — LOEO is the arbiter.
Results in `baseline_v14_perdensity_mtl_results.md`.
