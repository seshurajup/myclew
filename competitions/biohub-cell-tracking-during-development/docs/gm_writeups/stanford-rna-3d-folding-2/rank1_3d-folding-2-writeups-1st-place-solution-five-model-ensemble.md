## Acknowledgements

We are team_cp ( @naganohikaru, @yutaroito), a two-person team.  
This is our second time participating in the Stanford RNA 3D Folding series; we placed 25th in Part 1.  
The experience from Part 1 — particularly understanding which model combinations held up on unseen data and learning the power of template-based modeling — gave us a strong foundation for this competition.  
  
We are deeply grateful to the competition organizers, Stanford University, and the many Kagglers who shared public notebooks and insights throughout the competition!

---

## Summary

Our solution combines five independent structure prediction models into a single 5-prediction submission, with each model's allocation driven by sequence length.  
Since scoring is best-of-5, the goal is to maximize structural diversity across the five predictions.   
### The five models are:

| Model | Role |
|---|---|
| **TBM** (Template-Based Modeling) | Primary predictor for seq_len ≥ 250; fallback for all lengths |
| **Boltz2** | Primary predictor for seq_len < 250; secondary for 250–999 |
| **RNAPro** | Diffusion-based refinement for seq_len < 1000 |
| **DRFold2** | RNA-specific deep learning for short sequences |
| **Protenix** | AlphaFold3-based; short sequences and ≥ 1000 nt |
  
  
### Final ensemble composition (5 predictions per target):
| seq_len | Pred 1 | Pred 2 | Pred 3 | Pred 4 | Pred 5 |
|---|---|---|---|---|---|
| < 250 | Boltz2₁ | Boltz2₂ | RNApro₁ | Protenix₁ | DRFold2₁ |
| 250 – 999 | TBM₁ | Boltz2₁ | RNApro₁ | RNApro₂ | Boltz2₂ |
| ≥ 1000 | TBM₁ | TBM₂ | TBM₃ | Protenix₁ | Protenix₂ |

RNApro and Boltz2 skip sequences ≥ 1000 nt (OOM risk), so the ≥ 1000 nt tier uses TBM for 3 predictions and Protenix for the remaining 2.

### Final scores (selected submission):

| Comp. Public LB | Comp. Private LB | Final Public LB | Final Private LB |
|---|---|---|---|
| 0.43837 | 0.60037 | 0.42854 | **0.49669** |

---

## Phase 1: Template-Based Modeling (TBM)

TBM is the backbone of our pipeline for medium and long sequences.  
We search a combined train + validation pool using BioPython's `PairwiseAligner` (global mode), filtering to templates whose length is within 30% of the query length. Alignments are scored and normalized; the top-30 templates form the candidate pool.

**Diversity within TBM predictions:**
- Prediction 1: top-scoring template, used as-is.
- Predictions 2–5 (when TBM fills multiple slots): sampled from the top-12 with exponential weighting. Already-used templates receive a 0.1× weight penalty to discourage duplication.

**Gap filling:**
- Aligned gaps are filled by linear interpolation between the two nearest valid C1' coordinates.
- One-sided gaps (at chain termini) are extended by 5.95 Å steps along the extrapolated direction.
- If no template exists at all, a fallback linear structure is placed along the X-axis at 5.95 Å steps.

**Post-processing (`adaptive_rna_constraints`):**
Each TBM prediction is refined per-chain with:
1. Bond length correction: i→i+1 distance → 5.95 Å
2. Angle correction: i→i+2 distance → 10.2 Å
3. Laplacian smoothing to remove kinks
4. Steric self-avoidance (3.2 Å threshold) for chains of length ≥ 25

The correction strength scales with `(1 − template_similarity)` so that near-identical templates are barely modified while low-similarity matches are corrected more aggressively.

---

## Phase 2: DRFold2

DRFold2 (cfg97 configuration, 20 models) provides an RNA-specific deep-learning signal, contributing 1 prediction for sequences shorter than 250 nt.  
From Part 1, we confirmed that cfg97 outperformed other DRFold2 configurations.

Within that range, the number of models run scales with sequence length:
- L > 200 nt: 5 models `[13, 6, 14, 5, 3]`
- L > 100 nt: 10 models `[13, 6, 14, 12, 7, 2, 5, 19, 10, 9]`
- L ≤ 100 nt: all 20 models

For each model's output, we compute an energy score (bond lengths, bond angles, stacking) and normalize by the median absolute deviation across predictions. The top-5 scoring structures are selected.

---

## Phase 3: RNAPro

RNAPro (NVIDIA's Part 1 winner) is used for all sequences under 1000 nt.  
It contributes 2 predictions for mid-range sequences (250–999 nt) and 1 prediction for short sequences (< 250 nt), each conditioned on a different TBM template (top-1 and top-2).

Configuration:
- MSA enabled (`--use_msa true`)
- Pre-computed TBM templates (`--use_template ca_precomputed`)
- RibonanzaNet2 embeddings
- N_STEP = 200, N_CYCLE = 10, N_SAMPLE = 1

---

## Phase 4: Boltz2

Boltz2 is the primary model for short sequences (< 250 nt), contributing 2 predictions, and contributes 2 predictions for mid-range sequences (250–999 nt) as well.  
It is skipped for sequences ≥ 1000 nt to avoid OOM.

- `diffusion_samples = 10` structures generated per target
- Top-5 selected by combined plddt + confidence score
- Ligand information (SMILES) included in YAML input
- Multi-chain targets: only RNA chains are folded (`use_all_chains = False`) to avoid OOM with large complexes

---

## Phase 5: Protenix

Protenix (ByteDance, AlphaFold3 architecture) contributes 1 prediction for short sequences (< 250 nt) and 2 predictions for very long sequences (≥ 1000 nt).

- RNA MSA enabled (`USE_RNA_MSA = true`)

**Long-sequence handling (≥ 1000 nt):**

Protenix has a practical per-inference memory limit, so sequences longer than `MAX_SEQ_LEN = 512` nt are processed with a sliding-window chunking strategy:

1. **Chunking** — `split_sequence(seq, max_len=512, overlap=128)` divides the full sequence into overlapping windows.
   The step size is `512 − 128 = 384` nt, so a 1000 nt sequence produces three chunks: `[0:512]`, `[384:896]`, `[768:1000]`.

2. **Per-chunk inference** — each chunk is fed to Protenix independently as a separate short-sequence target (its own JSON input and `InferenceDataset`).
   `update_inference_configs` is called with the chunk token count so internal model buffers are sized correctly.

3. **C1' extraction** — from each chunk's raw atom-level output, C1' atoms are identified by matching `res_id` (1-indexed within the chunk) and `atom_name == "C1'"`.
   A two-stage fallback is used: first `centre_atom_mask`, then `atom_to_tokatom_idx == 12`.

4. **Assembly** — extracted C1' coordinates `(N_sample, chunk_len, 3)` are written into a pre-allocated `full_coords (N_sample, seq_len, 3)` array at `[:, start:end, :]`.
   Overlapping positions are overwritten by the later (right-side) chunk.

5. **Memory cleanup** — `torch.cuda.empty_cache()` and `gc.collect()` are called after every chunk to avoid GPU OOM on very long targets.

---

## What Worked

### The combination of five structurally diverse models.  
From Part 1 we confirmed that ensembling Boltz, Protenix, and DRFold2 was robustly better on unseen sequences than any single model alone.  
Building on that, Part 2 added RNAPro (the Part 1 winner) as a powerful refinement step, which significantly improved our mid-range sequences.

### TBM as the strong backbone for medium and long sequences.  
TBM proved highly competitive whenever a similar sequence existed in the training data.  
For sequences ≥ 250 nt, allocating multiple predictions to TBM and using other models to cover the remaining predictions was consistently better than the reverse.

### Length-adaptive allocation.
Rather than a one-size-fits-all ensemble, we tuned how many predictions each model contributes based on empirical performance at each length range.  
The "overlay" structure of the pipeline (models are run in order and later models overwrite earlier ones) made this easy to tune incrementally.  
The 250 nt boundary was selected by balancing inference time against score: models like Boltz2 and DRFold2 are fast enough for short sequences within the time limit but become impractical beyond this point, while TBM scales well to longer sequences at low cost.

### Diversity over TBM dependency.  
We treated TBM as a backbone only where it is clearly strongest (longer sequences with close training-set matches), and filled the remaining predictions with diverse models.  
For short sequences (< 250 nt), we deliberately avoided TBM as the primary predictor and instead relied on Boltz2, DRFold2, and Protenix — models that make predictions purely from sequence without template dependency.  
For the remaining slots outside of short sequences, we combined RNAPro — which starts from a TBM template but refines it into an independently generated structure — with template-free models (Boltz2 and Protenix), ensuring diversity across all five predictions.  
We believe this diversity contributed to our performance on the private leaderboard, which contained harder targets where no close template existed.

---

## Authors

team_cp: @naganohikaru, @yutaroito

---

## References

- [Our Part 1 solution (25th place)](https://www.kaggle.com/competitions/stanford-rna-3d-folding/writeups/25th-place-solution) — baseline ensemble strategy and lessons learned carried into Part 2.
- [Part 2 solution discussion](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/discussion/668412) — competition discussion thread for this solution.
- [Protenix v1 inference notebook](https://www.kaggle.com/code/qiweiyin/protenix-v1-inference-2026) — public notebook that formed the basis of our Protenix integration.
- [Beginner-to-advanced RNA 3D structure prediction](https://www.kaggle.com/code/qamarmath/beginner-to-advanced-rna-3d-structure-predicti) — general pipeline reference used during development.