# Acknowledgments
As a graduate student in bioinformatics at University of Science and Technology of China, "I have learned a lot from this competition, and it is a great help to me and my future. I am grateful for the help of my teammates and believe this was a great collaboration. Most of the open-source notebooks in this competition were built based on Qiwei's dataset; he has extraordinary code insight. Flop provided great ideas for our code improvement in the later stages. Other teammates also cleared up our confusion with professional RNA domain knowledge and provided computational power support.
Congrats to all the winners and participants!Here's a quick summary of what I did.

# conclusion
## Phase 1: Template-Based Modeling
We searched the combined training and validation set using Biopython's PairwiseAligner with RNA-tuned gap penalties. Atomic coordinates were transferred to the query using a geometry-aware gap-filling procedure: missing residues are interpolated along RNA-like helical trajectories (C1'–C1' step ~5.9 Å) rather than placed at zero or left as NaN. Diversity transforms — hinge rotations, chain jitter, smooth wiggle — are applied slot-by-slot from the same template source, followed by RNA geometry constraint refinement (bond-length correction + Laplacian smoothing + self-avoidance).
To avoid slot redundancy, we apply TM-score-based greedy diversity selection: from all TBM candidates, we greedily pick structures that maximize a quality-diversity trade-off score, ensuring the two TBM slots are structurally distinct rather than near-duplicates of the top hit.
Slot allocation is length-adaptive:
Short sequences (≤ 512 nt): max 2 TBM slots, 3 reserved for Protenix
Long sequences (> 512 nt): TBM-first, up to 5 TBM slots if sufficient templates exist

## Phase 2 Core Insight: MSA Depth
The mainstream open-source approach treats MSA as a binary switch — either full MSA or no MSA — . I argue this framing discards useful structure in the MSA quality signal.
My analysis of the 28 test targets revealed a clear depth distribution:
Depth ≥ 700  : 15 targets (54%) — strong evolutionary signal
Depth 10–699 :  7 targets (25%) — moderate, noise risk
Depth ≤ 9    :  6 targets (21%) — MSA nearly uninformative

I ran ablations across thresholds (100 / 300  / 700 / 1500) and found depth ≥ 700 to be the empirically optimal cutoff: below it, distant homologs inject alignment noise that degrades diffusion quality; above it, evolutionary co-variation provides reliable structural constraints.
This leads to our three-mode Protenix inference design:
~~~~
Slot 3: msa700 sample 1  — official MSA, depth ≥ 700
                            high-fidelity evolutionary constraint
Slot 4: msa700 sample 2  — same MSA policy, independent diffusion
                            diversity from stochastic diffusion,
                            NOT from MSA noise
Slot 5: msa_full         — relaxed threshold, depth ≥ 1
                            intentional diversity hedge:
                            noisy-but-different MSA features
                            only deployed AFTER slots 3&4 are secured
~~~~
The key distinction from the binary on/off approach: slots 3 and 4 share identical high-quality MSA features; diversity between them comes purely from independent diffusion sampling. Slot 5 is a deliberate hedge that accepts some MSA noise in exchange for a structurally distinct prediction.
For long sequences (> 512 nt), only msa700 is used with N_sample=3 — three independent diffusion samples in a single forward pass, sharing the expensive feature computation while diversifying outputs.
For targets requiring multi-chunk inference, we replaced the standard linear weight ramp with a quintic smoothstep function $$w(t) = 6t^5 - 15t^4 + 10t^3, \quad t \in [0, 1]$$. While linear blending results in first-derivative discontinuities at chunk boundaries, our approach guarantees  c2(curvature) continuity. Combined with Kabsch-based reference alignment in the 128-nt overlap regions, this eliminates geometric 'kinks' and ensures the global global topology remains smooth and physically valid.

# Phase 3
Standard Protenix deployments produce different outputs across runs due to non-deterministic CUDA operations, MSA subsampling, and floating-point ordering effects. We enforced full reproducibility through three mechanisms:
1.Disabling non-deterministic attention backends — FlashAttention and memory-efficient attention are both switched off, forcing deterministic standard attention
2.Disabling MC Dropout (mc_dropout_apply_rate=0.0) to remove inference-time stochastic regularization
3.Forcing torch-native LayerNorm and triangle attention kernels instead of cuDNN/cuEquivariance backends, which have non-deterministic implementations
This guarantees that any improvement in our pipeline comes from algorithmic changes, not RNG variation — a critical property for reliable ablation studies in a competition setting.

~~~~
test_sequences.csv
       |
       v
+----------------------+
|       Phase 1        |   PairwiseAligner, top_n=30
|         TBM          |   geometry-aware gap filling
|   Diversity Selection|   TM-score greedy selection
+----------------------+
       |
       |  short (≤512 nt): 2 TBM slots
       |  long  (>512 nt): up to 5 TBM slots (TBM-first)
       v
+----------------------+
|       Phase 2        |   Dual-GPU Protenix
|      Protenix        |
|   3 inference modes: |   slot 3: msa700 sample 1
|   msa700_1           |           (depth≥700, deterministic)
|   msa700_2           |   slot 4: msa700 sample 2
|   msa_full           |           (independent diffusion)
|                      |   slot 5: msa_full
|                      |           (depth≥1, diversity hedge)
+----------------------+
       |
       | long seqs (>512 nt):
       | chunk-level greedy bin-packing across 2 GPUs
       | → Kabsch SVD align + smoothstep stitch
       v
  submission.csv
  (5 slots per target, fully deterministic & reproducible)
~~~~

Reference

[1][https://www.kaggle.com/code/qiweiyin/protenix-v1-inference-2026](url)

[2][https://www.kaggle.com/code/nihilisticneuralnet/0-409-stanford-rna-folding-2-protenix-template](url)

[3][https://www.kaggle.com/code/alexxanderlarko/protenix-v1](url)