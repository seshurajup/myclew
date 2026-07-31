# 2nd Place Solution

First, I would like to extend my deepest gratitude to the competition organizing team and Kaggle staff for hosting this fascinating competition. Having previously worked in experimental sciences including organic chemistry, I am well aware of the challenges in obtaining molecular 3D structural data (ground-truth structures). Considering this, I'm deeply impressed that they were able to hold this Part II competition just less than a year after the [Part I competition](https://www.kaggle.com/competitions/stanford-rna-3d-folding).
During these approximately two months of trial-and-error work, I gained both meaningful experience and valuable learning.

## Summary of My Solution

I built "**BPP-Protenix**", a model that integrates Base Pair Probability (BPP) features into an AlphaFold3-style architecture. The model design was inspired by the architecture of [RNAPro](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2/discussion/668412) shared by @theoviel. My final pipeline is an ensemble of TBM, RNAPro, and BPP-Protenix (with plain Protenix as fallback for targets not suitable for BPP calculation). Although I dropped one place in the private LB, after completing the development of BPP-Protenix, I held 1st place on the public LB for the final two weeks of the competition. I guess almost no other participants modified the neural network architecture in this competition, so this appears to be a unique solution.

**Submission slot allocation is follows**:

| slot 1 | slot 2 | slot 3 |                    slot 4                     |                     slot5                     |
| :----: | :----: | :----: | :-------------------------------------------: | :-------------------------------------------: |
|  TBM   | RNAPro | RNAPro | **BPP-Protenix**<br>(with plain Protenix fallback) | **BPP-Protenix**<br>(with plain Protenix fallback) |

- **TBM**: I used [the notebook](https://www.kaggle.com/code/kami1976/stanford-rna-3d-folding-part-2a18) published by @kami1976 almost exactly as is. Since it appeared to offer good local scores and seed stability, I decided to borrow this one.

- **RNAPro**: Using the template created by the above TBM as input data, I applied the [the pipeline](https://www.kaggle.com/code/jaejohn/rnapro-inference-with-tbm) published by @jaejohn.

- **BPP-Protenix**: This is the model that required the most time to develop for this competition. I'll explain it below.

## Core Idea: BPP as Structural Prior

### Inspiration: Ribonanza Competition (2023)

The idea came from the [Stanford Ribonanza RNA Folding](https://www.kaggle.com/competitions/stanford-ribonanza-rna-folding) competition in 2023. @shujun717 (one of the hosts of the Ribonanza competition) had published about [RNAdegformer](https://academic.oup.com/bib/article/24/1/bbac581/6986359), a transformer predicting mRNA degradation at nucleotide level. In that paper, **BPP (Base Pair Probability) matrix was shown to be a very important feature** for predicting RNA stability. Inspired by this, almost all top Ribonanza participants added BPP embeddings as bias terms in their Transformers.

### From Stability Prediction to 3D Structure Prediction

The task of RNAdegformer and Ribonanza models was stability prediction, but RNA stability depends a lot on 3D structure. Furthermore, the base pair formation probability information obtained through secondary structure prediction provides proximity constraints on the 3D structure, thereby functioning as an effective inductive bias. Given that, providing BPP information to a 3D structure predictor seemed like a natural extension.

### BPP Injection into Pairformer

The BPP matrix `[N, N]` naturally corresponds to the pair representation `z [N, N, c]` in the Pairformer trunk of AlphaFold3/Protenix architecture — both are pairwise matrices indexed by residue positions. By passing BPP through a simple linear layer (1 → c) and adding the resulting embedding to `z_init`, we can inject RNA structural knowledge directly into the model.

```
BPP [N, N]
  → unsqueeze → [N, N, 1]
  → LinearNoBias(1 → 128) → [N, N, 128]
  → z_init += bpp_embedding
```

## BPP Calculator Selection: EternaFold vs ViennaRNA

I evaluated BPP quality against experimental 3D coordinates from the training data, checking if residue pairs with high BPP are actually close in 3D space. [ViennaRNA](https://github.com/ViennaRNA/ViennaRNA) and [EternaFold](https://github.com/WaymentSteeleLab/EternaFold), two of the primary RNA secondary structure computation tools, were tested. These secondary structure prediction tools assume RNA monomers as the target of prediction. Additionally, calculating BPP for long sequences during a Kaggle session is difficult in inference. Trherefore, it was considered that the training should also be performed using shorter sequences. The following filter was applied to the approximately 5,000 sequences of Kaggle's official training data:

**Filter conditions**
- RNA monomer (no multimer)
- Sequence length ≤ 1,000

The BPP quality was evaluated using the **1,532** sequences remaining after applying this filter. Here, a C1'-C1' distance < 12 Å was defined as "contact." However, this judge was only conducted for C1' combinations separated by 4 or more residues (because such residues exist spatially close regardless of base pair formation).

|                          | ViennaRNA | EternaFold |
| ------------------------ | --------- | ---------- |
| Contact rate (BPP ≥ 0.7) | 81.4%     | **96.2%**  |
| Contact rate (BPP ≥ 0.9) | 86.6%     | **97.4%**  |
| Random baseline*         | 3.6%      | 3.6%       |

*Random baseline: contact rate when residue pairs are selected randomly (no BPP information).

EternaFold BPP was clearly more accurate. Also, it is easily deployable in Kaggle notebooks via the [arnie](https://github.com/DasLab/arnie) library. So I chose EternaFold for feature engineering.

---

## BPP-Protenix training

### Setup

- **Pretrained checkpoint**: `protenix_base_20250630_v1.0.0`
- **Train data**: 619 PDBs (RNA-only)
  - From the 1,532 RNA monomers (≤1000 nt) used for BPP quality evaluation mentioned above, PDBs with protein/DNA partners were excluded.
- **Validation data**: 33 PDBs
  - Combined Kaggle's official validation set with my own collected PDBs released between 2025-12-03 and 2026-02-18. After applying the same filter, 33 records remained.

### Hyperparameters

| Parameter            | Value                           |
| -------------------- | ------------------------------- |
| diffusion_batch_size | 32                              |
| train_crop_size      | 550                             |
| lr                   | 1e-4                            |
| EMA decay            | 0.995                           |
| num_steps            | 4960                            |

---

## BPP-Protenix Inference

### BPP-Available Target Filter & Fallback to Plain Protenix

The following filters were applied to determine if the target was appropriate for BPP calculation by EternaFold:
- RNA monomer (no multimer)
- Total entity length ≤ 850 tokens
- No protein/DNA partners

Targets not passing the filter were predicted by **plain Protenix** as fallback. Targets exceeding 850 tokens were cropped to 850 residues, with remaining coordinates zero-padded.

### BPP-Protenix Standalone Performance

Before evaluating the ensemble, I assessed BPP-Protenix alone (all 5 slots filled with Protenix predictions). Inspired by RNAPro's TemplateEmbedder (Protenix v0.5), I tested three integration methods (A,B,C).

- **Method A**: just a linear layer from 1 to 128 dimensions, then embedding is added to `z_init`.
- **Method B**: In addition to the direct linear layer, I also converted BPP into a 20-dimensional one-hot vector by binning, embedded it with another linear layer, and added both features to z_init.
- **Method C**: In addition to method B, uses a BppEmbedder. It is the similar design pattern as TemplateEmbedder in RNAPro. Applies a small 2-block Pairformer before the main 48-block Pairformer in each recycling cycle.

| Method                   | Architecture                                                 | Public LB (standalone*) |
| ------------------------ | ------------------------------------------------------------ | ----------------------- |
| **A (Linear)**           | LinearNoBias(1→128), add BPP-emb to z_init                 | **0.340**               |
| B (+ Binning)            | Method A + one_hot(20 bins) → Linear(20→128),  add two embs to z_init | 0.317                   |
| C (+ BppEmbedder)        | Method B + BppEmbedder in recycling loop                     | 0.320                   |
| Plain Protenix (control) | -                                                            | 0.309                   |

*Standalone: all 5 slots filled with BPP-Protenix + plain Protenix fallback

The standalone BPP-Protenix (Method A) scored **0.34** on public LB. This is comparable to TBM-only public notebooks (~0.35), demonstrating that **NN-based predictions can match template-based approaches**.

Interesting finding: allowing protein/DNA partner targets in the BPP filter caused almost no change in public LB score. Possibly few such targets existed in the test set. Worth investigating the performance for RNA/protein assembly with late submissions.

---

## Pseudoknot Analysis

**Concern**: EternaFold does not predict pseudoknots. Could BPP-Protenix is not suitable for potential pseudoknot targets?

**Post-hoc validation**:
- Detected pseudoknots from ground-truth 3D coordinates using biotite (base pair detection + crossing pair check)
- Validation set (11 RNA-only targets): pseudoknots found in **7 out of 11** targets
- BPP-Protenix outperformed plain Protenix by **~0.06 TM-score** on average across all 11 targets
- Contrary to expectations, even within the pseudoknot group, BPP-Protenix showed higher TM-scores

**Interpretation**: BPP provides a "pairing tendency hint" as continuous values, not a complete secondary structure prediction. Even if pseudoknot base pairs are missed, non-pseudoknot BPP information is still valuable. And maybe, the 48-block Pairformer can absorb BPP inaccuracies.

Based on the local validation result, I have decided not to take any special measures specifically targeting pseudoknots.

---

## Ensemble Pipeline

### TBM (Slot 1) and RNAPro (Slot 2–3)
- Used a public notebook shared early in the competition as-is

### BPP-Protenix (Slot 4–5)
- Targets passing the BPP-available filter → BPP-Protenix (seed=101, N_sample=5, top 2 by confidence score)
- Targets failing the filter → Plain Protenix fallback (same config, no BPP, Longer targets were cropped to 850 tokens and zero-padded)

### Public/Private LB Results

| Configuration                                     | Public LB             |Private LB            |
| ------------------------------------------------- | --------------------- |--------------------- |
| TBM 2slot / RNAPro 3slot (baseline)               | ~0.43                 |—                     |
| TBM 1slot / RNAPro 2slot / Protenix 2slot         | 0.461                 |0.475                 |
| **TBM 1slot / RNAPro 2slot / BPP-Protenix 2slot** | **0.504 (1st place)** |**0.492 (2nd place)** |

- BPP-Protenix yielded **+0.04** over plain Protenix and **+0.07** over the TBM+RNAPro baseline in the public LB score
- BPP-Protenix demonstrated strong accuracy on the private test set as well

## Acknowledgments

Thank you to the competition hosts (Rhiju Das and team) for organizing this fascinating competition. Thanks also to the Kaggle community for sharing public notebooks (TBM, RNAPro) that formed the foundation of my ensemble.