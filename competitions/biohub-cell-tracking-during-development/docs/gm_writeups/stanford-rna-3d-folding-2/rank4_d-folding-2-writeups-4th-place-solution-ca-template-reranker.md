# 4th Place Solution — Cross-Attention Template Reranker + Protenix + RNAPro

## Summary

This was my first bioinformatics competition, and more generally my first serious Kaggle competition overall. I was curious to see how far I could go without domain expertise, and with AI tools as my main leverage. Big thanks to the organizers for the huge amount of work they put into Part 2, to the community for the excellent public datasets and kernels, and congratulations to everyone on the prize list!

The pipeline is **Protenix + RNAPro + template search reranked by a Cross-Attention TM predictor**. The core idea everything else hangs off is a **CUDA port of US-align** that made it cheap to label TM-scores for a very large number of template–template pairs (~1.42M). That labeled set is what the Cross-Attention model is trained on, which turns "take top-N by sequence identity" into a ranking that reorders candidates by **predicted structural TM**.

## Slot allocation (5 predictions per target)

| Slot | Source |
|------|--------|
| 1 | TBM — **Bio top-1** (`Bio.Align.PairwiseAligner` in global mode, competition-tuned scores) |
| 2 | TBM — **Model top-1** (Cross-Attn predicted TM, mmseqs-cluster-diverse from slot 1) |
| 3 | TBM — **Combined top-1** (`sim * model_score`, cluster-diverse from slots 1-2) |
| 4 | **Protenix v1** (co-fold with protein/DNA/ligands if they fit) |
| 5 | **RNAPro** |

For targets longer than 512 nt the Cross-Attn model isn't run (it was trained up to 512) and slots 2-3 fall back to Bio ranks 2 and 3 with cluster diversity.

## GPU US-align (the labeling engine)

https://www.kaggle.com/code/gapchenko/gpu-accelerated-usalign/notebook

I re-implemented `TMalign_main` from US-align in CUDA — faithful to the CPU source. All five phases (initial alignment, SS alignment, local fragment superposition, SS + distance matrix, fragment gapless threading) run on GPU; `DP_iter_gpu`, `gpu_TMscore8_search`, and `get_initial5_gpu_fast` are the heavy kernels. A binary stdin batch mode takes coordinate pairs directly via `struct.pack` — no PDB I/O. A separate PDB-batch mode uses a CPU thread pool with one thread per CUDA stream, so the GPU stays saturated while files are parsed.

On P100-class hardware this gets to hundreds of milliseconds per RNA pair, vs. seconds on the CPU binary — with the biggest speedup on longer molecules (>2k nt). That turned a "days on CPU" labeling problem into a "hours on one GPU" one, and is the prerequisite for the Cross-Attention predictor below.

**What got labeled**: every pair of available templates in the DB with length ratio in [0.75, 1.25] (~1.42M pairs). These pairs together with their C1' ground-truth coordinates are the supervised signal — TM-score is what I want the network to predict.

## Cross-Attention TM predictor

The model takes **two nucleotide sequences** (query and template) as independent inputs. Each passes through a shared encoder (embed -> 5 self-attention layers); the two encoded sequences then interact through one bidirectional cross-attention block. The query is pooled, fused with a small context vector, and mapped to a single logit for `P(TM >= 0.5)`.

```
        query seq                  template seq
      (N <= 512 nt)              (M <= 512 nt)
           |                           |
           |   shared encoder applied  |
           |   to each independently   |
           v                           v
    embed(vocab=5, d=128) + learned pos(max_len=512, d=128)
           |                           |
           v                           v
    5 x [ self-attention (h=4, d=128) + FFN ]
           |                           |
           v                           v
       q_enc (N, 128)           t_enc (M, 128)
           \                         /
            \                       /
             v                     v
      +---------------------------------+
      |   bidirectional cross-attention |
      |     q <- q + MHA(Q=q, K=t, V=t) |   # q attends to t
      |     t <- t + MHA(Q=t, K=q, V=q) |   # t attends to q
      +-----------------+---------------+
                        |   (t is then discarded)
                        v
              FFN + LayerNorm on q
                        |
                        v
            masked-mean pool over N
                        |
                        v
                pooled in R^128
                        |
                        |     context vector (10d):
                        |       4 common feats:
                        |         log(q_len)/6, log(t_len)/6, q_len/(t_len+1), Levenshtein similarity
                        |       6 auxiliary legacy feats (filled with zeros):
                        |         
                        |                       | 
                        |                       |
                        |                Linear(10 -> 128)
                        |                       |
                        +------- concat <-------+
                                 (256d)
                                   |
                                   v
                        Linear(256 -> 128) -> GELU
                                   |
                                   v
                           Linear(128 -> 1)
                                   |
                                   v
                                sigmoid
                                   |
                                   v
                             P(TM >= 0.5)
```

d = 128, 4 heads, 5 self-attention layers, 1 bidirectional cross-attention layer. Embedding is plain learned (vocab of 5 for `AUGC` + pad), positions are learned up to 512.

**Training.**
- Labels: 1.42M template-template pairs from GPU US-align, binarized at `TM >= 0.5`.
- Split: `mmseqs_0.300` clusters — 15% of clusters held out for validation; train pairs must cross clusters, val pairs must involve at least one val-cluster target.
- Filters: length ratio in [0.75, 1.25], TM in [0.12, 0.92] (drop trivially similar and trivially different), negatives downsampled to 1/3 of positives.
- Loss: `BCEWithLogitsLoss(pos_weight)` to compensate for residual class imbalance.
- Optimizer: AdamW (lr = 3e-4, wd = 1e-4), cosine schedule, early stop on val Spearman.
- This config (d=128, heads=4, layers=5) was the winner of a small architecture sweep over depth (1-8), width (24-192), and heads (2-8).

**Inference.** For each test target take top-500 templates by sequence similarity, score each one with
      the CA predictor in batches of 64, sigmoid the logits — that's the `model_score`. Slot 2 picks `argma
      x(model_score)` in a fresh `mmseqs_0.300` cluster; slot 3 picks `argmax(sim × model_score)` in yet ano
      ther fresh cluster. The three TBM slots thus cover three different reasons to like a template: high sequence identity, high predicted TM, and both agreeing at once.

## Protenix & RNAPro integration

Both run in-memory — model loaded once, configs updated per target, no subprocess restart. One caveat up front: **according to my comparisons, the marginal benefit of explicit MSA usage turned out to be minimal for both Protenix and RNAPro on this eval setup** — so a lot of the MSA wiring below is really about *not breaking* the models rather than squeezing more out of MSA.

On top of the public inference recipes:

- **Co-fold.** Non-RNA chains go in if they fit the token budget; if they miss by ≤100 tokens the largest protein ≥500 aa is trimmed symmetrically on both ends (and its MSA is column-trimmed to match).
- **Ligand filtering.** Buffer and crystallization artifacts (`GOL`, `EDO`, polyethylene glycols `PEG`/`PG4`/`P6G`/`1PE`/`PE4`/`EPE`, `SO4`, `PO4`, salts and buffers `CIT`/`ACT`/`FMT`/`ACY`/`TRS`/`MES`, solvents `MPD`/`IPA`/`DMS`/`BME`, polyamines `SPM`/`SPD`, crystallographic heavy-atom probes `NCO`/`IRI`/`RHD`, `HEZ`) are blacklisted. Everything else is injected as a `CCD_*` entity.
- **Heterodimer handling.** Multi-chain RNA targets use a separate `rnaSequence` entity per distinct chain, with the competition MSA column-sliced per chain.
- **Long sequences.** Overlapping chunks, per-chunk MSA column slicing, Kabsch alignment on the full overlap, and a narrow ±6-residue linear blend around each handover midpoint (hard switch outside that window — averaging over long overlaps hurt on my val).
- **Time budgets.** Both phases have wall-clock budgets with adaptive per-target limits so that a single slow target can't starve the rest.

## Findings which worked

- **GPU US-align → CA predictor.** The main lever. Without cheap TM labeling there's no supervised signal to rank templates *structurally*. Without the rerank, slot 2 is "take rank-2 by sequence", which is noisy.
- **Diverse TBM slot filling (bio / model / combined). I believe this was the biggest single source of Bo5 gain.** Strictly better than any single ranking on my held-out targets — the bio rank anchors, the model rank finds structurally similar but sequence-diverse templates, the combined rank catches the cases where both agree.
- **mmseqs-cluster diversity inside TBM.** Requiring the three TBM slots to come from different `mmseqs_0.300` clusters was bigger than any scoring tweak — it eliminates near-duplicate templates that all fail the same way.
- **Keeping PTX and RNAPro as independent single-slot de-novo predictions** beat letting either model take 2 slots on any consistent subset of targets in my evals. It would probably be more efficient with a conditional slot allocator, but I didn't have enough time to tune such a selector to the point where it consistently earned its keep.

## What didn't work (or helped only a little)

- **Coordinate-space blending.** Whether uniform, learned, or anchor-based, averaging Kabsch-aligned templates moved scores a little on close templates and hurt on diverse ones. Pick-one-template was simpler and better.
- **Explicit secondary-structure features.** I tried folding SS in via common tools like ViennaRNA — both as extra inputs to the Cross-Attention selector and as constraints when stitching chunk-wise de-novo predictions. Neither meaningfully moved the score. Entirely possible this is a skill issue on my end: I'm not confident I was using SS in the right way.
- **Better chunk-to-chunk blending.** Tried various stitching schemes — linear vs cubic-spline blends, wider overlap windows, even an L-BFGS fit over the overlap region — but the underlying de-novo chunks were probably just not accurate enough for this to show up in the end score.
- **Metadata-based routing of PTX vs. RNAPro slots.** Code computes `ptx_heavy` / `rnapro_heavy` / `balanced` labels from length, protein/DNA presence, and keywords, but the final allocation is flat (1 PTX + 1 RNAPro for everyone). The routing labels are reported for statistics only. Attempts to actually route to 2-of-one model lost consistency on the val set.

## Attachments

  - https://www.kaggle.com/datasets/gapchenko/rna3d2-pairwise-tm-labels — 1.42M ground-truth TM-scores for pairs of RNA
  structures from the competition training set
  - https://www.kaggle.com/code/gapchenko/gpu-accelerated-usalign/notebook —CUDA port of TMalign_main from
  US-align. Includes performance and accuracy comparison against CPU original version.
  - train_tm_cross_attn.py — training script for the Cross-Attention TM predictor. Reference
  implementation showing the full pipeline end-to-end (data loading, feature construction, model, training loop, eval by
   Spearman).

## Acknowledgements

- Organizers for running Part 2 so soon after Part 1, and for the clean data release.
- `@theoviel` and `@jaejohn` for the baseline TBM and RNAPro notebooks that everyone built on.
- Protenix (ByteDance) and RNAPro (NVIDIA Digital Bio) for the open checkpoints.
- Claude (Anthropic) was my pair-programmer for most of this, especially for the CUDA port.