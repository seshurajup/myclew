# Methods Survey: Closing the honest LOEO 0.73 → 0.897 gap

**Competition:** biohub-cell-tracking-during-development (3D+time light-sheet embryo microscopy).
**Metric:** adjusted edge Jaccard — an edge is TP only if BOTH endpoints match GT within 7 µm; `adj = J·(1 − 0.1·(predN − estN)/estN)`. Divisions scored separately (proven dead-end here).
**Current pipeline:** pilkwang UNet detector → edge-transformer linker → ILP (tracksdata/SCIP) → post-proc (min-track-len + gap-close).

## Where we actually are (target the RIGHT gap)

| Number | Value | Meaning |
|---|---|---|
| golden-12 CV | 0.9161 | Leaky, in-distribution. NOT the gap. |
| **honest embryo-disjoint LOEO** | **~0.73** | Real generalization. THIS is the gap. |
| Public LB target | ≈ 0.897 | Where we need to be. |

**Proven DEAD** (do not re-litigate): post-proc tuning, NMS / pool-kernel, recall-tilt, division classifiers, over-prediction fixes — all just slide the recall×precision Pareto or fail to transfer across embryos.
**Proven WORKS:** detector CONVERGENCE (more training iters) = +0.017 honest LOEO.

**Conclusion that frames this whole survey:** the LOEO collapse (0.9161 → 0.73) is a **cross-embryo domain-shift + detector-quality** problem, secondarily a **linking-quality** problem. `44b6` (late/dense, ≤1015 cells/frame) and `6bba` (early/sparse, ~40 cells/frame) are effectively two different domains. Every method below is judged first on: *does it improve detection/linking on an UNSEEN embryo?*

---

## 1. Recent detection + tracking architectures (2023–2026)

### 1a. Linajea / "sparse-annotation whole-embryo lineages" — HIGHEST RELEVANCE
Malin-Mayor et al., *Nature Biotechnology* 2022. [Paper](https://www.nature.com/articles/s41587-022-01427-7) · [PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614077/) · [code: funkelab/linajea](https://github.com/funkelab/linajea) · [zebrafish data](https://janelia.figshare.com/articles/dataset/Zebrafish_data_for_whole-embryo_lineage_reconstruction_with_linajea/24968724)

**What it is:** the closest published analogue to our exact task — 3D+time light-sheet embryo (mouse/Drosophila/**zebrafish**), trained from **SPARSE point annotations**, ending in ILP-selected lineages. A U-Net predicts two heads: (1) a **cell-indicator** Gaussian (max 1 at nucleus center), and (2) a **movement vector** per voxel pointing to the same nucleus in frame t−1. Candidate graph = local maxima as nodes, edges scored by agreement with the movement vector; global ILP selects nodes+edges.

**Why it targets our gap — two directly transferable ideas:**
- **Masked sparse loss.** They train the cell-indicator head *only inside a small radius around each annotation* (MSE within the mask), never penalizing un-annotated background. Our labels are ~4% of real cells — if our detector currently sees un-annotated true cells as negatives, that is a direct cause of both LOEO collapse AND over-prediction penalty. This is the single most important idea to check in our pipeline.
- **Learned movement-vector edges** replace/augment appearance-based linking with a *motion* prior that transfers across embryos (motion statistics generalize better than appearance). They report generalization across mouse/Drosophila/zebrafish *without organism-specific retraining* — evidence the cell-indicator + movement-vector formulation is domain-robust.

**Effort:** MED–HIGH (add a movement-vector head + masked loss to the detector; the ILP scaffolding we already have). **Priority: HIGH.**

### 1b. Trackastra — transformer linker, won CTC-2024 generalizable-linking
Gallusser & Weigert, ECCV 2024. [arXiv](https://arxiv.org/abs/2405.15700) · [HTML v2](https://arxiv.org/html/2405.15700v2) · [code: weigertlab/trackastra](https://github.com/weigertlab/trackastra)

**What it is:** a transformer that learns pairwise cell associations within a sliding temporal window (s=6 frames; s∈{3,6} all fine) from **shallow features only** — Fourier positional encoding of centroids + cheap appearance stats (mean intensity, area, inertia tensor). **No CNN feature extractor.** A **parental-softmax** enforces "≤1 parent" (allows division, forbids merges); division loss upweighted 10×. Linking via greedy / LAP / **ILP** (we already have ILP). Trainable on a single RTX 4090, batch 8.

**Why it targets our gap:** (1) It **won the 7th CTC on *generalizable* linking** — exactly our failure mode. (2) Reported gains are large: on bacteria, AOGM 872 (TrackMate LAP) → 23 (Trackastra ILP); on DeepCell nuclei, 18.1 (Caliban) → 7.9. Even *greedy* Trackastra cuts errors ~70% vs prior DL baselines. (3) Because features are shallow/positional, it is far less prone to appearance overfitting than a heavy appearance linker — good for cross-embryo transfer. (4) 3D: authors state it "is expected to scale well to 3D" (no dense image processing); only 2D shown, so 3D is on us.

**Two ways to use it:** (a) drop-in replacement for our edge-transformer linker on our detections, or (b) **fine-tune** its released pretrained weights on our sparse GT. Our detector stays; only linking changes → clean, isolated A/B.

**Effort:** MED (feature extraction from our detections + train/fine-tune; ILP already present). **Priority: HIGH.**

### 1c. Ultrack — multi-hypothesis segmentation + temporal-consistency ILP
Bragantini et al., *Nature Methods* 2025 (Chan Zuckerberg Biohub). [Nature Methods](https://www.nature.com/articles/s41592-025-02778-0) · [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.09.02.610652v1.full) · [ultrametric-contours paper](https://arxiv.org/pdf/2308.04526) · [code: royerlab/ultrack](https://github.com/royerlab/ultrack)

**What it is:** instead of committing to one segmentation, it feeds **many candidate segmentations** (multiple algorithms / thresholds / parameters) into an ILP that uses **temporal consistency** to pick the best segments per frame. Validated on **terabyte zebrafish/Drosophila/nematode** developmental time-lapses; reduces manual correction ~half on dense tissue; **no per-dataset retraining.** From the same Biohub lineage as this competition's data.

**Why it targets our gap:** our detector collapses on the unseen embryo because it commits to one thresholded output. Multi-hypothesis candidates let the temporal ILP *recover* cells the single-threshold detector misses on out-of-distribution density — directly attacks the LOEO detection hole, and its temporal selection naturally curbs over-prediction (helps the `estN` penalty). It is the most "generalizes without retraining" system in this list.

**Effort:** MED–HIGH (generate a candidate set from our UNet at several thresholds/scales; wire into ultrack's ILP, or borrow the multi-hypothesis idea into our SCIP ILP). **Priority: HIGH** (evaluate as a detector-generalization patch; can run alongside our linker experiments).

### 1d. EmbedTrack / VoxelEmbed — joint segment+track via learned offsets
Löffler & Mikut, IEEE TMI 2022. [arXiv](https://arxiv.org/abs/2204.10713) · [code](https://git.scc.kit.edu/kit-loe-ge/embedtrack). VoxelEmbed = 3D voxel-embedding variant.
**What it is:** one CNN predicts per-pixel offsets to cell center + clustering bandwidth; clustering yields instances AND frame-to-frame links jointly. Top-3 on 7/9 CTC 2D sets.
**Why/priority:** conceptually elegant and the offset-to-center idea overlaps Linajea's movement vector, but it is a full pipeline replacement and mainly 2D-proven. **Effort HIGH, Priority: LOW** (mine the offset-clustering idea, don't adopt wholesale).

### 1e. Detector backbone upgrades — StarDist-3D / Cellpose-SAM / Cellpose3
[StarDist-3D usability study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11495889/) · [Cellpose3 restoration, Nat. Methods 2025] · Cellpose-SAM "superhuman generalization" (bioRxiv 2025.04.28.651001).
**Notes:** StarDist-3D is strong for **dense, low-SNR, high-count** nuclei (our `44b6` regime) and is a candidate *candidate-generator* for an Ultrack-style multi-hypothesis set. Cellpose-SAM / Cellpose3 bring generalist priors + one-click restoration (denoise before detect can help cross-embryo SNR shift). Our own result says detector quality is the lever, so a stronger/generalist detector is on-thesis — but a full backbone swap is a big commitment vs. just training our UNet to convergence (already +0.017).
**Effort MED–HIGH, Priority: MED** (best used as extra hypotheses feeding 1c, not as a rip-and-replace).

### 1f. SAM2 zero-shot tracking, Cell-as-Point one-stage — noted, LOW
[Segment Anything for Cell Tracking](https://arxiv.org/html/2509.09943v1); [Cell as Point](https://arxiv.org/pdf/2411.14833). Zero-shot/one-stage are attractive for generalization in principle but weak on **dense 3D**; keep on watch list. **Priority: LOW.**

---

## 2. Better linking / graph methods beyond our ILP

### 2a. Learned edge affinities via message-passing GNN (MPNTrack family)
Brasó & Leal-Taixé, "MOT via Neural Message Passing" ([arXiv](https://arxiv.org/pdf/2207.07454)); [GNN for Cell Tracking in Microscopy, ECCV 2022](https://dl.acm.org/doi/10.1007/978-3-031-19803-8_36).
**What it is:** build the full spatio-temporal candidate graph, run message passing to produce context-aware node+edge embeddings, classify edges as active/inactive (≈ learned min-cost-flow). The cell-tracking variant explicitly models division and propagates information beyond adjacent frames.
**Why it targets our gap:** our ILP edge costs are largely *hand-weighted*; a GNN *learns* affinities with temporal context, which is exactly what a fixed-hyperparameter ILP cannot adapt across the two very different densities. Multi-frame context also fixes short gaps that our gap-close post-proc chases mechanically.
**Effort HIGH** (new training target + graph plumbing; but our candidate graph already exists). **Priority: MED–HIGH** — second-wave after Trackastra (Trackastra is the cheaper first cut at "learned linking").

### 2b. Min-cost-flow with learned costs
Message-passing above already subsumes MCF; classical MCF is a cheaper baseline. Our SCIP-ILP is at least as expressive. **Priority: LOW** (no new capability over what we have).

### 2c. Joint detection+tracking transformers (track queries) — MOTR / TrackFormer / Cell-TRACTR
[MOTR](https://arxiv.org/abs/2105.03247); [Cell-TRACTR (DETR-derivative for cells)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12101859/).
**What it is:** end-to-end detection+tracking; "track queries" carried frame-to-frame. Cell-TRACTR adapts this to segment+track cells with division.
**Why/priority:** elegant but **data-hungry, mostly 2D, and would discard our working detector+ILP**. High risk on sparse 4%-labeled 3D data. **Effort HIGH, Priority: LOW** (strategic watch, not a near-term experiment).

---

## 3. Competition / top-solution tricks for microscopy detection+tracking

Focus: which of these *actually help the UNSEEN-embryo LOEO failure*.

### 3a. Test-Time Augmentation (TTA) — cheap, on-thesis
[TTA for cell segmentation, Sci. Reports 2020](https://www.nature.com/articles/s41598-020-61808-3); DSB-2018 winners boosted their top score with TTA ([Nat. Methods 2019](https://www.nature.com/articles/s41592-019-0612-7)).
**Why for our gap:** averaging detector outputs over flips/rotations/3D-axis transposes + multi-scale reduces variance on an out-of-distribution embryo where a single forward pass is unstable. Directly lifts detection recall/precision jointly (not a Pareto slide) *on unseen data*. **Effort LOW, Priority: HIGH** — do this first; near-free.

### 3b. Multi-scale inference
Our two embryos differ ~25× in cells/frame → strong scale shift. Run detection at 2–3 scales and merge (feeds naturally into 1c's multi-hypothesis set). **Effort LOW–MED, Priority: HIGH.**

### 3c. Pseudo-labeling / self-training on the target embryo
[Dynamic Pseudo-Label Optimization for point-supervised nuclei seg, MICCAI 2024](https://arxiv.org/pdf/2406.16427).
**Why for our gap:** run the source-trained detector on the *unseen* embryo, keep high-confidence detections as pseudo-labels, retrain/fine-tune. This is the most direct antidote to LOEO domain shift when the target is unlabeled. Guard against confirmation bias (confidence + temporal-consistency filtering; the movement-vector prior from 1a is a natural filter). **Effort MED, Priority: HIGH.**

### 3d. Ensembling detectors
Average/union of a couple of detector checkpoints (or StarDist-3D + our UNet) → the candidate set for the ILP. On-thesis (detector is the lever) and de-correlates errors on the unseen embryo. **Effort LOW–MED, Priority: MED.**

### 3e. Heavy domain-invariant augmentation — see §4.

---

## 4. Generalization / limited-label training tricks (labels ~4% of real cells)

### 4a. Masked sparse-annotation loss (from Linajea §1a) — HIGHEST-LEVERAGE cheap fix
Restate because it is both a §1 and a §4 idea: **only supervise inside a radius of each annotated point; never treat un-annotated regions as hard negatives.** With 96% of true cells unlabeled, a naïve loss actively teaches the detector to *suppress* real cells → hurts recall AND worsens the `estN` count penalty AND fits embryo-specific label sparsity patterns (LOEO poison). **Effort LOW (loss-mask change), Priority: HIGH — verify our current loss first.**

### 4b. Domain-invariant / heavy augmentation
Linajea's exact recipe: **elastic deformation, mirroring, axis transpose, intensity augmentation** ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614077/)) let one model span mouse/Drosophila/zebrafish. For us, aggressive **intensity/contrast/gamma + noise + blur** augmentation simulates the SNR/brightness gap between `44b6` and `6bba`; **elastic + scale** simulates the density/morphology gap. Cheapest possible cross-embryo generalizer. **Effort LOW, Priority: HIGH.**

### 4c. Self-supervised 3D pretraining on unlabeled volumes
[SELMA3D — MICCAI 2024 self-supervised 3D light-sheet segmentation challenge](https://arxiv.org/html/2501.03880v2); [Self-Supervised Pretraining of Cell Segmentation Models](https://arxiv.org/pdf/2604.10609); DINOCell (DINOv2 continued-pretraining for cells, strong OOD zero-shot).
**Why for our gap:** we have far more *unlabeled* voxels (both embryos, all frames) than labels. SSL/MAE-style pretraining on all raw volumes → fine-tune the detector on sparse GT gives domain-relevant features that transfer to the held-out embryo. SELMA3D is the on-point precedent (3D light-sheet, self-supervised, generalization-focused). **Effort HIGH, Priority: MED** (biggest infra lift; bank the cheap wins first).

### 4d. Consistency regularization / test-time adaptation
[Survey of TTA under distribution shift](https://arxiv.org/pdf/2303.15361); single-image / continual TTA for medical seg (2024–2025). Adapt BN stats / a few params to the target embryo at test time with an entropy/consistency loss.
**Why/priority:** lighter-weight cousin of 3c; good if pseudo-labeling is too noisy. **Effort MED, Priority: MED.**

---

## RANKED shortlist — most likely to move honest LOEO 0.73 → 0.85+

Ordered by (expected LOEO gain × probability it transfers cross-embryo) ÷ effort.

1. **Fix the detector's sparse-annotation loss + heavy domain-invariant aug** (§4a + §4b + finish convergence). *Rationale:* convergence already gave +0.017; a masked sparse loss removes an active anti-recall / count-penalty poison, and intensity/elastic aug directly manufactures the `44b6`↔`6bba` domain gap in training. Lowest effort, on our proven lever (detector), attacks the exact LOEO mechanism. **Do first.**

2. **TTA + multi-scale inference at test time** (§3a + §3b). *Rationale:* near-free variance reduction on the unseen embryo; the ~25× density gap makes multi-scale especially apt. Pure upside, days of work, composes with everything else.

3. **Swap/fine-tune the linker to Trackastra** (§1b). *Rationale:* CTC-2024 *generalizable-linking* winner, shallow positional features (transfer-friendly), keeps our detector+ILP, clean isolated A/B. Best single shot at the linking half of the gap.

4. **Add a movement-vector detector head + candidate-graph motion edges (Linajea-style)** (§1a). *Rationale:* motion priors transfer across embryos better than appearance; couples detection and linking; it is the published method for our *exact* sparse-label light-sheet embryo setting. Higher effort but highest ceiling.

5. **Target-embryo pseudo-labeling / multi-hypothesis detection (Ultrack-style)** (§3c / §1c). *Rationale:* the two most direct "recover the cells the single-threshold detector misses on the unseen embryo" moves; run whichever the infra favors once 1–4 have de-risked the detector and linker.

**Sequencing:** land #1 and #2 immediately (cheap, detector-lever, cross-embryo by construction), then #3 as an isolated linker A/B, then invest in #4/#5. Divisions remain out of scope (proven dead-end).

---

## Sources
- Trackastra — [arXiv 2405.15700](https://arxiv.org/abs/2405.15700) · [code](https://github.com/weigertlab/trackastra)
- Ultrack — [Nature Methods 2025](https://www.nature.com/articles/s41592-025-02778-0) · [code](https://github.com/royerlab/ultrack) · [ultrametric-contours](https://arxiv.org/pdf/2308.04526)
- Linajea / sparse-annotation embryo lineages — [Nat. Biotech 2022](https://www.nature.com/articles/s41587-022-01427-7) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7614077/) · [code](https://github.com/funkelab/linajea)
- EmbedTrack — [arXiv 2204.10713](https://arxiv.org/abs/2204.10713) · [code](https://git.scc.kit.edu/kit-loe-ge/embedtrack)
- GNN cell tracking — [ECCV 2022](https://dl.acm.org/doi/10.1007/978-3-031-19803-8_36) · MPN MOT [arXiv 2207.07454](https://arxiv.org/pdf/2207.07454)
- MOTR / Cell-TRACTR — [MOTR arXiv 2105.03247](https://arxiv.org/abs/2105.03247) · [Cell-TRACTR PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12101859/)
- StarDist-3D vs Cellpose usability — [PMC 11495889](https://pmc.ncbi.nlm.nih.gov/articles/PMC11495889/)
- SAM2 cell tracking — [arXiv 2509.09943](https://arxiv.org/html/2509.09943v1)
- TTA for cell segmentation — [Sci. Reports 2020](https://www.nature.com/articles/s41598-020-61808-3) · DSB-2018 [Nat. Methods 2019](https://www.nature.com/articles/s41592-019-0612-7)
- Pseudo-labeling point-supervised nuclei — [MICCAI 2024, arXiv 2406.16427](https://arxiv.org/pdf/2406.16427)
- SELMA3D self-supervised 3D light-sheet — [arXiv 2501.03880](https://arxiv.org/html/2501.03880v2)
- Self-supervised pretraining of cell segmentation — [arXiv 2604.10609](https://arxiv.org/pdf/2604.10609)
- TTA survey — [arXiv 2303.15361](https://arxiv.org/pdf/2303.15361)
- Cell Tracking Challenge (CTC) — [celltrackingchallenge.net](https://celltrackingchallenge.net/) · [10-years benchmark, Nat. Methods 2023](https://www.nature.com/articles/s41592-023-01879-y)
