# Biohub — Cell Tracking During Development

> Slug: `biohub-cell-tracking-during-development` · Category: **Research** · Reward: **$60,000**
> Deadline: **2026-09-29** · Host: **Chan Zuckerberg Biohub** · Entered as `seshurajup`
> NOTE: `competition_setup.py` regenerates this file as a template — keep this rich version.

## 1. Problem
4D (3D+time) light-sheet microscopy of developing (zebrafish) embryos. Reconstruct cell
**lineage**: a directed graph `G=(V,E)`. Node `v=(t,z,y,x)` = a cell detection; edge `u→v` =
continuation or daughter; **division** = node with **out-degree ≥ 2**.

## 2. Data (focus on `train/` — has images AND labels)
| Item | Value |
|---|---|
| Image | Zarr v3, group `0/`, shape **(T,Z,Y,X)** uint16, one timepoint per chunk `0/c/{t}/0/0/0` (blosc2) |
| Voxel | **z=1.625, y=x=0.40625 µm** (anisotropic, Z ~4× coarser) |
| Labels | `train/<id>.geff` (GEFF graph). **Read:** `geff.read(path, backend='networkx')` or raw zarr (`nodes/ids`, `nodes/props/{t,z,y,x}/values`, `edges/ids`) |
| id | `<embryo>_<hash>`, embryo = `id.split('_')[0]` (e.g. 44b6, 6bba) |
| Labels are **SPARSE** | not a full census — don't treat unlabelled cells as background; use `estimated_number_of_nodes` (GEFF meta) for density |
| **Visible `test/` = copies of train** | ≥2 of 4 test datasets exist in train WITH labels (known artifact; real PRIVATE test is embryo-disjoint → not an exploit, but great for validation) |
| Size | ~87 GB total (train≈7.7 paged / test). Downloading locally; **do NOT touch the zip until user says done** |

## 3. Submission (`submission.csv`)
`id,dataset,row_type,node_id,t,z,y,x,source_id,target_id`. `node` rows set node_id,t,z,y,x
(source/target=-1); `edge` rows set source_id,target_id (others=-1). node_id unique per dataset.

## 4'. EXACT METRIC (from official metrics.md — authoritative, implement this)
```
node match: optimal bipartite assignment on scaled centroid dist, gate 7µm, scale (1.625,.40625,.40625)
edge TP  = pred edge whose BOTH endpoints match GT nodes joined by a GT edge
edge FP  = a "valid" pred edge (an endpoint matches a GT node that HAS a GT edge) that is not a TP
edge FN  = every GT edge with no matching pred edge ; all other pred edges are IGNORED
jaccard  = TP/(TP+FP+FN)
adjusted_jaccard = max(0, jaccard·(1 − 0.1·(T_pred − T_true)/T_true))   # a=0.1
   T_pred = total predicted nodes ; T_true = GEFF estimated_number_of_nodes (FULL est count)
   → predicting UNDER T_true gives a BONUS (score can exceed 1.0); OVER → penalty
adj_edge_jaccard(run) = Σ_i w_i·adjusted_jaccard_i / Σ_i w_i ,  w_i = TP_i+FP_i+FN_i  (weight-avg)
division_jaccard = micro Σtp/(Σtp+Σfp+Σfn) over samples
SCORE = adj_edge_jaccard + 0.1·division_jaccard
```
**Implication:** it's EDGE-Jaccard (not node recall) + a node-count term keyed to the EST total
(not sparse GT) + 10% divisions. Over-detection past the estimate is penalized (why kojimar's high
recall lost). Implemented locally in `src/metric.py:official_score` — NO tracksdata install needed.

## 4. Evaluation metric (REWARD — OFFICIAL CODE, exact)
**Authoritative source = the OFFICIAL repo** `royerlab/kaggle-cell-tracking-competition`
(linked in the organizers' Welcome discussion; cloned to `research/official_repo/`).
Metric in `src/tracking_cellmot/metrics.py`. Run it locally for a CV *identical* to the LB:
`pip install tracksdata polars` → `evaluate_datasets(graph_pairs, scale=(1.625,0.40625,0.40625), max_distance=7.0)`.

**Exact run-level (leaderboard) score:**
```
score = edge_jaccard + 0.1 * division_jaccard            # SCORE_DIVISION_WEIGHT=0.1
edge_jaccard     = Σ edge_tp / Σ(edge_tp + edge_fp + edge_fn)     # MICRO-avg across datasets
division_jaccard = Σ div_tp / Σ(div_tp + div_fp + div_fn)         # dropped if no divisions anywhere
```
- Node matching = `DistanceMatching(max_distance=7.0, scale=(z,y,x))` (the 7µm gate, scaled µm).
- Edge TP needs BOTH endpoints matched; FP counts only "valid" predicted edges (≥1 endpoint
  matched a GT node that has degree). So **edge-Jaccard is the dominant term** (explains why
  embryo-balanced edge_recall hit Pearson +0.996 vs LB).
- **Node-count penalty** is a PER-SAMPLE metric (`per_sample_metrics`): `total_node_ratio =
  (N_pred − N_gt)/N_gt`, `adj_edge_jaccard = max(0, J·(1 − 0.1·total_node_ratio))` (ADJUSTMENT_ALPHA=0.1).
  Penalizes over-prediction. (Run-level `evaluate_datasets` uses raw micro edge_jaccard; the
  adjustment is the per-sample diagnostic pilkwang referenced.)
- **MICRO-average ⇒ larger datasets dominate** the raw score (NOT embryo-balanced). Use micro-avg
  to estimate LB; keep embryo-balanced as the *generalization* check (unseen test embryo).
- Submissions scored on a **random sparse subset** of all cells (organizers) ⇒ public LB
  high-variance; inference must still track ALL cells.
- Divisions detected by `division_metrics.evaluate_divisions`; ALL public configs get div≈0 ⇒
  the +0.1·division_jaccard term is untapped headroom (where private 0.76 pulls ahead).

## 5. Proving local CV ↔ public LB (validation strategy)
We have train GT + public submissions with known LB (pilkwang **0.687**, yusuke **0.680**) that
predict the visible-test datasets (= train copies w/ labels). Score those submissions vs train
GT with the official metric → if numbers track the known LB, local CV is trusted, then sweep
configs on train (embryo-disjoint) to beat 0.687.
- Calibration NB: `notebooks/biohub_calibration_kaggle.ipynb` → kernel `seshurajup/biohub-metric-calibration`
- Public subs uploaded as dataset `seshurajup/biohub-public-submissions`

## 6. Leaderboard & public floor (EXACT public-notebook LB scores)
**NEW BAR = 0.725** (romanrozen "strong-start-beginner-guide-lb-top-10"). Winning lever =
**PER-SAMPLE COUNT CALIBRATION**: per-movie node BUDGET (topk/frame ≈ embryo's true count) instead
of a global threshold → counters the over-prediction penalty. Core = conservative (NMS 4.0,
MAX_LINK 10µm, USE_MOTION, **DIV OFF**, prune; THRESH_REL 0.32, generous 0.10 for topk path, SMOOTH 0.9).
Top cluster (0.720 xiaoleilian / 0.725 romanrozen) converges on: aggressive NMS + motion linking +
divisions OFF + count calibration. 0.720–0.725 within ±0.03 LB noise = tied top. **Divisions STILL
OFF at the very top ("false divisions cost") ⇒ our +0.1 division exploit is UNCLAIMED to the top.**
Lever to adopt: per-embryo count budget (~GT density 3/frame 44b6, 9/frame 6bba). Files:
research/romanrozen_top10/, research/xiaoleilian/.
- (prev) 0.707 lucifer19 "V11 strategy-switch", builds on pilkwang 0.687. Versions:
v1 0.586 → v2 0.618 → **v3/V11 0.707** (same-method calibration triplet). Recipe for 0.687→0.707 =
**precision EDGE tightening** (cut false-positive edges; edge-Jaccard=TP/(TP+FP+FN)):
`THRESH_REL 0.20→0.21`, `NMS 2.65→2.72µm`, `MAX_LINK_DIST→10.5µm` (tighter gate),
`DIV_PARENT/SISTER→8.25/5.75` (tighter divisions), isolated-node prune, **no motion/gap** (complexity
still OFF). Lever = edge PRECISION, not recall/complexity. Submission ~125k rows (≈pilkwang's 127k).
| LB | Notebook (author) |
|---|---|
| **0.707** | **V11 strategy-switch (lucifer19) ← THE BAR (precision edge tightening)** |
| **0.687** | Data Model, EDA, Baseline (pilkwang) |
| 0.641 | Starter Baseline (kojimar) |
| 0.637 | Metric-Aware Baseline (pavloivanin) |
| 0.628 | LB628 Clean-Room No-GPU (yusuke) |
| 0.618 | Cell Lineage Tracker (lucifer19) / V3 Metric (yaroslav) / V2 Sub-Voxel (pavloivanin) |
| 0.611 | **V3 Velocity Kalman + Gap Closing (pavloivanin)** |
| 0.581 | STRONG START guide (romanrozen) |
| 0.143 | Getting Started NN (inversion) |

**KEY INSIGHT — complexity HURTS:** the fanciest pipelines (V3 velocity-Kalman+gap-closing 0.611,
V2 sub-voxel 0.618) score *below* pilkwang's cleaner baseline (0.687). So **velocity prior + gap
closing in our `src/` likely hurt** — disable/A-B them. Beat 0.687 via detection quality + count
stabilization + conservative pruning, not added tracking machinery. Top private LB ≈ 0.761/0.760.
Pulled top-5 EXACT submissions (`research/public_submissions/`) + 10 kernels. Rank via
`kaggle kernels list --sort-by scoreDescending`; per-version LB needs web (CLI 403).

## 6a''. SHAKEUP RISK & FINAL-SUBMISSION RULE
Shakeup likely HIGH: public LB = 29% (~58 datasets, noise ≈0.03) of an unseen-embryo hidden test;
adversarial AUC 0.98 (embryos very different) ⇒ public & private rank methods differently. The
public 0.611–0.641 cluster (7 NBs within the 0.03 band) WILL reshuffle; only gaps >~0.05 are stable.
**Defense (decide finals by CV, not LB):** select submissions by embryo-disjoint `loeo` (corr +0.964,
std 0.017 < public 0.03, pessimistic = unseen-embryo-like); treat public-LB gaps <~0.03–0.05 as
noise; a change is real only if it beats prior CV by >~0.017; never tune to public LB.

**2 final slots = DIVERSE CV-hedge (not safe-vs-LB):** Slot 1 = best `loeo` CV (primary). Slot 2 =
a DIVERSE CV-good pick — ideally a different MODEL FAMILY (classical-best + deep-best) so the two
fail differently on an unseen 3rd embryo; alt = max-min-embryo (most robust operating point). Do
NOT spend a slot on "best public LB" (it's noise here). With adversarial AUC 0.98 the risk is a
structurally-different hidden embryo → method diversity hedges it; two operating points of one
method fail together. Public LB = sanity check only, never a selector.

## 6a'. CV vs TRAINING data (k=2 does NOT waste data)
- Current classical pipeline has NO training ⇒ loeo(k=2) is only an evaluation aggregation; all 199
  datasets always used, nothing lost.
- When we TRAIN a model: separate concerns — use embryo-disjoint k=2 for honest
  generalization/selection (pessimistic: each fold trains on 1 embryo), but FIT THE FINAL
  SUBMISSION MODEL ON ALL 199 (both embryos). loeo is a conservative lower bound (final model sees
  more embryos than a fold). For data-hungry hyperparam tuning, also run random k-fold (k=5, mixes
  embryos, lower variance) as a second scheme; trust embryo-disjoint for go/no-go. Only 2 embryos
  exist → embryo-disjoint k=2 is the honest max, not a waste.

## 6a-RESULT. ALIGNED CV (Spearman +0.90 vs all 9 public LBs) ✅
**The CV that aligns with all public notebook LBs:** official **micro adjusted-edge-jaccard**
(`evaluate_datasets`-style, weight-avg by TP+FP+FN) computed via config reproductions on the FULL
199-dataset TRAIN folder. Result: **Spearman +0.900, Pearson +0.871** across the 9 public notebooks
(xiaoleilian .720 → romanrozen .581). Requirements that made it work: (1) TRAIN folder only (not
dummy test); (2) official MICRO metric, NOT loeo-min (loeo on sparse 44b6 inverts the top tier);
(3) FAITHFUL config repros — a wrong kojimar threshold (0.22 vs real 0.34) over-detected and was the
main outlier (fixed: micro 0.616→0.538). Valid ONLY within the sane detection regime (see 6a-FIX:
over-detection breaks it). MLflow: cv_micro per run, kind=public_repro carries known_lb.

## 6a-GOLDEN (FROZEN 2026-06-30). `src/golden_cv.py` — single source of truth
**FROZEN CV** = official MICRO adjusted-edge-jaccard, **stratified 5-fold (embryo+size), 20 seeds,
averaged**, on full 199-train, with a DENSITY GUARDRAIL (valid iff pred density ≤ 190/frame 44b6,
95/frame 6bba = best-public xiaoleilian level). Validated: Spearman(golden_cv,LB)=+0.900 on the 9
public configs (all valid), fold_std 0.017–0.04; the 6 egregious over-detection configs (density
203–250) correctly flagged valid=False. Significance: Δgolden_cv > ~0.03. Call `golden_cv(per_dataset,
frames)`. DO NOT change without re-validating vs the 9 public LBs.

## 6a-PROTOCOL. LOWEST-VARIANCE CV (use this) — repeated stratified 5-fold, averaged
Best strategy (`experiments/cv_stability_v2.py`): **stratified 5-fold** (each fold balances embryo
AND dataset-size) **repeated over ≥20 seeds, averaged**. Proven:
- Stratified 5-fold alignment = **0.910 ± 0.033 (min 0.883)** vs plain-random 0.891 ± 0.066 — higher
  mean, HALF the variance. (k=3/6 stratified don't help — too few folds for size-balancing.)
- Repeated-averaged per-config CV: estimator std across 10 independent batches = **0.0000**
  (perfectly reproducible). Single-fold std ~0.02–0.03 → averaging ~100 folds drives it to ~0.
⇒ Report each config's CV as the repeated-stratified-5-fold AVERAGE micro (deterministic & stable);
significance floor on a single fold ~0.03, ~0 for the averaged estimate.

## 6a-STABILITY. k-fold × seeds proof (`experiments/cv_stability.py`)
Alignment Spearman(fold micro-CV, public LB) across k∈{3,5,6,9} × 30 seeds (90–270 folds each):
k=3 0.898±0.044 (min .75, 94%≥.8); k=5 .891±.067; k=6 .891±.069; k=9 .886±.076 (min .65, 79%≥.8).
⇒ the +0.90 alignment is STABLE across folds/seeds, not a lucky split. Per-config CV value std
≈0.032–0.046 (k=5×30 seeds) = significance floor: a change must beat prior CV by >~0.04 to be real;
adjacent-LB configs (<~0.04 apart) are at the noise limit (≈ public-LB ±0.03). Scope: proves
ranking stability WITHIN the sane regime (does not cover the over-detection blind spot, 6a-FIX).

## 6a-RULE. CV uses the TRAIN folder ONLY (199 labeled datasets)
The 4 visible `test/` datasets are DUMMY example copies (swapped out at rerun) — do NOT score on
them (only 4 noisy points, unrepresentative; e.g. romanrozen LB 0.581 scores ~0.04 there). All CV =
config reproductions run on the full 199-dataset TRAIN folder (`official_cv.py`), correlated vs the
public notebooks' known LB. Public submission.csv files only cover the 4 dummies, so we reproduce
each notebook's CONFIG on train instead (must use FAITHFUL params — see kojimar fix below).

## 6a-FIX. CV WEAK POINT (sparse-GT over-detection inflation) — READ THIS
**Discovered via xiaoleilian (real LB 0.720) reproduction:** the official metric computed on our
SPARSE TRAIN GT is **systematically fooled by over-detection**. Evidence (official micro CV vs real LB):
det-more configs 0.62 > kojimar 0.616 (LB .641) > xiaoleilian 0.604 (LB **.720**) > lucifer 0.560
(LB **.707**) > pilkwang 0.548 (LB **.687**). The REAL top-3 score LOWEST locally; over-detection
scores HIGHEST. **Root cause:** sparse GT can't count false-positive edges — extra detections miss
the few labeled cells, so their wrong edges are IGNORED not penalized → more detection = higher local
score regardless of quality. No aggregation (loeo/6bba/micro) fixes it. `loeo` on the SPARSE 44b6 is
the worst (inverts the top-3).
**Consequences:** (1) our CV is only valid WITHIN the sane-detection regime the public notebooks
occupy; it CANNOT rank configs that differ in detection density. (2) The detection-sweep "gains"
(cv 0.49→0.58, det_thr0.12_peak1_sm0.8) ESCAPED that regime → NOT real; discard. (3) Earlier
"+0.97 corr" held only because the 8 public NBs all use sane density.
**THE FIX (stronger CV):** impose a node-density PRIOR (xiaoleilian: raise THRESH_REL ~0.32, cap
nodes ≈ GT density ~3/frame 44b6, ~12/frame 6bba, NMS ~4µm) so over-detection can't game it; then use
the official metric to tune linking/divisions/refinement WITHIN that fixed-density regime. Calibrate
the absolute level with a REAL submission (sparse labels structurally can't separate precision from
lucky-recall). Best CV ordering signal among sane configs: official micro / denser-6bba (NOT loeo-min).

## 6a. (SUPERSEDED by 6a-FIX) earlier LOCKED CV STRATEGY
**Test structure:** hidden test ≈ TRAIN size (~199 datasets) on NEW (unseen) embryos; visible
`test/` = 4 format-example copies of train (NOT scored). Public LB = **29% of hidden ≈ ~58
datasets**; private = ~71% ≈ ~141 — both unseen-embryo. ⇒ exact LB NOT locally reproducible; goal
= high CV↔LB **correlation** (Deotte), not identity.

**Adversarial validation (`experiments/cv_sampling_search.py`):** 44b6 vs 6bba separable at
**AUC 0.980** (44b6 density 0.01/est~37k; 6bba density 0.09/est~16k) = STRONG embryo shift ⇒ the
unseen-embryo private test WILL differ ⇒ expect a public↔private SHAKEUP; use the pessimistic
`loeo` CV and never overfit the public LB.

**k-fold conclusion (data-driven):** only 2 embryos ⇒ embryo-disjoint CV is fixed at **k=2 (loeo)**.
Random-dataset k-fold at k∈{2,3,5,6,9} ALL give Spearman +0.893 (no gain) while per-fold noise
GROWS with k (0.012→0.052). Only embryo-grouping lifts to +0.964/Pe +0.868. ⇒ do NOT use k=3/6/9;
matching the EMBRYO split (not more folds) is what tracks the LB.

**Cell-subsampling (`experiments/cv_node_subsample.py`):** scoring vs a random K% subset of GT
CELLS (the metric's described eval) at frac 1.0/0.5/0.29 keeps loeo corr +0.964; cell-sampling
noise ~0.01. CV robust to which cells are labeled. **Organizer-aligned split:** the official repo
harness reads `dataset_splits.json` (folds with train/test lists) but ships none — competitor
provides it. Created `input/.../train/dataset_splits.json` = 2 embryo-disjoint folds (fold0 test=44b6,
fold1 test=6bba) → loeo, and plugs into the official train/predict/evaluate scripts. (Deotte OTTO
lesson: replicate the organizer's split/process — here the harness is embryo-disjoint by design.)

**Noise floors (from seeded resampling, `experiments/cv_resampling.py`):**
- KFold(5) fold-assignment std ≈ **0.0016** (CV is stable to splitting).
- Bootstrap-over-199 std ≈ **0.017** = CV estimation noise → a config change must beat prior CV by
  **> ~0.017** to be real.
- 29%-subset (≈ public-LB sampling, ~58 datasets) std ≈ **0.031** → public-LB gaps < ~0.03–0.06 are
  NOISE (the 0.611–0.641 public cluster is indistinguishable). Full-train CV (0.017) is MORE stable
  than the public LB (0.031) ⇒ **trust CV over a single public-LB reading**.

**THE CV = exact official metric (`src/metric.py official_counts`/`official_score`) on the FULL
199-dataset train, aggregated as leave-one-embryo-out WORST case (`loeo_min` = min over the 2
embryos of the official run-score).** Empirically **Spearman +0.964** with the 9 public LB scores
(via faithful config reproductions); micro-avg official = +0.893. Runner: `experiments/official_cv.py`
(full 199 train × a config in ~50s local, 24 cores). Mirrors the unseen-embryo private test.
- Don't use pure node/edge recall as the CV — it's gamed by over-detection (kojimar: highest
  recall, lower LB). Use the official adjusted-edge-jaccard.
- Scoring real public submissions on the 4 example datasets only is noisy (+0.61) — too few points.
- Decision rule (Deotte): keep a change only if it improves `loeo_min` beyond noise; never tune on
  a single public-LB reading; submit to Kaggle only when `loeo_min` clearly beats the 0.687-config.

## 6b. (superseded by 6a) recall-proxy CV exploration
Ran each public config on FULL train (199 datasets) → per-dataset node/edge/div recall, then
correlated aggregations vs known LB (`experiments/cv_analysis.py`). On the first 4 configs:

| Aggregation | Spearman vs LB | Pearson |
|---|---|---|
| mean_node_recall | +0.80 | +0.93 |
| **embryo_edge** (embryo-balanced edge recall) | **+1.00** | **+0.996** |
| embryo_node+edge | +1.00 | +0.975 |

- **Best CV = embryo-balanced recall** (weight 44b6 & 6bba equally; train is imbalanced 71/128 but
  test has both). Edge recall correlates best (metric is tracking-centric: edge needs both endpoints).
- **LOEO robust:** pilkwang is top config scored on 44b6-only AND 6bba-only → ranking generalizes to
  an unseen embryo (the private-test condition). Only **2 embryos** in train (44b6, 6bba), so
  embryo-disjoint CV = effectively 2-fold; private test = unseen 3rd embryo.
- **Recall not gamed by over-detection:** count-penalized variant still ranks pilkwang #1.
- **Divisions ≈ 0 for ALL public configs** → division recall is unused headroom = where top private
  (0.76) pulls ahead. pilkwang CV (node 0.698 / pooled 0.72) ≈ its LB 0.687.
- Tooling: `experiments/cv_analysis.py --results <dir>`; per-config CSVs archived by config name.

## 6c. TRAINING PATH (domain research — for the learned phase)
**Official deep baseline** (`research/official_repo`, GPU): `TemporalUNet3D` (3D U-Net + temporal
attention → per-voxel detection map → local-max centers) + `SimpleNodeTransformer` (cross-attention
scores every (t,t+1) node pair), trained JOINTLY end-to-end; sparse supervision (only GT-annotated
edges backprop). **Division-aware** (`softmax(dim=0)`=divisions-allowed-not-merges + division-row
upweight) → directly targets the +0.1·division term that ALL classical configs score 0 on.
- Plugs into our `dataset_splits.json` (`--split 0/1` = embryo-disjoint folds); scored by exact
  official metric ⇒ directly comparable to classical loeo (~0.42). Train fold0+fold1, gate by loeo;
  FINAL model trains on all 199 (see §6a').
- Run: `nvidia-smi -pl 400` first; `--dry-run` (1 epoch/1 video) to validate; then full. Env via
  `uv sync` or pip the repo + torch into `kaggle_vision`.
- SOTA upgrade paths (organizer-named): `trackastra` (transformer tracker), `ultrack`/`motile`
  (graph-opt linking on any detector), `byotrack`. Official U-Net+Transformer = pragmatic start.

## 6c2. DOMAIN RESEARCH (see notebooks/references/) — KEY: the GT pipeline is known
**Zebrahub** (CZ Biohub Royer) = zebrafish nuclear light-sheet 3D+time + lineage, GT made with
**Ultrack** → almost certainly THIS competition's GT pipeline (same org/organism/modality). ⇒
(1) Zebrahub = pretraining set + sparse-label Rosetta stone; (2) **Ultrack** (division+death aware,
offline, BSD-3) reproduces the GT-generation → strong baseline + likely best division handling.
Detectors: Cellpose nuclei/cpsam (anisotropy=4.0) or StarDist3D fine-tuned (init Xenopus/BlastoSPIM
embryo-nuclei). Linker: **Trackastra `ctc`** (learned, DIVISION-AWARE, offline) or Ultrack. Datasets:
Zebrahub, Fluo-N3DH-CE (dense), Fluo-N3DL-DRO (sparse light-sheet analog), SIM+. Full notes:
`notebooks/references/research_models_datasets.md`. (ArXiv + past-CTC-solutions agents pending.)

## 6d. METRIC EXPLOIT — division term (+0.1, UNCLAIMED) = #1 opportunity
score = adj_edge_jaccard + **0.1·division_jaccard**, and ALL public notebooks score division_jaccard=0
(disabled — false divs crater it). Headroom: 30–50% division recall at high precision → div_j 0.3–0.5
→ **+0.03–0.05 score** (> noise 0.03) → pushes 0.720→~0.75–0.77, points nobody collects.
CATCH: divisions RARE (151 total / 199 datasets, 0.12% of edges, only 87/199 have any; ~0.4/ds 44b6,
~1.0/ds 6bba) → few FPs crush the jaccard ⇒ it's a PRECISION problem (fire only when confident).
Minor exploits: node-count BONUS (predict under est → up to +10% multiplier, scores>1.0; never exceed
est); weight-avg by (TP+FP+FN) → dense datasets dominate, optimize there. TRAP (not usable):
"other pred edges ignored" → free FPs on sparse train only; does NOT transfer (=over-detection blind
spot; guardrail blocks). Develop a conservative division detector vs the GOLDEN CV.

## 7. Public approach (classical; what we reproduced in `src/`)
detect (XY block-mean ×4 → isotropic, smooth, `θ=max(Otsu,P50+0.20·(P99.8−P50))`, local-maxima,
sub-voxel CoM refine on raw vol, physical NMS `~2.65µm`, border filter) → link (Hungarian in µm,
gate `~11µm`, **V3 velocity prior**, **gap-close T→T+2**) → conservative division
(`DIV_PARENT 8.75`, `DIV_SISTER 6.25µm`) → **prune track-isolated nodes**. Count-stabilizer caps
per-frame detections. Key: `MATCH_GATE_UM=7`, `XY_DS=4`, `SMOOTH_SIGMA=1.0`, `MIN_PEAK_DIST=2`,
`THRESH_REL=0.20`, `NMS_RADIUS_UM=2.65`.

## 8. Code (kaggle_vision env)
- `src/`: config, io (geff+blosc2), detect, link (velocity+gap-close+division+prune), metric
  (replica — recall-focused; calibrate to traccuracy), cv (embryo-disjoint), submission, pipeline
- `experiments/run.py --dry-run` (synthetic; validated), `experiments/validate.py` (train CV)
- `notebooks/`: `make_*_notebook.py` generators → CV / submission / calibration Kaggle notebooks
  (self-contained, validated). Kernels: `biohub-cv-embryo-disjoint`, `biohub-metric-calibration`.

## 9. Strategy / next
- Biggest lever = node recall × localization (quadratic on edges) + keep density ≈ estimate.
- Validate metric (calibration NB) → sweep config on train CV → reproduce 0.687 floor → beat it.
- Real jump (per public authors): GPU 3D nucleus detector trained on GEFF labels (RTX 5090 local).
- Always CV **embryo-disjoint** (group by `id.split('_')[0]`) to mirror private test.

## 10. Our experiments  (all logged to MLflow exp 13 = kaggle-biohub-cell-tracking-during-development)
Tracking: `src/track.py log_cv(...)` → one per-comp MLflow table accumulating CV configs now +
future public notebooks (kind=public_repro, known_lb) + our submissions (kind=our_submission,
kaggle_lb) + trained models. Decision metric = `cv_loeo` (worst embryo, official metric).
View: http://localhost:5000/#/experiments/13.

| Run | loeo CV | vs bar | Notes |
|---|---|---|---|
| lucifer_v11_0707 (bar, LB 0.707) | 0.4367 | — | precision edge tightening |
| pilkwang_0687 (LB 0.687) | 0.4184 | — | clean classical baseline |
| **sw_link9.0** | **0.4820** | **+0.045** | tighter link gate 9.0µm → fewer FP edges. BEST so far. |
| sw_link9.5 | 0.4727 | +0.036 | |
| sw_combo (link9.5+thr0.23+nms3.0) | 0.4698 | +0.033 | |
| sw_link10.0 | 0.4563 | +0.020 | |

**Lever confirmed = edge PRECISION** (tighter link gate cuts FP edges). GT motion: median 1.8µm,
global p99 8.4µm (44b6 p99 7.2; 6bba p99 8.5). **CHOSEN CANDIDATE: `ours_v1_link8.5`** = v11 + link
gate **8.5µm** → loeo **0.4882 (+0.045 over the 0.707 bar's CV)**. Chose 8.5 (not the raw loeo max
at 7.0µm, loeo 0.495) on PRINCIPLE: 44b6(loeo) keeps rising to 7µm, but **6bba PEAKS at 8.5µm and
degrades below it** (tightening past its motion cuts true edges). 6bba = proxy for a higher-motion
unseen private embryo ⇒ 7.0µm overfits 44b6 / is fragile; 8.5µm (≈global p99) is robust for both
embryos. Extra +0.013 loeo from 7.0µm not worth the generalization risk (Deotte: don't overfit the
worst fold). Logged MLflow kind=our_candidate.

## 7. THE 0.810 BASELINE (UNet + Transformer + ILP) — new bar, pretrained, no-training
A public reproduction scores **LB 0.810** (`yaroslavkholmirzayev/biohub-cell-tracking-v4-unet-ilp-reproduction`),
far above the old 0.725 public bar. Author of the method: `thibautgoldsborough`; all artifacts public in
dataset **`cellmot-baseline-artifacts`** (repo code + `weights/unet_transformer/split_0/edge_predictor_best.pth`
8.3MB + offline wheels incl. pyscipopt/SCIP). Runs on Kaggle with internet OFF.

- **Pipeline:** `TemporalUNet3D` detector + `UNetNodeTransformer` edge-predictor → **ILP** linking
  (pyscipopt). ILP is the jump over greedy and models divisions (ILP_DIVISION_WEIGHT=1.0). One state_dict
  holds both UNet + transformer. Knobs: DET_THRESHOLD=0.99, ILP edge=-1.0/appear=0.1/disappear=0.1/div=1.0.
- **METRIC VERIFIED == LB:** scored the author's `bundle_verify/split_0/6bba_268e1230.geff` with our
  `src/metric.official_score` → **adj_edge_jaccard 0.8014 ≈ LB 0.810** (edge tp/fp/fn 522/32/87; raw
  0.814 × node-count-adj 0.984 for predicting 5921 vs 5107 estimated nodes). Confirms src/metric.py AND golden CV.
- **REPRODUCED locally (no training):** ran `predict_unet_transformer.py` in an isolated venv
  (`research/cellmot_venv`, inherits our torch+cu128, wheels installed offline) on the same dataset →
  **0.8014, identical** (5921 nodes, 4964 edges). Full pipeline runs on our GPU.
- **Divisions still ≈0** even at 0.810 (GT 1 div, pred 14, 0 matched) → the +0.1 term remains unclaimed.
- **Tuning (no-train) in progress:** sweeping DET_THRESHOLD + ILP div weight vs golden CV
  (`experiments/baseline_repro/tune.py`) to trim node over-prediction and try to beat 0.810 before submitting.
