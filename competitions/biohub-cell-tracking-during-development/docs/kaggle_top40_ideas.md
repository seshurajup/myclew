# Biohub Cell Tracking — Top Public Notebooks Knowledge Base

**Date:** 2026-07-05
**Competition:** `biohub-cell-tracking-during-development`
**Metric:** `score = adjusted_edge_jaccard + 0.1 * division_jaccard` (7 µm node matching, embryo-disjoint test)
**Notebooks covered:** 40 (pulled to `research/public_notebooks/`)

## Leaderboard context (public LB, top 20)

| rank | team | LB |
|---|---|---|
| 1 | doheon114 | **0.906** |
| 2 | TWEAK / James Weatherhead | 0.899 |
| 4 | Rahul Parmeshwar / Giuseppe Lentini | 0.898 |
| 6 | ChenWenSheng | 0.897 |
| 7 | Maher el Ouahabi / Matt Goldfield | 0.896 |
| 9 | PowerPuff Girls | 0.895 |
| 10 | Ryo / AL Najafi | 0.894 |
| 12 | **Pilkwang Kim / Kun Zhang (beicicc)** + 6 others | 0.893 |

- **Public-notebook ceiling ≈ 0.891** (lucifer19 / yusuke internal presets); **best submitted public authors = 0.893** (Pilkwang, Kun Zhang).
- **Our stack = 0.857** (rule-based v14). Gap to public-notebook ceiling ≈ **0.034**; gap to LB #1 ≈ 0.049.
- LB range across the notebooks studied: **~0.72 (DoG starters) → 0.891 (learned-graph)**.

---

## Ranked table (by LB, then family)

| notebook | LB | CV | detector | linking | division | key params | novel idea |
|---|---|---|---|---|---|---|---|
| **lucifer19** / biohub-cell-lineage-tracker | **0.891** (106ep) / 0.886 (129ep) | node-row ablation | pilkwang `unet_transformer` UNet | ILP + motion-relink | ILP div + `safe_divisions` | det 0.99, gap-close 1, gap2, linefit; SHA-pinned checkpoint + weight-soup | **Checkpoint provenance IS the lever** (which .pth = 0.891 vs 0.886) |
| **ravi123a321at** / lineage-tracker | ~0.886–0.891 | — | pilkwang support-pack | ILP + motion-relink | safe_divisions | identical `score_push` preset to lucifer19 | code-only fork (viz stripped) |
| **yusuketogashi** / lb886 micro-safe-div | **0.886** | — | TemporalUNet3D center head | learned node-transformer edge → ILP | `safe_divisions`, tightened gates | det 0.99, sister 6.85 / child 7.45 µm, frame-cap 0.0072 | isolate safe-division precision as the ONLY changed axis |
| **yusuketogashi** / lb885 micro-safe-div | **0.885** | — | same UNet | learned edge → ILP + motion-relink | safe_divisions | det 0.99 / 0.985 recall, gap2 9.7/4.05 µm | tiny A/B on gap + safe-div FP risk |
| **boristown** / AGI | 0.891 (129ep) / 0.886 (106ep) / **0.884** default (159ep) | — | same UNet | `score_push`, learned-prob bonus → ILP | safe_div 4.8/7.0/7.6 | det 0.99 | **proves identical code + different weight artifact = whole LB spread** |
| **beicicc (Kun Zhang)** submitted best | **0.893** (LB) | — | learned-graph fork (Yusuke/Pilkwang) | ILP + motion-relink | safe_divisions | see exp029/033 below | top public LB via learned backbone |
| **beicicc / exp029** lb884-prune50 | 0.884 | — | learned-graph 0.890 base | ILP + edge-consensus prune | safe_div | prune dist≥9.9 / motion≥7.5 / prob≤0.006, cap 50 edges | tighter prune < base → **pruning HURTS** |
| **beicicc / exp033** yusuke-score-push | n/s (0.89x lineage) | — | learned-graph | ILP + motion-relink | safe_div 4.8/7.0/7.6 | det 0.99, motion tight 6.2/relaxed 10.4, gap2 ON 9.7/4.05 | balanced score-push preset (stable/score_push/high_recall) |
| **beicicc / exp028** yusuke-lb873-repro | 0.873 | — | learned-graph 0.890 base | ILP + `ultra_tiny_edge_prune` | safe_div | prune dist≥9.6 / motion≥7.2 / prob≤0.008, cap 65 | pruning weak long edges → **0.873 < 0.890, hurt** |
| **beicicc / exp031** pilkwang-precision-repair | n/s (200ep test) | — | pilkwang 200ep | ILP, gap2 OFF | safe_div 4.5/6.7/7.2 | **det 0.992** (highest), edge_max 13.4, fewer synthetic | diagnostic: is 200ep loss from over-repair? precision-first |
| **beicicc / exp032** pilkwang-recall-clean | n/s (200ep test) | — | pilkwang 200ep candidate-17 | ILP, gap2 ON conservative | safe_div 4.6/6.9/7.4 | det 0.985 recall, edge_max 14.2, frame-node cap 3200 | isolate recall vs over-repair as loss source |
| **beicicc / exp027** v10-thr033-gap55 | ~0.856–0.858 | — | DoG 2-scale | two-pass Hungarian + gap2 | safe_div post-link | DoG **thr 0.033**, gap 5.5 µm, linefit w=0.8 | lower det threshold = recall; div in linker regressed |
| **beicicc / exp006** rule-safe-div | ~0.839 | — | DoG 2-scale | two-pass | safe-div parent≤5 / sib≤8 / child≤8.5, caps 0.6% | rel_thr 0.045, min_dist 4.0 | guarded post-link 2nd-daughter, +0.004 claimed |
| **yunusgmsoy** / v10 | **0.858** | proxy≈LB | DoG 2-scale, XY_DS=4 | two-pass (6/8 µm, vel_blend 0.5) | **safe_divisions ON (+0.004 LB)** | scales [[1.5,4.0],[2.2,5.5]], rel_thr 0.030, min_dist 4.0, gap 6.0 | conservative safe-div, ≤0.6% edges, hard caps |
| **nomannic19** / temporal 3D-UNet | **0.857** | embryo-strat calib | **Temporal** 3D-UNet in_ch=3 (t−1,t,t+1), base=20 | two-pass Hungarian (6/8) + gap interp | gap only (no true div) | thr auto 0.04–0.20, NMS 4.4, PU-BCE, 4 ep, 24 fr/movie | **3-frame temporal input**; per-embryo count-penalty calib |
| **seshurajup (ours)** / v14 | **0.857** | — | DoG 2-scale, XY_DS=2 | two-pass (tight 6 / loose 8) | none active | rel_thr 0.025, min_dist 3.0, gap 6.0, smooth_w 0.8 | `recover_gap2` (t→t+3) + `linefit_smooth` |
| **amanatar** / EMA intensity cost | **0.855** | node-calib 1.12 | DoG enhanced 2–4 scale, XY_DS=2 | **EMA-velocity** two-pass + **intensity-sim cost** (w=0.12) | geometric (angle≥60°, survival≥5, div≤5, sib≤8) | rel_thr 0.025, min_dist 2.8, gap1+gap2, savgol smooth | **EMA velocity + intensity-diff blended edge cost** |
| **hosen42** / cv6 | ~0.842–0.847 | — | DoG 2-scale, XY_DS=4 | two-pass; div-linker OFF | present but OFF (regressed) | rel_thr 0.045, min_dist 4.0, gap 6.0 | `validate_divisions` (prune divisions whose daughters die) |
| **rahuljiwane** / rj2 | ~0.847 | — | DoG 2-scale | two-pass | none | rel_thr 0.045, min_dist 4.0, filter_short 4 | minimal clean two-pass + short-track filter |
| **xiaoleilian** / 3D-UNet (train+infer) | **0.841** | TEST4 0.795; ensemble VAL 0.815 | compact 3D-UNet base=24, XY-pool×4→64³ | two-pass µm Hungarian (6→10) | none (bijective → div=0) | UNET_THRESH 0.10–0.15, NMS 4, PU-BCE W_POS12/W_IGN0.05, σ1.0, 40ep | **PU-aware loss** (ignore bright-unlabelled); learned det beats DoG on det/frame |
| **koushikrudra** / robust centroid | ~0.826 | — | DoG 2-scale, XY_DS=4 | single-pass Hungarian (8 µm) | none | rel_thr 0.045, min_dist 3.2, gap 6.0 | **bg-subtracted robust centroid** (p20 baseline, reject shift>2.8) |
| **isakatsuyoshi** / rule-based baseline | ~0.826 | — | DoG 2-scale, XY_DS=4 | single-pass Hungarian (8) | available, OFF | rel_thr 0.045, min_dist 3.2, gap 6.0 | **canonical modular baseline everyone forked** |
| **thibautgoldsborough** / UNet baseline | ~0.79 ILP / 0.73 greedy | — | royerlab `unet_transformer` | **ILP** (global flow) or greedy | ILP div_weight=1.0 (native) | **DET_THRESHOLD 0.99** (sparse GT poorly calibrated) | official learned UNet + node-transformer + ILP |
| **romanrozen** / DoG band-pass | **0.73** | embryo-grouped proxy | DoG band-pass, XY_DS=4, peak_local_max | two-pass (6/10), motion 0.5 | OFF | DOG_SIGMAS (1.0,1.8,3.0), K=1.6, THR_PCT 80 | **per-sample count calibration** (cap each movie to ⌈1.15·f·D⌉ peaks) |
| **jirkaborovec** / DoG + Trackastra | n/s (trails classical) | proxy 0.5·nodeF1+0.4·edgeJ | DoG → **watershed instance masks** | **Trackastra** (ctc graph transformer) | **native learned** (out-degree 2) | watershed basin cap 5, greedy/ILP | **ONLY Trackastra user** — learned association, no distance gates |
| **jirkaborovec** / EDA + DoG detect | ~0.73 | — | multi-scale DoG (σ 1.0/1.8/3.0) | two-pass Hungarian + velocity | off | DOG_THR_PCT 80, NMS 4, per-movie topk budget | per-movie count calibration as topk budget |
| **avikdas567** / scale-space Hungarian | low | seed EDA | single-scale DoG (σ 1.1/2.2) | single-pass global LSA (7 µm) | crude (nearest child ≤6, no sister gate) | pct 93.5, max_det 2000 | simplest inline division engine (FP-prone) |
| **xiaoleilian** / classical baseline | **0.720** | proxy (0.5,0.4,0.1) | single-scale local-max + Otsu (NOT DoG) | two-pass (7/11) | OFF (FP div tanked div_f1) | smooth σ1.0, thresh_rel 0.18, NMS 4 | **lesson: DoG over-detection (~700/fr) DROPPED 0.72→0.59** |
| **pilkwang** / data-model EDA-baseline | — | — | multi-scale DoG [[1.5,4],[2.2,5.5]] | Hungarian / motion | optional geometric (off) | rel_thr 0.045, min_dist 3.2, gap 6.0 | clean config-driven rule-based reference |
| **inversion** / NN getting-started | very low | — | percentile-thresh + connected-comp | single Hungarian (15 µm) | none | PERCENTILE 90 | official minimal starter |
| **tom99763** / tracksdata tutorial | n/a | n/a | — | — | explains div = out-degree 2 | — | **exact-metric walkthrough** (7µm → edge-J → node-count penalty → +0.1 div-J) |
| **aman5153684** / tuning the ILP | n/s | embryo-disjoint sweep | official `unet_transformer` | **ILP** | **ILP division_weight sweep** | CV_DET [0.99,0.95,0.90] × DIV_W [1.0,0.5] | **lowering div_weight floods spurious divisions, hurts edge-J** → keep ≈1.0 |
| **biohack44** / UNet + gap-closing | n/s | runtime bounds | xiaolei UNet + DoG fallback | two-pass gated (KD-tree) | **`recover_divs_g`** (DIV 6, SIB 9, SYM 2.2) | GAP 1@6, max_det 1500 | explicit gap-close + division-recovery post-passes |
| **tamerlanomralinov** / centroid-refine ILP-div | ~0.884–0.886 | — | pilkwang UNet | ILP + motion-relink | safe_div **+ keep ILP divisions** (parent≤6, sister≤9, min_prob 0.5) | det 0.99, **node centroid re-snap** win_yx 4, shift 2.0 µm | full-res intensity-centroid re-snapping (undo 4× xy-downsample) |
| **yaroslavkholmirzayev** / v4 UNet-ILP repro | ~0.884 | — | pilkwang UNet | ILP + motion-relink | safe_div | det 0.99, **clamp node coords ≥0** | non-negative coord clamp fixing boundary-refine bug |
| **canqiang** / ZebraCellTrace | ~0.85 cluster | — | 3D-UNet + DoG band-pass, peak_local_max | greedy + Hungarian | — | full-frame submit | full-frame inference variant |
| **shubhamveer** / math foundations | n/a | n/a | DoG (1.5/4, 2.2/5.5) | gated Hungarian α=0.5 | precision-first repair (P–C2≤5, C1–C2≤8, capped) | gate 8 µm | failure-mode framework; division as capped regularizer |

*(pilkwang / learned-graph-w-gap-recovery — the new release — analyzed in its own section below.)*

---

## Consensus techniques

**Two distinct solution families, and the ceiling lives entirely in the learned-graph one:**

### Family A — Rule-based DoG (LB ceiling ≈ 0.858)
Almost all forked from `isakatsuyoshi/biohub-rule-based-baseline` (identical io/metric/detect modules verbatim in 8+ notebooks).
- **Detector:** multi-scale DoG on an XY-downsampled near-isotropic grid, voxel `SCALE=[1.625, 0.40625, 0.40625]` µm. The winning `dog_scales` is universally **[[1.5,4.0],[2.2,5.5]]** (µm).
- **Detection threshold splits by era:** top cluster (0.855–0.858) uses **rel_threshold 0.025–0.030 + min_dist 3–4 µm** (recall-tilted); older 0.826–0.847 baselines use conservative **rel_threshold 0.045**. Lowering the threshold + controlling node count is the lever that moved 0.826 → 0.858.
- **Linking:** two-pass velocity-aware Hungarian, **tight ≈6 µm / loose ≈8 µm**, all costs in physical µm. Two-pass beat single-pass by ~+0.012.
- **Gap closing:** max_gap 1 at **~6 µm**; the top notebooks add a strict `recover_gap2` (t→t+3, capped ~2%).
- **Post-proc = the separator:** `filter_short_tracks(min_len≈4)` + `prune_isolated` + `linefit_smooth (w=0.8)` + `recover_gap2` lifted the family from ~0.847 to ~0.858 (all node-count-neutral, exploiting the over-prediction penalty).

### Family B — Learned graph (LB ceiling ≈ 0.891, public best submitted 0.893)
All share the royerlab `unet_transformer` backbone: a `TemporalUNet3D` center-detection head (single-voxel GT, class-balanced weighted BCE, neg α≈0.01, local-max decode) → a **bidirectional node-transformer edge predictor** → **ILP** (`pyscipopt`/`ilpy`; edge=−1.0, appearance=disappearance=0.1, **division=1.0**), then heavy conservative post-proc.
- **DET_THRESHOLD = 0.99 by default** (0.985 for recall presets, up to 0.992 for precision). The GT is sparse, so the UNet sigmoid is poorly calibrated and the threshold sits very high.
- **`add_safe_divisions` is universal in this family** — it adds ONLY a *second* outgoing edge to a node that already has one child, gated by parent–child ≤ ~4.6–4.9 µm, sister ≤ ~6.85–7.2 µm, existing-child ≤ ~7.4–7.8 µm, with per-frame (~0.7–0.9%) and global (~0.4%) fraction caps. **This is the sole mechanism claiming the +0.1·division_jaccard term.**
- **Gap recovery is two-tier and heavily capped:** `gap_close` (single missing frame, ≤2·6 µm, ≤5% added nodes) + strict `gap2` (two missing frames, total ≤9.7 µm, per-step ≤4 µm, ≤~0.3% of edges).
- **The dominant LB lever is the detector CHECKPOINT (epoch count), not the linker.** boristown/lucifer19 prove identical code with a different `.pth` gives 0.884 / 0.886 / 0.891 (159 / 106/129 / 106 ep). Linker/prune tweaks are near-flat or negative.

### Cross-family consensus on divisions
- **Divisions are dangerous and most notebooks leave them OFF.** In-linker / low-div_weight division handling consistently **regressed** LB (hosen42, xiaoleilian, aman's ILP sweep). The metric's 0.1× weight rarely pays for the edge-FP cost of extra splits.
- The **only** positive-delta division method is a **guarded post-link "safe-division" pass** with hard geometric caps (+0.004 LB, yunusgmsoy) or the native ILP division at weight≈1.0 (not lowered).
- **No one uses StarDist or Cellpose.** Exactly one notebook runs **Trackastra** (jirka), and it trails the tuned classical baseline (pretrained-ctc + fabricated watershed masks; fine-tuning proposed but not done).

---

## Pilkwang's new release — `biohub-cell-tracking-learned-graph-w-gap-recovery`

This is the canonical, most-upvoted (135 votes) base of the entire learned-graph family and the current best public reference (Pilkwang LB **0.893**). It is the same `predict_unet_transformer.py` backbone as Yusuke/boristown/lucifer19, shipped with the "candidate-17 / candidate-20" recall-clean weight profiles. First-hand parameter read:

- **Detector:** TemporalUNet3D center-logit head, single-voxel targets, class-balanced weighted-BCE (negative weight α=0.01), local-maxima decode. Detection loss `BCELogit`; edge loss = focal-weighted BCE `(1−P*)²·BCE` with per-parent softmax normalization; total `L_edge + λ_det·L_det`, λ_det=1.
- **`DET_THRESHOLD = 0.985`** (recall profile; the score-push presets use 0.99). UNet batch 4.
- **Linking:** learned edge predictor → ILP; cost `C_ij = d_motion + 0.05·d_raw − β·P_learned`. Motion relink tight **5.95 µm** / relaxed **10.0 µm**; `OUTPUT_EDGE_MAX_UM = 14.2`.
- **Gap recovery (the headline feature):**
  - `gap_close` (max_gap 1) at **5.9 µm** (≤2g); reuse existing isolated node within **3.1 µm** or insert a synthetic midpoint node re-centered by local intensity centroid (accepted only if shift ≤ **3.2 µm**); capped at **4.5% added nodes / 1900 abs**.
  - `gap2` recovery (two missing frames) total ≤ **9.5 µm**, per-step ≤ **4.0 µm**, requires non-contradictory velocity context, ≤ **120 links / 0.26% of frame**.
- **Divisions:** `OUTPUT_SAFE_DIVISIONS=1` with `SAFE_DIV_MAX_UM=4.6`, `SISTER=6.9`, `EXISTING_CHILD=7.4 µm`; `ILP_DIVISION_WEIGHT=1.0`; geometry filter OFF by default (`DIV_PARENT_MAX 10.5`, `DIV_SISTER 8.0`).
- All geometry measured in microns using anisotropic voxel scale `[1.625, 0.40625, 0.40625]`.

**Takeaway:** it externalizes ~40 env-var knobs; the forks (Yusuke lb886, beicicc exp028-033, tamerlan, yaroslav) only nudge these presets. The genuine content vs our stack is (1) the **learned node-transformer edge predictor** and (2) **ILP-native + safe-division** producing a nonzero division_jaccard.

---

## What we're missing (vs our stack)

Our stack: rule-based DoG tracker at **LB 0.857** + a from-scratch UNet detector. Our metric decomposition: **`div_J = 0` is our biggest unclaimed headroom** (+0.1·div_J ≈ up to +0.05 if perfectly claimed, realistically +0.004–0.01).

1. **We are at the rule-based ceiling (0.857 ≈ family cap 0.858).** Every rule-based notebook tops out at 0.858. To break past it we must move to the **learned-graph family** — no amount of DoG tuning closes the 0.034 gap. The evidence is unambiguous: the entire 0.88–0.89 cluster is one UNet+node-transformer+ILP pipeline.
2. **We claim zero division_jaccard; the top notebooks all run `add_safe_divisions`.** This is the single most-transferable, lowest-risk win. It is a **post-link pass** (no retraining): for any track node with exactly one child, attempt to attach a second daughter gated by parent≤4.6, sister≤6.9, existing-child≤7.4 µm, with hard per-frame (~0.8%) and global (~0.4%) caps. Verified positive delta **+0.004 LB** (yunusgmsoy) and it is universal in the 0.885–0.891 family. It bolts straight onto our existing two-pass tracker output.
3. **Our detector is DoG; the ceiling detector is a learned UNet center-head with a PU-aware loss.** The xiaolei/nomannic PU-BCE recipe (W_POS=12, W_BG=1, **W_IGN=0.05**, Gaussian σ=1.0 stamps, ignore bright-unlabelled voxels) is the consensus training recipe and directly applicable to our from-scratch UNet.
4. **The linker is the real ceiling-setter — a learned node-transformer edge predictor + ILP, not Hungarian.** This is the heavy lift, but it is what separates 0.858 from 0.891. Cheaper intermediate: the `unet_transformer` + ILP is fully public (thibaut's official-repo notebook) and reproducible.
5. **Detection checkpoint epoch count dominates the learned-family LB** (0.884↔0.891 from `.pth` alone). If we train a UNet detector, checkpoint selection / weight-soup is a first-class lever, not an afterthought.

### Ranked adoption shortlist
1. **`add_safe_divisions` post-link pass** — claims div_J with verified +0.004, zero training, bolts onto our current tracker. **Do this first.**
2. **PU-aware weighted BCE + Gaussian center-heatmap loss** for our from-scratch UNet detector (W_IGN=0.05 is the key trick for sparse GT).
3. **Reproduce the `unet_transformer` + ILP linker** (thibaut/pilkwang public code) as the path off the 0.858 rule-based ceiling.
4. Cheap rule-based squeezes we may not have: **per-sample count calibration** (romanrozen), **EMA-velocity + intensity-similarity edge cost** (amanatar), **full-res centroid re-snap** (tamerlan) — each worth ~0.001–0.003.

**Do NOT:** add divisions inside the linker or lower ILP division_weight to force more splits — this backfired in ≥3 independent notebooks (hosen42, xiaoleilian, aman). Divisions must be precision-first and capped.
