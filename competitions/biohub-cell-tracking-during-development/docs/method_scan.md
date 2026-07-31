# Method scan — closing adj_edge_jaccard 0.872 → 0.906

**Date 2026-07-05. Research only (no code edits).** Metric ≈ pure adj_edge_jaccard (division confirmed dead:
safe-div = 0 div_tp, floods FP, hurts even pilkwang). Test is embryo-disjoint; our measurable CV = fold0
(train 6bba → test 44b6).

## Headline finding (from the public-kernel scan)
**Every top public kernel (ravi 0.891, yusuke/beicicc 0.884, tamerlan, drkong, boristown ~0.885) is the SAME
pre-trained pilkwang model** (TemporalUNet3D detector + `SimpleNodeTransformer` learned edge head + thin ILP).
They are inference-only; **the entire 0.884→0.891 LB spread is post-processing knob-tuning.** Top LB =
doheon114 **0.906**; dense cluster 0.891–0.898. The public post-proc chain (`filter_output_graph`):
`drop non-consecutive → MOTION-RELINK (constant-velocity Hungarian, OVERRIDES the ILP edges; ILP demoted to
edge_prob prior) → single-parent repair → 1&2-frame gap recovery → safe-div → consensus prune → line-fit smooth`.

## Reconciling with our decomposition (important, honest)
We have TWO separable gaps:
- **Gap A — DETECTOR (our #1):** our `loeo_no_aug` (10ep, 6bba-only) = **0.000** vs pilkwang (129ep, same
  6bba-only split_0) = **0.872**. This is UNDERTRAINING → `loeo_moreep`(50ep)/`loeo_129ep` (in progress).
  **This is the prerequisite** — none of the post-proc below matters until the detector converges.
- **Gap B — POST-PROC (public frontier):** pilkwang raw-ILP 0.872 → tuned post-proc 0.891. NOTE our own
  decomposition measured *default* post-proc at **−0.016** on the dense-8 subset (safe-div dominated). The
  public gain comes specifically from the **MOTION-RELINK + gap-recovery** with ravi's tuned µm-gates, NOT
  safe-div. So: adopt the *positive* post-proc pieces (motion-relink, gap-close, centroid-refine), skip safe-div.

## Ranked shortlist (idea | why | expected Δ | test on fold0 | effort)

| # | Idea | Why it moves edge-jaccard | Δ (est) | Test on fold0 (train6bba/test44b6) | Effort |
|:--|:--|:--|:--:|:--|:--:|
| 0 | **Converge the detector** (loeo_moreep→loeo_129ep) | Gap A: 0.000→~0.872. Prerequisite for everything | **+0.87** | score adjJ vs pilk 0.872 (running) | Med (GPU) |
| 1 | **Motion-relink + 1&2-frame gap recovery** (ravi score_push chain, on the converged detector) | Gap B core: velocity-Hungarian overrides weak ILP edges; recovers dropped links → more edge-TP | +0.010–0.018 | apply postproc to converged preds, official_counts vs no-postproc | **Low** |
| 2 | **Sub-voxel centroid refine on ALL nodes** (tamerlan `refine_all_nodes`) | De-quantizes the 4× XY downsample (~1.6µm bias) → more endpoints land within the 7µm gate; edge=TP needs BOTH ends → compounds | +0.005–0.015 | intensity-weighted CoM, cap ±2µm; re-score | **Low** |
| 3 | **Consensus prune (prune50 / edge-consensus)** = over-prediction knob | Deletes long+motion-inconsistent+learned-unsupported relaxed edges under a hard cap; over-prediction is #2 binding constraint | +0.003–0.008 | sweep cap + (dist≥10, motion≥7.8, prob≤0.006) on CV | **Low** |
| 4 | **Detection heatmap-avg ensemble + TTA + det_thr/pool_kernel sweep** | Avg logits BEFORE peak-find over folds/ckpts; keep flip-XY TTA (never Z); fix pool_kernel=5.0 | +0.005–0.012 | 2–3 ckpt avg, re-score fold0 | Low-Med |
| 5 | **Motion features in the learned edge head + rich cross-embryo aug** | Folds the hand-tuned velocity model INTO the learned scorer; bias-field/contrast/gamma aug (currently brightness±0.1 only) targets embryo-disjoint generalization. The 0.891→0.906 path | +0.005–0.012 | retrain edge head, score fold0 | Med-High |

**Skip:** ultrack / EmbedTrack (need segmentation hypotheses, not point heatmaps — big rewrite, poor fit);
Trackastra pretrained (our Thread-2 exp#1 = LOSS, mask/blob confound; our SimpleNodeTransformer already IS a
Trackastra-style linker); safe-div (empirically dead); division fine-tune (weakest binding constraint).

## Recommendation / sequencing
1. **Gate on loeo_moreep** (detector convergence) — the +0.87 prerequisite.
2. On the converged detector, **ideas 1+2+3 are the cheap, high-confidence 0.872→~0.891 path** (all
   post-proc / localization, testable on fold0 with no retrain).
3. **Reaching 0.906 needs a better MODEL** (idea 4 ensemble + idea 5 motion-aware linker), not more geometric
   heuristics — the whole public field plateaus at 0.884–0.898 on the shared 50ep weights.

**Top 3 to act on (post-convergence):** (1) motion-relink+gap-recovery chain, (2) sub-voxel centroid refine,
(3) consensus-prune tuning — all low-effort, fold0-measurable, and target the two binding constraints
(linking, over-prediction). Extracted kernels: `scratchpad/kernels/`; local ravi `learning/public_pull/ravi_lineage_0891/code.py`.

---

# Step C — beat 0.89 → 0.906 (model win). Ranked plan (2026-07-06)

**Prereq:** the converged loeo_129ep base (~0.872) + Step-B chain (~0.885-0.89) must land first. Step-C is a
MODEL win (the public field plateaus at 0.884-0.898 on shared post-proc; 0.906 needs a better model).
Ranked by expected-Δ × effort × fold0-testability (fold0 = train 6bba → test 44b6, embryo-disjoint, measurable).

## CRITICAL CORRECTION (drives the ranking)
**Every augmentation verdict we produced was at 10 EPOCHS = UNDERTRAINED = CONFOUNDED.** The screen_matched
Phase-1 "all augs REJECTED" and stagebridge results were measured on a base whose edge head was degenerate
(12 edges, adjJ 0.000-0.67). Augmentation's ENTIRE purpose is generalization, which only manifests once the
model converges. So those verdicts say NOTHING about aug at 129ep. Aug is our **most under-explored lever**,
not a dead one.

## Ranked Step-C levers

### #1 (HIGHEST) — AUG AT CONVERGENCE  ·  Δ +0.005–0.015 (unknown, high-value)  ·  effort Med  ·  fold0 ✅
Re-test the best augs at 129ep on embryo-disjoint fold0 — do they lift the CONVERGED cross-embryo adjJ vs
no-aug? This directly targets our fundamental challenge (6bba→44b6 generalization) and resolves the biggest
open uncertainty (the confounded 10ep verdicts).
- **A/B design:** anchor = loeo_129ep (no-aug, 128 6bba, 129ep, fold0). Treatments, ONE aug each, same recipe:
  `loeo_129ep_cropscale` (density-UP zoom, s∈[0.5,1.0]), `loeo_129ep_translate` (translate_static), 
  `loeo_129ep_flip` (flip_xy), `loeo_129ep_rot` (rot90_yx). Score fold0 adjJ_44b6; KEEP if aug > no-aug-129ep.
- **Priority within:** crop_scale (density = the physically-motivated 6bba→44b6 bridge) + flip_xy first; then
  translate/rot. **Cost:** each is a full 129ep run (~2.5h GPU) → sequence, don't fan out; start with the 2 highest-prior.
- **Why #1:** highest-value uncertainty, fold0-measurable, and the ONE lever that attacks cross-embryo
  generalization at the model level (not post-hoc geometry).

### #2 — DETECTION ENSEMBLE / heatmap-avg  ·  Δ +0.003–0.010 (reliable)  ·  effort Med  ·  fold0 ✅
Train 2–3 detectors (different seeds; optionally the 2 LOEO folds), average the detection **heatmap logits
BEFORE peak-finding** (NOT point-union — point-union floods FP edges), then link. Reduces detection variance →
better peak positions + recall → more edge-TP. Keep flip-XY TTA (never Z, anisotropy). Robust to the unseen
embryo. **A/B:** score single-model vs 2-3-model heatmap-avg on fold0. Reliable, cheap-ish, low-risk.

### #3 — TRAIN-ON-BOTH (loeo_both)  ·  Δ +0.003–0.008 (LB-only)  ·  effort Low  ·  fold0 ❌ (unmeasurable)
Final-coverage submission model (train 44b6+6bba, 129ep, golden-12 checkpoint — config authored + GREEN).
NOT embryo-disjoint-validatable (2 embryos, both in train) → validate the RECIPE (levers #1/#2) on fold0, then
apply best recipe here for submission. **Applied LAST.**

### #4 (LOWEST) — MOTION-AWARE LEARNED LINKER  ·  Δ +0.003–0.010 (uncertain marginal)  ·  effort HIGH  ·  fold0 ✅
Fold velocity/motion features into the SimpleNodeTransformer edge head so the learned scorer subsumes the
hand-tuned motion model. **BUT the Step-B postproc motion-relink ALREADY captures the motion lever (+0.05 on
the 0.67 base) cheaply.** So a learned motion linker is HIGH effort (architecture + retrain) for uncertain
MARGINAL gain over the free postproc. Deprioritize unless #1/#2 stall.

## Recommended execution order (post-129ep + Step-B)
1. **#1 aug-at-convergence** (crop_scale + flip_xy first) — resolve the confound, biggest upside.
2. **#2 detection ensemble** — reliable, parallelizable with #1's GPU queue.
3. **#3 train-on-both** — final submission model with the best recipe from #1/#2.
4. **#4 learned linker** — only if the plateau isn't broken by #1–#3.

---

# Step D — the 0.89 → 0.906 gap (the hard model win). Ranked (2026-07-06)

**Leaderboard reality:** doheon114 **0.906** is a PRIVATE OUTLIER (no public kernel — only unrelated 2025 work);
the public cluster is **0.896–0.898** (Rahul/Giuseppe 0.898, Maher/Matt 0.896). So 0.906 = beat the public
cluster by ~+0.008 with an UNKNOWN method (no writeup to mine). This is genuine open headroom, not a recipe copy.

## #1 re-examined — DIVISION is DEAD even via T+2 RANKING (empirically closed)
Tested boristown's T+2 coherence RANKING on golden-12 (8 GT divisions): **div_tp = 0** (unchanged), div_fp
3→32 (flood), div_fn 8 (all GT divisions still missed), adj_edge_J 0.8527→0.8518 (regress). Same failure as
geometric safe-div. WHY: the metric's div_tp needs a fork AT the GT-division node (matched ≤7µm) with the GT
DAUGHTER matched. T+2 ranks divisions that *persist* coherently, but with only ~8 GT divisions among ~25k cells
and no knowledge of WHERE they are, geometric/temporal addition never lands on them. The ONLY way to claim
div_tp is a LEARNED division head that predicts divisions at the right places — a big model change, and even
pilkwang (0.87) claims 0 div_tp. **DROP division entirely.** The +0.1 term is effectively unreachable on this GT.

## Ranked 0.906 levers (delta × cost)

### #1 — DETECTION ENSEMBLE (heatmap-logit avg)  ·  Δ +0.005–0.010  ·  cost 12–18h GPU  ·  fold0 ✅
The clearest 0.89→0.90+ path (ensembles reliably add ~0.005–0.01; the public cluster is single-model post-proc).
- **Mechanism:** train K detectors (different seeds), predict each to detection HEATMAP LOGITS, AVERAGE the
  logits ACROSS K *before* peak-finding (NOT point-union — that floods FP edges), then peak-find → nodes → link
  (add-only relink) → full recipe → score. Averaging logits reduces detection variance → cleaner peaks + recall.
- **Cost:** each seed = a ~6h 129ep run. K=3 (seeds 1234/2/3) = ~18h GPU (diminishing returns beyond 3).
- **Fold0 test:** heatmap-avg predict of the 3 seeds vs single-model, apply recipe, score adjJ. Needs a
  predict-side change to emit + average logits (moderate). KEEP if > single-model + recipe.

### #2 — INFERENCE TTA + det_threshold/pool_kernel fine-tune  ·  Δ +0.002–0.005  ·  cost LOW (inference only)  ·  fold0 ✅
Expand flip-XY TTA (never Z), re-sweep det_threshold around 0.99 (writeup says 0.99 optimal WITH postproc) and
pool_kernel_um on the converged base. Cheap, no retrain — do FIRST as a low-cost probe before the expensive ensemble.

### #3 — AUG AT CONVERGENCE (Step-C #1, re-slotted for 0.906)  ·  Δ +0.005–0.015 (uncertain)  ·  cost N×6h  ·  fold0 ✅
Our 10ep aug verdicts were confounded; at 129ep aug targets cross-embryo generalization (our fundamental
challenge) at the MODEL level — the kind of diversity a 0.906 model likely has. Expensive (each aug = 6h). Slot
after the ensemble unless #2 stalls.

### DEAD / skip
Division (T+2 + geometric both 0 div_tp), learned motion-linker (postproc relink captures it), ultrack/EmbedTrack
(point-not-segmentation), higher-res (1,2,2) (the weights trap).

## Ranked plan
1. **#2 TTA + threshold fine-tune** (cheap, do first) → +0.002–0.005 toward the 0.896–0.898 cluster.
2. **#1 detection ensemble** (heatmap-avg, K=3, ~18h) → the +0.005–0.010 that reaches ~0.90.
3. **#3 aug-at-convergence** → diversity for the last push, if #1/#2 plateau short of 0.906.
Reaching doheon's exact 0.906 may need a differentiator we can't see (private) — but ensemble + aug-diversity is
the principled route to the 0.896–0.90+ band, the best we can validate on our embryo-disjoint fold0.

---

# Step-D #2 — ENSEMBLE seed plan (cost/benefit, 2026-07-06)

**Heatmap-logit-avg REQUIRES converged (129ep) models** — a 50ep/undertrained base has a degenerate edge head
(adjJ 0.67), so averaging undertrained detectors dilutes rather than denoises. Ruling out the cheap-but-invalid
shortcuts up front:
- ❌ **50ep seeds** (~2.5h each): undertrained base → invalid ensemble (averaging weak detectors ≠ denoising a good one).
- ❌ **seed-avg of loeo_moreep(50ep)+loeo_129ep(129ep)**: mixes EPOCHS not seeds → confounded + one member undertrained.
- ❌ **multi-checkpoint avg of loeo_129ep**: trainer saves only `best.pth`, no per-epoch checkpoints → unavailable
  without a re-run that saves them.

**Valid options (converged 129ep, 1 GPU serial):**
| plan | members | new GPU cost | expected Δ | verdict |
|:--|:--|:--:|:--:|:--|
| K=2 | loeo_129ep(seed1234, DONE) + 1 new seed | ~6h | +0.003–0.006 | **RECOMMENDED first** — half the gain, 1/3 the cost |
| K=3 | + a 2nd new seed | +6h (12h total) | +0.005–0.010 | only if K=2 shows a clear gain |

**Recommendation:** run **K=2** (loeo_129ep + one seed=2, ~6h) as the cost-effective ensemble; escalate to K=3
(seed=3, +6h) only if K=2 lifts fold0 adjJ clearly. **Mechanism (locked):** predict each seed → detection heatmap
LOGITS → average logits ACROSS seeds *before* peak-finding (never point-union) → peak-find → link (add-only relink)
→ full recipe → score. Needs a predict-side change to emit + average logits across the K weight files (moderate,
authored when Step-D executes). Avoids the 18h up-front by proving K=2 first.
