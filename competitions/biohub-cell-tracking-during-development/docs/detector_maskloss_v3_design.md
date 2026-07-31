# Detector-train v3 — [A] Sparse-annotation MASKED detection loss (design, 2026-07-10)

**Status:** code done + CPU-unit-tested, config dryrun-GREEN, **GPU-PARKED** (runs after the 2-fold
convergence verdict + push-iters). Clean one-variable A/B vs **EXP_162 = 0.7322** (v2 convergence).

## The pathology (measured, not assumed)
The training GT is **extremely sparse** — a subset of cell centres, not a complete labelling. Direct
node counts from the GT geffs:

| train volume | total GT nodes | frames | GT cells / frame (min·med·max) |
|---|---|---|---|
| `6bba_0c7fa718` | 704 | 94 | 1 · **8** · 13 |
| `44b6_0b24845f` | 51 | 40 | 1 · **1** · 2 |

The volumes (64×256×256) actually hold **dozens (early 6bba) to ~1000+ (late 44b6)** true nuclei per
frame. So the GT labels only ~1–8 cells/frame while **~99 % of real nuclei are unlabelled.**

`compute_detection_loss` builds a **dense** target: GT voxels = positive (weight `1/n_pos`), **every
other voxel = negative** (weight `neg_weight/n_neg`). → every unlabelled true nucleus is supervised as
**background**. That is a systematic downward bias on real cells. Two independent confirmations:
- **EXP_158 (v1):** raising `det_neg_weight` 0.01→0.05 (more negative penalty) *hurt* −0.0084 —
  node_recall 0.969→0.939, edge_J 0.742→0.720. The extra penalty suppressed **true** nuclei ⇒ the
  negative set demonstrably contains real cells. `det_neg_weight` only **slides along** the
  recall×precision frontier (net-negative); it cannot push it.
- **Dense 44b6 under-detection:** `144b256d` predN/estN = 0.506 (misses half). The most crowded
  regions are where the "unlabelled cells taught as background" bias bites hardest.

The current `neg_weight=0.01` is tiny (per-neg-voxel weight ≈ `0.01/n_neg` ≈ 1e-8), which is *why* the
detector still over-fires at all — but the bias is still there, degrading peak sharpness/calibration
exactly where nuclei crowd. Neither convergence (EXP_162, a real +0.017 but a *training-length* lever)
nor `neg_weight` addresses the **label pathology** itself.

## The lever: ignore bright unlabelled voxels
`CELLMOT_DET_FG_IGNORE=P` (opt-in; default unset ⇒ behaviour byte-identical). When set, the detection
loss stops penalising **bright** non-GT voxels:
- `fg = image ≥ per-sample P-th percentile` (P=98 ⇒ top 2 % brightest) — bright blobs ≈ likely cells.
- `ignore = fg & (target==0)` → weight **0** (removed from the loss).
- Only **dark background** (below the percentile, non-GT) stays negative; `n_neg` is renormalised over
  the *kept* negatives so their weight is honest.
- Positives (GT voxels) untouched — even a dim GT cell keeps weight `1/n_pos`.

**Per-sample percentile = self-calibrating ⇒ embryo-agnostic** (no absolute intensity threshold that
would fail to transfer to a 3rd embryo — the key requirement from `detector_quality_lever_scope.md`).
This is the frontier-**push** `neg_weight` cannot give: it removes the bias against real cells, so recall
*and* precision (sharper, better-separated peaks in crowded 44b6) can rise together.

### CPU unit-test (done, no GPU)
`compute_detection_loss` on synthetic (B=2, 8×16×16, bright non-GT region):
- DEFAULT (env unset): loss finite, grad finite, all 4096 voxels contribute — **unchanged**.
- MASKED (P=95): loss finite, grad finite, **3840/4096** voxels contribute; the bright non-GT region has
  **exactly zero** gradient (ignored) vs >0 in default. ✔ mechanism verified.

## The experiment (matched A/B)
`config/loeo_maskloss_f0.yml` = **EXP_162 verbatim** (fold0 train-6bba→test-44b6, 20ep/300it convergence,
`det_neg_weight 0.01`, E50 domain-invariant aug, leak-clean 6bba-val-holdout selection) **+ the single
change** `det_fg_ignore: 98`. `num_workers: 0` (hang-fix default). Score: canonical LOEO fold0 vs
**EXP_162 = 0.7322**; ledger win-gate; promote only on canonical lift.
- **Dryrun-GREEN:** launch cmd + env validated (num-workers 0, max-iters 300, `CELLMOT_DET_FG_IGNORE=98`
  wired via env, 11 augs matched). No `--dry-run` in this trainer ⇒ the real smoke = a 1-iter GPU run
  when the GPU frees (loss-path already CPU-tested).
- **Route:** `bash start_train.sh config/loeo_maskloss_f0.yml` via :7788 (researcher hands off GPU).

### Readouts / risk
- **Primary:** canonical LOEO fold0 > 0.7322 (frontier-push). Watch **predN/estN** on 44b6 (does the
  dense under-detection 0.506 recover?) and **edge_J**.
- **Over-ignore risk:** P too low ignores real background ⇒ *more* over-prediction (already a tendency).
  P=98 is conservative (ignores only the top 2 %). If predN explodes, raise P (99, 99.5); if no effect,
  lower (95). One knob, cheap to sweep on the same fold.
- **Fold1 confirm:** if fold0 wins, mirror as `loeo_maskloss_f1.yml` (train-44b6→test-6bba) for the
  2-fold verdict before promotion.
- **Honest caveat:** node_recall on the *sparse GT* is a useless signal (~1.0 everywhere,
  `golden12-sparse-gt-vs-estN`); judge strictly by canonical adj-edge-Jaccard + predN/estN.

## [B] TTA / multi-scale test-time inference (cheap, stackable follow-up)
Near-free variance cut on the unseen embryo; **no retrain**, stacks on ANY detector (incl. [A]):
- **Flip/rot90 TTA:** run detection over the 4–8 in-plane dihedral transforms (Y/X flips + rot90 —
  the same symmetries the aug asserts are valid; Z-flip OFF, unphysical), invert, **average the logit
  heatmaps** pre-NMS. Reduces per-orientation detector noise.
- **Multi-scale:** re-detect at ±1 downsample step (e.g. [1,3,3] and [1,5,5]) and merge peaks — apt for
  the ~25× density gap between 6bba and 44b6 where one fixed scale can't fit both.
- **Cost:** k× forward passes at inference only (k≈4–8), zero training. **Effort:** a wrapper around the
  existing predict path (peak-detect on averaged logits); no loss/model change.
- **Sequence:** spec now, implement after [A]'s verdict so we stack TTA on the *best* detector, not a
  baseline we're about to replace.

## Provenance
Density counts: `input/.../train/{6bba_0c7fa718,44b6_0b24845f}.geff` (node `t` arrays). Loss code:
`research/official_repo/scripts/train_unet_transformer.py::compute_detection_loss` (env-gated). Baseline:
EXP_162 (`beat-ceiling`/`gap-decomposition-detector-is-lever`; `detector_quality_lever_scope.md`).
Survey source: `docs/methods_survey_toward_0897.md` rec [A] (Linajea-style masked sparse-label loss).
