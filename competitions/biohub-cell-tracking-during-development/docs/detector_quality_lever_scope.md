# Detector-quality lever — scope (2026-07-10)

**Why:** honest LOEO ceiling = EXP_157 = 0.7276 (embryo-disjoint), vs EXP_156 golden-12 proxy 0.9161.
The gap is DETECTOR quality on truly-unseen embryos, not postproc (mtl/gap pruning already exhausted).
Target: a TRANSFERABLE >0.85 that survives LOEO.

## Diagnosis (golden-12 fresh re-detect, support-pack pool5/det0.99, per-ds)
predN/estN is **embryo + density dependent**, NOT pure density:
- **44b6 dense → UNDER-detect:** 144b256d (estN 65376, predN/estN 0.506 — misses HALF), 12dfb391 (58672, 0.908).
- **6bba → OVER-detect:** 1.05–1.74; even the densest overall, 05db0fb1 (estN 69800), over-detects at 1.070.
- node_recall ≈ 1.0 everywhere (sparse-GT artifact, [[golden12-sparse-gt-vs-estN]]) — useless as a detector signal;
  the real detector signal is **predN/estN** (count) + **edge_jaccard** (linking).
- Score-limiters are a MIX: under-detection (dense 44b6: 12dfb391 adj 0.817, 144b256d), over-detection
  (6bba_07e24132 1.735→adj 0.793), and LINKING (0b24845f adj 0.764 at predN/estN 0.984 = good count but low
  edge_J → an edge-precision failure, not detection).

## PER-DS CHARACTERIZATION (2026-07-09, CPU, fresh re-detect + zarr image stats — the fleet data-audit
## agent runs a FIXED flow_gt audit and ignored the SNR request, so this was computed directly)
Per golden-12 ds: predN/estN, edge_J, image contrast (zarr `image_statistics.quantiles` q999−q001), and
post-NMS median nearest-neighbor distance (µm). Findings that DECIDE the lever:
- **CONFIRMED (44b6 over-merge):** the two big under-detectors are DENSE 44b6 — 144b256d (estN 65,376 →
  predN/estN 0.506), 12dfb391 (58,672 → 0.908). Within 44b6 it's density-graded: dense→under, sparse→over
  (0db75fae 15k→1.256). Consistent with pool5 (5µm NMS) MERGING crowded late-stage 44b6 nuclei.
- **REFUTED (clean SNR/density proxy):** 6bba OVER-detects at ALL densities — even 05db0fb1 (estN 69,800,
  the densest overall) → 1.070. And CONTRAST does NOT separate: 05db0fb1 has the HIGHEST contrast (3401) yet
  over-detects, opposite to 144b256d (contrast 2556, under-detects). So it is an **embryo(nucleus-appearance)
  × density INTERACTION**, not a contrast/SNR knob and not a global density knob. 44b6 = late/dense/larger
  nuclei the detector under-segments; 6bba = early/sparse/distinct → over-detects on texture.
- IMPLICATION: **no clean embryo-AGNOSTIC pool rule exists** (density & contrast both fail to separate the
  two embryos). So an "SNR/density-adaptive NMS" would NOT transfer to a 3rd unseen embryo. The only
  embryo-agnostic transferable fix is a BETTER/multi-scale detector (more training), not a pool knob.

## DETECTOR-TRAIN v1 RESULT — FP-penalty (det_neg_weight) is a DEAD lever (2026-07-10, fold0)
Trained loeo_negw05_f0 (det_neg_weight 0.01→0.05, else = loeo_detector_aug), honest LOEO fold0 canonical:
negw05 = **0.7189** vs baseline(0.01) EXP_157 = 0.7273 → **-0.0084**. The lever WORKED mechanically
(predN/estN 1.0-1.49 → 0.90-1.20, over-detection fixed) BUT net-lost: node_recall 0.9687→0.9394 (-0.029),
edge_J 0.7419→0.7197 (-0.022) — the FP penalty suppressed TRUE nuclei, and that recall/edge loss outweighed
the small count-relief (baseline over-detection cost only a few % via the count term). VERDICT: over-PRODUCTION
is NOT the honest bottleneck; the detector's recall×precision PARETO is. det_neg_weight just MOVES ALONG the
frontier (net-negative). The real lever = a frontier-PUSHING change (higher recall AND precision): CONVERGENCE
(max_iters=150 cap = undertraining, [[gap-decomposition-detector-is-lever]]) / multi-scale / more data — NOT a
loss-weight tune. Next proposed = detector-train v2 'longer training' A/B (max_iters 150→400, epochs 12→30,
det_neg_weight back to 0.01).

## DETECTOR-TRAIN v2 RESULT — CONVERGENCE WINS (2026-07-10, leak-clean, fold0) — FIRST real detector gain
Clean A/B (both val-selected on 6bba-val, NEVER the 44b6 test → no test-selection confound): v2 (300it/20ep)
EXP_162 = **0.7322** vs clean-baseline (150it/12ep) EXP_161 = **0.7152** → **+0.0170 leak-clean lift**. Both
peaked @ep10 (v2 early-stop @ep14) → the lever is **max_iters (300 vs 150 iters/EPOCH = more data-passes)**,
NOT more epochs. TRUE honest fold0 baseline = 0.7152 (the old biased EXP_157=0.7276 carried +0.0124
test-selection inflation). NEXT: fold1 (44b6-val-holdout) to confirm the +0.017 on the other embryo (2-fold
LOEO win); push iters further (500-600/ep) for more lift. Convergence is the confirmed frontier-PUSH lever.

## Levers, ranked
1. **Better EMBRYO-DISJOINT detector training (THE honest lever).** The LOEO ceiling (~0.73) is set by
   one-embryo-trained detectors having less data than the both-embryo support-pack model. Path to a
   transferable >0.85: embryo-invariance augmentation (the loeo_detector_aug recipe — contrast/brightness/
   bias/blur, per E50), MORE training (loeo_129ep overfit at max_iters=150 cap → need more real epochs w/ aug),
   external data (zebrahub / box-sampled) for density+staining diversity, and possibly a stronger detector head.
   **Cost: GPU training (2 folds).** This is the only lever that raises the HONEST number.
2. **Adaptive NMS / pool_kernel_um — leader's LEAD HYPOTHESIS (2026-07-09).** The 44b6-under / 6bba-over split
   is INDEPENDENT of density (6bba_05db0fb1 is densest, estN 69,800, yet over-detects at 1.070; 44b6_144b256d
   estN 65,376 under-detects at 0.506). So pool5 OVER-MERGES distinct 44b6 nuclei (larger pool = more NMS
   suppression = fewer peaks → under-detect), while 6bba wants ~5.0. Directionally: 44b6 → SMALLER pool
   (recover merged peaks), 6bba → ~5.0. Verify via predN/estN + edge_J (node_recall useless on sparse GT).
   Could even REFINE the pool5 win. **Cost: GPU re-detect pool sweep (PARKED until LB).**
   TRANSFER CAVEAT — the fork the LB decides: a HARD per-EMBRYO pool (44b6=x, 6bba=5.0) only transfers if the
   LB test is the SAME two embryos (likely — the local test/ is 44b6+6bba); if the LB is a 3rd unseen embryo,
   per-embryo hard-coding is a proxy trap. The TRANSFERABLE form = **SNR/density-ADAPTIVE pool** (choose pool
   per-dataset from local SNR/crowding, embryo-agnostic) — the fleet data-audit's SNR/size characterization
   feeds exactly this. Prefer the adaptive form; use per-embryo only as the diagnostic upper bound.
3. **Linking / edge-precision (0b24845f-type).** Separate from detection; the div/flow + consensus levers —
   lower priority (div ceiling ~+0.005 [[beat-ceiling-postproc-and-div-classifier]]; consensus-prune parquet
   still 4/12).

## (a)-BRANCH RESULT — DEAD (2026-07-10): NMS-kernel tuning is NOT the honest lever.
Corrected fold0 LOEO sweep (--splits, DISTINCT kernels — the fine {3.0-4.5} sweep was a no-op, all k3):
k1 (pool2.0/kernel1) EXPLODES (predN 619k-1.68M, 24-60× over-detect → CUDA OOM); k3 (pool5.0/kernel3) =
0.7273 (control, predN parity 27630, BEST); k5 (pool6.0/kernel5) n=8 = 0.7117 (-0.0156, more suppression
hurts). The pool→kernel quantization is COARSE (only k1/k3/k5) and k3 (current EXP_156/157 setting) is
already optimal. REFRAME: on the honest --splits path loeo_detector_aug fold0 44b6 OVER-detects (predN/estN
1.0-1.5); the "under-detection" that motivated "smaller pool" was a --debug-video ARTIFACT. → NMS pool
lever CLOSED, no promotion. The only honest 44b6 lever left = detector training (lever 1 below).
[Superseded readiness note below kept for provenance.]

## (a)-BRANCH READY TO ROUTE (dryrun-GREEN, GPU-PARKED till LB)
Per-embryo pool sweep BUILT + dryrun-validated: predict_pool_by_embryo.sh (per-dataset --debug-video, pool
by embryo prefix — mechanism confirmed, no blocker) + config/pool_sweep_golden12.yml + reuse
config/loeo_redetect_f0.yml. Sweep 44b6 pool ∈ {3.0,3.5,4.0,4.5}, 6bba=5.0; score EACH on golden-12
canonical AND honest LOEO fold0 (--split-file fleet_loeo_mini8 --fold 0). pool=5.0 controls already exist
(golden-12 EXP_156=0.9161, LOEO fold0 EXP_157=0.7273). 8 GPU predict jobs handed to leader, NOT launched.
Routes the instant the LB confirms same-2-embryos; if LB shows a 3rd embryo, pivot to detector-training.

## What needs GPU (currently HELD pending human submission-fork)
Levers 1 and 2 both require GPU (training / re-detect sweeps). While GPU is held: (a) this diagnosis is CPU-done;
(b) can scope the loeo_detector_aug training-recipe deltas (epochs/aug/data) on paper; (c) can prep external-data
density-match (box-sample) for lever 1 without GPU. Execution waits for GPU GO.

## Recommendation
Prioritize **lever 1 (better embryo-disjoint training)** — it's the only path to a transferable number. Lever 2
is a cheap diagnostic only (proxy trap). Hold all GPU until the human's submission-fork decision; meanwhile
prep the training-recipe + external-data scope so lever 1 is queue-ready the moment GPU frees.
