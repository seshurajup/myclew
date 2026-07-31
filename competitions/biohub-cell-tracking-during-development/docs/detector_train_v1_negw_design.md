# Detector-training v1 — FP-penalty (det_neg_weight) A/B (design+dryrun, GPU-PARKED)

**Target the MEASURED honest failure:** loeo_detector_aug fold0 44b6 OVER-detects on the honest --splits
path (predN/estN 1.0–1.5) → the count penalty drags adj down; honest LOEO ceiling EXP_157 = 0.7276.
Postproc/NMS/recall/division all proven dead — detector training is the only remaining honest lever.

## Root cause (from the trainer) → the lever
Detection loss = per-voxel weighted BCE: `weight_pos=1/n_pos`, `weight_neg = det_neg_weight/n_neg`
(train_unet_transformer.py:590-597). loeo_detector_aug uses **det_neg_weight=0.01** — negatives are
barely penalized → the model predicts foreground liberally → OVER-DETECTS (exactly the predN 1.0–1.5 we
measured). This is a count-aware / harder-negative lever (leader's option 1), directly grounded in the data.

**Bounded change (clean A/B, ONE lever):** det_neg_weight **0.01 → 0.05** (5× FP penalty). Also batch_size
16→8 (dense fold OOMs 32GB@16; the k1-explosion showed OOM risk). Everything else identical to
loeo_detector_aug (12 epochs, lr 1e-4, det_loss_weight 10.0, pool 5.0, [1,4,4], the E50 aug stack). So the
result isolates the FP-penalty's effect on predN→estN.

## Configs (dryrun-GREEN)
- `config/loeo_negw05_f0.yml` — split 0 (train 6bba → test UNSEEN 44b6), the fold where over-detection was measured.
- `config/loeo_negw05_f1.yml` — split 1 (train 44b6 → test UNSEEN 6bba).
Both: `src/baseline/train.py --dry-run` PASS — aug OK, trainer import-check PASS (CUDA_VISIBLE_DEVICES=''),
resolved command carries `--det-neg-weight 0.05 --batch-size 8`, no GPU used.

## Eval (honest, embryo-disjoint)
Per fold: predict (--splits) → mtl10/gap5.5 → `score_golden12_official.py --split-file fleet_loeo_mini8
--fold {0,1}` (canonical) + `fleet_agents.cv_contract` leak-assert. PROMOTE only on canonical LOEO lift
over EXP_157=0.7276 (mean of fold0/1), ledger win-gate. Report per-ds predN/estN (did it tighten toward 1.0
without over-correcting into under-detection).

## GPU wall-time estimate
Training: 12 epochs × 150 iters/epoch = 1800 iters, bs8, 3D UNet → ~40–60 min/fold (the deferred 1-iter
smoke from the dry-run measures per-iter to confirm before the full run). Eval predict+score: ~15 min/fold.
**Total ≈ 2–2.5 hr GPU for the 2-fold A/B.** Route fold0 first (the measured failure).

## Expected honest-LOEO delta + risk
The count-penalty term (adj ∝ edge_J·min(1, estN/predN)) was dragging adj down at predN 1.0–1.5. A 5× FP
penalty should tighten predN→~1.0 → count-relief ≈ **+0.01 to +0.04** on honest LOEO over 0.7276, IF it
doesn't over-correct. RISK: 0.05 may over-shoot into under-detection (predN<1) → lost real nuclei → edge_J
drops. MITIGATION: bounded A/B — the per-ds predN/estN readout tells us directly; if fold0 under-detects,
drop to 0.03; if still over, raise to 0.1. This is a small honest gain at best (0.74 is far from 0.897); it
tests whether the detector's over-production is cheaply fixable before committing to bigger changes
(multi-scale head / more data), which are the next step if the FP-penalty saturates.

## Status
DESIGN + DRYRUN-GREEN, NOT launched. Fire-ready — route via the config-driven trainer on human GO (or LB).
