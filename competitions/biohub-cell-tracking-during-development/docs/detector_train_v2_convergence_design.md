# Detector-train v2 — CONVERGENCE A/B (design+dryrun, GPU-PARKED/fire-ready)

**Why:** v1 proved det_neg_weight is a dead frontier-SLIDE (over-production not the bottleneck; recall×precision
Pareto is). v2 tests a frontier-PUSH: does CONVERGING the aug'd model (the max_iters=150 cap is a known
undertraining artifact, [[gap-decomposition-detector-is-lever]]) lift the honest LOEO ceiling (EXP_157=0.7276).

## The A/B (dryrun-GREEN): `config/loeo_conv_f0.yml`
max_iters 150→**400**, epochs 12→**30**, det_neg_weight back to **0.01**, bs 16→8 (OOM safety), else identical
to loeo_detector_aug. `src/baseline/train.py --dry-run` PASS (resolved cmd: `--epochs 30 --max-iters 400
--det-neg-weight 0.01 --batch-size 8`, import-check PASS, no GPU).

## Overfit mitigation — the trainer ALREADY has it (set patience)
- **best-ckpt, NOT last:** trainer saves `edge_predictor_best.pth` = the highest eval edge-Jaccard epoch
  (train_unet_transformer.py:951/1210), so "longer" cannot ship an overfit-last model.
- **early-stop:** env `CELLMOT_EARLY_STOP_PATIENCE=4` → stop after 4 epochs with no new best (line 1216/1300).
- **EMA** (best ckpt = EMA weights, line 1184) + the E50 aug stack = regularizers already on.
  → Set `CELLMOT_EARLY_STOP_PATIENCE=4` in the training dispatch env; nothing else needed.

## ⚠ CONFOUND — best-ckpt/early-stop select on the LOEO TEST embryo (line 1254 evals `test_loader` = 44b6)
This means (a) the absolute LOEO is optimistically biased (selecting ON test), and (b) it CONFOUNDS the
convergence A/B: v2's 30 epochs get more "max-over-test-epochs" luck than v1's 12 → a v2>v1 could be selection
inflation, not real convergence. **LEAK-CLEAN UPGRADE (recommended for a rigorous verdict):** a train-embryo
(6bba) VAL HOLDOUT — a training splits file `[{train: 6bba-minus-16, test: 6bba-val-16}]` so best-ckpt/early-stop
select on 6bba-val (never touches 44b6), then eval the fixed ckpt on the real 44b6 LOEO test. I can build this in
~5 min on request. Without it, interpret v2's LOEO conservatively (best-on-test optimistic). — Leader's call:
fire the simple best-on-test v2 now, or the leak-clean val-holdout version.

## GPU wall-time
30 ep × 400 iters = 12,000 iters (6.7× v1's 1,800) → ~**3–5 hr** training at v1's ~1–1.5 s/iter; early-stop
(patience 4) likely cuts to ~**2–3.5 hr** if it plateaus/overfits by epoch ~15–20. Eval predict+score ~15 min.
(If ≤4 hr is a hard cap, drop to max_iters 300 / epochs 20 = 6,000 iters ~ 2–3 hr.)

## Overfit DETECTION (the fold0 curve, req 3)
The trainer prints per-epoch test edge-Jaccard + recall (line 1254). I capture the per-epoch curve →
**overfit = the test-eval curve PEAKS then DECLINES** (best-epoch << final-epoch); I report the epoch-of-best,
the peak-vs-final gap, and whether early-stop triggered. (With the leak-clean upgrade, this becomes the honest
6bba-val curve.) A monotone-or-plateau curve = converged, not overfit.

## Eval + promotion
Predict 44b6 LOEO test (fleet_loeo_mini8 fold0) with the v2 ckpt → mtl10/gap5.5 → canonical `--split-file
fleet_loeo_mini8 --fold0` + cv_contract. PROMOTE only on canonical lift over baseline 0.7273, ledger win-gate.

## Status
DESIGN + DRYRUN-GREEN, NOT launched. Fire-ready. Recommend the leak-clean val-holdout before launch for a
trustworthy convergence verdict — one word and I build it.
