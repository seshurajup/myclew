# Aug-ablation `contrast` (train-9da500d0ec) — CONFOUNDED / SUPERSEDED, not a journey row

**Status: INVALID as an ablation. Do NOT use as a kept/rejected data point.**

## What ran
- Job `train-9da500d0ec`, `config/aug_ablation/contrast.yml`, fleet-auto-submitted (deterministic fleet runner), succeeded exit 0, ~33 min GPU, **split_0 only** (single fold).
- Weights: `research/official_repo/weights/augabl_contrast/split_0/edge_predictor_best.pth`.
- Final **training-proxy** metric only: `acc*recall = 0.9513` (det_loss=0.0141, edge_loss=0.0009). **Never scored through `official_scorer`** → no official CV number, no by-embryo adjJ.

## Why it's invalid (per human aug_journey directive, 2026-07-05)
The existing `config/aug_ablation/*.yml` are **confounded**: `base.yml` = brightness+flip (NOT no-aug), and `contrast.yml` (like noise/gamma/rot90) **bakes brightness+flip in on top of contrast**. So this run does not isolate the `contrast` augmentation — it measures contrast+brightness+flip together. It also ran out-of-order (Stage-3 aug work before the Stage-1 baseline was scored) and jumped the `:7799` queue via fleet auto-submit.

## Disposition
- **Superseded** by the clean, isolated re-authoring in the ordered **aug_journey**: `00_no_aug.yml` reference + `31_contrast.yml` = no-aug + contrast ONLY, screened on `splits_screen_matched` (+ k6), judged by-embryo (adjJ_44b6 / adjJ_6bba).
- Ledger: log with `observation=confounded (brightness+flip baked in), single-fold, training-proxy only — superseded by clean 31_contrast`. Do not count toward kept/rejected.
