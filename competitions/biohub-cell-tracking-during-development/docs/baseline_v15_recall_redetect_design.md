# baseline_v15 — RECALL-recovery GPU re-detect (pilk support-pack, golden-12)

**Date:** 2026-07-09 · **Author:** researcher · Bar: **EXP_154 = 0.8837 canonical** (mtl10/gap6.0).

## Goal
node_recall is saturated on the cached pilk pipeline (~0.977–0.988; canonical full-12 mtl10 = 0.9774),
but the metric is edge_J ≈ node_rec²·Q_link, so recall still has leverage (0.984→1.0 alone ≈ 0.91 per
human decomposition). A **real GPU re-detect at a LOWER decode threshold** recovers the last ~1.6% missed
nuclei (dim/crowded) that the cached preds cannot — `BIOHUB_DET_THRESHOLD` is a **no-op on cache**, it only
bites at detection time (`predict_unet_transformer.py:285` `sigmoid(logits) > det_threshold`). Then re-run
ILP + the mtl10/gap6.0 post-proc and **canonically score full-12** (`score_golden12_official.py`),
promoting only past 0.8837.

## Weights mechanism — OPTION (a), PROVEN, predict_and_score.sh UNMODIFIED
`predict_and_score.sh` builds `WEIGHTS=research/official_repo/weights/${METHOD}/split_0/edge_predictor_best.pth`
from `train.method` only (no config override for the weights path). To force the **support-pack** checkpoint
without editing the shared script (it must keep resolving official_repo for other `score` dispatches), a
method-indirection **symlink**:

```
research/official_repo/weights/pilk_redetect/split_0/edge_predictor_best.pth
   -> ../../../../pilkwang_support_pack/weights/unet_transformer/split_0/edge_predictor_best.pth
research/official_repo/weights/pilk_redetect/split_0/config.json
   -> ../../../../pilkwang_support_pack/weights/unet_transformer/split_0/config.json   # downsample [1,4,4], window 2
```

Config `config/pilk_redetect.yml` sets `train.method: pilk_redetect`. **Proof:**
`readlink -f research/official_repo/weights/pilk_redetect/split_0/edge_predictor_best.pth`
→ `.../pilkwang_support_pack/weights/unet_transformer/split_0/edge_predictor_best.pth`. Arch from the
symlinked config.json = **[1,4,4]** (NOT the (1,2,2) official_repo-copy trap). Provenance-clean (versioned
in the config), other methods unaffected.

**GPU-free wiring validation (dryrun-GREEN):** the exact `predict_and_score.sh` parse resolves
METHOD=pilk_redetect, SPLITS=golden12_splits.json (12 ds = 6 44b6 + 6 6bba), POOL=5.0, WEIGHTS EXISTS via
symlink, config.json downsample [1,4,4].

## BLOCKER — det-threshold (the recall lever) is hardwired
`predict_and_score.sh` hardwires `--det-threshold 0.99`. The recall-tilt sweep (0.985/0.98/0.97) **cannot**
be expressed through the `score` kind (`score_step.py` also hardwires `script_path=predict_and_score.sh`,
`script_args=[cfg]`). At 0.99 this config re-detects at the SAME threshold as the cached baseline → reproduces
~0.87, i.e. NO recall lever. Options:
- **(A) RECOMMENDED — non-breaking:** add a dedicated `predict_and_score_pilk.sh` (support-pack predict +
  `--det-threshold $2`) and a ~3-line backward-compatible override in `score_step.py` to honor optional
  `spec["script"]` + `spec["extra_args"]` (defaults to `predict_and_score.sh`; existing dispatches unchanged).
  Shared `predict_and_score.sh` stays official-repo-only.
- (B) edit `predict_and_score.sh` to read det-threshold from config/env — **vetoed** (corrupts other `score` jobs).

## Plan (once threshold mechanism approved + routed via :7788)
1. SCREEN (fast, GPU): re-detect one golden-12 dataset (`--slice ':1'`) across det-threshold {0.985, 0.98, 0.97}
   + `recall_proxy` → pick the recall/count knee.
2. CONFIRM (GPU): full-12 re-detect at the chosen threshold → ILP → mtl10/gap6.0 post-proc.
3. SCORE (CPU): `score_golden12_official.py --pred-dir <out>` (canonical), promote via `promote_to_ledger.py`
   only if > 0.8837. bs=8 in the config only matters if we ever fine-tune (dense 44b6 OOMs @ bs16).

## Status
Weights path GREEN + proven. Detector re-detect + threshold sweep held for leader's approval of mechanism (A)
and the :7788 route (GPU serialized behind the CPU levers).
