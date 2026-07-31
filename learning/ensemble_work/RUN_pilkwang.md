# Reproduce pilkwang 0.885 locally — runbook

Env: `research/cellmot_venv/bin/python` (has tracksdata, pyscipopt, geff, torch/CUDA).

## 1. Detection + edge-transformer + ILP → geffs
```
cd research/pilkwang_support_pack/repo
PYTHONPATH=src BIOHUB_DET_THRESHOLD=0.99 <venv> scripts/predict_unet_transformer.py \
  --data-dir <TRAIN> --splits <splits.json> --split 0 \
  --weights ../weights/unet_transformer/split_0/edge_predictor_best.pth \
  --det-threshold 0.99 --use-ilp
```
splits.json = `[{"train":[], "test":["<stem>.zarr", ...]}]`.
Output geffs → `predictions/seshu/unet_transformer/split_0/<stem>.zarr.geff`.
Timing: ~14s/dataset sparse 44b6; dense 6bba slower. ~48 min for all 199 (detection+ILP).

## 2. Post-process (motion-relink + gap + safe-div + smooth) + score
`learning/ensemble_work/score_pilkwang.py` — loads geffs, applies `pilk_post.filter_output_graph`
(verbatim notebook post-proc, env-var defaults), scores vs GT with `src.metric`, saves node
detections to `pilkwang_nodes/<stem>.csv`.
Post-proc is slow on dense 6bba (~1-2 min each, frame-reads for gap refine).
