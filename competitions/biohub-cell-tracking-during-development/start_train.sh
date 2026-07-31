#!/usr/bin/env bash
# Config-driven detector training, logged to MLflow.
#   bash start_train.sh config/exp1.yml
# The YAML fully declares the experiment (paths, cache, MLflow, hyperparams). No CLI overrides needed.
# Extra trainer flags may be appended, but the exp way is: one config per experiment.
set -uo pipefail
ROOT=/home/seshu/kaggle/2026/biohub-cell-tracking-during-development
CFG=${1:?"usage: bash start_train.sh config/exp1.yml"}
PY="$ROOT/research/cellmot_venv/bin/python"
# model_scratch/config/* are from-scratch DETECTOR configs → run via train_v0.py (NOT train_from_config.py,
# which expects a `paths:` block). This is the fix heal diagnosed for `KeyError: 'paths'`.
case "$CFG" in
  model_scratch/config/*)
    # topk/set decode configs need the POINTS (peak) cache; add --peak-cache + the prebuilt cache-dir
    # (fix heal diagnosed for `KeyError: 'tgt'` — the .npz had keys [img,points], not [img,tgt,mask]).
    EXTRA=""
    if grep -qE "decode:\s*(topk|set)" "$ROOT/$CFG" 2>/dev/null; then
      EXTRA="--peak-cache --cache-dir $ROOT/model_scratch/results/cache_peak_v3"
    fi
    exec "$PY" "$ROOT/model_scratch/train_v0.py" --config "$CFG" $EXTRA "${@:2}" ;;
  *)
    exec "$PY" "$ROOT/scripts/train_from_config.py" "$@" ;;
esac
