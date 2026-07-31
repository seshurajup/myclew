#!/bin/bash
cd /home/seshu/kaggle/2026/biohub-cell-tracking-during-development
VENV=research/cellmot_venv/bin/python
GT=$(pwd)/input/biohub-cell-tracking-during-development/train
score() { CELLMOT_DATA_DIR=$GT USER=seshu PYTHONPATH=research/official_repo/src $VENV research/official_repo/scripts/evaluate.py --method "$1" 2>&1 | grep "$1/split_0:" | grep -oE "score=[0-9.]+ .*FN=[0-9]+\)"; }
run() { $VENV learning/ensemble_work/recover_div.py "$@" 2>&1 | tail -1; }

# method prob_thr frac reassign
declare -a CFG=(
  "divbase 1.1 0.02 0"
  "divadd05 0.5 0.03 0"
  "divre05 0.5 0.03 1"
  "divre03 0.3 0.05 1"
  "divre07 0.7 0.03 1"
)
for c in "${CFG[@]}"; do
  set -- $c
  echo ">>> $1 (thr=$2 frac=$3 reassign=$4)"
  run "$@"
  echo "    $(score $1)"
done
echo "=== base reference (with pilkwang safe-div ON) ==="
echo "    base_fast: $(score base_fast)"
echo "DONE_SWEEP"
