#!/bin/bash
# Build + score rule-based variants on pilkwang base (golden-12, official metric).
cd /home/seshu/kaggle/2026/biohub-cell-tracking-during-development
GT=$(pwd)/input/biohub-cell-tracking-during-development/train
VENV=research/cellmot_venv/bin/python
EW=learning/ensemble_work
PRED=research/official_repo/predictions/seshu

score() { CELLMOT_DATA_DIR=$GT USER=seshu PYTHONPATH=research/official_repo/src $VENV research/official_repo/scripts/evaluate.py --method "$1" 2>&1 | grep -oE "score=[0-9.]+ .*node_recall=[0-9.]+"; }
build() { # method + env assignments
  local m=$1; shift
  ( cd $EW && env "$@" BIOHUB_GAP_REFINE_SYNTHETIC=0 ../../$VENV rule_variant.py "$m" >/dev/null 2>&1 )
}

# wait for base_fast (already launched) to finish 12
until [ "$(ls -d $PRED/base_fast/split_0/*.geff 2>/dev/null | wc -l)" -ge 12 ]; do sleep 5; done
echo "base_fast   : $(score base_fast)"

# variant: safe divisions OFF (remove the 30 FP divisions)
build nodiv BIOHUB_OUTPUT_SAFE_DIVISIONS=0
echo "nodiv       : $(score nodiv)"

# variant: tighter safe-div gates (fewer FP divisions, keep only very-safe)
build divtight BIOHUB_SAFE_DIV_MAX_UM=3.5 BIOHUB_SAFE_DIV_SISTER_MAX_UM=5.0 BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM=5.0
echo "divtight    : $(score divtight)"

# variant: motion relink tighter (fewer wrong edges)
build relinktight BIOHUB_MOTION_RELINK_TIGHT_UM=5.0 BIOHUB_MOTION_RELINK_RELAXED_UM=8.0 BIOHUB_OUTPUT_SAFE_DIVISIONS=0
echo "relinktight : $(score relinktight)"

# variant: motion relink looser
build relinkloose BIOHUB_MOTION_RELINK_TIGHT_UM=7.0 BIOHUB_MOTION_RELINK_RELAXED_UM=11.0 BIOHUB_OUTPUT_SAFE_DIVISIONS=0
echo "relinkloose : $(score relinkloose)"

# variant: edge max prune tighter (drop long wrong edges)
build prunetight BIOHUB_OUTPUT_EDGE_MAX_UM=11.0 BIOHUB_OUTPUT_SAFE_DIVISIONS=0
echo "prunetight  : $(score prunetight)"

echo "DONE"
