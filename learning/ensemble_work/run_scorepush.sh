#!/bin/bash
cd /home/seshu/kaggle/2026/biohub-cell-tracking-during-development
export BIOHUB_OUT_METHOD=scorepush
export BIOHUB_GAP_REFINE_SYNTHETIC=0
# lucifer19 score_push preset (0.888): gap2 ON + relaxed gates + lighter smoothing
export BIOHUB_OUTPUT_GAP2_RECOVERY=1
export BIOHUB_GAP2_MAX_TOTAL_UM=9.7
export BIOHUB_GAP2_MAX_STEP_UM=4.05
export BIOHUB_GAP2_MAX_LINKS_FRAC=0.0032
export BIOHUB_GAP2_MAX_LINKS_ABS=140
export BIOHUB_GAP2_FRAME_FRAC_CAP=0.0045
export BIOHUB_OUTPUT_LINEFIT_WEIGHT=0.72
export BIOHUB_OUTPUT_EDGE_MAX_UM=14.5
export BIOHUB_MOTION_RELINK_TIGHT_UM=6.2
export BIOHUB_MOTION_RELINK_RELAXED_UM=10.4
export BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT=0.52
export BIOHUB_MOTION_RELINK_LEARNED_BONUS=0.78
export BIOHUB_MOTION_RELINK_MAX_FRAME_NODES=2800
export BIOHUB_GAP_CLOSE_UM=6.2
export BIOHUB_GAP_CLOSE_REUSE_UM=3.4
research/cellmot_venv/bin/python learning/ensemble_work/build_pilk_full_geff.py
