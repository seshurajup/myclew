#!/usr/bin/env bash
# Config-driven TEACHER pseudo-label generation (exp2 way).
#   bash gen_labels.sh config/exp2_labels.yml
# All teacher settings (TTA, union merge, linker, filter, smoothing) come from the YAML.
set -euo pipefail
ROOT=/home/seshu/kaggle/2026/biohub-cell-tracking-during-development
CFG=${1:?"usage: bash gen_labels.sh config/exp2_labels.yml"}
export PYTHONPATH="$ROOT/research/official_repo/src:$ROOT/research/official_repo/scripts:$ROOT"
exec "$ROOT/research/cellmot_venv/bin/python" "$ROOT/experiments/zebrahub/exp2_gen_pseudolabels.py" --config "$CFG"
