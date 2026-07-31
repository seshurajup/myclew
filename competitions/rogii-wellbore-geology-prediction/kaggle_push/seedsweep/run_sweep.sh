#!/bin/bash
set -e
ROOT=/home/seshu/kaggle/2026/rogii-wellbore-geology-prediction
PY=$ROOT/.venv_rogii/bin/python
SCRIPT=$ROOT/kaggle_push/seedsweep/rogii_local.py
cd $ROOT/kaggle_local/working
for K in 2 3 4 5; do
  echo "===== SWEEP_SEEDS=$K start $(date +%T) ====="
  rm -f submission.csv
  t0=$SECONDS
  SWEEP_SEEDS=$K $PY -u $SCRIPT > $ROOT/kaggle_push/seedsweep/runs/seed$K.log 2>&1
  rc=$?
  cp submission.csv $ROOT/kaggle_push/seedsweep/out/sub_seed$K.csv
  echo "===== SWEEP_SEEDS=$K rc=$rc elapsed=$((SECONDS-t0))s rows=$(wc -l < submission.csv) $(date +%T) ====="
done
echo "ALL SWEEP DONE"
