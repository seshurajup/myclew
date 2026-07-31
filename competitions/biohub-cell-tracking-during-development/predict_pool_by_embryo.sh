#!/usr/bin/env bash
# PER-EMBRYO POOL SWEEP driver (readiness prep for the (a)-branch; GPU-PARKED until LB).
# 44b6-dense over-merges at pool5 → sweep a SMALLER 44b6 pool while 6bba stays 5.0. Predict is PER-DATASET
# (--debug-video), so per-dataset pool control = pick pool by embryo prefix in the loop. Collects each geff
# (predict cleans its out-dir per call — same trap as the submission notebook). Researcher post-procs
# (mtl10/gap5.5) + canonical-scores after. Does NOT score here; does NOT auto-run on the fleet queue.
#
#   bash predict_pool_by_embryo.sh <config.yml> [det=0.99] [pool_44b6=4.0] [pool_6bba=5.0]
#   BIOHUB_DRYRUN=1 -> print the resolved per-ds plan, no GPU.
set -uo pipefail
CFG="${1:?usage: predict_pool_by_embryo.sh <config.yml> [det] [pool_44b6] [pool_6bba] [only_embryo]}"
DET="${2:-0.99}"; POOL44="${3:-4.0}"; POOL6B="${4:-5.0}"; ONLY_EMB="${5:-}"
# only_embryo (e.g. 44b6) = GPU-EFFICIENCY: on golden-12 re-detect ONLY 44b6 (reuse EXP_156's 6bba@5.0
# geffs unchanged); the scorer then combines the 6 swept 44b6 + the 6 EXP_156 6bba. Empty = all datasets.
ROOT="/home/seshu/kaggle/2026/biohub-cell-tracking-during-development"
PY="$ROOT/research/cellmot_venv/bin/python"
TEST_TRAIN="$ROOT/input/biohub-cell-tracking-during-development/train"

read -r METHOD FOLD SPLITS_REL < <("$PY" - "$ROOT/$CFG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1])); tr = cfg.get("train", {})
splits = (cfg.get("paths", {}) or {}).get("splits") or tr.get("splits") or cfg.get("splits")
print(tr.get("method") or cfg.get("name", "run"), str(tr.get("split", "0")), splits)
PYEOF
)
SPLITS="$ROOT/$SPLITS_REL"; [[ "$SPLITS_REL" = /* ]] && SPLITS="$SPLITS_REL"
WEIGHTS="$ROOT/research/official_repo/weights/${METHOD}/split_${FOLD}/edge_predictor_best.pth"
[[ -f "$WEIGHTS" ]] || { echo "ERROR: weights missing $WEIGHTS"; exit 3; }
# test datasets = this fold's test list from the splits file
mapfile -t DSS < <("$PY" -c "import json,sys; f=json.load(open('$SPLITS'))[$FOLD]; print('\n'.join(t.replace('.zarr','').replace('.geff','') for t in f['test']))")
TAG="${METHOD}_44b6p${POOL44//./}_6bbap${POOL6B//./}_t${DET//./}"
COLL="$ROOT/tools/researchpapers/output/pool_sweep/$TAG"

echo "########## POOL-SWEEP PREDICT :: $TAG  (det=$DET, 44b6 pool=$POOL44, 6bba pool=$POOL6B) ##########"
echo "  weights -> $(readlink -f "$WEIGHTS" 2>/dev/null || echo "$WEIGHTS")  | ${#DSS[@]} datasets"
if [[ "${BIOHUB_DRYRUN:-0}" == "1" ]]; then
  echo "[DRYRUN] per-dataset pool plan (NOT executed, no GPU)${ONLY_EMB:+  [only_embryo=$ONLY_EMB]}:"
  for ds in "${DSS[@]}"; do
    emb="${ds%%_*}"; [[ -n "$ONLY_EMB" && "$emb" != "$ONLY_EMB" ]] && { echo "   $ds  ->  SKIP (reuse EXP_156 ${emb}@${POOL6B})"; continue; }
    p=$([[ "$emb" == "44b6" ]] && echo "$POOL44" || echo "$POOL6B")
    echo "   $ds  ->  pool_kernel_um=$p"
  done
  echo "[DRYRUN OK] method=$METHOD fold=$FOLD weights_exist=yes splits=$SPLITS collect->$COLL"
  exit 0
fi
mkdir -p "$COLL"; cd "$ROOT/research/official_repo"
for ds in "${DSS[@]}"; do
  emb="${ds%%_*}"; [[ -n "$ONLY_EMB" && "$emb" != "$ONLY_EMB" ]] && continue   # reuse EXP_156 geffs for skipped embryo
  p=$([[ "$emb" == "44b6" ]] && echo "$POOL44" || echo "$POOL6B")
  z="$TEST_TRAIN/$ds.zarr"
  PYTHONPATH="src:scripts" BIOHUB_OUTPUT_FILTER_SHORT_TRACKS=1 BIOHUB_OUTPUT_MIN_TRACK_LEN=10 BIOHUB_GAP_CLOSE_UM=5.5 \
    "$PY" scripts/predict_unet_transformer.py --debug-video "$z" --weights "$WEIGHTS" \
      --method poolsweep --det-threshold "$DET" --pool-kernel-um "$p" --use-ilp 2>&1 | grep -E "ILP|Saved" | tail -1
  g=$(ls -d predictions/*/poolsweep/split_0/*.geff | head -1); cp -r "$g" "$COLL/$ds.geff"
done
echo "collected $(ls -d $COLL/*.geff | wc -l)/${#DSS[@]} -> $COLL"
echo "NEXT (researcher, CPU): stack_postproc_on_fresh.py mtl10/gap5.5 on $COLL, then score_golden12_official.py --pred-dir <stacked> [--split-file $SPLITS --fold $FOLD]"
