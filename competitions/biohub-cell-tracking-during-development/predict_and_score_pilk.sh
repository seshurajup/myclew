#!/usr/bin/env bash
# RECALL-recovery variant of predict_and_score.sh — PARAMETERISED det-threshold + optional --slice.
# Dedicated to the pilk support-pack re-detect (config/pilk_redetect.yml). The shared
# predict_and_score.sh is UNTOUCHED (it stays official_repo for every other {kind:score} dispatch);
# this script is invoked only when a fleet spec sets spec["script"]=predict_and_score_pilk.sh.
#
# Uses the OFFICIAL_REPO predict copy (it exposes --pool-kernel-um and --slice; the support-pack copy
# pins pool_kernel_um=3.0 → over-detects → ~0.75) with the SUPPORT-PACK WEIGHTS resolved via the
# method-indirection symlink (research/official_repo/weights/<method>/split_0 → pilkwang_support_pack).
# So: support-pack WEIGHTS + [1,4,4] config + pool 5.0, only the det-threshold changes.
#
#   bash predict_and_score_pilk.sh <config.yml> [det_threshold=0.99] [slice=""]
#   e.g.  bash predict_and_score_pilk.sh config/pilk_redetect.yml 0.98
#         bash predict_and_score_pilk.sh config/pilk_redetect.yml 0.98 ':1'   # fast screen, 1 dataset
set -uo pipefail

CFG="${1:?usage: predict_and_score_pilk.sh <config.yml> [det_threshold] [slice] [pool_override]}"
DET="${2:-${BIOHUB_DET_THRESHOLD:-0.99}}"
SLICE="${3:-${BIOHUB_SLICE:-}}"
POOL_OVERRIDE="${4:-}"
ROOT="/home/seshu/kaggle/2026/biohub-cell-tracking-during-development"
PY="$ROOT/research/cellmot_venv/bin/python"
TRAIN="$ROOT/input/biohub-cell-tracking-during-development/train"
SPLITS="$ROOT/learning/ensemble_work/finetune/splits_ft.json"

read -r METHOD FOLD SPLITS_REL POOL < <("$PY" - "$ROOT/$CFG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
tr = cfg.get("train", {})
splits = (cfg.get("paths", {}) or {}).get("splits") or tr.get("splits") or cfg.get("splits") \
         or "learning/ensemble_work/finetune/splits_ft.json"
print(tr.get("method") or cfg.get("name", "run"), str(tr.get("split", "0")), splits,
      str(tr.get("pool_kernel_um", "5.0")))
PYEOF
)
if [[ -n "${SPLITS_REL:-}" ]]; then
  [[ "$SPLITS_REL" = /* ]] && SPLITS="$SPLITS_REL" || SPLITS="$ROOT/$SPLITS_REL"
fi
# optional pool override (pool3 control vs pool5 treatment for the LOEO delta)
[[ -n "$POOL_OVERRIDE" ]] && POOL="$POOL_OVERRIDE"
POOL_TAG="p${POOL//./}"
# per-(threshold,pool) method label so sweep outputs don't collide; weights resolve from the FOLD's split dir
DET_TAG="${DET//./}"
OUT_METHOD="${METHOD}_t${DET_TAG}_${POOL_TAG}"
WEIGHTS="$ROOT/research/official_repo/weights/${METHOD}/split_${FOLD}/edge_predictor_best.pth"
# mirror the base method's weights symlink dir (from the FOLD's split) to the output method
mkdir -p "$ROOT/research/official_repo/weights/${OUT_METHOD}/split_${FOLD}"
ln -sf "../../${METHOD}/split_${FOLD}/edge_predictor_best.pth" "$ROOT/research/official_repo/weights/${OUT_METHOD}/split_${FOLD}/edge_predictor_best.pth"
ln -sf "../../${METHOD}/split_${FOLD}/config.json"            "$ROOT/research/official_repo/weights/${OUT_METHOD}/split_${FOLD}/config.json"
LOGDIR="$ROOT/tools/researchpapers/output/score/${OUT_METHOD}"
mkdir -p "$LOGDIR"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: checkpoint not found: $WEIGHTS  (is the pilk_redetect symlink in place?)"; exit 3
fi
echo "########## PREDICT (GPU) :: ${OUT_METHOD} det-threshold=${DET} pool=${POOL} slice='${SLICE}' ##########"
echo "  weights -> $(readlink -f "$WEIGHTS")"
cd "$ROOT/research/official_repo"
SLICE_ARG=(); [[ -n "$SLICE" ]] && SLICE_ARG=(--slice "$SLICE")
if [[ "${BIOHUB_DRYRUN:-0}" == "1" ]]; then
  echo "[DRYRUN] resolved predict command (NOT executed, no GPU):"
  echo "  PYTHONPATH=src:scripts $PY scripts/predict_unet_transformer.py --data-dir $TRAIN --splits $SPLITS --split ${FOLD:-0} --weights $WEIGHTS --method $OUT_METHOD --det-threshold $DET --pool-kernel-um ${POOL:-5.0} ${SLICE_ARG[*]}"
  echo "[DRYRUN OK] method=$OUT_METHOD det=$DET pool=$POOL weights_exist=$([[ -f $WEIGHTS ]] && echo yes || echo NO) splits=$SPLITS"
  echo "PRED_GLOB=$ROOT/research/official_repo/predictions/*/${OUT_METHOD}/split_0"
  exit 0
fi
PYTHONPATH="src:scripts" "$PY" scripts/predict_unet_transformer.py \
  --data-dir "$TRAIN" --splits "$SPLITS" --split "${FOLD:-0}" \
  --weights "$WEIGHTS" --method "$OUT_METHOD" \
  --det-threshold "$DET" --pool-kernel-um "${POOL:-5.0}" "${SLICE_ARG[@]}" 2>&1 | tee "$LOGDIR/predict.log"

# Predictions land at research/official_repo/predictions/<user>/${OUT_METHOD}/split_0/*.geff.
# CANONICAL scoring is done by the researcher via score_golden12_official.py --pred-dir <that dir>
# (NOT src.metric); this script only re-detects. Echo the pred dir for the caller.
PRED_DIR="$ROOT/research/official_repo/predictions"
echo ""
echo "PRED_METHOD=${OUT_METHOD}"
echo "PRED_GLOB=${PRED_DIR}/*/${OUT_METHOD}/split_0"
echo "done: re-detect ${OUT_METHOD} @ det=${DET}  (predict.log in $LOGDIR)"
