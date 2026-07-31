#!/usr/bin/env bash
# GENERIC post-training predict+score for ONE config (the loop-closer, competition-level).
#   Mirror of start_train.sh but for the SCORE half: given the SAME config that trained a model,
#   predict its detections from the trained weights (GPU) then run the official golden-CV scorer
#   (CPU) -> logs golden_cv/official_score to MLflow 'kaggle-biohub-cell-tracking' + a score.json.
#
# Works for any config whose train.method + official_repo trainer wrote
#   research/official_repo/weights/<method>/split_0/edge_predictor_best.pth
# (baseline_v1_*, aug_ablation/*, and any config using train_unet_transformer.py).
#
#   bash predict_and_score.sh config/aug_ablation/noise.yml
set -uo pipefail

CFG="${1:?usage: predict_and_score.sh <config.yml>}"
ROOT="/home/seshu/kaggle/2026/biohub-cell-tracking-during-development"
PY="$ROOT/research/cellmot_venv/bin/python"
TRAIN="$ROOT/input/biohub-cell-tracking-during-development/train"
SPLITS="$ROOT/learning/ensemble_work/finetune/splits_ft.json"

# method + fold + splits come FROM THE CONFIG (generic — no hardwired experiment set/split)
read -r METHOD FOLD SPLITS_REL POOL_KERNEL < <("$PY" - "$ROOT/$CFG" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
tr = cfg.get("train", {})
splits = (cfg.get("paths", {}) or {}).get("splits") or tr.get("splits") or cfg.get("splits") \
         or "learning/ensemble_work/finetune/splits_ft.json"
# predict_unet_transformer.py takes pool_kernel_um from the CLI (default 3.0) and IGNORES config.json;
# must pass the CONFIG's value (5.0 = pilkwang) or predict over-detects (3.0 -> ~x1.3 nodes -> adjJ ~0.75)
print(tr.get("method") or cfg.get("name", "run"), str(tr.get("split", "0")), splits,
      str(tr.get("pool_kernel_um", "5.0")))
PYEOF
)
# journey contract: PREDICT on the CONFIG's own split (screen_matched / stagebridge), never hardwired splits_ft
if [[ -n "${SPLITS_REL:-}" ]]; then
  [[ "$SPLITS_REL" = /* ]] && SPLITS="$SPLITS_REL" || SPLITS="$ROOT/$SPLITS_REL"
fi
WEIGHTS="$ROOT/research/official_repo/weights/${METHOD}/split_0/edge_predictor_best.pth"
LOGDIR="$ROOT/tools/researchpapers/output/score/${METHOD}"
mkdir -p "$LOGDIR"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "ERROR: checkpoint not found: $WEIGHTS"
  echo "  (the official_repo trainer writes weights/<train.method>/split_0/; has training for '$METHOD' finished?)"
  exit 3
fi

echo "########## PREDICT (GPU) :: ${METHOD} (from $CFG) ##########"
# config.json beside the weights sets downsample/window automatically; same post-proc as the 0.8708 baseline.
cd "$ROOT/research/official_repo"
PYTHONPATH="src:scripts" "$PY" scripts/predict_unet_transformer.py \
  --data-dir "$TRAIN" --splits "$SPLITS" --split "${FOLD:-0}" \
  --weights "$WEIGHTS" --method "$METHOD" \
  --det-threshold 0.99 --pool-kernel-um "${POOL_KERNEL:-5.0}" 2>&1 | tee "$LOGDIR/predict.log"

echo ""
echo "########## SCORE (CPU, official) :: ${METHOD} on its OWN test split ##########"
cd "$ROOT/tools/researchpapers"
# CRITICAL: score on the SAME split we predicted on (--split-file) — NOT hardwired golden-12. This is
# what makes stagebridge (train 6bba / test 44b6) actually measure adjJ_44b6; else predict(44b6) vs
# score(golden-12) mismatch → 0 geffs → cv=None. golden-12-based configs still resolve to golden-12.
"$PY" baseline/score_v1.py --method "$METHOD" --run-name "$METHOD" \
  --config-file "$ROOT/$CFG" --split-file "$SPLITS" 2>&1 | tee "$LOGDIR/score.log"
echo ""
echo "done: ${METHOD}  (predict.log + score.log in $LOGDIR)"
