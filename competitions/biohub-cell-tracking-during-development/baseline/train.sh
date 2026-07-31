#!/bin/bash
#
# baseline/train.sh — Complete training orchestrator
#
# This script captures ALL training knowledge in one place:
# - Config selection, dry-run validation, GPU setup, MLflow tracking
# - Result logging, reproducibility checks
#
# Usage:
#   ./baseline/train.sh --quick              (10-min dry-run)
#   ./baseline/train.sh --loeo-cv            (LOEO training)
#   ./baseline/train.sh --detector-tune      (Recall-tilted detector)
#   ./baseline/train.sh --help               (Full help)
#

set -euo pipefail

COMP="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$COMP"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[✓]${NC} $*"; }
log_warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ============================================================================
# CONFIG
# ============================================================================

MLFLOW_TRACKING_URI="http://localhost:5000"
TRAINER_SCRIPT="model_scratch/train_v0.py"
LAUNCHER_SCRIPT="scripts/train_from_config.py"
DATA_ROOT="input/biohub-cell-tracking-during-development/train"
LEDGER_FILE="docs/experiment_ledger.jsonl"
CUDA_DEVICE="${CUDA_VISIBLE_DEVICES:-0}"
POWER_LIMIT_W=400

# ============================================================================
# REPRODUCIBILITY CHECK
# ============================================================================

check_reproducibility() {
  local config="$1"
  
  log_info "Reproducibility checklist..."
  
  [[ -f "$config" ]] && log_success "Config exists" || { log_error "Config not found"; return 1; }
  [[ -d "$DATA_ROOT" ]] && log_success "Data exists" || { log_warn "Data path may not exist"; }
  
  if python "$TRAINER_SCRIPT" --dry-run --config "$config" &>/dev/null; then
    log_success "Dry-run passed (YAML valid, paths resolve)"
  else
    log_error "Dry-run failed"
    return 1
  fi
}

# ============================================================================
# GPU & MLFLOW
# ============================================================================

setup_gpu() {
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
  log_success "CUDA_VISIBLE_DEVICES=$CUDA_DEVICE"
  
  if command -v nvidia-smi &>/dev/null; then
    nvidia-smi -pl "$POWER_LIMIT_W" &>/dev/null && \
      log_success "Power limit: ${POWER_LIMIT_W}W" || \
      log_warn "Could not set power limit"
  fi
}

setup_mlflow() {
  export MLFLOW_TRACKING_URI
  
  if curl -s "$MLFLOW_TRACKING_URI" >/dev/null 2>&1; then
    log_success "MLflow ready at $MLFLOW_TRACKING_URI"
  else
    log_warn "MLflow not running (optional, but recommended)"
  fi
}

# ============================================================================
# TRAINING
# ============================================================================

run_training() {
  local config="$1"
  shift
  
  log_info "========== TRAINING: $config =========="
  
  check_reproducibility "$config" || return 1
  setup_gpu
  setup_mlflow
  
  log_info "Starting training..."
  python "$LAUNCHER_SCRIPT" "$config" "$@" || return 1
  
  log_success "Training completed"
  [[ -f "$LEDGER_FILE" ]] && log_success "Results logged" || true
}

# ============================================================================
# RECIPES
# ============================================================================

recipe_quick_test() {
  log_info "Quick test (10 min, GPU-safe)..."
  python "$TRAINER_SCRIPT" --dry-run
  log_success "✓ Schema valid, paths resolve"
}

recipe_loeo_cv() {
  log_info "LOEO CV (train one embryo, eval other)..."
  run_training "config/loeo_detector.yml" --epochs 25 --patience 5
}

recipe_detector_tune() {
  log_info "Detector tuning (recall-tilted, pilkwang recipe)..."
  run_training "model_scratch/config/exp_det_v2_recall.yml" \
    --epochs 25 --patience 5 --eval-every 2
}

recipe_single_aug() {
  local aug="${1:-flip_xy}"
  log_info "Testing augmentation: $aug"
  run_training "config/aug_ablation/${aug}.yml" --epochs 10
}

recipe_replicate_public() {
  local nb="${1:-pilkwang}"
  log_info "Replicating: $nb"
  run_training "config/exp/public/${nb}.yml"
}

recipe_ablation() {
  log_info "Full ablation suite (35 augmentations, ~24 hours)..."
  for cfg in config/aug_ablation/*.yml; do
    log_info "Running: $(basename $cfg)"
    run_training "$cfg" --epochs 10 || log_warn "Failed: $cfg"
  done
  log_success "Ablation complete"
}

list_configs() {
  echo -e "${YELLOW}ALL 130 CONFIGS:${NC}\n"
  echo "AUGMENTATION (35): config/aug_ablation/"
  ls config/aug_ablation/*.yml | head -3 | sed 's|^|  |'; echo "  ... + 32 more"
  echo ""
  echo "AUTO-GENERATED (15): config/_auto/"
  ls config/_auto/*.yml | head -3 | sed 's|^|  |'; echo "  ... + 12 more"
  echo ""
  echo "PUBLIC NOTEBOOKS (60): config/exp/public/"
  ls config/exp/public/*.yml | head -3 | sed 's|^|  |'; echo "  ... + 57 more"
  echo ""
  echo "DETECTOR (5): model_scratch/config/"
  ls model_scratch/config/exp_det*.yml | sed 's|^|  |'
  echo ""
  echo "For full list: find config/ model_scratch/config/ -name '*.yml'"
}

show_help() {
  cat << 'EOF'
TRAINING SCRIPT — Complete training orchestrator for biohub cell tracking

═══════════════════════════════════════════════════════════════════════════

QUICK START:

  # Dry-run (validate, no GPU needed)
  ./baseline/train.sh --quick

  # Standard training (LOEO CV)
  ./baseline/train.sh --loeo-cv

  # Detector tuning (recall-tilted, pilkwang recipe)
  ./baseline/train.sh --detector-tune

═══════════════════════════════════════════════════════════════════════════

RECIPES (Common patterns):

  --quick                 10-min dry-run (GPU-safe)
  --loeo-cv               Leave-one-embryo-out (train one, eval other)
  --detector-tune         Detector tuning (recall-tilted BCE + topk decode)
  --single-aug <aug>      Test one augmentation (e.g. flip_xy, noise)
  --replicate <nb>        Replicate public notebook (pilkwang, yusuketogashi, etc.)
  --ablation              Full suite (35 augmentations, ~24 hours)
  --list                  Show all 130 configs

═══════════════════════════════════════════════════════════════════════════

RUN SPECIFIC CONFIG:

  # By path
  ./baseline/train.sh config/loeo_detector.yml
  ./baseline/train.sh model_scratch/config/exp_det_v2_recall.yml

  # With overrides
  ./baseline/train.sh config/loeo_detector.yml --epochs 30 --patience 5

═══════════════════════════════════════════════════════════════════════════

WHAT THIS SCRIPT DOES:

  1. Reproducibility Check
     ✓ Verifies config exists
     ✓ Checks data paths resolve
     ✓ Runs dry-run (YAML validation, GPU-safe)
     
  2. GPU Setup
     ✓ Sets CUDA device
     ✓ Manages power limit (thermal safety on RTX 5090)
     
  3. MLflow Setup
     ✓ Connects to http://localhost:5000 (optional)
     ✓ Logs metrics, hyperparams, artifacts
     
  4. Training
     ✓ Runs config via scripts/train_from_config.py
     ✓ Monitors golden-CV (full pipeline metric)
     ✓ Saves checkpoint on best golden-CV
     ✓ Logs result to docs/experiment_ledger.jsonl

═══════════════════════════════════════════════════════════════════════════

TRAINING DISCIPLINE (What makes it reproducible):

  ✓ Seed fixed             (experiment.seed from config)
  ✓ Data paths resolved    (absolute paths)
  ✓ CV split verified      (embryo-disjoint, no leakage)
  ✓ Metric is golden-CV    (NOT loss, NOT recall alone)
  ✓ Checkpoint on golden-CV (end-to-end through linker)
  ✓ Dry-run validates      (schema, paths, GPU not needed)
  ✓ Results logged         (to experiment_ledger.jsonl + MLflow)

═══════════════════════════════════════════════════════════════════════════

CONFIG STRUCTURE (All 130):

  📂 Augmentation Ablations (35)
     config/aug_ablation/00_no_aug.yml, 10_crop_scale.yml, ..., loeo_*.yml
  
  ⚡ Auto-Generated (15)
     config/_auto/auto_flip_xy.yml, auto_noise.yml, auto_mix_*.yml, ...
  
  🏆 Public Notebooks (60)
     config/exp/public/pilkwang.yml, yusuketogashi.yml, beicicc.yml, ...
  
  🧠 Detector Variants (5)
     model_scratch/config/exp_det_v0.yml, exp_det_v2_recall.yml, ...
  
  📊 LOEO/LOSO (10)
     config/loeo_detector.yml, loeo_detector_aug.yml, loso_*.yml
  
  🎯 Main (4)
     config/exp1.yml, exp2_*.yml, exp/winning_inference_div.yml

═══════════════════════════════════════════════════════════════════════════

EXAMPLES:

  # Quick sanity check (2 min)
  ./baseline/train.sh --quick

  # Test one aug (1 hour)
  ./baseline/train.sh --single-aug flip_xy

  # LOEO CV (4 hours)
  ./baseline/train.sh --loeo-cv

  # Detector tuning (6 hours)
  ./baseline/train.sh --detector-tune

  # Replicate pilkwang (4 hours)
  ./baseline/train.sh --replicate pilkwang

  # Run specific config
  ./baseline/train.sh config/loeo_detector.yml --epochs 30

  # Full ablation (24 hours, parallel on 5 workers recommended)
  ./baseline/train.sh --ablation

═══════════════════════════════════════════════════════════════════════════

MONITORING:

  MLflow dashboard:  http://localhost:5000
  Results ledger:    docs/experiment_ledger.jsonl
  Saved weights:     model_scratch/results/
  Training logs:     model_scratch/results/*/train.log

═══════════════════════════════════════════════════════════════════════════

TROUBLESHOOTING:

  "CUDA not available"
    → Check torch: python -c "import torch; print(torch.cuda.is_available())"
    → Reinstall torch: pip install torch --index-url https://download.pytorch.org/whl/cu128
  
  "Data not found"
    → Check path: ls input/biohub-cell-tracking-during-development/train/
  
  "MLflow not running"
    → Optional! Start: mlflow ui --host 0.0.0.0 --port 5000
    → Or disable: export MLFLOW_TRACKING_URI=''
  
  "Dry-run failed"
    → Check YAML: python -m yaml config/your_config.yml
    → Check paths in config are relative (no hardcoded /home/...)

═══════════════════════════════════════════════════════════════════════════

MORE INFO:

  Full learning guide: docs/LEARNING_COMPLETE_TRAINING_SYSTEM.md
  Model architecture:  model_scratch/README.md
  Config reference:    model_scratch/config/base.yml

═══════════════════════════════════════════════════════════════════════════
EOF
}

# ============================================================================
# MAIN
# ============================================================================

main() {
  [[ $# -eq 0 ]] && { show_help; return 0; }
  
  case "${1:-}" in
    --help|-h)      show_help;;
    --quick)        recipe_quick_test;;
    --list|-l)      list_configs;;
    --loeo-cv)      recipe_loeo_cv;;
    --detector-tune) recipe_detector_tune;;
    --single-aug)   recipe_single_aug "${2:-flip_xy}";;
    --replicate)    recipe_replicate_public "${2:-pilkwang}";;
    --ablation)     recipe_ablation;;
    *)              run_training "$@";;
  esac
}

main "$@"
