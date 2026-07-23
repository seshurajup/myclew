#!/usr/bin/env bash
# Config-driven experiment runner for rogii-wellbore-geology-prediction.
# Every experiment is a YAML in config/experiments/. Results append to results/ledger.csv,
# then we print a goal table so we can SEE at a glance whether we're reaching the 6.858 bar.
#
# Usage:
#   ./run_experiments.sh                 # run every config/experiments/*.yml
#   ./run_experiments.sh trackA_gbm      # run one (by name, no path/extension)
#   ./run_experiments.sh --dry-run       # list what would run, no compute
#   ./run_experiments.sh --smoke         # force limit=8 wells for a fast end-to-end check
set -euo pipefail
cd "$(dirname "$0")"

source /home/seshu/miniconda3/etc/profile.d/conda.sh
conda activate kaggle_tabular

# GPU thermal safety (RTX 5090) — non-fatal if not permitted
sudo -n nvidia-smi -pl 400 >/dev/null 2>&1 || true

EXP_DIR="config/experiments"
DRY=0; SMOKE=0; SELECT=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --smoke)   SMOKE=1 ;;
    *)         SELECT+=("$a") ;;
  esac
done

# resolve which configs to run (in dependency order: A, B, then blend)
mapfile -t ALL < <(ls "$EXP_DIR"/*.yml | sort)
CONFIGS=()
for f in "${ALL[@]}"; do
  n=$(basename "$f" .yml)
  if [ ${#SELECT[@]} -eq 0 ] || printf '%s\n' "${SELECT[@]}" | grep -qx "$n"; then
    CONFIGS+=("$f")
  fi
done
# order: non-blend first so blend sees fresh OOF
ORDERED=(); for f in "${CONFIGS[@]}"; do grep -q '^track: *blend' "$f" || ORDERED+=("$f"); done
for f in "${CONFIGS[@]}"; do grep -q '^track: *blend' "$f" && ORDERED+=("$f"); done

echo "=== experiments to run ==="; printf '  %s\n' "${ORDERED[@]##*/}"
[ "$DRY" -eq 1 ] && { echo "(dry-run) nothing executed."; exit 0; }

export ROGII_SMOKE=$SMOKE
for f in "${ORDERED[@]}"; do
  echo; echo "############################################################"
  if [ "$SMOKE" -eq 1 ]; then
    tmp=$(mktemp --suffix=.yml)
    python - "$f" "$tmp" <<'PY'
import sys, yaml
e = yaml.safe_load(open(sys.argv[1]))
e.setdefault("params", {})["limit"] = 8
yaml.safe_dump(e, open(sys.argv[2], "w"))
PY
    python experiment.py --config "$tmp" || echo "!! FAILED: $(basename "$f")"
    rm -f "$tmp"
  else
    python experiment.py --config "$f" || echo "!! FAILED: $(basename "$f")"
  fi
done

echo; echo "############################################################"
echo "=== GOAL LEDGER (results/ledger.csv) ==="
python - <<'PY'
import pandas as pd
from pathlib import Path
p = Path("results/ledger.csv")
if not p.exists():
    print("no ledger yet"); raise SystemExit
df = pd.read_csv(p).drop_duplicates("name", keep="last").sort_values("cv_rmse")
def flag(r): return "✅ BEATS GOAL" if r.cv_rmse < r.goal else ("↑ beats baseline" if r.beats_baseline is True or str(r.beats_baseline)=="True" else "· below baseline")
df["status"] = df.apply(flag, axis=1)
cols = ["name","track","cv_rmse","baseline","goal","status","ts"]
print(df[cols].to_string(index=False))
best = df.iloc[0]
print(f"\nBEST: {best['name']} CV RMSE {best.cv_rmse}  (goal {best.goal}) -> "
      + ("GOAL REACHED ✅" if best.cv_rmse < best.goal else "not yet at goal"))
PY
