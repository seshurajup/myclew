#!/usr/bin/env bash
# myclew — mirror ALL fleet agent code + competition experiment code into this repo and make a
# STABLE commit every run IF there are pending changes. "Stable" = every .py byte-compiles first;
# a syntactically-broken tree is NEVER committed. Intended to run every 3 hours via cron.
set -uo pipefail
REPO="/home/seshu/kaggle/2026/myclew"
BIOHUB="/home/seshu/kaggle/2026/biohub-cell-tracking-during-development"
ROGII="/home/seshu/kaggle/2026/rogii-wellbore-geology-prediction"
cd "$REPO" || exit 1

RSYNC_EXCLUDES=(--exclude '__pycache__' --exclude '*.pyc' --exclude '.venv' --exclude 'venv'
  --exclude 'cellmot_venv' --exclude 'site-packages' --exclude '*.log' --exclude '.ipynb_checkpoints'
  --exclude 'input' --exclude 'output' --exclude 'wheels' --exclude '*.whl' --exclude '.git'
  --exclude 'mlruns' --exclude '*.pt' --exclude '*.pth' --exclude '*.ckpt' --exclude '*.csv'
  --exclude '*.parquet' --exclude '*.npy' --exclude '*.npz' --exclude 'research' --exclude 'kernels'
  --exclude 'external' --exclude 'notebooks' --exclude 'extracted.py' --exclude 'scratchpad')

# 1) the agents + their verifiers (source of truth = biohub/fleet_agents)
rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$BIOHUB/fleet_agents/"       "$REPO/fleet_agents/"
rsync -a --delete "${RSYNC_EXCLUDES[@]}" "$BIOHUB/test_fleet_agents/"  "$REPO/test_fleet_agents/"

# 2) per-competition experiment code we build (no data, just code + configs)
mkdir -p "$REPO/competitions/rogii-wellbore-geology-prediction"
rsync -a "${RSYNC_EXCLUDES[@]}" --include '*/' --include '*.py' --include '*.yml' --include '*.yaml' \
  --include '*.sh' --include '*.md' --exclude '*' \
  "$ROGII/" "$REPO/competitions/rogii-wellbore-geology-prediction/"

# 3) STABLE guard — every tracked .py must byte-compile, else abort the commit
if ! python -m py_compile $(find "$REPO/fleet_agents" "$REPO/test_fleet_agents" "$REPO/competitions" -name '*.py') 2>/tmp/myclew_pycompile.err; then
  echo "[myclew] STABLE-GUARD FAILED — .py compile errors, NOT committing:"; cat /tmp/myclew_pycompile.err; exit 2
fi

# 4) commit only if there are pending changes
git add -A
if git diff --cached --quiet; then
  echo "[myclew] no pending changes — nothing to commit ($(date '+%F %T'))"; exit 0
fi
N=$(git diff --cached --name-only | wc -l)
git -c user.name="seshurajup" -c user.email="seshuraju.p@ipauthor.com" \
  commit -q -m "stable snapshot $(date '+%F %H:%M') — ${N} files

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
git push -q origin HEAD 2>&1 | tail -2 || { echo "[myclew] push failed"; exit 3; }
echo "[myclew] committed + pushed ${N} changed files ($(date '+%F %T'))"
