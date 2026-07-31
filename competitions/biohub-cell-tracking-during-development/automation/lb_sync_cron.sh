#!/usr/bin/env bash
# 3-hourly official-LB snapshot for biohub-cell-tracking. Runs the lb-sync fleet agent
# DIRECTLY (not via the board queue) so it captures a snapshot without needing a worker loop.
# Writes to Postgres (lb_snapshot/lb_team) + docs/lb_history.jsonl. Installed as a cron job.
set -euo pipefail
ROOT="/home/seshu/kaggle/2026/biohub-cell-tracking-during-development"
cd "$ROOT"
PY="$ROOT/research/cellmot_venv/bin/python"
export PYTHONPATH="$ROOT/tools/researchpapers:$ROOT"
export USER="${USER:-seshu}"
ts="$(date -u +%FT%TZ)"
echo "[$ts] lb-sync start" >> "$ROOT/logs/lb_sync.cron.log"
"$PY" -c "from fleet_agents import lb_sync; s,r,t,m=lb_sync.run({'question':'3-hourly cron snapshot','spec':{}}, 'cron'); print(s, m)" \
  >> "$ROOT/logs/lb_sync.cron.log" 2>&1
echo "[$ts] lb-sync done rc=$?" >> "$ROOT/logs/lb_sync.cron.log"
