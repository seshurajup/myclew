#!/usr/bin/env bash
# YouTube Shorts factory — build videos one by one from their input folders (see README.md).
#   ./build_youtube.sh all                  # every playlist, numeric order
#   ./build_youtube.sh 01-learn-python      # one playlist
#   ./build_youtube.sh 01-learn-python 001  # one video (prefix match)
#   FORCE=1 ./build_youtube.sh ...          # rebuild even if already in gallery
set -uo pipefail
YT="$(cd "$(dirname "$0")" && pwd)"
FLEET="/home/seshu/kaggle/2026/biohub-cell-tracking-during-development"
PY="python"; export PYTHONPATH="$FLEET:$FLEET/tools/researchpapers"

build_video() {
  local pl_dir="$1" vid_dir="$2"
  local name="$(basename "$pl_dir")-$(basename "$vid_dir")"
  local out="$YT/gallery/${name}_final.mp4"
  if [ -f "$out" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "[skip] $name (already built; FORCE=1 to rebuild)"; return 0
  fi
  # HARD GATE: retention rules + sync/pace sanity (tools/retention_rules.py, README §3.9).
  # Runs before any GPU work — a label hook or a missing next-video tail is cheap to fix now
  # and impossible to fix after upload (the hook IS the title and the thumbnail).
  if ! $PY "$YT/tools/gate.py" "$vid_dir"; then
    echo "[GATE-REJECT] $name — fix the above, then rebuild"; return 1
  fi
  echo "=== building $name ==="
  if [ -f "$vid_dir/artifacts.py" ]; then
    (cd "$vid_dir" && $PY artifacts.py) || { echo "[FAIL] artifacts: $name"; return 1; }
  fi
  (cd "$FLEET" && $PY - "$vid_dir" "$YT/gallery/${name}.mp4" <<'PYEOF'
import json, sys
from pathlib import Path
from fleet_agents.shorts_builder import run_shorts
vdir, out = Path(sys.argv[1]), sys.argv[2]
spec = json.loads((vdir / "spec.json").read_text())
for k in ("code_path", "transcript_path", "outputs_path"):
    if spec.get(k):
        spec[k] = str(vdir / spec[k])
if spec.get("outputs_path"):
    outs = json.loads(Path(spec["outputs_path"]).read_text())
    for ev in outs:
        if ev.get("image") and not str(ev["image"]).startswith(("/", "data:", "http")):
            ev["image"] = str(vdir / ev["image"])
    spec["outputs"] = outs; spec.pop("outputs_path")
if spec.get("transcript_path"):
    spec["transcript"] = json.loads(Path(spec["transcript_path"]).read_text())
    spec.pop("transcript_path")
spec["out"] = out
status, data, _, msg = run_shorts({"spec": spec}, "build_youtube")
print(msg)
sync = data.get("sync") or {}
sys.exit(0 if status == "done" and (not sync.get("checked") or sync.get("ok")) else 1)
PYEOF
  ) || { echo "[FAIL] $name"; return 1; }
  echo "[done] $name"
}

MODE="${1:-all}"

# hidden mode: build exactly one (playlist_dir, video_dir) — used by the parallel dispatcher
if [ "$MODE" = "__one" ]; then build_video "$2" "$3"; exit $?; fi

# JOBS>1 → build videos in parallel. Split cores between jobs so total render threads ~= nproc.
JOBS="${JOBS:-1}"
NCPU="$(nproc)"
# GPU safety cap: each job peaks ~8.9GB VRAM (TTS+music). Cap JOBS to what free VRAM allows
# (with a 3GB margin) so a parallel batch can never OOM. Measured 2026-07-26 on the RTX 5090 (32GB).
PER_JOB_MIB=9000; MARGIN_MIB=3000
FREE_MIB="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)"
if [ -n "${FREE_MIB:-}" ]; then
  MAXJ=$(( (FREE_MIB - MARGIN_MIB) / PER_JOB_MIB )); [ "$MAXJ" -lt 1 ] && MAXJ=1
  if [ "$JOBS" -gt "$MAXJ" ]; then
    echo "[guard] JOBS=$JOBS exceeds safe max for ${FREE_MIB}MiB free VRAM → capping to $MAXJ (each job ~${PER_JOB_MIB}MiB)"
    JOBS="$MAXJ"
  fi
fi
if [ "$JOBS" -gt 1 ]; then
  export SHORTS_CONCURRENCY="${SHORTS_CONCURRENCY:-$(( NCPU / JOBS > 1 ? NCPU / JOBS : 2 ))}"
fi

collect_pairs() {  # emit "pl_dir<TAB>vid_dir" lines for the requested MODE
  if [ "$MODE" = "all" ]; then
    # "all" = the 81-video Python curriculum. The older NN-learn-* playlists are drafts and are
    # NOT part of the channel; build them explicitly by name if you ever revive them.
    for pl in "$YT"/[0-9][0-9]-python-*/; do
      for v in "$pl"[0-9][0-9][0-9]-*/; do [ -d "$v" ] && printf '%s\t%s\n' "${pl%/}" "${v%/}"; done
    done
  else
    for v in "$YT/$MODE"/[0-9][0-9][0-9]-*/; do [ -d "$v" ] && printf '%s\t%s\n' "$YT/$MODE" "${v%/}"; done
  fi
}

if [ "$JOBS" -gt 1 ] && [ -z "${2:-}" ]; then
  echo "== parallel build: JOBS=$JOBS, per-video concurrency=$SHORTS_CONCURRENCY (of $NCPU cores) =="
  collect_pairs | xargs -P "$JOBS" -d '\n' -I{} bash -c 'IFS=$'"'"'\t'"'"' read -r pl v <<<"{}"; FORCE="'"${FORCE:-0}"'" SHORTS_CONCURRENCY="'"$SHORTS_CONCURRENCY"'" "'"$0"'" __one "$pl" "$v"'
  exit 0
fi

if [ "$MODE" = "all" ]; then
  for pl in "$YT"/[0-9][0-9]-python-*/; do
    for v in "$pl"[0-9][0-9][0-9]-*/; do [ -d "$v" ] && build_video "${pl%/}" "${v%/}"; done
  done
elif [ -n "${2:-}" ]; then
  pl="$YT/$MODE"
  v=$(ls -d "$pl"/${2}*/ 2>/dev/null | head -1)
  [ -z "$v" ] && { echo "video $2 not found in $MODE"; exit 1; }
  build_video "$pl" "${v%/}"
else
  for v in "$YT/$MODE"/[0-9][0-9][0-9]-*/; do [ -d "$v" ] && build_video "$YT/$MODE" "${v%/}"; done
fi
