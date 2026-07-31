#!/usr/bin/env bash
# YouTube Shorts factory health check — run before a big batch:  ./healthcheck.sh
# Confirms every dependency the pipeline needs is present & the system has headroom.
YT="$(cd "$(dirname "$0")" && pwd)"
FLEET="/home/seshu/kaggle/2026/biohub-cell-tracking-during-development"
PY="/home/seshu/miniconda3/envs/llm/bin/python"
ok(){ echo "  ✅ $1"; }; bad(){ echo "  ❌ $1"; FAIL=1; }; warn(){ echo "  ⚠️  $1"; }
FAIL=0
echo "== YouTube Shorts factory health =="

echo "[GPU]"
if command -v nvidia-smi >/dev/null; then
  read -r FREE TOTAL <<<"$(nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader,nounits|head -1|tr ',' ' ')"
  MAXJ=$(( (FREE-3000)/9000 )); [ "$MAXJ" -lt 1 ] && MAXJ=1
  ok "GPU free ${FREE}/${TOTAL} MiB → safe max JOBS=$MAXJ (each job ~9GB)"
else bad "nvidia-smi missing"; fi

echo "[render deps]"
command -v npx >/dev/null && ok "npx present" || bad "npx (Node) missing → Remotion can't render"
[ -d "$YT/remotion_shorts/node_modules/remotion" ] && ok "remotion installed" || warn "remotion node_modules absent (first run will npm install)"
$PY -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" >/dev/null 2>&1 && ok "ffmpeg (imageio) ok" || bad "imageio-ffmpeg missing → no audio mux"

echo "[ML voices/music]"
if [ -x "$YT/remotion_shorts/.venv-tts/bin/python" ] || [ -x "/home/seshu/kaggle/2026/external/voicebox/.venv-tts/bin/python" ]; then ok "TTS venv present"; else warn "TTS venv not found → narration falls back"; fi
$PY - <<PY 2>/dev/null && ok "torch+CUDA usable" || bad "torch CUDA not usable"
import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)
PY

echo "[hub + db]"
ss -ltn 2>/dev/null | grep -q ':9090 ' && ok "hub listening on gpu:9090" || warn "hub down → run ~/Sites/start_youtube.sh"
export PYTHONPATH="$FLEET:$FLEET/tools/researchpapers"
$PY -c "from fleet_agents.shorts_builder import ensure_db,_connect; ensure_db(); c=_connect();cur=c.cursor();cur.execute('SELECT count(*) FROM videos');print('  ✅ postgres kaggle_shorts ok, rows='+str(cur.fetchone()[0]));c.close()" 2>/dev/null || bad "postgres kaggle_shorts unreachable"

echo "[gallery integrity]"
NV=$(ls "$YT"/gallery/*.mp4 2>/dev/null | grep -vc _silent)
NBAD=$($PY - <<PY 2>/dev/null
import json,glob,os
bad=0
for j in glob.glob("$YT/gallery/*.json"):
    try:
        m=json.load(open(j)); s=m.get("sync") or {}
        if s.get("checked") and not s.get("ok"): bad+=1
    except Exception: bad+=1
print(bad)
PY
)
[ "${NBAD:-0}" = "0" ] && ok "$NV videos, all sync=OK" || warn "$NV videos, $NBAD with sync!=OK (won't upload)"

echo "=================================="
[ "$FAIL" = "1" ] && { echo "RESULT: ❌ issues found — fix the ❌ lines before batch"; exit 1; } || echo "RESULT: ✅ system solid"
