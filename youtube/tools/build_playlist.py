import json, sys, os
from pathlib import Path
sys.path.insert(0, "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/tools/researchpapers")
os.chdir("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
from fleet_agents.shorts_builder import run_shorts
import imageio.v3 as iio

PLAYLIST = os.environ["PLAYLIST"]              # e.g. 01-python-basics
START = os.environ.get("START", "000")          # build dirs with name >= START
VOICE_REF = "/home/seshu/kaggle/2026/youtube/common/voice/indian_hf_alpha_ref.wav"
BASE = Path("/home/seshu/kaggle/2026/youtube")/PLAYLIST
GAL = Path("/home/seshu/kaggle/2026/youtube/gallery")

dirs = sorted(d for d in BASE.iterdir() if d.is_dir() and d.name >= START
              and (d/"code.py").exists())
for d in dirs:
    name = d.name
    for ext in (".mp4",".json",".png"):
        Path(str(GAL/name)+ext).unlink(missing_ok=True)
    spec = json.load(open(d/"spec.json"))
    spec["code"] = open(d/"code.py").read()
    spec["transcript"] = json.load(open(d/"transcript.json"))
    spec["outputs"] = json.load(open(d/"outputs.json"))
    spec["max_seconds"] = 120           # allow full 60-120s, don't cap at 90
    spec["tts_engine"] = "chatterbox"
    spec["voice_ref"] = VOICE_REF
    spec["out"] = str(GAL/f"{name}.mp4")
    print(f"=== BUILD {PLAYLIST}/{name} ===", flush=True)
    status, data, to, msg = run_shorts({"spec": spec}, "seshu")
    sync_ok = bool(data.get("sync",{}).get("ok"))
    dur = round(iio.immeta(spec["out"], plugin="pyav")["duration"],1) if Path(spec["out"]).exists() else 0
    segs = data["props"]["segments"]
    print(f"RESULT {name} sync={sync_ok} dur={dur}s segs={len(segs)}", flush=True)
    if not sync_ok or dur < 60 or dur > 120:
        for ext in (".mp4",".json",".png"):
            Path(str(GAL/name)+ext).unlink(missing_ok=True)
        print(f"FAIL {name}: sync={sync_ok} dur={dur} — DELETED, STOPPING", flush=True)
        sys.exit(1)
    print(f"OK {name}", flush=True)
print(f"PLAYLIST_DONE {PLAYLIST}", flush=True)
