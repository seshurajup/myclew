import json, sys, os
from pathlib import Path
sys.path.insert(0, "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/tools/researchpapers")
sys.path.insert(0, "/tmp")
os.chdir("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
from fleet_agents.shorts_builder import run_shorts
from prebuild_check import prebuild_check
import imageio.v3 as iio

VOICE_REF = "/home/seshu/kaggle/2026/youtube/common/voice/indian_hf_alpha_ref.wav"
YT = Path("/home/seshu/kaggle/2026/youtube"); GAL = YT/"gallery"
PLAYLISTS = ["01-python-basics","02-python-functions","03-python-loops-iteration",
             "04-python-oop","05-python-advanced","06-python-testing-tools","07-python-libraries"]

for pl in PLAYLISTS:
    base = YT/pl
    for d in sorted(x for x in base.iterdir() if x.is_dir() and (x/"code.py").exists()):
        name = d.name
        probs = prebuild_check(d)
        if probs:
            print(f"CRITIC-REJECT {pl}/{name}: {probs}", flush=True); continue
        for ext in (".mp4",".json",".png"):
            Path(str(GAL/name)+ext).unlink(missing_ok=True)
        spec = json.load(open(d/"spec.json"))
        spec["code"] = open(d/"code.py").read()
        spec["transcript"] = json.load(open(d/"transcript.json"))
        spec["outputs"] = json.load(open(d/"outputs.json"))
        spec["max_seconds"] = 120; spec["tts_engine"] = "chatterbox"; spec["voice_ref"] = VOICE_REF
        spec["out"] = str(GAL/f"{name}.mp4")
        print(f"=== BUILD {pl}/{name} ===", flush=True)
        try:
            status, data, to, msg = run_shorts({"spec": spec}, "seshu")
        except Exception as e:
            print(f"ERROR {name}: {e}", flush=True); continue
        sync_ok = bool(data.get("sync",{}).get("ok"))
        dur = round(iio.immeta(spec["out"], plugin="pyav")["duration"],1) if Path(spec["out"]).exists() else 0
        print(f"RESULT {pl}/{name} sync={sync_ok} dur={dur}s segs={len(data['props']['segments'])}", flush=True)
        if not sync_ok or dur < 60 or dur > 120:
            for ext in (".mp4",".json",".png"): Path(str(GAL/name)+ext).unlink(missing_ok=True)
            print(f"FAIL {pl}/{name}: sync={sync_ok} dur={dur} — DELETED (continuing)", flush=True)
    print(f"PLAYLIST_DONE {pl}", flush=True)
print("ALL_PLAYLISTS_DONE", flush=True)
