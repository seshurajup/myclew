"""ACE-Step music worker — run inside ACE-Step/.venv.
Usage: python ace_music.py <request.json>  where request = {"prompt", "duration", "seed", "out"}
Includes the torchaudio-2.11 save shim (needs torchcodec otherwise).
"""
import json
import sys

import soundfile
import torchaudio


def _save(*a, **k):
    uri = a[0] if a else k.get("uri", k.get("filepath"))
    t = a[1] if len(a) > 1 else k.get("src")
    sr = a[2] if len(a) > 2 else k.get("sample_rate")
    soundfile.write(str(uri), t.detach().cpu().numpy().T, int(sr))


torchaudio.save = _save

def main():
    req = json.load(open(sys.argv[1]))
    from acestep.pipeline_ace_step import ACEStepPipeline
    p = ACEStepPipeline(dtype="bfloat16", torch_compile=False)
    p(prompt=req["prompt"], lyrics="[inst]", audio_duration=float(req["duration"]),
      infer_step=27, guidance_scale=15.0, manual_seeds=[int(req.get("seed", 7))],
      save_path=req["out"])
    print("ace_music: saved", req["out"])


if __name__ == "__main__":
    main()
