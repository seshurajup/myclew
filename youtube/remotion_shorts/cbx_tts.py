"""Chatterbox Turbo batch TTS worker — run inside voicebox/.venv-tts.
Usage: python cbx_tts.py <request.json>   where request = {"texts": [...], "ref": wav, "out": npz}
Loads the model once, generates every segment with the cloned reference voice (non-verbal tags
like [chuckle] render as real sounds), saves all clips to one npz {sr, "0", "1", ...}.
"""
import json
import sys

import numpy as np


def main():
    req = json.load(open(sys.argv[1]))
    from chatterbox.tts_turbo import ChatterboxTurboTTS
    m = ChatterboxTurboTTS.from_pretrained(device="cuda")
    outs = {}
    for i, t in enumerate(req["texts"]):
        w = m.generate(t, audio_prompt_path=req["ref"])
        outs[str(i)] = w.squeeze(0).cpu().numpy().astype("float32")
    np.savez(req["out"], sr=m.sr, **outs)
    print(f"cbx_tts: {len(outs)} clips @ {m.sr}Hz")


if __name__ == "__main__":
    main()
