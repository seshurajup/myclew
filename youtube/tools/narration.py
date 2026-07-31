"""Record your own voice instead of TTS — scaffolding and validation.

Why: every one of the 81 videos is narrated by Chatterbox cloning a synthetic Kokoro reference.
That is the single biggest monetisation-review risk on the channel, and it is also the easiest to
remove: the transcripts, the timing pins and the code are already written, so the only missing
piece is your voice reading lines that already exist.

You do NOT need to match any timing. The builder retimes captions and typing speed from the real
length of each clip, exactly as it does for TTS — read at whatever pace sounds natural.

    python tools/narration.py script 01-python-basics/001-hello-world   # what to read, numbered
    python tools/narration.py check  01-python-basics/001-hello-world   # validate before building
    python tools/narration.py status                                    # coverage across all 81

Recording: one file per segment, named 001.wav, 002.wav, … in <video>/narration/.
Any sample rate, mono or stereo. Phone voice-memo or Audacity is fine — quiet room matters far
more than the microphone. Leave a beat of silence at each end; the builder trims it.

Then add to that video's spec.json:   "narration_dir": "narration"
(build_youtube.sh resolves it relative to the video folder). If ANY segment is missing the whole
video falls back to TTS, so you can never ship a half-human, half-synthetic narration.
"""
import json
import re
import sys
from pathlib import Path

YT = Path(__file__).resolve().parent.parent
PLAYLISTS = ["01-python-basics", "02-python-functions", "03-python-loops-iteration",
             "04-python-oop", "05-python-advanced", "06-python-testing-tools", "07-python-libraries"]
TAG = re.compile(r"\[[a-z ]+\]")


def _segments(d: Path):
    return json.loads((d / "transcript.json").read_text())


def script(d: Path):
    """Print the numbered read script — tags stripped, since those are TTS-only directions."""
    segs = _segments(d)
    spec = json.loads((d / "spec.json").read_text())
    print(f"\n{d.parent.name}/{d.name} — \"{spec.get('hook', '')}\"")
    print(f"record {len(segs)} clips into {d / 'narration'}/\n")
    for i, s in enumerate(segs, 1):
        text = " ".join(TAG.sub(" ", s["text"]).split())
        print(f"  {i:03d}.wav  {text}")
    print(f"\nthen: add \"narration_dir\": \"narration\" to {d / 'spec.json'}")


def check(d: Path, verbose=True):
    """Validate the recordings. Returns (have, want, problems)."""
    segs = _segments(d)
    nd = d / "narration"
    want = len(segs)
    problems, have = [], 0
    if not nd.exists():
        return 0, want, [f"no {nd} directory yet"]
    try:
        import soundfile as sf
    except ImportError:
        return 0, want, ["soundfile not installed (pip install soundfile)"]
    for i, s in enumerate(segs, 1):
        f = nd / f"{i:03d}.wav"
        if not f.exists():
            problems.append(f"{i:03d}.wav missing")
            continue
        have += 1
        try:
            info = sf.info(str(f))
        except Exception as e:                                     # noqa: BLE001
            problems.append(f"{i:03d}.wav unreadable: {e}")
            continue
        words = len(TAG.sub(" ", s["text"]).split())
        # ~2.6 words/sec is a natural read; flag clips far outside that for a re-take
        expect = words / 2.6
        if info.duration < expect * 0.45:
            problems.append(f"{i:03d}.wav only {info.duration:.1f}s for {words} words "
                            f"(~{expect:.1f}s expected) — clipped or wrong line?")
        elif info.duration > expect * 2.4:
            problems.append(f"{i:03d}.wav {info.duration:.1f}s for {words} words "
                            f"(~{expect:.1f}s expected) — long pause or wrong line?")
    if verbose:
        print(f"{d.parent.name}/{d.name}: {have}/{want} clips")
        for p in problems:
            print("   -", p)
        if have == want and not problems:
            spec = json.loads((d / "spec.json").read_text())
            ready = spec.get("narration_dir") == "narration"
            print("   ✓ complete" + ("" if ready else "  (add \"narration_dir\": \"narration\" to spec.json)"))
    return have, want, problems


def status():
    tot = done = partial = 0
    for pl in PLAYLISTS:
        for d in sorted((YT / pl).iterdir()):
            if not (d.is_dir() and (d / "code.py").exists()):
                continue
            tot += 1
            have, want, _ = check(d, verbose=False)
            if have == want and want:
                done += 1
            elif have:
                partial += 1
    print(f"real-voice coverage: {done}/{tot} complete, {partial} partial, "
          f"{tot - done - partial} not started")
    if done < tot:
        print("videos without complete narration fall back to TTS automatically")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "status":
        status()
    else:
        script_dir = YT / sys.argv[2]
        (script if cmd == "script" else check)(script_dir)
