"""Install YOUR voice as the narrator for every video, and check the sample is good enough.

Chatterbox Turbo clones zero-shot from one short reference clip, so the quality of every one of
the 81 narrations is decided by this single file. A noisy or clipped reference produces a noisy
or clipped narrator, 81 times over — hence the checks below rather than a blind copy.

    python tools/set_voice.py check  <sample.wav>     # inspect before committing to it
    python tools/set_voice.py set    <sample.wav>     # install + point all 81 specs at it
    python tools/set_voice.py show                    # what is the narrator right now?

What makes a good reference (aim for all of these):
  • 15-30s of continuous speech — too short gives the model little to imitate
  • ONE speaker, no music, no background noise, no reverb (a soft room, not a bathroom)
  • read in the tone you want narration to have: calm, explaining something technical
  • normal headroom — not clipped, not whisper-quiet
  • the same language as the tutorials (English), so the accent transfers cleanly

NOTE this is still synthetic speech, generated in your voice. It removes the consent and
copyright problems entirely and gives the channel your identity and accent, but a listener may
still perceive TTS. Only genuinely recorded narration (tools/narration.py) removes that.
"""
import json
import shutil
import sys
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".ogg", ".m4a", ".mp3"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

YT = Path(__file__).resolve().parent.parent
VOICE_DIR = YT / "common" / "voice"
DEST = VOICE_DIR / "my_voice_ref.wav"
PLAYLISTS = ["01-python-basics", "02-python-functions", "03-python-loops-iteration",
             "04-python-oop", "05-python-advanced", "06-python-testing-tools", "07-python-libraries"]
# the builder's fallback when no spec sets voice_ref
DEFAULT_REF = VOICE_DIR / "af_heart_ref.wav"


def extract_audio(src: Path, out: Path, max_seconds=180):
    """Pull mono 24 kHz audio out of a video (or non-wav audio) with PyAV — ffmpeg is not
    installed on this machine, and PyAV is already a dependency of the render verification."""
    import av
    import numpy as np
    import soundfile as sf
    with av.open(str(src)) as c:
        if not c.streams.audio:
            raise ValueError(f"{src.name} has no audio track")
        st = c.streams.audio[0]
        rs = av.AudioResampler(format="fltp", layout="mono", rate=24000)
        chunks = []
        for frame in c.decode(st):
            for f in rs.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
            if sum(len(x) for x in chunks) > max_seconds * 24000:
                break
    if not chunks:
        raise ValueError("decoded no audio")
    x = np.concatenate(chunks).astype("float32")
    sf.write(str(out), x, 24000)
    return out


def best_window(path: Path, seconds=30):
    """Keep the most consistently-voiced `seconds` of a long sample.

    Frame levels are computed ONCE and the window slides over frames, not over raw samples — the
    naive version re-analysed 720k samples at every 0.5s offset, which is minutes of work on a
    6-minute take.
    """
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    win = int(seconds * sr)
    if x.size <= win:
        return path
    fr = int(0.05 * sr)                                    # 50 ms analysis frames
    nf = x.size // fr
    frames = x[:nf * fr].reshape(nf, fr)
    lv = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    clip = (np.abs(frames) > 0.999).mean(axis=1)
    thr = lv.max() * 0.08
    voiced = (lv > thr).astype(np.float64)
    wf = int(seconds / 0.05)                               # frames per window
    # sliding sums via cumulative sums → O(n) instead of O(n * window)
    cv = np.concatenate([[0.0], np.cumsum(voiced)])
    cc = np.concatenate([[0.0], np.cumsum(clip)])
    starts = np.arange(0, nf - wf)
    score = ((cv[starts + wf] - cv[starts]) / wf) - 2.0 * ((cc[starts + wf] - cc[starts]) / wf)
    k = int(starts[int(np.argmax(score))])
    best = k * fr
    out = path.with_name(path.stem + "_win.wav")
    sf.write(str(out), x[best:best + win], sr)
    print(f"   picked the best {seconds}s window at {best / sr:.0f}s "
          f"({float(score.max()) * 100:.0f}% voiced)")
    return out


def prepare(src: Path) -> Path:
    """Normalise any input (video or audio, any length) into a clean wav reference."""
    tmp = VOICE_DIR / "_incoming.wav"
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in VIDEO_EXT or src.suffix.lower() in (AUDIO_EXT - {".wav"}):
        print(f"   extracting audio from {src.name} ...")
        extract_audio(src, tmp)
        src = tmp
    return best_window(src, 30)


def inspect(path: Path):
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = x.mean(axis=1)
    dur = len(mono) / sr
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    rms = float(np.sqrt((mono.astype(np.float64) ** 2).mean())) if mono.size else 0.0
    clipped = int((np.abs(mono) > 0.999).sum())
    # noise floor: the quietest 10% of 50ms frames — speech pauses should be near silence
    fr = max(1, int(0.05 * sr))
    frames = mono[:len(mono) // fr * fr].reshape(-1, fr)
    fl = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1)) if frames.size else np.array([1.0])
    floor = float(np.quantile(fl, 0.10))
    snr = 20 * np.log10(rms / floor) if floor > 0 else 99.0

    problems, notes = [], []
    notes.append(f"duration {dur:.1f}s, {sr} Hz, {x.shape[1]} channel(s)")
    notes.append(f"peak {peak:.2f}, rms {rms:.3f}, est. SNR {snr:.0f} dB")
    if dur < 12:
        problems.append(f"only {dur:.1f}s — aim for 15-30s so the clone has enough to imitate")
    elif dur > 60:
        notes.append(f"{dur:.0f}s is longer than needed; the first ~30s carries the clone")
    if peak >= 0.999 or clipped > 20:
        problems.append(f"clipped ({clipped} samples at full scale) — re-record with lower gain")
    if peak < 0.15:
        problems.append(f"very quiet (peak {peak:.2f}) — raise the level or move closer to the mic")
    if snr < 18:
        problems.append(f"noisy background (SNR ~{snr:.0f} dB) — that hiss WILL be cloned")
    if sr < 16000:
        problems.append(f"{sr} Hz is low; 24 kHz or higher is much safer for cloning")
    return problems, notes


def show():
    refs = sorted(VOICE_DIR.glob("*.wav"))
    n = 0
    for pl in PLAYLISTS:
        for d in sorted((YT / pl).iterdir()):
            if d.is_dir() and (d / "code.py").exists():
                if json.loads((d / "spec.json").read_text()).get("voice_ref"):
                    n += 1
    print(f"specs pointing at a voice_ref : {n}/81")
    print(f"builder default when unset    : {DEFAULT_REF.name}")
    print(f"your voice installed          : {'yes' if DEST.exists() else 'no'} ({DEST})")
    print("\navailable references:")
    for r in refs:
        print(f"   {r.name}")


def install(src: Path):
    src = prepare(src)
    problems, notes = inspect(src)
    for n in notes:
        print("  ", n)
    if problems:
        print("\nnot installing — fix these first:")
        for p in problems:
            print("   -", p)
        return 1
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, DEST)
    n = 0
    for pl in PLAYLISTS:
        for d in sorted((YT / pl).iterdir()):
            if not (d.is_dir() and (d / "code.py").exists()):
                continue
            p = d / "spec.json"
            s = json.loads(p.read_text())
            s["voice_ref"] = str(DEST)
            p.write_text(json.dumps(s, indent=2) + "\n")
            n += 1
    print(f"\n✓ installed {DEST.name} and pointed {n} specs at it")
    print("  rebuild to hear it:  FORCE=1 ./build_youtube.sh 01-python-basics 001")
    return 0


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "show":
        show()
    elif cmd == "check":
        probs, notes = inspect(prepare(Path(sys.argv[2])))
        for n in notes:
            print("  ", n)
        print("\n" + ("usable ✓" if not probs else "problems:"))
        for p in probs:
            print("   -", p)
        sys.exit(1 if probs else 0)
    elif cmd == "set":
        sys.exit(install(Path(sys.argv[2])))
    else:
        print(__doc__)
