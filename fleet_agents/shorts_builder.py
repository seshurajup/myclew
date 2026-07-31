"""shorts_builder — tutorial (code, transcript, language) → YouTube Short (1080×1920 MP4) with a
Remotion-rendered typing-code animation styled with facebook/astryx neutral-theme tokens.

The REAL renderer is the Node/Remotion project at myclew/remotion_shorts/ (remotion-dev/remotion:
React composition `CodeShort` types the code char-by-char with Prism syntax highlighting in the
astryx dark syntax palette, blinking cursor, auto-scroll, and timed transcript captions). This agent
is the Python orchestrator: it normalizes the 3 spec inputs into props.json, ensures node_modules,
and shells `npx remotion render`. When Node/Chromium is unavailable (offline test box, Kaggle), it
falls back to a Python-native typing renderer (pygments + PIL, same astryx palette) streamed through
video_builder's imageio writer — so the data-wise test stays green offline.

Spec (3 core inputs, all reusable/config-driven):
  • code       — tutorial source text, or code_path to a file
  • transcript — narration; plain lines (spread over duration), "start|end|text" timed lines,
                 or a list of {start,end,text} dicts; or transcript_path
  • language   — "python" | "rust" (any Prism/pygments lexer name works)
Optional: out, title, cps (chars/sec typing, 30), fps (30), tail_seconds (2), max_seconds (90; hard
          YouTube Shorts cap 180), voice (af_heart), music/speech ("off" to disable), audio (user wav),
          force_pil (skip Remotion), size ("WxH" for the PIL fallback only; Remotion is 1080×1920).
Post-render verify_sync() MEASURES caption↔narration↔video alignment (audio RMS in/out of caption
windows + caption-bubble pixels at segment midpoints) and reports sync=OK/FAIL in the message.
"""
from __future__ import annotations
import re
import json
import os
import shutil
import subprocess
from pathlib import Path

from .base import BaseAgent, AUTO, COMP

YT_ROOT = Path.home() / "kaggle" / "2026" / "youtube"      # the video factory home (not the comp dir)
GALLERY = YT_ROOT / "gallery"
REMOTION_DIR = YT_ROOT / "remotion_shorts"

# astryx neutral theme, dark stops (mirror of remotion_shorts/src/astryxTheme.ts)
ASTRYX = {
    "body": (27, 27, 27), "surface": (38, 38, 38), "code_bg": (10, 10, 10),
    "border": (64, 64, 64), "text": (229, 229, 229), "muted": (163, 163, 163),
}
ASTRYX_SYNTAX = {  # pygments token-name prefix → RGB (astryx dark syntax stops)
    "Keyword": (239, 168, 255), "String": (166, 210, 162), "Comment": (163, 163, 163),
    "Number": (255, 179, 127), "Name.Function": (160, 202, 255), "Name.Class": (239, 168, 255),
    "Name.Builtin": (239, 168, 255), "Name.Decorator": (238, 193, 47), "Operator": (163, 163, 163),
    "Punctuation": (82, 82, 82), "Name": (229, 229, 229),
}


def _strip_fillers(t):
    """Drop non-verbal filler interjections (hmm/uh/um/erm/mmm/ahh…) so they never get
    spoken by the TTS or shown in captions, and tidy up the punctuation they leave behind."""
    import re
    s = re.sub(r"(?i)\b(h+m+|m{2,}|u+h+|u+m+|er+m*|a{2,}h*|o+h+|w+e+l+p)\b", "", t)
    s = re.sub(r"\s*,\s*,", ",", s)                     # collapse doubled commas
    s = re.sub(r"\s+([,.!?—-])", r"\1", s)              # no space before punctuation
    s = re.sub(r"([,—-])\s*([,.!?])", r"\2", s)         # comma+dash then period → period
    s = re.sub(r"^\s*[,—-]\s*", "", s)                  # no leading comma/dash
    return re.sub(r"\s{2,}", " ", s).strip()


def parse_transcript(raw):
    """Normalize transcript input → [{start,end,text}] (seconds). Accepts list-of-dicts,
    'start|end|text' lines, or plain lines (timed later against the typing duration)."""
    if isinstance(raw, list):
        out = []
        for s in raw:
            e = {"start": float(s.get("start", -1)), "end": float(s.get("end", -1)), "text": _strip_fillers(str(s["text"]))}
            if s.get("until_line") is not None:
                e["until_line"] = int(s["until_line"])                  # pace-lock override
            out.append(e)
        return out
    lines = [ln.strip() for ln in str(raw or "").splitlines() if ln.strip()]
    timed, plain = [], []
    for ln in lines:
        parts = ln.split("|", 2)
        try:
            if len(parts) == 3:
                timed.append({"start": float(parts[0]), "end": float(parts[1]), "text": _strip_fillers(parts[2].strip())})
                continue
        except ValueError:
            pass    # a line that fails float() is prose, not a timing row — the plain-text path below owns it
        plain.append(ln)
    if timed and not plain:
        return timed
    return [{"start": -1.0, "end": -1.0, "text": _strip_fillers(t)} for t in plain]  # timed by build_props vs duration


def build_props(code, transcript, language, title=None, cps=30, tail_seconds=2.0, max_seconds=90.0):
    """Assemble the CodeShort props dict; spread untimed transcript lines evenly over the typing time."""
    segs = parse_transcript(transcript)
    max_seconds = min(float(max_seconds), 180.0)                        # YouTube Shorts hard limit (3 min)
    typing = len(code) / max(float(cps), 1.0)
    if segs and segs[0]["start"] < 0:                                   # plain lines → even spread
        dur = min(max_seconds, typing + float(tail_seconds))
        step = dur / len(segs)
        segs = [{**s, "start": round(i * step, 2), "end": round((i + 1) * step, 2)}
                for i, s in enumerate(segs)]                            # keep until_line etc.
    return {"code": code, "language": str(language).lower(), "title": title or f"tutorial.{ 'rs' if str(language).lower()=='rust' else 'py'}",
            "segments": segs, "cps": float(cps), "tailSeconds": float(tail_seconds),
            "maxSeconds": max_seconds}


def pace_lock(props):
    """Attach charEnd to every segment: by a segment's end, exactly that many chars are typed —
    typing can NEVER run ahead of (or lag) the narration. Mapping = explicit until_line per
    transcript line when given, else code lines spread evenly across segments; always ends at
    the full code length."""
    code, segs = props["code"], props["segments"]
    if not segs:
        return props
    lines = code.split("\n")
    line_end = []
    n = 0
    for ln in lines:
        n += len(ln) + 1
        line_end.append(min(n, len(code)))
    n_lines, n_segs = len(lines), len(segs)
    prev = 0
    for i, sg in enumerate(segs):
        li = int(sg.get("until_line") or round((i + 1) * n_lines / n_segs))
        li = max(1, min(li, n_lines))
        ce = line_end[li - 1]
        sg["charEnd"] = max(ce, prev)                                   # monotonic
        prev = sg["charEnd"]
    segs[-1]["charEnd"] = len(code)                                     # everything typed by the last caption
    return props


def normalize_outputs(raw):
    """4th input: interpreter outputs / media events. JSON list of
    {after_line | at, text?, image?, caption?, marks?:[{x,y,w,h} normalized]}. Image file paths are
    inlined as base64 data URIs (works in Remotion AND keeps props self-contained); marks draw
    highlight boxes on the image (paper-figure-with-markers use case)."""
    import base64
    import mimetypes
    out = []
    for ev in (raw or []):
        e = {}
        if ev.get("after_line") is not None:
            e["afterLine"] = int(ev["after_line"])
        if ev.get("at") is not None:
            e["at"] = float(ev["at"])
        for k in ("text", "caption", "marks"):
            if ev.get(k):
                e[k] = ev[k]
        img = ev.get("image")
        if img:
            if str(img).startswith(("data:", "http")):
                e["image"] = img
            else:
                mime = mimetypes.guess_type(str(img))[0] or "image/png"
                e["image"] = f"data:{mime};base64," + base64.b64encode(Path(img).read_bytes()).decode()
        if e:
            out.append(e)
    return out


def render_remotion(props, out_path, fps=30, timeout=1800):
    """Render props via `npx remotion render` in remotion_shorts/. Raises on any failure so the
    caller can fall back to the PIL renderer."""
    if not shutil.which("npx"):
        raise RuntimeError("npx not on PATH")
    if not (REMOTION_DIR / "node_modules" / "remotion").exists():
        r = subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=str(REMOTION_DIR),
                           capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"npm install failed: {r.stderr[-300:]}")
    props_file = AUTO / "shorts_props.json"
    props_file.parent.mkdir(parents=True, exist_ok=True)
    props_file.write_text(json.dumps(props))
    # frame rendering is CPU-bound (headless Chromium); use more cores. Lower it when several videos
    # build in parallel so total threads stay ~= nproc. Override with SHORTS_CONCURRENCY.
    conc = os.environ.get("SHORTS_CONCURRENCY") or str(max(2, (os.cpu_count() or 4) // 2))
    cmd = ["npx", "remotion", "render", "src/index.ts", "CodeShort", str(out_path),
           f"--props={props_file}", "--crf=18", "--jpeg-quality=95", f"--concurrency={conc}"]
    gl = os.environ.get("SHORTS_GL")                                    # e.g. angle / angle-egl to try GPU raster
    if gl:
        cmd.append(f"--gl={gl}")
    r = subprocess.run(cmd, cwd=str(REMOTION_DIR), capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0 or not Path(out_path).exists():
        raise RuntimeError(f"remotion render failed: {(r.stderr or r.stdout)[-400:]}")
    return {"path": str(out_path), "renderer": "remotion"}


def render_pil(props, out_path, fps=30, size=(540, 960)):
    """Python-native fallback: pygments-highlighted typing frames (astryx palette) streamed to MP4
    (GIF if ffmpeg absent). Same layout idea as Short.tsx at reduced fidelity."""
    import numpy as np
    import imageio
    from PIL import Image, ImageDraw, ImageFont
    from pygments import lex
    from pygments.lexers import get_lexer_by_name
    W, H = size
    fs = max(10, W // 45)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", fs)
    except OSError:
        font = ImageFont.load_default()
    lexer = get_lexer_by_name(props["language"], stripall=False)
    spans = []
    for tok, text in lex(props["code"], lexer):
        name = str(tok).replace("Token.", "")
        color = ASTRYX_SYNTAX.get(name) or next(
            (c for pfx, c in ASTRYX_SYNTAX.items() if name.startswith(pfx)), ASTRYX["text"])
        spans.append((text, color))
    code = props["code"]
    typing = len(code) / max(props["cps"], 1.0)
    dur = min(props.get("maxSeconds", 90.0), max(typing, max((s["end"] for s in props["segments"]), default=0)) + props["tailSeconds"])
    n_frames = max(int(fps), int(round(dur * fps)))
    pad, top = W // 20, H // 8
    line_h = fs + 4
    is_gif = str(out_path).endswith(".gif")
    if is_gif:
        writer = imageio.get_writer(str(out_path), duration=1.0 / fps, loop=0)
    else:
        import imageio_ffmpeg  # noqa: F401  (probe: no ffmpeg plugin → caller falls back to GIF)
        writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264")
    for f in range(n_frames):
        t = f / fps
        n_typed = min(len(code), int(t * props["cps"]))
        img = Image.new("RGB", (W, H), ASTRYX["body"])
        d = ImageDraw.Draw(img)
        d.rectangle([pad, top, W - pad, H - top], fill=ASTRYX["code_bg"], outline=ASTRYX["border"], width=2)
        d.rectangle([pad, top - int(2.2 * line_h), W - pad, top], fill=ASTRYX["surface"],
                    outline=ASTRYX["border"], width=2)
        d.text((pad + 8, top - int(1.8 * line_h)), f"{props['title']}  [{props['language']}]",
               font=font, fill=ASTRYX["muted"])
        x, y, left = pad + 8, top + 6, n_typed
        for text, color in spans:
            if left <= 0:
                break
            chunk, left = text[:left], left - len(text)
            for i, ln in enumerate(chunk.split("\n")):
                if i > 0:
                    x, y = pad + 8, y + line_h
                if ln and y < H - top - line_h:
                    d.text((x, y), ln, font=font, fill=color)
                    x += d.textlength(ln, font=font)
        if (int(t * 2.5) % 2 == 0 or n_typed < len(code)) and y < H - top - line_h:
            d.rectangle([x, y, x + fs // 2, y + fs], fill=ASTRYX["text"])
        seg = next((s for s in props["segments"] if s["start"] <= t < s["end"]), None)
        if seg:
            d.text((pad, H - top + line_h), seg["text"][: W // (fs // 2)], font=font, fill=ASTRYX["text"])
        writer.append_data(np.asarray(img))
    writer.close()
    return {"path": str(out_path), "renderer": "pil", "n_frames": n_frames}


# ---------------------------------------------------------------- audio (music bed + clicks + narration)
SR = 24000  # kokoro native rate; music synthesized at the same rate so the mix is trivial


def synth_music(duration, n_chars, cps, sr=SR, seed=0):
    """Copyright-free procedural bed: lo-fi chord pad (Am–F–C–G, detuned sines, slow attack) + soft
    key-click transients at the typing rate while code is being typed. Pure numpy — always works.
    (HF alternative ACE-Step 1.5 is MIT but 4B-class — overkill for a background bed; MusicGen is
    CC-BY-NC so unusable on YouTube.)"""
    import numpy as np
    rng = np.random.RandomState(seed)
    n = int(duration * sr)
    t = np.arange(n) / sr
    out = np.zeros(n, np.float32)
    chords = [[220.0, 261.63, 329.63], [174.61, 220.0, 261.63],
              [130.81, 164.81, 196.0], [196.0, 246.94, 293.66]]      # Am F C G
    bar = 60.0 / 72 * 4                                              # 72 bpm, one chord per bar
    for i in range(int(np.ceil(duration / bar))):
        s0, s1 = int(i * bar * sr), min(int((i + 1) * bar * sr), n)
        if s0 >= n:
            break
        seg_t = t[s0:s1] - t[s0]
        env = np.minimum(seg_t / 0.8, 1.0) * np.minimum((bar - seg_t) / 0.6, 1.0).clip(0, 1)
        for f in chords[i % 4]:
            for det in (-1.5, 1.5):                                   # gentle detune → warm pad
                out[s0:s1] += 0.045 * env * np.sin(2 * np.pi * (f + det) * seg_t
                                                   + rng.uniform(0, 2 * np.pi))
    typing_end = min(duration, n_chars / max(cps, 1.0))
    click = (rng.randn(int(0.02 * sr)) * np.exp(-np.arange(int(0.02 * sr)) / (0.004 * sr))).astype(np.float32)
    for k in range(int(typing_end * cps)):
        p = int(k / cps * sr + rng.randint(0, sr // 100))
        if p + click.size < n:
            out[p:p + click.size] += 0.05 * click
    return out


ACE_DIR = COMP.parent / "external" / "ACE-Step"


def music_gen(duration, prompt=None, seed=7):
    """Generate a REAL music bed with ACE-Step v1-3.5B (Apache-2.0 weights, MIT repo — safe for
    YouTube) in its isolated venv via remotion_shorts/ace_music.py; cached by (prompt,duration,seed)
    so each style is generated once. Returns mono float32 @SR, or None → synth fallback."""
    import hashlib
    import numpy as np
    prompt = prompt or ("lofi hip hop, chill study beat, soft piano, vinyl crackle, mellow, "
                        "instrumental, no vocals")
    key = hashlib.md5(f"{prompt}|{int(duration)}|{seed}".encode()).hexdigest()[:16]
    cache = AUTO / "music_cache"; cache.mkdir(parents=True, exist_ok=True)
    wav = cache / f"ace_{key}.wav"
    py = ACE_DIR / ".venv" / "bin" / "python"
    if not wav.exists():
        if not py.exists():
            return None
        req = cache / f"ace_{key}_req.json"
        req.write_text(json.dumps({"prompt": prompt, "duration": float(duration),
                                   "seed": int(seed), "out": str(wav)}))
        r = subprocess.run([str(py), str(REMOTION_DIR / "ace_music.py"), str(req)],
                           capture_output=True, text=True, cwd=str(ACE_DIR), timeout=1200)
        if r.returncode != 0 or not wav.exists():
            return None
    try:
        import soundfile as sf
        a, in_sr = sf.read(str(wav), dtype="float32", always_2d=True)
        a = a.mean(axis=1)
        if in_sr != SR:
            n_out = int(round(a.size * SR / in_sr))
            a = np.interp(np.linspace(0, a.size - 1, n_out), np.arange(a.size), a).astype(np.float32)
        return a
    except Exception:  # noqa: BLE001
        return None


VOICEBOX_DIR = COMP.parent / "external" / "voicebox"
VOICE_REF = YT_ROOT / "common" / "voice" / "af_heart_ref.wav"


def tts_batch_chatterbox(texts, ref_wav=None):
    """Natural human narration: Chatterbox Turbo 350M (MIT, voicebox .venv-tts) voice-cloned from
    our af_heart reference — supports REAL non-verbal sounds via [chuckle] [sigh] [laugh] tags and
    fillers like "hmm". One subprocess (remotion_shorts/cbx_tts.py) loads the model once and
    renders every segment. Returns list of float32 mono @SR, or None → Kokoro fallback."""
    import numpy as np
    py = VOICEBOX_DIR / ".venv-tts" / "bin" / "python"
    ref = str(ref_wav or VOICE_REF)
    if not py.exists() or not Path(ref).exists():
        return None
    req = AUTO / "cbx_req.json"; out = AUTO / "cbx_out.npz"
    if out.exists():
        out.unlink()
    req.write_text(json.dumps({"texts": list(texts), "ref": ref, "out": str(out)}))
    r = subprocess.run([str(py), str(REMOTION_DIR / "cbx_tts.py"), str(req)],
                       capture_output=True, text=True, cwd=str(VOICEBOX_DIR), timeout=1800)
    if r.returncode != 0 or not out.exists():
        return None
    z = np.load(out)
    in_sr = int(z["sr"])
    clips = []
    for i in range(len(texts)):
        a = z[str(i)]
        if in_sr != SR:
            n_out = int(round(a.size * SR / in_sr))
            a = np.interp(np.linspace(0, a.size - 1, n_out), np.arange(a.size), a).astype(np.float32)
        clips.append(a)
    return clips


def tts_line(text, voice="af_heart"):
    """One narration line via Kokoro-82M (Apache-2.0, af_heart = top female voice). Returns float32
    mono @24k. Guarded import — caller treats any failure as 'no speech'."""
    import numpy as np
    from kokoro import KPipeline
    global _KPIPE
    if "_KPIPE" not in globals() or _KPIPE is None:
        _KPIPE = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return np.concatenate([np.asarray(a, dtype=np.float32) for _, _, a in _KPIPE(text, voice=voice)])


def _strip_tags(t):
    """Remove [chuckle]-style non-verbal tags for engines that would read them aloud (Kokoro)."""
    import re
    return re.sub(r"\s*\[[a-z ]+\]\s*", " ", t).strip()


def _load_narration(d, n_segments):
    """Load hand-recorded narration: <dir>/001.wav ... one per transcript segment, any sample rate.

    Returns None (→ fall back to TTS) unless EVERY segment has a file, so a half-finished
    recording session can never ship a video that is half human and half synthetic.
    """
    import numpy as np
    from pathlib import Path as _P
    d = _P(d)
    files = [d / f"{i + 1:03d}.wav" for i in range(n_segments)]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        if len(missing) < n_segments:
            print(f"shorts-builder: narration incomplete ({len(missing)} of {n_segments} missing: "
                  f"{missing[:3]}...) → using TTS for the whole video")
        return None
    try:
        import soundfile as sf
    except ImportError:
        print("shorts-builder: soundfile missing → cannot read narration, using TTS")
        return None
    out = []
    for f in files:
        x, sr = sf.read(str(f), dtype="float32", always_2d=True)
        x = x.mean(axis=1)                                   # stereo → mono
        if sr != SR:                                         # linear resample to the mix rate
            m = int(round(x.size * SR / sr))
            x = np.interp(np.linspace(0, x.size - 1, m), np.arange(x.size), x).astype(np.float32)
        x = np.trim_zeros(x, "fb") if x.size else x          # drop dead air at the edges
        out.append(x)
    print(f"shorts-builder: using REAL narration from {d} ({n_segments} clips, "
          f"{sum(c.size for c in out) / SR:.1f}s)")
    return out


def build_audio(props, wav_path, voice="af_heart", speech=True, music=True, gap=0.35, voice_ref=None,
                narration_dir=None):
    """Full audio track for the Short. When speech is on and the transcript was untimed, RETIME the
    caption segments from the actual narration lengths (perfect caption↔voice sync) and stretch cps
    so typing spans the narration. Music is ducked to 55% under speech. Returns updated props."""
    import numpy as np
    clips = None
    if speech and narration_dir:
        # REAL VOICE: one wav per transcript segment (001.wav, 002.wav, ...). Everything downstream
        # is unchanged — captions retime to these real speech lengths exactly as they do for TTS,
        # so a human recording needs no timing work at all.
        clips = _load_narration(narration_dir, len(props["segments"]))
    if speech and clips is None:
        if props.get("ttsEngine", "chatterbox") == "chatterbox":
            try:
                clips = tts_batch_chatterbox([s["text"] for s in props["segments"]], ref_wav=voice_ref)
            except Exception:  # noqa: BLE001
                clips = None
        if clips is None:
            try:
                clips = [tts_line(_strip_tags(s["text"]), voice=voice) for s in props["segments"]]
            except Exception:  # noqa: BLE001  (kokoro absent → music-only, never breaks the render)
                clips = None
    if clips:
        for sg in props["segments"]:
            sg["text"] = _strip_tags(sg["text"])                        # tags are audio-only, not captions
        speech_total = sum(c.size / SR for c in clips)
        target = props.get("targetSeconds")
        if target:                                                     # nudge gaps toward target, but
            lead = 1.7 if props.get("hook") else 0.5                   # NEVER pad long silence — a big
            budget = min(float(target), props.get("maxSeconds", 90.0)) - props["tailSeconds"] - lead
            # cap the auto-stretch gap at a natural pause (0.8s): silence longer than that reads as
            # dead air / "voice finished but caption frozen". Reach the target with narration length,
            # not padding — if speech is too short for the target, the video is simply shorter.
            gap = max(gap, min(0.8, (budget - speech_total) / max(len(clips), 1)))
        cursor = 1.7 if props.get("hook") else 0.5                     # narration starts after the card
        for s, c in zip(props["segments"], clips):                     # retime to narration
            s["start"] = round(cursor, 2)
            s["speechEnd"] = round(cursor + c.size / SR, 2)             # real end of voice (no gap)
            s["end"] = round(s["speechEnd"] + gap, 2)                   # +silence gap to hit target length
            cursor = s["end"]
        total = min(props.get("maxSeconds", 90.0), cursor + props["tailSeconds"])
        props["cps"] = (len(props["code"]) / max(total - props["tailSeconds"], 1) if target
                        else max(props["cps"], len(props["code"]) / max(total - props["tailSeconds"], 1)))
    dur = min(props.get("maxSeconds", 90.0), max(len(props["code"]) / max(props["cps"], 1),
                        max((s["end"] for s in props["segments"]), default=0)) + props["tailSeconds"])
    n = int(dur * SR)

    def _rms_norm(x, target):                                          # loudness-normalize a track
        r = float(np.sqrt((x.astype(np.float64) ** 2).mean())) or 1.0
        return (x * (target / r)).astype(np.float32)

    bed = None
    if music:
        bed = music_gen(dur, prompt=props.get("musicPrompt"))          # ACE-Step (cached) → synth fallback
        if bed is None:
            bed = synth_music(dur, len(props["code"]), props["cps"])
    if bed is None:
        bed = np.zeros(n, np.float32)
    bed = bed[:n] if bed.size >= n else np.pad(bed, (0, n - bed.size))
    if clips:
        bed = _rms_norm(bed, 0.018)                                    # bed clearly UNDER the voice
        duck = np.ones(n, np.float32)
        speech_track = np.zeros(n, np.float32)
        for s, c in zip(props["segments"], clips):
            p = int(s["start"] * SR)
            m = min(c.size, n - p)
            if m > 0:
                speech_track[p:p + m] += c[:m]
                duck[p:p + m] = 0.08                                   # music near-silent while speaking so
                #                                                        the voice ALWAYS dominates regardless
                #                                                        of run-to-run TTS loudness variance
        sp = speech_track[speech_track != 0]
        if sp.size:                                                    # normalize speech on ITS windows only
            speech_track = speech_track * (0.15 / (float(np.sqrt((sp.astype(np.float64) ** 2).mean())) or 1.0))
        mix = bed * duck + speech_track                                # duck the bed under narration only
    else:
        mix = _rms_norm(bed, 0.06) if music else bed
    peak = float(np.abs(mix).max()) or 1.0
    mix = (0.9 / peak) * mix if peak > 0.9 else mix                    # never clip
    from .video_builder import write_wav
    write_wav(mix, wav_path, sample_rate=SR)
    return props, str(wav_path), bool(clips)


def mux(video_path, wav_path, out_path):
    """Attach the wav using the ffmpeg binary bundled with imageio-ffmpeg (no system ffmpeg needed)."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    # explicit -map: without it this ffmpeg build muxed a silent track (default best-audio pick bug)
    try:
        r = subprocess.run([ff, "-y", "-i", str(video_path), "-i", str(wav_path),
                            "-map", "0:v:0", "-map", "1:a:0",
                            "-c:v", "copy", "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k",
                            "-shortest", "-movflags", "+faststart", str(out_path)],
                           capture_output=True, text=True,
                           timeout=600)  # ffmpeg can hang on a corrupt input; stream-copy mux is fast
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"mux timed out after 600s: {video_path}")
    # 48kHz stereo = YouTube's recommended ingest spec
    if r.returncode != 0:
        raise RuntimeError(f"mux failed: {r.stderr[-300:]}")
    return str(out_path)


YT_TOKEN = AUTO / "youtube_token.json"
YT_SECRETS = Path.home() / ".google" / "oauth.json"
YT_SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
             "https://www.googleapis.com/auth/youtube"]           # upload + playlist management


def _yt_service():
    """Authorized YouTube client. First run needs the USER to complete the browser OAuth consent
    (we never touch their password); the refresh token is then cached at YT_TOKEN (chmod 600)."""
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build
    creds = None
    if YT_TOKEN.exists():
        creds = Credentials.from_authorized_user_file(str(YT_TOKEN), YT_SCOPES)
    if creds and creds.expired and creds.refresh_token:
        from google.auth.transport.requests import Request
        creds.refresh(Request())
    if not creds or not creds.valid:
        if not YT_SECRETS.exists():
            raise RuntimeError(f"no OAuth client secrets at {YT_SECRETS} — see agent docs")
        from google_auth_oauthlib.flow import InstalledAppFlow
        flow = InstalledAppFlow.from_client_secrets_file(str(YT_SECRETS), YT_SCOPES)
        # remote/SSH-friendly: fixed port, no local browser — user tunnels the port from their
        # laptop (ssh -L 8765:localhost:8765 <host>) and opens the printed URL there.
        creds = flow.run_local_server(port=8765, open_browser=False,
                                      authorization_prompt_message="\nOPEN THIS URL in the browser "
                                      "on your LAPTOP (after: ssh -L 8765:localhost:8765 <this-host>):"
                                      "\n\n{url}\n")
    YT_TOKEN.write_text(creds.to_json()); YT_TOKEN.chmod(0o600)
    return build("youtube", "v3", credentials=creds)


def build_metadata(props, hook=None):
    """YouTube-optimized Shorts metadata: hook-first title ≤100 chars with #Shorts, a description
    that front-loads the learning outcome + hashtags, focused tags. Used when the upload spec
    doesn't override."""
    import re
    lang = props["language"]
    hook = hook or props.get("hook") or f"{props['title']} explained"
    secs = int(max((x["end"] for x in props["segments"]), default=30))
    # topic keyword from the file title (fibonacci.py -> "fibonacci") — drives search/discovery
    topic = re.sub(r"\.[a-z0-9]+$", "", str(props.get("title") or "")).replace("_", " ").strip()
    lang_t = lang.capitalize() if lang.islower() else lang
    # TITLE: keyword-first (YouTube weights the front), concrete, <=100 chars, #Shorts at the end
    title = hook if hook.rstrip().endswith("#Shorts") else f"{hook} #Shorts"
    if len(title) > 100:
        title = (hook[:100 - len(" …#Shorts")].rstrip() + " …#Shorts")
    # DESCRIPTION: value in line 1 (YouTube shows ~the first line), what-you-learn, CTA, tight hashtags
    learn = " ".join(s["text"] for s in props["segments"][:3]).strip()
    if len(learn) > 400:                                   # trim to a whole sentence — never a mid-word "…"
        cut = learn[:400]
        p = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        learn = (cut[:p + 1] if p > 80 else cut).rstrip()
    topic_tag = "#" + re.sub(r"[^a-z0-9]", "", (topic or lang).lower())[:24]
    description = (
        f"{hook}\n\n"
        f"{learn}\n\n"
        f"▶ Watch the {lang_t} code typed live and RUN with real output — {secs}s, no fluff.\n"
        f"🔔 Subscribe for a new {lang_t} Short every day.\n\n"
        f"#Shorts #{lang} #coding #programming #learntocode {topic_tag}"
    ).strip()
    # TAGS: topical + broad, deduped, capped at 15 (YouTube limit)
    raw = [topic, f"{topic} {lang}", lang, f"{lang} tutorial", f"{lang} shorts", f"learn {lang}",
           "coding", "programming", "code", "tutorial", "shorts", "learntocode",
           "coding shorts", "software development", "developer"]
    seen, tags = set(), []
    for t in raw:
        t = (t or "").strip()
        if t and t.lower() not in seen:
            seen.add(t.lower()); tags.append(t)
    return {"title": title, "description": description, "tags": tags[:15]}


def youtube_upload(video_path, title, description="", tags=None, playlist="Learn Python V2",
                   privacy="private"):
    """Upload a rendered Short PRIVATE (never public from here) and add it to `playlist` (created if
    missing). Returns {video_id, playlist_id, url}. NOTE: brand-new API projects are additionally
    locked private by YouTube until the project passes their audit — fine for draft workflow."""
    from googleapiclient.http import MediaFileUpload
    if privacy != "private":
        raise RuntimeError("shorts-builder only uploads PRIVATE; publish manually from Studio")
    yt = _yt_service()
    body = {"snippet": {"title": title[:100], "description": description[:4900],
                        "tags": (tags or ["python", "coding", "shorts"])[:15], "categoryId": "27"},
            "status": {"privacyStatus": "private", "selfDeclaredMadeForKids": False}}
    media = MediaFileUpload(str(video_path), mimetype="video/mp4", resumable=True)
    req = yt.videos().insert(part="snippet,status", body=body, media_body=media)
    resp = None
    while resp is None:
        _, resp = req.next_chunk()
    vid = resp["id"]
    pl_id = None
    if playlist:
        found = yt.playlists().list(part="id,snippet", mine=True, maxResults=50).execute()
        for it in found.get("items", []):
            if it["snippet"]["title"].strip().lower() == playlist.strip().lower():
                pl_id = it["id"]; break
        if not pl_id:
            pl = yt.playlists().insert(part="snippet,status", body={
                "snippet": {"title": playlist, "description": "Auto-built code-typing Shorts"},
                "status": {"privacyStatus": "private"}}).execute()     # playlist private too
            pl_id = pl["id"]
        yt.playlistItems().insert(part="snippet", body={"snippet": {
            "playlistId": pl_id, "resourceId": {"kind": "youtube#video", "videoId": vid}}}).execute()
    return {"video_id": vid, "playlist_id": pl_id, "url": f"https://studio.youtube.com/video/{vid}/edit"}


def verify_sync(video_path, segments, fps=30):
    """CONFORMANCE CHECK — measure, don't assume, that captions/narration/video line up:
      (a) AUDIO: decode the muxed track; narration RMS inside caption windows must dominate the
          ducked bed outside them (ratio > 1.5).
      (b) VIDEO: decode the frame at each caption midpoint — the caption bubble (bright text in the
          bottom band) must be present; and absent in the final tail frame (after the last caption).
    Deterministic because Remotion renders the same segments JSON, so the only real risk is a mux
    offset — exactly what (a) catches. Guarded: missing pyav/ffmpeg → {"checked": False}."""
    import numpy as np
    res = {"checked": False, "audio_ok": None, "captions_ok": None}
    try:
        import tempfile
        import wave
        import imageio_ffmpeg
        import imageio.v3 as iio
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        with tempfile.TemporaryDirectory() as td:
            wavp = f"{td}/a.wav"
            try:
                r = subprocess.run([ff, "-y", "-v", "error", "-i", str(video_path), "-map", "0:a:0",
                                    "-ac", "1", "-ar", "24000", wavp], capture_output=True, timeout=300)
            except subprocess.TimeoutExpired:   # corrupt container: skip sync-check, not the whole build
                r = subprocess.CompletedProcess([], returncode=1)
            if r.returncode == 0:
                w = wave.open(wavp)
                a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float64)
                sr = 24000
                # VAD, not segment windows: the retimed caption windows drift out of alignment with the
                # muxed audio (TTS clip lengths vary per build → the timeline stretches, remotion trims
                # audio to video length), so a window-based mask catches bed-as-"in"/speech-as-"out" and
                # FALSELY FAILS a mix that is objectively clean. Detect the speech-active frames directly
                # from the energy envelope and check narration clearly dominates the ducked bed.
                win = max(1, sr // 20)
                env = np.sqrt(np.convolve(a ** 2, np.ones(win) / win, "same"))
                # reference off p99, not the absolute max: a single loud transient (a hard consonant
                # burst / music onset) inflates env.max() ~3x above p99, pushing 0.30*peak past all real
                # speech → speech_frac collapses to ~0.01 and a clean mix FALSELY fails (hit 033-sorted-key).
                peak = float(np.percentile(env, 99))
                sp = env > 0.30 * peak if peak > 0 else np.zeros(a.size, bool)
                frac = float(sp.mean())
                rin = float(np.sqrt((a[sp] ** 2).mean())) if sp.any() else 0.0
                rout = float(np.sqrt((a[~sp] ** 2).mean())) if (~sp).any() else 1.0
                res["rms_in"], res["rms_out"], res["speech_frac"] = round(rin), round(rout), round(frac, 2)
                # narration must dominate the bed by >1.5×, AND there must actually be speech (not all/none)
                res["audio_ok"] = (rin > 1.5 * max(rout, 1.0)) and (0.12 <= frac <= 0.75)
        frames = iio.imread(str(video_path), plugin="pyav")             # (T,H,W,3)
        T, H = frames.shape[0], frames.shape[1]
        band = slice(int(0.06 * H), int(0.22 * H))                      # caption zone (top band below YT nav-safe ~112px)
        def bright(fr):                                                 # bubble text = near-white pixels
            return int((fr[band].max(axis=-1) > 200).sum())
        # sample across the whole timeline (drift-proof); captions gap-fill so one is shown at all times,
        # so we require the vast majority of sampled frames to carry a caption + the tail still present.
        sampT = [min(T - 1, max(0, int(T * x))) for x in np.linspace(0.05, 0.98, 24)]
        hits = [bright(frames[i]) > 50 for i in sampT]
        tail_present = bright(frames[T - 1]) > 0
        res["captions_ok"] = (sum(hits) / max(len(hits), 1) >= 0.8) and tail_present
        res["caption_hits"] = [sum(hits), len(hits)]
        res["checked"] = True
        res["ok"] = bool(res["captions_ok"]) and (res["audio_ok"] is not False)
    except Exception as e:  # noqa: BLE001  (verification must never break the render)
        res["error"] = str(e)[:200]
    return res


def verify_layout(video_path, props, fps=30):
    """LAYOUT CONFORMANCE CHECKLIST — measure that the frame uses its space correctly at every stage:
      • caption sits in the top safe band (never above YouTube's ~112px nav-safe);
      • the code block is present and actually FILLS its pane (no big dead space) when there is no
        output; when an output block exists it must be visible in the lower overlay region;
      • nothing readable bleeds into YouTube's unsafe zones (top 112px, right action-rail, bottom CTA).
    Returns {checks: {name: bool}, warnings: [...], ok: bool}. Hard checks gate keep/reject; the
    "code_fills" check is a warning only (text wrapping makes exact fill fraction noisy)."""
    import numpy as np
    out = {"checked": False, "checks": {}, "warnings": []}
    try:
        import imageio.v3 as iio
        frames = iio.imread(str(video_path), plugin="pyav")             # (T,H,W,3)
        T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
        has_output = bool(props.get("outputs"))
        def bright_rect(fr, y0, y1, x0, x1):                            # near-white pixel count in a rect
            r = fr[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)]
            return int((r.max(axis=-1) > 200).sum()) if r.size else 0
        mid, late, tail = frames[T // 2], frames[int(T * 0.85)], frames[T - 1]
        c = out["checks"]
        # Presence checks sample MANY frames across the timeline, not one fixed instant: code is typed
        # progressively and outputs appear sequentially tied to their line, so a single frame is brittle
        # (a real, correct video can show 0 output pixels at exactly t=0.85 between two output blocks).
        sampT = [min(T - 1, max(0, int(T * x))) for x in np.linspace(0.30, 0.99, 24)]
        # caption present in the top safe band, and nothing above the 112px nav-safe line
        c["caption_present"] = bright_rect(mid, 0.06, 0.22, 0.0, 1.0) > 50
        c["top_nav_safe"] = bright_rect(mid, 0.0, 0.058, 0.0, 1.0) < 40
        # code block present and using its pane (code region ≈ y0.24..0.66, full width) — peak over samples
        code_hits = max(bright_rect(frames[i], 0.24, 0.66, 0.02, 0.98) for i in sampT)
        c["code_present"] = code_hits > 400
        # FILL: fraction of code-region rows that carry ink — warn (don't fail) if the pane is mostly empty
        cr = late[int(0.24 * H):int(0.66 * H)]
        row_ink = (cr.max(axis=-1) > 90).mean(axis=1) if cr.size else np.array([0.0])
        fill = float((row_ink > 0.02).mean())
        c["code_fills"] = fill > 0.45
        if not c["code_fills"] and not has_output:
            out["warnings"].append(f"code fills only {fill:.0%} of its pane (dead space)")
        # output block visible in the lower overlay region when declared — present in ANY sampled frame
        # (outputs are transient/sequential; require it to appear, not to be up at one hard-coded instant)
        if has_output:
            out_present = sum(1 for i in sampT if bright_rect(frames[i], 0.55, 0.74, 0.02, 0.98) > 200)
            c["output_present"] = out_present >= 2
            out["output_frames"] = [out_present, len(sampT)]
        # YouTube unsafe zones must stay clear: right action-rail (lower 60%) and bottom CTA at the tail
        # POSTER: YouTube uses frame 0 as the Shorts thumbnail (custom thumbs are ignored in the
        # feed), so the hook title card must be FULLY rendered at frame 0 — no fade-in. A 0.15s
        # fade once made the poster of every video a blank editor; this is a hard gate now.
        if props.get("hook"):
            c["poster_shows_hook"] = bright_rect(frames[0], 0.35, 0.60, 0.0, 1.0) > 500
        # Sample MANY frames, not just the midpoint: the typing caret sweeps right as each line
        # is typed, so a long line bleeds into the action-rail only during its own few seconds.
        # A single-frame check passed 7 of 8 genuinely-bleeding videos.
        c["right_rail_clear"] = max(bright_rect(frames[i], 0.40, 1.0, 0.90, 1.0)
                                    for i in sampT) < 60
        # band starts at the layout's SAFE_BOT (Short.tsx height-480 = 0.75); content legally fills to
        # there, so checking from 0.74 clipped the last output line in a 19px overlap above the boundary
        c["bottom_cta_clear"] = bright_rect(tail, 0.75, 1.0, 0.0, 1.0) < 60
        # HARD gate: everything except the informational fill check
        hard = {k: v for k, v in c.items() if k != "code_fills"}
        out["ok"] = all(hard.values())
        out["fill_frac"] = round(fill, 2)
        out["checked"] = True
    except Exception as e:  # noqa: BLE001  (verification must never break the render)
        out["error"] = str(e)[:200]
    return out


# ═════════════════════════════════════════════════ LONG-FORM: a paper → a 16:9 explainer video
# One agent, two formats. `format="short"` (the DEFAULT) is byte-for-byte the pipeline above — vertical
# 1080×1920, YouTube's Shorts safe zones, the typing-code composition. `format="video"` reuses everything
# that is format-agnostic (TTS, music, caption retiming, sync verification, the Postgres gallery) and
# swaps only what is genuinely format-specific: canvas, safe zones, caption placement and the renderer.
#
# The long-form storyboard is NOT written by hand: it is READ from a paper's lesson series produced by
# `paper-md` → `paper-learn`. Those lessons already hold, per section, the teaching prose, the formula in
# LaTeX, a crop of the formula as typeset in the PDF, a runnable proof and its REAL captured output, plus
# the figures. That is exactly a storyboard, so the video inherits the lessons' correctness: if a proof
# failed, it never reached the lesson, so it cannot reach the video.
FORMATS = {
    "short": dict(width=1080, height=1920, pil=(540, 960), composition="CodeShort",
                  max_seconds=180.0, caption="top",
                  # YouTube Shorts chrome: nav-safe top, the right action-rail, and the bottom CTA band
                  safe=dict(top_frac=0.058, bottom_frac=0.22, right_frac=0.13)),
    "video": dict(width=1920, height=1080, pil=(1280, 720), composition="PaperVideo",
                  max_seconds=1800.0, caption="lower-third",
                  # 16:9 watch page: only the title-safe top and the progress-bar strip are unusable
                  safe=dict(top_frac=0.04, bottom_frac=0.08, right_frac=0.02)),
}


def fmt(name):
    """Format spec by name; unknown names fall back to `short` so an old spec can never break."""
    return dict(FORMATS.get(str(name or "short"), FORMATS["short"]))


def _ffmpeg_exe():
    """Find an ffmpeg binary wherever it lives on this box.

    The lesson venv has no ffmpeg while the `llm` env ships imageio_ffmpeg's bundled build, so a renderer
    that assumes its own interpreter can encode silently degrades to a GIF. Probe: PATH → this env's
    imageio_ffmpeg → the other envs' bundled binaries. Returns a path or None.
    """
    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        pass
    for env in ("llm", "kaggle_vision", "nlp", "audio"):
        base = Path.home() / "miniconda3" / "envs" / env / "lib"
        for cand in sorted(base.glob("python3.*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*")):
            if os.access(cand, os.X_OK):
                return str(cand)
    return None


def parse_learning(path):
    """Parse a Pattern-B `.learning` file → (meta, cells). Same template the :7777 hub reads."""
    meta, cells, cur, mode, buf = {}, [], None, None, []

    def flush():
        if cur is not None and mode:
            cur[mode] = "\n".join(buf).strip("\n")

    for ln in Path(path).read_text(errors="replace").split("\n"):
        if cur is None and ln.startswith("@ ") and ":" in ln:
            k, v = ln[2:].split(":", 1)
            meta[k.strip()] = v.strip()
            continue
        m = re.match(r"^--- (note|code|output|image|shape)\s*$", ln)
        if m:
            flush()
            if m.group(1) == "note":
                if cur is not None:
                    cells.append(cur)
                cur = {}
            mode, buf = m.group(1), []
            continue
        if cur is not None or mode:
            buf.append(ln)
    flush()
    if cur is not None:
        cells.append(cur)
    return meta, cells


def _narratable(md, limit=340):
    """Markdown/LaTeX note → something a TTS voice can actually say.

    Reading `\\mathcal{M}_{t-1}` aloud is unlistenable, so display math is replaced by a spoken pointer
    ("the formula on screen") — the viewer READS the equation crop while the voice explains it. Bold and
    code markers are dropped, headings become the sentence they are.
    """
    t = re.sub(r"\$\$.*?\$\$", " the formula on screen. ", md or "", flags=re.S)
    t = re.sub(r"\$[^$]*\$", " ", t)
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"^\s*>\s?", "", t, flags=re.M)                       # blockquote markers
    t = re.sub(r"^#+\s*", "", t, flags=re.M)                         # headings
    t = re.sub(r"\|.*?\|", " ", t)                                   # markdown tables read terribly
    t = re.sub(r"[*_`\[\]]", "", t)
    t = re.sub(r"\((?:eq\.|equation)\s*(\d+)\)", r"equation \1", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:limit]


def paper_storyboard(slug, prefix, lessons_dir=None, papers_dir=None, ids=None,
                     seconds_per_scene=9.0, max_scenes=64):
    """A paper's lesson series → an ordered storyboard: title, chapters, scenes.

    Scene kinds: `title` (paper), `chapter` (one per lesson), `formula` (the PDF crop of that equation,
    with the note as narration), `figure` (a paper figure or one of our generated charts), `code` (a proof
    with its real captured output). Returns {title, subtitle, chapters, scenes, source}.
    """
    comp = COMP
    ld = Path(lessons_dir) if lessons_dir else comp / "learning" / "annotated"
    pd_ = Path(papers_dir) if papers_dir else comp / "docs" / "papers" / slug
    man = json.loads((pd_ / "manifest.json").read_text()) if (pd_ / "manifest.json").exists() else {}
    files = [ld / f"{i}.learning" for i in ids] if ids else sorted(ld.glob(f"{prefix}*.learning"))
    files = [f for f in files if f.exists()]
    scenes = [dict(kind="title", title=man.get("title") or slug,
                   subtitle=f"{man.get('pages', '?')} pages · {man.get('equations', '?')} formulas · "
                            f"every one proved in PyTorch",
                   narration=f"{man.get('title') or slug}. "
                             f"We go through this paper formula by formula, and every claim you hear is "
                             f"checked by code that runs.", seconds=6.0)]
    chapters = []
    for f in files:
        meta, cells = parse_learning(f)
        order = int(meta.get("order", 999))
        chapters.append(dict(id=meta.get("id", f.stem), title=meta.get("title", f.stem), order=order))
        scenes.append(dict(kind="chapter", title=meta.get("title", f.stem),
                           subtitle=meta.get("subtitle", ""), lesson=meta.get("id", f.stem),
                           narration=_narratable(meta.get("title", "")), seconds=4.0))
        for c in cells:
            note, code, out, img = c.get("note", ""), c.get("code"), c.get("output"), c.get("image")
            spoken = _narratable(note)
            if img:
                first, *rest = [x for x in img.split("\n") if x.strip()]
                scenes.append(dict(kind="formula" if "/eq/" in first else "figure",
                                   image=first.strip(), caption=" ".join(rest).strip(),
                                   narration=spoken, seconds=seconds_per_scene, lesson=meta.get("id")))
            elif code:
                scenes.append(dict(kind="code", code=code, output=(out or "")[:1200],
                                   narration=spoken, seconds=max(seconds_per_scene, len(code) / 45.0),
                                   lesson=meta.get("id")))
            elif spoken:
                scenes.append(dict(kind="note", text=note[:600], narration=spoken,
                                   seconds=seconds_per_scene * 0.7, lesson=meta.get("id")))
            if len(scenes) >= max_scenes:
                break
        if len(scenes) >= max_scenes:
            break
    chapters.sort(key=lambda c: c["order"])
    return dict(title=man.get("title") or slug, subtitle=man.get("slug", slug), scenes=scenes,
                chapters=chapters, source=str(pd_), lessons=[str(f) for f in files])


def storyboard_chapters(props):
    """YouTube chapter markers (`0:00 Title`) from the rendered scene timings — the long-form affordance
    a Short does not have. YouTube requires the first marker at 0:00 and at least three chapters."""
    lines, seen = ["0:00 Intro"], set()
    for sc in props.get("scenes", []):
        if sc.get("kind") != "chapter":
            continue
        t = float(sc.get("start", 0.0))
        if t < 10.0 or sc.get("title") in seen:
            continue
        seen.add(sc.get("title"))
        lines.append(f"{int(t) // 60}:{int(t) % 60:02d} {sc['title']}")
    return "\n".join(lines) if len(lines) >= 3 else ""


def build_paper_props(storyboard, format="video", cps=30, tail_seconds=2.5, target_seconds=None):
    """Storyboard → props that the FORMAT-AGNOSTIC machinery already understands.

    `segments` (what TTS retimes and `verify_sync` measures) and `code` (what the typing renderer needs)
    are produced from the scenes, so audio, caption sync and the gallery all work unchanged — only the
    canvas and the renderer differ.
    """
    F = fmt(format)
    scenes = [dict(s) for s in storyboard["scenes"]]
    segs, t = [], 0.0
    for sc in scenes:
        sc["start"] = round(t, 2)
        sc["end"] = round(t + float(sc.get("seconds", 8.0)), 2)
        if sc.get("narration"):
            segs.append({"start": sc["start"], "end": sc["end"], "text": sc["narration"]})
        t = sc["end"]
    code_scenes = [s for s in scenes if s.get("kind") == "code"]
    props = {
        "format": format, "width": F["width"], "height": F["height"], "fps": 30,
        "composition": F["composition"], "caption": F["caption"], "safe": F["safe"],
        "title": storyboard["title"], "hook": storyboard["title"][:80],
        "subtitle": storyboard.get("subtitle", ""),
        "language": "python", "cps": float(cps), "tailSeconds": float(tail_seconds),
        "maxSeconds": F["max_seconds"], "segments": segs,
        "code": "\n\n".join(s["code"] for s in code_scenes) or "# (no proof cells in this selection)\n",
        "outputs": [], "scenes": scenes, "chapters": storyboard.get("chapters", []),
        "sourceLessons": storyboard.get("lessons", []),
    }
    if target_seconds:
        props["targetSeconds"] = float(target_seconds)
    return props


def retime_scenes(props):
    """After `build_audio` retimed the narration, push those real timings back onto the scenes so the
    renderer, the captions and the chapter markers stay in lockstep (the sync bug we already fixed once
    for Shorts: a caption must never outlive its voice)."""
    segs = [s for s in props.get("segments", [])]
    i = 0
    for sc in props.get("scenes", []):
        if sc.get("narration") and i < len(segs):
            sc["start"], sc["end"] = float(segs[i]["start"]), float(segs[i]["end"])
            sc["speechEnd"] = float(segs[i].get("speechEnd", sc["end"]))
            i += 1
    props["scenes"] = sorted(props.get("scenes", []), key=lambda s: float(s.get("start", 0)))
    return props


def render_paper_pil(props, out_path, fps=30, size=None):
    """Render the 16:9 explainer with PIL + imageio: title card, chapter cards, formula crops (the REAL
    typeset math lifted from the PDF), figures/charts, and code panes with their captured output.

    Deliberately PIL rather than a new Remotion composition: this must run offline in the data-wise test
    and on a box with no Node, exactly like the Shorts fallback does.
    """
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont
    W, H = size or fmt(props.get("format", "video"))["pil"]
    BG, FG, MUTE, ACCENT = (14, 16, 22), (235, 238, 245), (140, 146, 160), (11, 108, 255)
    PANEL = (22, 25, 33)

    def font(px, mono=False):
        p = ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf" if mono
             else "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")
        try:
            return ImageFont.truetype(p, px)
        except OSError:
            return ImageFont.load_default()

    f_title, f_h1, f_body, f_mono, f_cap = (font(int(W * 0.040)), font(int(W * 0.030)),
                                            font(int(W * 0.020)), font(int(W * 0.0145), True),
                                            font(int(W * 0.0175)))

    def wrap(draw, text, fnt, max_w):
        words, lines, cur = str(text).split(), [], ""
        for w in words:
            t = (cur + " " + w).strip()
            if draw.textlength(t, font=fnt) <= max_w:
                cur = t
            else:
                lines.append(cur); cur = w
        if cur:
            lines.append(cur)
        return lines

    def fit(img, box_w, box_h):
        r = min(box_w / img.width, box_h / img.height, 1.0)
        return img.resize((max(1, int(img.width * r)), max(1, int(img.height * r))), Image.LANCZOS)

    safe = props.get("safe", fmt(props.get("format", "video"))["safe"])
    cap_top = int(H * (1 - safe["bottom_frac"]) - H * 0.11)              # lower-third caption band
    total = max((float(s.get("end", 0)) for s in props.get("scenes", [])), default=10.0)
    total = min(total + props.get("tailSeconds", 2.0), props.get("maxSeconds", 1800.0))

    def frame_for(t):
        im = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(im)
        sc = next((s for s in props["scenes"] if float(s.get("start", 0)) <= t < float(s.get("end", 0))),
                  None) or (props["scenes"][-1] if props.get("scenes") else {})
        d.rectangle([0, 0, W, int(H * 0.006)], fill=ACCENT)              # brand strip
        kind = sc.get("kind", "note")
        pad = int(W * 0.05)
        if kind == "title":
            for i, ln in enumerate(wrap(d, sc.get("title", ""), f_title, W - 2 * pad)[:3]):
                d.text((pad, int(H * 0.30) + i * int(W * 0.052)), ln, font=f_title, fill=FG)
            d.text((pad, int(H * 0.30) + 3 * int(W * 0.052) + 8), sc.get("subtitle", ""),
                   font=f_cap, fill=ACCENT)
        elif kind == "chapter":
            d.text((pad, int(H * 0.42)), "CHAPTER", font=f_cap, fill=ACCENT)
            for i, ln in enumerate(wrap(d, sc.get("title", ""), f_h1, W - 2 * pad)[:3]):
                d.text((pad, int(H * 0.47) + i * int(W * 0.040)), ln, font=f_h1, fill=FG)
        elif kind in ("formula", "figure"):
            p = COMP / sc.get("image", "")
            box_h = cap_top - int(H * 0.10)
            if p.exists():
                try:
                    img = fit(Image.open(p).convert("RGB"), W - 2 * pad, box_h)
                    if kind == "formula":                                # crops are dark-on-white
                        bg = Image.new("RGB", (img.width + 28, img.height + 28), (250, 250, 252))
                        bg.paste(img, (14, 14)); img = fit(bg, W - 2 * pad, box_h)
                    im.paste(img, ((W - img.width) // 2, int(H * 0.10) + (box_h - img.height) // 2))
                except Exception:  # noqa: BLE001
                    pass
            label = sc.get("caption") or ("the formula, as typeset in the paper" if kind == "formula" else "")
            d.text((pad, int(H * 0.055)), label[:110], font=f_cap, fill=MUTE)
        elif kind == "code":
            box = [pad, int(H * 0.10), W - pad, cap_top - int(H * 0.02)]
            d.rounded_rectangle(box, 10, fill=PANEL)
            y, lh = box[1] + 14, int(W * 0.0185)
            for ln in str(sc.get("code", "")).split("\n")[:22]:
                d.text((box[0] + 16, y), ln[:110], font=f_mono, fill=FG); y += lh
            outp = str(sc.get("output", "")).strip()
            if outp:
                oy = box[3] - int(H * 0.20)
                d.rounded_rectangle([box[0] + 8, oy, box[2] - 8, box[3] - 8], 8, fill=(18, 30, 24))
                d.text((box[0] + 22, oy + 8), "REAL OUTPUT", font=f_cap, fill=(0, 200, 140))
                yy = oy + 8 + int(W * 0.022)
                for ln in [l for l in outp.split("\n") if l.strip()][:6]:
                    d.text((box[0] + 22, yy), ln.replace("```", "")[:105], font=f_mono, fill=(190, 235, 210))
                    yy += lh
        else:
            for i, ln in enumerate(wrap(d, sc.get("text", ""), f_body, W - 2 * pad)[:9]):
                d.text((pad, int(H * 0.18) + i * int(W * 0.028)), ln, font=f_body, fill=FG)
        # caption: lower third, above the progress bar — the 16:9 counterpart of the Shorts top band
        cap = sc.get("narration", "")
        if cap and t < float(sc.get("speechEnd", sc.get("end", 0))):
            lines = wrap(d, cap, f_cap, int(W * 0.86))[:2]
            bh = len(lines) * int(W * 0.024) + 18
            d.rounded_rectangle([pad, cap_top, W - pad, cap_top + bh], 10, fill=(0, 0, 0))
            for i, ln in enumerate(lines):
                d.text((pad + 14, cap_top + 9 + i * int(W * 0.024)), ln, font=f_cap, fill=FG)
        return im

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    import numpy as np
    step = max(1, int(fps // 10))                                        # 10 distinct frames/s is plenty
    exe = _ffmpeg_exe()
    if exe:                             # pipe raw frames straight into H.264 — no temp files, any env
        cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               str(out_path)]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
        arr = None
        for i in range(int(total * fps)):
            if i % step == 0 or arr is None:
                arr = np.asarray(frame_for(i / fps), dtype=np.uint8).tobytes()
            pr.stdin.write(arr)
        pr.stdin.close()
        if pr.wait() != 0:
            raise RuntimeError("ffmpeg failed: " + pr.stderr.read().decode()[-300:])
        return str(out_path), float(total)
    out_path = str(Path(out_path).with_suffix(".gif"))                   # last resort (offline test box)
    wr = imageio.get_writer(out_path, duration=1.0 / fps, loop=0)
    arr = None
    for i in range(int(min(total, 20) * fps)):                           # a GIF of a 10-min video is absurd
        if i % (fps // 2) == 0 or arr is None:
            arr = np.asarray(frame_for(i / fps))
        wr.append_data(arr)
    wr.close()
    return out_path, float(total)


def verify_layout_video(video_path, props, fps=30):
    """Layout conformance for the 16:9 format: the counterpart of the Shorts checklist, with the bands
    that actually matter on a watch page (title-safe top, progress-bar bottom, caption in the lower
    third, content actually filling the frame)."""
    import numpy as np
    W0, H0 = int(props.get("width", 1920)), int(props.get("height", 1080))
    frames = []
    exe = _ffmpeg_exe()
    if exe:                    # decode via ffmpeg: imageio cannot READ mp4 without its plugin either,
        try:                   # and a verifier that silently returns None verifies nothing
            r = subprocess.run([exe, "-loglevel", "error", "-i", str(video_path),
                                "-vf", "fps=1/3", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                               capture_output=True, timeout=900)
            buf, n = r.stdout, W0 * H0 * 3
            frames = [np.frombuffer(buf[i * n:(i + 1) * n], dtype=np.uint8).reshape(H0, W0, 3)
                      for i in range(min(len(buf) // n, 40))]
        except Exception:  # noqa: BLE001
            frames = []
    if not frames:
        try:
            import imageio.v2 as imageio
            rd = imageio.get_reader(video_path)
            frames = [np.asarray(f) for i, f in enumerate(rd) if i % max(1, int(fps)) == 0][:40]
            rd.close()
        except Exception as e:  # noqa: BLE001
            return {"checks": {}, "ok": None, "note": f"unreadable: {type(e).__name__}"}
    if not frames:
        return {"checks": {}, "ok": None, "note": "no frames"}
    H, W = frames[0].shape[:2]
    safe = props.get("safe", FORMATS["video"]["safe"])

    def band(fr, y0, y1):
        return float(np.asarray(fr)[int(y0 * H):int(y1 * H)].mean())

    mid = frames[len(frames) // 2]
    c = {
        "aspect_16_9": abs(W / H - 16 / 9) < 0.02,
        "title_safe_top_clear": band(mid, 0.0, safe["top_frac"]) < 60,
        "progress_bar_strip_clear": band(mid, 1 - safe["bottom_frac"] * 0.4, 1.0) < 70,
        "content_fills_middle": band(mid, 0.15, 0.75) > 12,
        "caption_in_lower_third": any(band(f, 0.66, 0.90) > band(f, 0.05, 0.12) for f in frames),
        "not_a_still": len({round(band(f, 0.1, 0.9), 1) for f in frames}) > 1,
    }
    return {"checks": c, "ok": all(v for v in c.values()), "frames": len(frames), "size": [W, H]}


class ShortsBuilder(BaseAgent):
    name = "shorts-builder"
    thread = "S"; kind = "finding"

    def run(self, q, worker):
        import time
        t_build0 = time.time()
        s = self.spec(q)
        if str(s.get("format", "short")) != "short":                    # ── LONG-FORM branch (paper video)
            return self.run_video(s, worker, t_build0)
        code = s.get("code") or (Path(s["code_path"]).read_text() if s.get("code_path") else None)
        transcript = s.get("transcript") or (Path(s["transcript_path"]).read_text()
                                             if s.get("transcript_path") else "")
        language = s.get("language", "python")
        if not code:                                                    # demo spec so a bare fleet run works
            code = 'def hello(name: str) -> str:\n    return f"hi {name}"\n'
            transcript = transcript or "A tiny typed function\nf-strings format inline"
        props = build_props(code, transcript, language, title=s.get("title"),
                            cps=s.get("cps", 30), tail_seconds=s.get("tail_seconds", 2.0),
                            max_seconds=s.get("max_seconds", 90.0))
        if s.get("target_seconds"):
            props["targetSeconds"] = float(s["target_seconds"])
        props["hook"] = s.get("hook") or ""                             # title card = Shorts poster frame
        if s.get("music_prompt"):
            props["musicPrompt"] = s["music_prompt"]
        props["ttsEngine"] = s.get("tts_engine", "chatterbox")
        if s.get("accent"):
            props["accent"] = s["accent"]   # per-playlist chrome variant (astryxTheme.accents)
        raw_outputs = s.get("outputs")
        pace_lock(props)
        if not raw_outputs and s.get("outputs_path"):
            raw_outputs = json.loads(Path(s["outputs_path"]).read_text())
        props["outputs"] = normalize_outputs(raw_outputs)
        AUTO.mkdir(parents=True, exist_ok=True)
        out = s.get("out") or str(AUTO / "code_short.mp4")
        fps = int(s.get("fps", 30))
        # audio FIRST: narration retimes captions/cps, so props must be final before the render
        wav, has_speech = None, False
        if s.get("music", "auto") != "off" or s.get("speech", "auto") != "off":
            try:
                props, wav, has_speech = build_audio(
                    props, AUTO / "shorts_audio.wav", voice=s.get("voice", "af_heart"),
                    speech=s.get("speech", "auto") != "off", music=s.get("music", "auto") != "off",
                    voice_ref=s.get("voice_ref"),
                    narration_dir=s.get("narration_dir"))
            except Exception:  # noqa: BLE001  (numpy-only path failing would be a real bug, but never
                wav = None     # let audio kill the video)
        # render to a "_silent" temp path; the deliverable ends up at the plain `out` path (no suffix)
        render_out = str(Path(out).with_name(Path(out).stem + "_silent" + Path(out).suffix))
        try:
            if s.get("force_pil") or os.environ.get("SHORTS_FORCE_PIL"):
                raise RuntimeError("PIL renderer forced")
            r = render_remotion(props, render_out, fps=fps)
        except Exception as e:  # noqa: BLE001  (no Node/Chromium/npm → offline fallback)
            size = tuple(int(v) for v in str(s.get("size", "540x960")).split("x"))
            try:
                r = render_pil(props, render_out, fps=fps, size=size)
            except Exception:  # noqa: BLE001                            # ffmpeg absent → GIF
                render_out = str(Path(render_out).with_suffix(".gif"))
                r = render_pil(props, render_out, fps=fps, size=size)
            r["fallback_reason"] = str(e)[:200]
        track = s.get("audio") or wav                                   # user wav wins over generated mix
        if track and not str(render_out).endswith(".gif"):
            try:
                r["path"] = mux(render_out, track, out)
                r["audio"] = "speech+music" if has_speech else "music"
                Path(render_out).unlink(missing_ok=True)  # silent pre-mux render is regenerable, don't keep it
            except Exception as e:  # noqa: BLE001
                r["audio_error"] = str(e)[:200]
                r["path"] = render_out                    # mux failed → silent render is the only deliverable
        else:
            r["path"] = render_out                        # gif, or no audio track at all: silent IS the output
        sync = verify_sync(r["path"], props["segments"], fps=fps) if r["renderer"] == "remotion" else \
            {"checked": False}
        if r["renderer"] == "remotion":                                 # layout conformance checklist
            layout = verify_layout(r["path"], props, fps=fps)
            sync["layout"] = layout
            if layout.get("checked"):
                sync["ok"] = bool(sync.get("ok")) and bool(layout.get("ok"))
        sync_word = ("OK" if sync.get("ok") else "FAIL") if sync.get("checked") else "skipped"
        msg = (f"shorts-builder: rendered {props['language']} typing short via {r['renderer']} → {r['path']} "
               f"({len(props['segments'])} caption segments, cps={props['cps']:.0f}, "
               f"audio={r.get('audio', 'none')}, sync={sync_word})")
        self.log(msg, kind="finding",
                 recommendation="spec: code/code_path + transcript(+timings) + language; Remotion render "
                                "(remotion_shorts/, astryx theme) when Node present, PIL fallback offline")
        up = s.get("upload")
        if up and sync.get("ok"):                                       # sync-gated: never upload a FAIL
            try:
                meta = build_metadata(props, hook=up.get("hook"))
                info = youtube_upload(r["path"], title=up.get("title") or meta["title"],
                                      description=up.get("description") or meta["description"],
                                      tags=up.get("tags") or meta["tags"],
                                      playlist=up.get("playlist", "Learn Python V2"))
                r["upload"] = info
                msg += f" | uploaded PRIVATE → {info['url']} (playlist {up.get('playlist', 'Learn Python V2')})"
            except Exception as e:  # noqa: BLE001
                r["upload_error"] = str(e)[:300]; msg += f" | upload failed: {r['upload_error']}"
        elif up:
            msg += " | upload SKIPPED (sync not OK)"
        build_seconds = round(time.time() - t_build0, 1)
        msg += f" | built in {build_seconds}s"
        # publish to the review gallery (:9090 shorts_hub): copy + sidecar meta + PG row
        try:
            GALLERY.mkdir(exist_ok=True)
            final = Path(r["path"])
            gal = final if final.parent == GALLERY else Path(shutil.copy2(final, GALLERY / final.name))
            playlist = ((s.get("upload") or {}).get("playlist") if isinstance(s.get("upload"), dict)
                        else None) or s.get("playlist") or "Learn Python V2"
            side = {k: v for k, v in props.items() if k != "outputs"}   # keep sidecar small (no base64)
            side["playlist"] = playlist
            prev_side = gal.with_suffix(".json")                        # re-render of same clip → bump version
            try:                                                        # a 0-byte/corrupt sidecar (e.g. build
                prev = json.loads(prev_side.read_text()) if prev_side.exists() else {}
            except (json.JSONDecodeError, ValueError):                  # killed mid-write) must not abort the
                prev = {}                                               # re-render's sidecar+PG persist step
            if not isinstance(prev, dict):
                prev = {}
            prev_up = prev.get("upload") or {}
            version = int(prev.get("version") or 1) + 1 if prev else 1
            carry_up = r.get("upload") or (prev_up or None)             # keep the existing YouTube link if not re-uploaded
            prev_side.write_text(json.dumps(
                {"props": side, "n_outputs": len(props.get("outputs") or []),
                 "build_seconds": build_seconds, "sync": sync, "upload": carry_up,
                 "version": version, "renderer": r["renderer"]}, indent=2))
            r["upload"] = carry_up
            r["gallery"] = str(gal)
            try:                                                        # poster = the title-card frame
                import imageio.v3 as _iio
                _iio.imwrite(str(gal.with_suffix(".png")),
                             _iio.imread(str(gal), plugin="pyav", index=8))
            except Exception:  # noqa: BLE001
                pass
            try:                                                        # PG = queryable source for the hub
                segs = props["segments"]
                record_video(gal.name, title=props["title"], language=props["language"],
                             duration_s=round(max((x["end"] for x in segs), default=0)
                                              + props["tailSeconds"], 1),
                             captions=len(segs), n_outputs=len(props.get("outputs") or []),
                             build_seconds=build_seconds, renderer=r["renderer"], sync=sync,
                             upload=r.get("upload"), playlist=playlist, version=version,
                             status="uploaded" if r.get("upload") else "draft")
            except Exception:  # noqa: BLE001  (PG down → sidecar still has everything)
                pass
        except Exception:  # noqa: BLE001
            import traceback, sys as _sys
            print("PUBLISH_BLOCK_ERROR:", traceback.format_exc(), file=_sys.stderr)
        return self.done({"path": r["path"], "renderer": r["renderer"], "props": props, "sync": sync,
                          "upload": r.get("upload"), "upload_error": r.get("upload_error"),
                          "gallery": r.get("gallery")}, msg)


def _run_video(self, s, worker, t_build0):
    """Build the long-form paper explainer. Shares TTS/music/caption-retiming/sync/gallery with Shorts.

    Spec: {"format":"video", "paper":"kimi-k3", "prefix":"k3", "ids":[…]?, "target_seconds":600,
           "max_scenes":64, "out":…, "upload":false}
    """
    import time
    paper = s.get("paper") or s.get("slug")
    if not paper:
        return self.escalate(worker, "leader",
                             f"[{worker}] shorts-builder format=video needs `spec.paper` (a paper-md slug, "
                             f"e.g. 'kimi-k3'); its lesson series is the storyboard.")
    sb = paper_storyboard(paper, s.get("prefix") or paper[:2], ids=s.get("ids"),
                          seconds_per_scene=float(s.get("seconds_per_scene", 9.0)),
                          max_scenes=int(s.get("max_scenes", 64)))
    if len(sb["scenes"]) < 3:
        return self.escalate(worker, "leader",
                             f"[{worker}] no lessons found for paper='{paper}' prefix='{s.get('prefix')}' — "
                             f"run kind=paper-learn first (the lessons ARE the storyboard).")
    props = build_paper_props(sb, format=s.get("format", "video"), cps=s.get("cps", 30),
                              tail_seconds=s.get("tail_seconds", 2.5),
                              target_seconds=s.get("target_seconds"))
    if s.get("music_prompt"):
        props["musicPrompt"] = s["music_prompt"]
    if s.get("action") == "prepare" or s.get("render") is False:
        # PREPARE ONLY: write every input a build needs into the paper's own folder and stop. Rendering a
        # 10-minute video costs minutes of GPU/CPU and a voice pass; preparing costs nothing and is what
        # you actually want to review first. `kind=shorts-builder {format:"video", paper:…}` renders it
        # later from exactly these files, with no re-derivation.
        vdir = (COMP / "docs" / "papers" / paper / "video")
        vdir.mkdir(parents=True, exist_ok=True)
        assets = [sc["image"] for sc in props["scenes"] if sc.get("image")]
        missing = [a for a in assets if not (COMP / a).exists()]
        (vdir / "storyboard.json").write_text(json.dumps(sb, indent=2))
        (vdir / "props.json").write_text(json.dumps(props, indent=2))
        (vdir / "narration.txt").write_text("\n\n".join(
            f"[{i + 1:03d}] {sg['text']}" for i, sg in enumerate(props["segments"])))
        (vdir / "chapters.txt").write_text(storyboard_chapters(props) or "")
        (vdir / "assets.txt").write_text("\n".join(assets))
        words = sum(len(sg["text"].split()) for sg in props["segments"])
        readme = (f"# Video build inputs — {props['title']}\n\n"
                  f"Prepared by `shorts-builder` (format={props['format']}) from this paper's lesson series.\n"
                  f"Nothing here is rendered; this is the reviewable input set.\n\n"
                  f"- `storyboard.json` — scenes in order (title/chapter/formula/figure/code/note)\n"
                  f"- `props.json` — render props: canvas {props['width']}×{props['height']}, safe zones,"
                  f" caption band, segment timings\n"
                  f"- `narration.txt` — the spoken script, one block per caption segment\n"
                  f"- `chapters.txt` — YouTube chapter markers\n"
                  f"- `assets.txt` — every image the render needs (formula crops, figures, charts)\n\n"
                  f"**Scenes** {len(props['scenes'])} · **segments** {len(props['segments'])} · "
                  f"**assets** {len(assets)}"
                  + (f" (⚠ {len(missing)} MISSING)" if missing else " (all present)") +
                  f" · **script** ~{words} words ≈ {words / 150:.1f} min of speech\n\n"
                  f"Build later:\n```\nkind=shorts-builder  spec={{\"format\":\"video\","
                  f"\"paper\":\"{paper}\",\"prefix\":\"{s.get('prefix') or paper[:2]}\"}}\n```\n")
        (vdir / "README.md").write_text(readme)
        data = dict(prepared=str(vdir.relative_to(COMP)), scenes=len(props["scenes"]),
                    segments=len(props["segments"]), assets=len(assets), missing_assets=missing,
                    chapters=len(props.get("chapters", [])), script_words=words,
                    est_speech_minutes=round(words / 150.0, 1), lessons=len(props.get("sourceLessons", [])),
                    size=[props["width"], props["height"]], format=props["format"])
        msg = (f"[{worker}] VIDEO INPUTS READY for `{paper}` → `{data['prepared']}/` "
               f"({data['scenes']} scenes from {data['lessons']} lessons, {data['segments']} narration "
               f"segments ≈ {data['est_speech_minutes']} min, {data['assets']} assets"
               + (f", ⚠ {len(missing)} missing" if missing else ", all present") +
               f", {data['chapters']} chapters). NOT rendered — build it later from these files.")
        self.post(worker, "all", msg)
        return self.done(data, msg)
    props["ttsEngine"] = s.get("tts_engine", "chatterbox")
    AUTO.mkdir(parents=True, exist_ok=True)
    out = s.get("out") or str(AUTO / f"paper_{paper}.mp4")
    fps = int(s.get("fps", 30))
    wav, has_speech = None, False
    if s.get("music", "auto") != "off" or s.get("speech", "auto") != "off":
        try:
            props, wav, has_speech = build_audio(
                props, AUTO / f"paper_{paper}_audio.wav", voice=s.get("voice", "af_heart"),
                speech=s.get("speech", "auto") != "off", music=s.get("music", "auto") != "off",
                voice_ref=s.get("voice_ref"), narration_dir=s.get("narration_dir"))
        except Exception:  # noqa: BLE001 — audio must never kill the render (same rule as Shorts)
            wav = None
    props = retime_scenes(props)                                        # captions/scenes/chapters in lockstep
    silent = str(Path(out).with_name(Path(out).stem + "_silent" + Path(out).suffix))
    rendered, dur = render_paper_pil(props, silent, fps=fps, size=tuple(s.get("size") or ()) or None)
    renderer = "pil-16x9"
    final = out
    try:
        if wav is not None and str(rendered).endswith(".mp4"):
            mux(rendered, wav, out)
        else:
            final = rendered
    except Exception:  # noqa: BLE001
        final = rendered
    sync = verify_sync(final, props, fps=fps) if has_speech else {"ok": None, "note": "no speech"}
    layout = verify_layout_video(final, props, fps=fps)
    chapters = storyboard_chapters(props)
    meta = build_metadata(props) if "build_metadata" in globals() else {}
    if isinstance(meta, dict) and chapters:
        meta["description"] = (meta.get("description", "") + "\n\nChapters\n" + chapters).strip()
    data = dict(out=final, format=props["format"], size=[props["width"], props["height"]],
                seconds=round(float(dur), 2), scenes=len(props["scenes"]),
                chapters=len(props.get("chapters", [])), chapter_markers=chapters,
                lessons=len(props.get("sourceLessons", [])), renderer=renderer,
                sync=sync, layout=layout, build_seconds=round(time.time() - t_build0, 1),
                paper=paper, metadata=meta)
    try:
        record_video(name=f"paper_{paper}", title=props["title"], language="python",
                     spec={**{k: v for k, v in s.items() if k != "narration_dir"}, "format": props["format"]},
                     renderer=renderer, sync=sync, duration_s=float(dur),
                     captions=len(props["segments"]), n_outputs=sum(
                         1 for sc in props["scenes"] if sc.get("output")),
                     build_seconds=float(data["build_seconds"]))
    except Exception:  # noqa: BLE001 — no Postgres on this box is not a build failure
        pass
    msg = (f"[{worker}] PAPER VIDEO ✅ `{props['title'][:60]}` → `{final}` "
           f"({props['width']}×{props['height']}, {data['seconds']}s, {data['scenes']} scenes from "
           f"{data['lessons']} lessons, {data['chapters']} chapters; renderer {renderer}; "
           f"layout {'OK' if layout.get('ok') else layout.get('ok')}, sync {sync.get('ok')}). "
           f"Shorts pipeline untouched (format='short' is still the default).")
    self.post(worker, "all", msg)
    return self.done(data, msg)


ShortsBuilder.run_video = _run_video

_AGENT = ShortsBuilder()


def run_shorts(q, worker):
    return _AGENT.run(q, worker)


# ---------------------------------------------------------------- postgres (kaggle_shorts)
from .db import PG

DB = "kaggle_shorts"

DDL = """
CREATE TABLE IF NOT EXISTS videos (
    id BIGSERIAL PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    title TEXT, language TEXT,
    duration_s REAL DEFAULT 0, captions INT DEFAULT 0, n_outputs INT DEFAULT 0,
    build_seconds REAL DEFAULT 0,
    renderer TEXT, sync JSONB, upload JSONB, spec JSONB,
    playlist TEXT DEFAULT 'Learn Python V2',
    version INT DEFAULT 1,
    status TEXT DEFAULT 'draft',
    created_at TIMESTAMPTZ DEFAULT now()
);
"""


def _connect(dbname=DB):
    import psycopg2
    return psycopg2.connect(dbname=dbname, **PG)


def ensure_db():
    import psycopg2
    con = psycopg2.connect(dbname="postgres", **PG); con.autocommit = True
    try:
        cur = con.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{DB}"')
    finally:
        con.close()
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute(DDL)
        cur.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS playlist TEXT DEFAULT 'Learn Python V2'")
        cur.execute("ALTER TABLE videos ADD COLUMN IF NOT EXISTS version INT DEFAULT 1")
        con.commit()
    finally:
        con.close()


def record_video(name, *, title=None, language=None, duration_s=0, captions=0, n_outputs=0,
                 build_seconds=0, renderer=None, sync=None, upload=None, spec=None, status="draft",
                 playlist="Learn Python V2", version=1):
    """Upsert one video row (re-render of the same name overwrites — latest build wins)."""
    ensure_db()
    con = _connect()
    try:
        con.cursor().execute("""
            INSERT INTO videos (name,title,language,duration_s,captions,n_outputs,build_seconds,
                                renderer,sync,upload,spec,status,playlist,version)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (name) DO UPDATE SET
                title=EXCLUDED.title, language=EXCLUDED.language, duration_s=EXCLUDED.duration_s,
                captions=EXCLUDED.captions, n_outputs=EXCLUDED.n_outputs,
                build_seconds=EXCLUDED.build_seconds, renderer=EXCLUDED.renderer,
                sync=EXCLUDED.sync, upload=EXCLUDED.upload, spec=EXCLUDED.spec,
                status=EXCLUDED.status, playlist=EXCLUDED.playlist, version=EXCLUDED.version,
                created_at=now()
        """, (name, title, language, duration_s, captions, n_outputs, build_seconds, renderer,
              json.dumps(sync or {}), json.dumps(upload) if upload else None,
              json.dumps(spec or {}), status, playlist, version))
        con.commit()
    finally:
        con.close()


_YT_LIST_RE = re.compile(r"[?&]list=([A-Za-z0-9_-]+)")
_YT_VID_RE = re.compile(r"(?:v=|/video/|youtu\.be/|/shorts/)([A-Za-z0-9_-]{6,})")


def _merge_upload(name, patch):
    """Read the videos.upload JSONB for `name`, merge `patch` into it, write back; also update the
    gallery sidecar so PG and sidecar stay in lockstep. Creates a row if none exists."""
    ensure_db()
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute("SELECT upload,playlist FROM videos WHERE name=%s", (name,))
        row = cur.fetchone()
        up = (row[0] if row and row[0] else {}) or {}
        up.update({k: v for k, v in patch.items() if v is not None})
        if row:
            cur.execute("UPDATE videos SET upload=%s, status=%s WHERE name=%s",
                        (json.dumps(up), "uploaded" if up.get("url") else "draft", name))
        else:
            cur.execute("INSERT INTO videos (name,upload,status) VALUES (%s,%s,%s)",
                        (name, json.dumps(up), "uploaded" if up.get("url") else "draft"))
        con.commit()
    finally:
        con.close()
    side = GALLERY / (Path(name).stem + ".json")                       # keep sidecar consistent
    try:
        m = json.loads(side.read_text()) if side.exists() else {}
        cur_up = (m.get("upload") or {})
        cur_up.update({k: v for k, v in patch.items() if v is not None})
        m["upload"] = cur_up
        side.write_text(json.dumps(m, indent=2))
    except Exception:  # noqa: BLE001
        pass
    return up


def set_video_link(name, url):
    """Manually record (or clear, with "") the YouTube URL for one video (name = the .mp4 file name)."""
    m = _YT_VID_RE.search(url or "")
    return _merge_upload(name, {"url": (url or ""), "video_id": (m.group(1) if m else "")})


def set_playlist_link(playlist, url):
    """Manually record the YouTube playlist URL — extracts list=<id> and stamps it on every video
    in that playlist so the hub's playlist link resolves."""
    m = _YT_LIST_RE.search(url or "")
    pl_id = m.group(1) if m else None
    ensure_db()
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM videos WHERE playlist=%s", (playlist,))
        names = [r[0] for r in cur.fetchall()]
    finally:
        con.close()
    for n in names:
        _merge_upload(n, {"playlist_id": pl_id, "playlist_url": url or None})
    return {"playlist": playlist, "playlist_id": pl_id, "videos": len(names)}


def list_videos():
    """All non-deleted videos, newest first → list of dicts for the hub."""
    ensure_db()
    con = _connect()
    try:
        cur = con.cursor()
        cur.execute("""SELECT name,title,language,duration_s,captions,n_outputs,build_seconds,
                              renderer,sync,upload,status,created_at,playlist
                       FROM videos WHERE status != 'deleted' ORDER BY created_at DESC""")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


def mark_deleted(name):
    ensure_db()
    con = _connect()
    try:
        con.cursor().execute("UPDATE videos SET status='deleted' WHERE name=%s", (name,))
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------- review hub (:9090)
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler  # noqa: E402


PORT = 9090

# astryx neutral tokens, light/dark pairs via prefers-color-scheme (system theme, no default forced)
_FAVICON = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#a0caff"/><stop offset="1" stop-color="#efa8ff"/></linearGradient></defs><rect x="6" y="6" width="52" height="52" rx="16" fill="url(#g)"/><path d="M26 21l19 11-19 11z" fill="#fff"/></svg>'
_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Learn Python V2</title>
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
@import url('https://fonts.googleapis.com/css2?family=Figtree:wght@400;600;700;800&display=swap');
:root{color-scheme:light dark;
  /* astryx neutral theme (Meta, MIT) — exact tokens, follows system theme.
     grayscale spine 50:#fafafa 100:#f5f5f5 200:#e5e5e5 400:#a3a3a3 500:#737373 900:#171717 950:#0a0a0a
     syntax = OKLCH T30 (light) stops from neutralTheme.ts */
  --bg:#fafafa;--surface:#ffffff;--elev:#f5f5f5;--border:#e5e5e5;--text:#171717;--muted:#737373;--faint:#a3a3a3;
  --accent:#00458c;--purple:#700084;--green:#005600;--orange:#6e3500;--yellow:#584400;--teal:#005348;
  --ok:#005600;--okbg:#e4f2e2;--fail:#89001a;--failbg:#ffe8e6;
  --up:#00458c;--upbg:#e2edfb;--shadow:0 1px 3px rgba(0,0,0,.08),0 8px 24px rgba(0,0,0,.06)}
@media (prefers-color-scheme:dark){:root{
  /* astryx dark — grayscale spine + OKLCH T80 dark syntax stops */
  --bg:#1b1b1b;--surface:#262626;--elev:#0a0a0a;--border:#404040;--text:#e5e5e5;--muted:#a3a3a3;--faint:#737373;
  --accent:#a0caff;--purple:#efa8ff;--green:#a6d2a2;--orange:#ffb37f;--yellow:#eec12f;--teal:#83dac9;
  --ok:#a6d2a2;--okbg:#1c2a1b;--fail:#ffaeaa;--failbg:#301b1a;
  --up:#a0caff;--upbg:#16233a;--shadow:0 1px 3px rgba(0,0,0,.5),0 8px 24px rgba(0,0,0,.35)}}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--text);margin:0;
  font-family:Figtree,-apple-system,'Segoe UI',Roboto,Helvetica,sans-serif}
header{display:flex;align-items:center;gap:12px;padding:14px 24px;
  border-bottom:1px solid var(--border);background:var(--surface);position:sticky;top:0;z-index:2}
.logo{width:26px;height:26px;border-radius:8px;background:linear-gradient(135deg,#a0caff,#efa8ff)}
h1{font-size:15px;margin:0;font-weight:700}
.count{color:var(--muted);font-size:12px;margin-left:auto}
.wrap{display:flex;height:calc(100vh - 55px);overflow:hidden}
.col{border-right:1px solid var(--border);padding:16px;overflow-y:auto;min-height:0}
.pl-col{width:240px}
.vid-col{width:340px}
.colhead{font-size:11px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  color:var(--muted);margin:4px 4px 12px}
.pl{display:block;text-decoration:none;color:var(--text);border:1px solid transparent;
  border-radius:12px;padding:12px 14px;margin-bottom:6px}
.pl:hover{background:var(--surface)}
.pl.active{background:var(--surface);border-color:transparent;box-shadow:var(--shadow);
  border-left:3px solid var(--accent);background-image:linear-gradient(90deg,rgba(160,202,255,.08),transparent 40%)}
.pl .nm{font-weight:700;font-size:14px}
.pl .info{color:var(--muted);font-size:11.5px;margin-top:3px}
.item{display:block;text-decoration:none;color:var(--text);background:var(--surface);
  border:1px solid var(--border);border-radius:14px;padding:12px 14px;margin-bottom:10px;
  transition:transform .12s,box-shadow .12s}
.item:hover{transform:translateY(-1px);box-shadow:var(--shadow)}
.item.active{border-color:var(--accent);box-shadow:var(--shadow)}
.iname{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px;
  word-break:break-all;margin-bottom:8px}
.row{display:flex;gap:10px;align-items:center}
.grow{flex:1;min-width:0}
.thumb{width:44px;height:78px;object-fit:cover;border-radius:8px;border:1px solid var(--border)}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{font-size:10.5px;padding:2px 9px;border-radius:999px;background:var(--elev);
  color:var(--muted);border:1px solid var(--border)}
.chip.ok{background:var(--okbg);color:var(--ok);border-color:transparent}
.chip.fail{background:var(--failbg);color:var(--fail);border-color:transparent}
.chip.up{background:var(--upbg);color:var(--up);border-color:transparent}
.player{flex:1;padding:28px;display:flex;flex-direction:column;align-items:center;overflow-y:auto;min-height:0}
.stage{background:var(--surface);border:1px solid var(--border);border-radius:20px;
  padding:18px;box-shadow:var(--shadow);max-width:520px;width:100%}
.vtitle{font-weight:700;font-size:15px;margin:2px 4px 12px}
.dlall{margin-left:auto;padding:6px 14px;border-radius:999px;background:linear-gradient(135deg,#a0caff,#efa8ff);color:#171717;font-weight:700;font-size:12px;text-decoration:none}.dlall:hover{filter:brightness(1.05)}
.count{margin-left:14px}
.dlbtn{display:inline-block;margin:14px auto 0;padding:9px 18px;border-radius:999px;background:linear-gradient(135deg,#a0caff,#efa8ff);color:#171717;font-weight:700;font-size:13px;text-decoration:none;box-shadow:var(--shadow)}.dlbtn:hover{filter:brightness(1.05)}
video{height:70vh;max-width:100%;width:auto;aspect-ratio:9/16;border-radius:14px;background:#000;
  display:block;margin:0 auto}
.meta{margin-top:14px;display:flex;flex-wrap:wrap;gap:8px;justify-content:center}
.meta .chip{font-size:12px;padding:4px 12px}
.empty{color:var(--muted);margin-top:28vh;font-size:14px}
a.studio{color:var(--up);text-decoration:none;font-weight:600}
.details{max-width:520px;width:100%;margin-top:20px;display:flex;flex-direction:column;gap:14px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:16px 18px;box-shadow:var(--shadow)}
.card .lbl{font-size:10.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:8px}
.kv{display:flex;gap:10px;font-size:13px;margin:5px 0}
.kv .k{color:var(--muted);min-width:60px}
.kv .v{color:var(--text);word-break:break-word}
.kv .mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.desc{white-space:pre-wrap;font-size:13px;line-height:1.5;color:var(--text)}
.copy{cursor:pointer;border:1px solid var(--border);background:var(--elev);color:var(--muted);border-radius:8px;font-size:11px;padding:3px 10px;float:right}
.lnkrow{display:flex;gap:6px;margin-top:8px}
.lnk{flex:1;min-width:0;border:1px solid var(--border);background:var(--bg);color:var(--text);border-radius:8px;font-size:12px;padding:5px 8px;font-family:ui-monospace,Menlo,monospace}
.lnkrow .copy{float:none;padding:5px 12px}
.copy:hover{color:var(--text)}
.rel{display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:10px}
.rel a{text-decoration:none;color:var(--text)}
.rel img{width:100%;aspect-ratio:9/16;object-fit:cover;border-radius:10px;border:1px solid var(--border)}
.rel a.cur img{border-color:var(--accent);box-shadow:var(--shadow)}
.rel .rt{font-size:10.5px;color:var(--muted);margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
__VOICECSS__
</style></head><body>
<header><div class="logo"></div><h1>Learn Python V2</h1>
<a id="dlall" class="dlall" download>&#8681; Playlist</a>
__VOICE__
<span class="count">__N__ videos · say "ready to push &lt;name&gt;" in chat to upload private</span></header>
<div class="wrap">
  <nav class="col pl-col"><div class="colhead">Playlists</div><div id="pls"></div></nav>
  <nav class="col vid-col"><div class="colhead">Videos</div><div id="links"></div></nav>
  <main class="player">
    <div class="stage" id="stage" style="display:none">
      <div class="vtitle" id="vtitle"></div>
      <video id="vid" controls></video>
 <a id="dl" class="dlbtn" download>&#8681; Download MP4</a>
      <div id="meta" class="meta"></div>
    </div>
    <div class="details" id="details" style="display:none">
      <div class="card">
        <div class="lbl">Details</div>
        <div class="kv"><span class="k">ID</span><span class="v mono" id="d-id"></span></div>
        <div class="kv"><span class="k">Title</span><span class="v" id="d-title"></span></div>
        <div class="kv"><span class="k">Playlist</span><span class="v" id="d-pl"></span></div>
        <div class="kv"><span class="k">Version</span><span class="v" id="d-ver"></span></div>
        <div class="kv"><span class="k">YouTube</span><span class="v" id="d-yt"></span></div>
      </div>
      <div class="card">
        <button class="copy" id="d-copy">copy</button>
        <div class="lbl">Description</div>
        <div class="desc" id="d-desc"></div>
      </div>
      <div class="card">
        <div class="lbl">Tags</div>
        <div class="chips" id="d-tags"></div>
      </div>
      <div class="card">
        <div class="lbl" id="d-rel-lbl">Related in this playlist</div>
        <div class="rel" id="d-rel"></div>
      </div>
    </div>
    <div id="empty" class="empty">select a playlist and a video</div>
  </main>
</div>
<script>
let VIDS = __VIDS__;
const slugify = (x) => x.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
const vslug = (v) => v.name.slice(0, -4);
function buildPls(vids){
  const pls = [];
  for (const v of vids) {
    let pl = pls.find((p) => p.name === (v.playlist || 'Learn Python V2'));
    if (!pl) { pl = {name: v.playlist || 'Learn Python V2', vids: []}; pls.push(pl); }
    pl.vids.push(v);
  }
  return pls;
}
let PLS = buildPls(VIDS);
const chip = (txt, cls) => '<span class="chip ' + (cls || '') + '">' + txt + '</span>';
function chipsFor(v, big){
  const c = [];
  if (v.sync && v.sync.checked) c.push(chip(v.sync.ok ? 'sync OK' : 'sync FAIL', v.sync.ok ? 'ok' : 'fail'));
  c.push(chip(v.upload ? 'UPLOADED' : 'draft', v.upload ? 'up' : ''));
  if (v.lang) c.push(chip(v.lang));
  if (big) {
    c.push(chip(v.captions + ' captions'));
    c.push(chip(v.outputs + ' outputs'));
    c.push(chip(v.mb + ' MB'));
    if (v.playlist) c.push(chip('playlist: ' + v.playlist));
    if (v.upload) c.push('<a class="studio" href="' + v.upload + '">open in Studio &#8599;</a>');
  }
  return c.join('');
}
// URL scheme: /playlist/<playlist-id>/<video-id>  (legacy /<pl>/<vid> still parses)
function plUrl(pl, v){ return '/playlist/' + slugify(pl.name) + (v ? '/' + vslug(v) : ''); }
function fmtDur(secs){
  secs = Math.round(secs || 0);
  const h = Math.floor(secs / 3600), m = Math.floor((secs % 3600) / 60), s = secs % 60;
  const p = [];
  if (h) p.push(h + 'h');
  if (m) p.push(m + 'm');
  if (s || !p.length) p.push(s + 's');
  return p.join(' ');
}
function parseUrl(){
  let parts = location.pathname.split('/').filter(Boolean).map(decodeURIComponent);
  if (parts[0] === 'playlist') parts = parts.slice(1);   // strip the /playlist prefix
  const pl = PLS.find((p) => slugify(p.name) === parts[0]) || PLS[0];
  const v = pl ? (pl.vids.find((x) => vslug(x) === parts[1]) || pl.vids[0]) : null;
  return {pl, v};
}
function go(pl, v){
  history.pushState({}, '', plUrl(pl, v));
  render();
}
function render(){
  const {pl, v} = parseUrl();
  const _dla = document.getElementById('dlall');
  if (_dla && pl) { _dla.href = '/download-all?playlist=' + encodeURIComponent(pl.name);
    _dla.textContent = '\u2681 ' + ((pl.vids&&pl.vids.length)||0) + ' videos';
    _dla.style.display = (pl.vids && pl.vids.length) ? '' : 'none'; }
  const pls = document.getElementById('pls'); pls.innerHTML = '';
  for (const p of PLS) {
    const a = document.createElement('a');
    a.className = 'pl' + (pl === p ? ' active' : ''); a.href = plUrl(p, p.vids[0]);
    const secs = p.vids.reduce((t, x) => t + (x.dur || 0), 0);
    const built = p.vids.reduce((t, x) => t + (x.build || 0), 0);
    a.innerHTML = '<div class="nm">' + p.name + '</div><div class="info">' + p.vids.length +
      ' videos &middot; ' + fmtDur(secs) + ' total &middot; ' + fmtDur(built) + ' to build &middot; ' +
      p.vids.filter((x) => x.upload).length + ' uploaded</div>';
    a.onclick = (e) => { e.preventDefault(); go(p, p.vids[0]); };
    pls.appendChild(a);
  }
  const links = document.getElementById('links'); links.innerHTML = '';
  if (pl) for (const x of pl.vids) {
    const a = document.createElement('a');
    a.className = 'item' + (v === x ? ' active' : ''); a.href = plUrl(pl, x);
    const img = document.createElement('img');
    img.className = 'thumb'; img.src = '/' + vslug(x) + '.png';
    img.onerror = () => { img.style.display = 'none'; };
    const grow = document.createElement('div'); grow.className = 'grow';
    grow.innerHTML = '<div class="iname">' + x.name + '</div><div class="chips">' + chipsFor(x, false) + '</div>';
    const row = document.createElement('div'); row.className = 'row';
    row.appendChild(img); row.appendChild(grow); a.appendChild(row);
    a.onclick = (e) => { e.preventDefault(); go(pl, x); };
    links.appendChild(a);
  }
  const stage = document.getElementById('stage'), empty = document.getElementById('empty'),
        details = document.getElementById('details'), vid = document.getElementById('vid');
  if (!v) { stage.style.display = 'none'; details.style.display = 'none'; empty.style.display = ''; return; }
  empty.style.display = 'none'; stage.style.display = ''; details.style.display = '';
  document.getElementById('vtitle').textContent = v.name + '  \u2014  ' + pl.name;
  if (!vid.src.endsWith('/' + encodeURIComponent(v.name))) vid.src = '/' + v.name;
  document.getElementById('dl').href = '/download/' + encodeURIComponent(v.name);
  vid.poster = '/' + vslug(v) + '.png';  // same title-card frame shown in the list, before play
  document.getElementById('meta').innerHTML = chipsFor(v, true);
  // ---- details panel: id / title / description / tags / related videos ----
  document.getElementById('d-id').textContent = v.id || vslug(v);
  document.getElementById('d-title').textContent = v.yt_title || v.title || v.name;
  const plu = (pl.vids.find((x) => x.pl_url) || {}).pl_url;
  const dpl = document.getElementById('d-pl');
  dpl.innerHTML = pl.name + (plu ? '  <a href="' + plu + '" target="_blank" rel="noopener">open playlist \u2197</a>' : '')
    + '<div class="lnkrow"><input id="pl-in" class="lnk" placeholder="paste playlist URL" value="'
    + (plu || '').replace(/"/g, '&quot;') + '"><button class="copy" id="pl-save">save</button></div>';
  document.getElementById('pl-save').onclick = async () => {
    const url = document.getElementById('pl-in').value.trim();
    await fetch('/api/setlink', {method: 'POST', body: JSON.stringify({playlist: pl.name, url})});
    location.reload();
  };
  document.getElementById('d-ver').textContent = 'v' + (v.version || 1);
  const yt = document.getElementById('d-yt');
  yt.innerHTML = (v.upload ? '<a href="' + v.upload + '" target="_blank" rel="noopener">open on YouTube \u2197</a>'
    : '<span class="muted">not uploaded yet</span>')
    + '<div class="lnkrow"><input id="yt-in" class="lnk" placeholder="paste YouTube URL" value="'
    + (v.upload || '').replace(/"/g, '&quot;') + '"><button class="copy" id="yt-save">save</button></div>';
  document.getElementById('yt-save').onclick = async () => {
    const url = document.getElementById('yt-in').value.trim();
    await fetch('/api/setlink', {method: 'POST', body: JSON.stringify({video: v.name, url})});
    location.reload();
  };
  const desc = v.description || '(no description yet)';
  document.getElementById('d-desc').textContent = desc;
  document.getElementById('d-copy').onclick = () => navigator.clipboard &&
    navigator.clipboard.writeText((v.yt_title ? v.yt_title + String.fromCharCode(10,10) : '') + desc);
  const tw = document.getElementById('d-tags');
  tw.innerHTML = (v.tags && v.tags.length) ? v.tags.map((t) => chip('#' + t)).join('') : chip('no tags');
  const rel = document.getElementById('d-rel');
  const others = pl.vids;
  document.getElementById('d-rel-lbl').textContent = 'Related in ' + pl.name + ' (' + others.length + ')';
  rel.innerHTML = '';
  for (const x of others) {
    const a = document.createElement('a');
    a.href = plUrl(pl, x); a.className = (x === v ? 'cur' : '');
    a.innerHTML = '<img src="/' + vslug(x) + '.png" onerror="this.style.visibility=\\'hidden\\'">' +
      '<div class="rt mono">' + x.name + '</div>';
    a.onclick = (e) => { e.preventDefault(); go(pl, x); };
    rel.appendChild(a);
  }
}
window.addEventListener('popstate', render);
if (location.pathname === '/' && PLS.length)
  history.replaceState({}, '', plUrl(PLS[0], PLS[0].vids[0]));
render();
// live auto-reload: poll the gallery JSON; re-render only when it actually changed, and never
// while a video is playing (so a new build landing mid-watch doesn't interrupt playback).
let _sig = JSON.stringify(VIDS);
async function refresh(){
  try {
    const r = await fetch('/api/vids', {cache: 'no-store'});
    if (!r.ok) return;
    const next = await r.json();
    const sig = JSON.stringify(next);
    if (sig === _sig) return;                         // nothing changed
    const vid = document.getElementById('vid');
    const playing = vid && !vid.paused && !vid.ended && vid.currentTime > 0;
    if (playing) return;                              // defer until playback stops/pauses
    _sig = sig; VIDS = next; PLS = buildPls(VIDS);
    render();
  } catch (e) { /* transient fetch error — try again next tick */ }
}
setInterval(refresh, 4000);
</script></body></html>"""


def _pg_rows():
    """PG is the source of truth; {} if it is down (sidecars carry on)."""
    try:
        return {r["name"]: r for r in list_videos()}
    except Exception:  # noqa: BLE001
        return {}


def _vid_entry(mp4: Path, pg: dict) -> dict:
    e = {"name": mp4.name, "id": mp4.stem, "mb": round(mp4.stat().st_size / 1e6, 1), "sync": None,
         "lang": None, "captions": 0, "dur": 0, "upload": None, "build": None, "outputs": 0,
         "yt_title": None, "description": None, "tags": [], "version": 1, "pl_url": None}
    row = pg.get(mp4.name)
    if row:                                                             # PG row (authoritative)
        e["playlist"] = row.get("playlist") or "Learn Python V2"
        e["title"] = row.get("title")
        e.update(lang=row.get("language"), captions=row.get("captions") or 0,
                 dur=round(row.get("duration_s") or 0), sync=row.get("sync"),
                 build=row.get("build_seconds"), outputs=row.get("n_outputs") or 0,
                 upload=(row.get("upload") or {}).get("url") if row.get("upload") else None,
                 version=int(row.get("version") or 1))
        _rup = row.get("upload") or {}
        if _rup.get("playlist_id"):
            e["pl_url"] = "https://www.youtube.com/playlist?list=" + _rup["playlist_id"]
    else:
        side = mp4.with_suffix(".json")                                 # sidecar fallback
        if side.exists():
            try:
                m = json.loads(side.read_text())
                e["sync"] = m.get("sync")
                e["playlist"] = (m.get("props") or {}).get("playlist") or m.get("playlist") or "Learn Python V2"
                e["build"] = m.get("build_seconds")
                e["outputs"] = m.get("n_outputs") or 0
                p = m.get("props") or {}
                segs = p.get("segments") or []
                e["lang"] = p.get("language")
                e["captions"] = len(segs)
                e["dur"] = round(max((s.get("end", 0) for s in segs), default=0) + p.get("tailSeconds", 0))
                up = m.get("upload")
                e["upload"] = up.get("url") if up else None
            except Exception:  # noqa: BLE001
                pass
    # YouTube publish metadata (id/title/description/tags) so the hub shows exactly what will be
    # uploaded — from the stored upload spec if present, else the same build_metadata() the uploader
    # uses, computed from the sidecar props. Best-effort; never breaks the listing.
    try:
        side = mp4.with_suffix(".json")
        m = json.loads(side.read_text()) if side.exists() else {}
        up = m.get("upload") or {}
        props = m.get("props")
        meta = build_metadata(props, hook=props.get("hook")) if props else {}
        e["yt_title"] = up.get("title") or meta.get("title") or e.get("title")
        e["description"] = up.get("description") or meta.get("description")
        e["tags"] = up.get("tags") or meta.get("tags") or []
        if up.get("id"):
            e["id"] = up["id"]                                          # real YouTube video id once uploaded
        e["version"] = int(m.get("version") or up.get("version") or e.get("version") or 1)  # bumped each re-render
        if up.get("playlist_id"):                                      # link to the whole playlist on YouTube
            e["pl_url"] = "https://www.youtube.com/playlist?list=" + up["playlist_id"]
        e["upload"] = up.get("url") or e.get("upload")                 # keep PG/sidecar url, don't blank it
    except Exception:  # noqa: BLE001
        pass
    return e


class Handler(SimpleHTTPRequestHandler):
    """Adds HTTP Range support (RFC 7233) on top of SimpleHTTPRequestHandler — browsers demand
    ranges to play/seek mp4; the stdlib handler alone leaves the <video> element dead."""

    def __init__(self, *a, **k):
        super().__init__(*a, directory=str(GALLERY), **k)

    def end_headers(self):  # inject CORS on every response so :9090 never trips a CORS error
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Max-Age", "86400")
        super().end_headers()

    def do_OPTIONS(self):  # noqa: N802 — CORS preflight
        self.send_response(204)
        self.end_headers()

    def _serve_range(self, path: Path):
        rng = self.headers.get("Range", "")
        size = path.stat().st_size
        try:
            lo, hi = rng.split("=")[1].split("-")
            lo = int(lo or 0); hi = min(int(hi) if hi else size - 1, size - 1)
        except Exception:  # noqa: BLE001
            lo, hi = 0, size - 1
        self.send_response(206)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Range", f"bytes {lo}-{hi}/{size}")
        self.send_header("Content-Length", str(hi - lo + 1))
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(lo)
            left = hi - lo + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def do_POST(self):  # noqa: N802
        from urllib.parse import urlparse
        clean = urlparse(self.path).path
        if clean == "/api/voiceref":
            from urllib.parse import parse_qs, urlparse as _up
            q = parse_qs(_up(self.path).query)
            try:
                import sys as _sys; _sys.path.insert(0, str(YT_ROOT / "tools"))
                import recorder_ui
                if q.get("install"):
                    out = recorder_ui.install_voice_ref()
                else:
                    n = int(self.headers.get("Content-Length") or 0)
                    out = recorder_ui.save_voice_ref(self.rfile.read(n),
                                                     q.get("ref_text", [""])[0])
                body = json.dumps(out).encode(); self.send_response(200)
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"ok": False, "notes": [], "problems": [str(e)[:200]]}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if clean == "/api/narration":
            from urllib.parse import parse_qs, urlparse as _up
            q = parse_qs(_up(self.path).query)
            try:
                import sys as _sys; _sys.path.insert(0, str(YT_ROOT / "tools"))
                from recorder_ui import save_clip
                n = int(self.headers.get("Content-Length") or 0)
                out = save_clip(q.get("playlist", [""])[0], q.get("video", [""])[0],
                                q.get("index", ["0"])[0], self.rfile.read(n))
                body = json.dumps(out).encode(); self.send_response(200)
            except Exception as e:  # noqa: BLE001
                body = json.dumps({"ok": False, "error": str(e)[:200]}).encode()
                self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body); return
        if clean != "/api/setlink":
            self.send_response(404); self.end_headers(); return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            data = json.loads(self.rfile.read(n) or b"{}")
            if data.get("playlist") is not None and data.get("video") is None:
                out = set_playlist_link(data["playlist"], data.get("url", ""))
            else:
                out = {"upload": set_video_link(data["video"], data.get("url", ""))}
            body = json.dumps({"ok": True, **out}).encode()
            self.send_response(200)
        except Exception as e:  # noqa: BLE001
            body = json.dumps({"ok": False, "error": str(e)[:200]}).encode()
            self.send_response(400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        _c = self.path.split("?")[0]
        if _c == "/me":
            try:
                import sys as _sys; _sys.path.insert(0, str(YT_ROOT / "tools"))
                import recorder_ui
                from urllib.parse import parse_qs, urlparse as _up2
                _m = parse_qs(_up2(self.path).query).get("mode", ["quick"])[0]
                b = recorder_ui.me_html(_m).encode(); code = 200
            except Exception as e:  # noqa: BLE001
                b = f"<pre>recorder error: {e}</pre>".encode(); code = 500
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b); return
        if _c == "/record" or _c.startswith("/record/"):
            try:
                import sys as _sys; _sys.path.insert(0, str(YT_ROOT / "tools"))
                import recorder_ui
                parts = [x for x in _c.split("/") if x]
                html = (recorder_ui.booth_html(parts[1], parts[2]) if len(parts) >= 3
                        else recorder_ui.index_html())
                b = html.encode(); code = 200
            except Exception as e:  # noqa: BLE001
                b = f"<pre>recorder error: {e}</pre>".encode(); code = 500
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b); return
        if _c in ("/favicon.svg", "/favicon.ico"):
            b = _FAVICON.encode(); self.send_response(200)
            self.send_header("Content-Type", "image/svg+xml"); self.send_header("Cache-Control", "max-age=86400")
            self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); return
        if _c.startswith("/download/"):
            p = GALLERY / Path(_c).name
            if p.exists():
                data = p.read_bytes(); self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Disposition", 'attachment; filename="' + p.name + '"')
                self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            self.send_response(404); self.end_headers(); return
        if _c.startswith("/download-all"):
            from urllib.parse import parse_qs, urlparse
            import io, zipfile
            pl = parse_qs(urlparse(self.path).query).get("playlist", [""])[0]
            rows = _pg_rows()
            finals = {p.stem for p in GALLERY.glob("*.mp4") if not p.stem.endswith("_silent")}
            vids = [p for p in GALLERY.glob("*.mp4")
                    if not (p.stem.endswith("_silent") and p.stem[:-len("_silent")] in finals)]
            sel = [v for v in vids if (not pl) or (_vid_entry(v, rows).get("playlist") == pl)]
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
                for v in sel: z.write(v, arcname=v.name)
            data = buf.getvalue(); fn = (pl or "all").replace(" ", "_") + "_shorts.zip"
            self.send_response(200); self.send_header("Content-Type", "application/zip")
            self.send_header("Content-Disposition", 'attachment; filename="' + fn + '"')
            self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
        if self.path.endswith(".mp4") and "Range" in self.headers:
            p = GALLERY / Path(self.path.split("?")[0]).name       # flat dir; basename kills traversal
            if p.exists():
                try:
                    self._serve_range(p)
                except (BrokenPipeError, ConnectionResetError):
                    pass                                            # player seeked away mid-stream
                return
        clean = self.path.split("?")[0]
        if clean == "/api/vids":                                       # live gallery JSON for auto-reload
            finals = {p.stem for p in GALLERY.glob("*.mp4") if not p.stem.endswith("_silent")}
            vids = sorted((p for p in GALLERY.glob("*.mp4")
                           if not (p.stem.endswith("_silent") and p.stem[:-len("_silent")] in finals)),
                          key=lambda p: p.stat().st_mtime, reverse=True)
            body = json.dumps([_vid_entry(v, _pg_rows()) for v in vids], default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if clean in ("/", "/index.html") or "." not in Path(clean).name:   # path-routed video pages
            finals = {p.stem for p in GALLERY.glob("*.mp4") if not p.stem.endswith("_silent")}
            vids = sorted((p for p in GALLERY.glob("*.mp4")
                           if not (p.stem.endswith("_silent") and p.stem[:-len("_silent")] in finals)),
                          key=lambda p: p.stat().st_mtime, reverse=True)  # skip _silent when its final twin exists
            pg = _pg_rows()
            try:
                import sys as _sys; _sys.path.insert(0, str(YT_ROOT / "tools"))
                import recorder_ui as _ru
                _vnav, _vcss = _ru.hub_header_html(), _ru.HUB_CSS
            except Exception:  # noqa: BLE001
                _vnav, _vcss = "", ""
            page = (_PAGE.replace("__VOICE__", _vnav).replace("__VOICECSS__", _vcss)
                         .replace("__N__", str(len(vids)))
                    .replace("__VIDS__", json.dumps([_vid_entry(v, pg) for v in vids], default=str)))
            body = page.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")                # never serve a stale UI again
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def log_message(self, *a):  # quiet
        pass


HTTPS_PORT = 9443


def hub_main():
    GALLERY.mkdir(exist_ok=True)
    # HTTPS alongside HTTP: browsers only expose the microphone on a secure origin, so the
    # narration recorder is unusable over the LAN on plain http. Self-signed is fine — once the
    # exception is accepted the origin counts as secure, which is all getUserMedia asks for.
    crt = YT_ROOT / "common" / "certs" / "hub.crt"
    key = YT_ROOT / "common" / "certs" / "hub.key"
    if crt.exists() and key.exists():
        import ssl
        import threading

        def _tls():
            try:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ctx.load_cert_chain(str(crt), str(key))
                srv = ThreadingHTTPServer(("0.0.0.0", HTTPS_PORT), Handler)
                srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
                print(f"shorts-hub: https on :{HTTPS_PORT} (mic-capable)", flush=True)
                srv.serve_forever()
            except Exception as e:  # noqa: BLE001  (TLS must never take down the plain hub)
                print(f"shorts-hub: https unavailable ({e})", flush=True)

        threading.Thread(target=_tls, daemon=True).start()
    else:
        print(f"shorts-hub: no cert at {crt} — run tools/make_cert.py for mic support", flush=True)
    # threaded: one playing video stream must never block the index (single-thread wedge bug)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()



if __name__ == "__main__":                                    # python -m fleet_agents.shorts_builder --hub
    hub_main()
