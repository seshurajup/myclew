"""video_builder — frames → video assembler, the Python-native answer to remotion (which is a React/TS
programmatic-video framework needing Node + a Chromium render farm — a heavy foreign runtime to bolt onto an
ML fleet). For a fleet the useful, dependency-light primitive is: take a stack of numpy RGB frames (e.g.
cell-tracking overlays, a training-curve animation, an attention-map rollout) and write an animated GIF (always
available via imageio) or an MP4 (when ffmpeg is present, import-guarded). Deterministic, offline, no Node.

Primitives (deps = imageio; numpy):
  • write_gif(frames, path, fps)          — animated GIF from (T,H,W,3) uint8 frames (always works).
  • write_mp4(frames, path, fps)          — MP4 via imageio-ffmpeg (guarded; raises if ffmpeg absent).
  • frames_from_arrays(list_of_HxWx3)     — normalize/stack a list of arrays to a (T,H,W,3) uint8 clip.
  • overlay_points(frame, pts, radius)    — draw tracked points on a frame (the cell-tracking use case).
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


def frames_from_arrays(arrays):
    """Stack a list of HxW (grayscale) or HxWx3 arrays into a (T,H,W,3) uint8 clip, min-max scaling floats."""
    out = []
    for a in arrays:
        a = np.asarray(a)
        if a.ndim == 2:
            a = np.stack([a] * 3, axis=-1)
        if a.dtype != np.uint8:
            a = a.astype(float)
            lo, hi = a.min(), a.max()
            a = (255 * (a - lo) / (hi - lo)) if hi > lo else np.zeros_like(a)
            a = a.astype(np.uint8)
        out.append(a)
    return np.stack(out, axis=0)


def overlay_points(frame, pts, radius=2, color=(255, 0, 0)):
    """Draw square markers of `radius` at each (y,x) in pts on an HxWx3 uint8 frame (returns a copy)."""
    f = np.asarray(frame).copy()
    H, W = f.shape[:2]
    for y, x in pts:
        y, x = int(round(y)), int(round(x))
        y0, y1 = max(0, y - radius), min(H, y + radius + 1)
        x0, x1 = max(0, x - radius), min(W, x + radius + 1)
        f[y0:y1, x0:x1] = color
    return f


def write_gif(frames, path, fps=8):
    """Write (T,H,W,3) uint8 frames to an animated GIF. Returns {"path","n_frames"}. Always available."""
    import imageio
    frames = np.asarray(frames)
    imageio.mimsave(str(path), list(frames), duration=1.0 / max(fps, 1), loop=0)
    return {"path": str(path), "n_frames": int(frames.shape[0])}


def write_wav(samples, path, sample_rate=16000):
    """Write a mono waveform (numpy float [-1,1] or int16) to a WAV file — stdlib `wave`, no deps. Lets the
    fleet's audio agents (audio_pack/audio_infer produce/consume waveforms) feed video_builder's muxer."""
    import wave, struct
    s = np.asarray(samples).ravel()
    if s.dtype.kind == "f":
        s = np.clip(s, -1, 1); s = (s * 32767).astype(np.int16)
    else:
        s = s.astype(np.int16)
    with wave.open(str(path), "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(int(sample_rate))
        w.writeframes(s.tobytes())
    return {"path": str(path), "n_samples": int(s.shape[0]), "sample_rate": int(sample_rate)}


def mux_audio_video(video_path, audio_path, out_path):
    """Attach an audio track to a video (frames→GIF/MP4 from write_mp4/write_gif) via ffmpeg, producing a
    video-with-audio MP4. Requires ffmpeg on PATH (import-guarded). This is the 'video WITH audio attached'
    step: render frames with video_builder, synth/collect a waveform with the audio agents + write_wav, mux."""
    import shutil, subprocess
    ff = shutil.which("ffmpeg")
    if not ff:
        raise RuntimeError("mux_audio_video needs ffmpeg on PATH (install ffmpeg / imageio-ffmpeg)")
    cmd = [ff, "-y", "-i", str(video_path), "-i", str(audio_path),
           "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {r.stderr[-300:]}")
    return {"path": str(out_path)}


def write_mp4(frames, path, fps=12):
    """Write frames to an MP4 (needs imageio-ffmpeg / ffmpeg). Raises RuntimeError if unavailable."""
    try:
        import imageio
        frames = np.ascontiguousarray(np.asarray(frames, dtype=np.uint8))
        w = imageio.get_writer(str(path), fps=fps, codec="libx264")
        for fr in frames:
            w.append_data(fr)
        w.close()
    except Exception as e:  # noqa: BLE001  (ffmpeg/imageio-ffmpeg absent, or a plugin write error)
        raise RuntimeError(f"mp4 unavailable (needs ffmpeg/imageio-ffmpeg): {e}")
    return {"path": str(path), "n_frames": int(frames.shape[0])}


# ---------------------------------------------------------------- agent
class VideoBuilder(BaseAgent):
    name = "video-builder"
    thread = "S"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        try:
            import imageio  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return self.escalate(q, "leader", f"video-builder needs imageio: {e}")
        from .base import AUTO
        AUTO.mkdir(parents=True, exist_ok=True)
        rng = np.random.RandomState(int(s.get("seed", 0)))
        T = int(s.get("n_frames", 12)); H = W = int(s.get("size", 64))
        # synthetic tracking clip: a moving point on a drifting-noise background
        frames = []
        for t in range(T):
            bg = (rng.rand(H, W) * 60).astype(np.uint8)
            fr = np.stack([bg] * 3, axis=-1)
            y = int(H * (0.2 + 0.6 * t / max(T - 1, 1))); x = int(W * (0.5 + 0.3 * np.sin(t / 2)))
            fr = overlay_points(fr, [(y, x)], radius=3)
            frames.append(fr)
        clip = np.stack(frames, axis=0)
        out = s.get("out") or str(AUTO / "video_builder_demo.gif")
        r = write_gif(clip, out, fps=int(s.get("fps", 8)))
        mp4 = None
        try:
            mp4 = write_mp4(clip, str(AUTO / "video_builder_demo.mp4"), fps=12)["path"]
        except RuntimeError:
            mp4 = None                                          # ffmpeg absent → GIF only (expected here)
        msg = (f"video-builder: wrote {r['n_frames']}-frame animation → {r['path']}"
               f"{' (+mp4)' if mp4 else ' (GIF only; ffmpeg absent)'}. Python-native frames→video "
               f"(overlay_points for tracking overlays); no Node/remotion needed")
        self.log(msg, kind="finding",
                 recommendation="render tracking/training animations with frames_from_arrays + write_gif; "
                                "install imageio-ffmpeg for mp4")
        return self.done({"path": r["path"], "n_frames": r["n_frames"], "mp4": mp4}, msg)


_AGENT = VideoBuilder()


def run_video(q, worker):
    return _AGENT.run(q, worker)
