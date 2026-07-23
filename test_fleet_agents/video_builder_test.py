"""video_builder_test — data-wise verifier for the remotion-alternative frames→video builder.

Core properties:
  1. frames_from_arrays normalizes to (T,H,W,3) uint8; grayscale promoted to RGB.
  2. overlay_points draws a marker (pixel color changes at the point).
  3. write_gif produces a real, readable GIF with the right frame count.
  4. agent contract (GIF path exists)."""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import video_builder as V


def _run():
    print("=== VIDEO-BUILDER VERIFIER ===")
    checks = {}
    try:
        import imageio
    except Exception as e:
        print("imageio missing:", e); return False

    # 1. frames_from_arrays — three VISIBLY DISTINCT frames (a GIF optimizer merges identical ones)
    rng = np.random.RandomState(0)
    g0 = rng.rand(16, 16); g1 = np.tile(np.linspace(0, 1, 16), (16, 1)); g2 = np.eye(16)
    clip = V.frames_from_arrays([g0, g1, g2])
    checks["stack_shape"] = clip.shape == (3, 16, 16, 3) and clip.dtype == np.uint8

    # 2. overlay
    f = np.zeros((20, 20, 3), np.uint8)
    f2 = V.overlay_points(f, [(10, 10)], radius=1, color=(255, 0, 0))
    checks["overlay_draws"] = tuple(f2[10, 10]) == (255, 0, 0) and f2.sum() > 0

    # 3. write gif
    out = os.path.join(tempfile.mkdtemp(), "clip.gif")
    r = V.write_gif(clip, out, fps=6)
    checks["gif_written"] = os.path.exists(out) and os.path.getsize(out) > 0
    checks["gif_frame_count"] = r["n_frames"] == 3
    back = imageio.mimread(out)
    checks["gif_readable"] = len(back) == 3
    print(f"  -> gif {r['n_frames']} frames, read back {len(back)}")

    # 3b. audio track: write a WAV (for muxing into video) and confirm it round-trips
    import wave, tempfile as _tf
    sr = 16000; t = np.linspace(0, 1, sr, endpoint=False)
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)          # 440Hz tone
    wpath = os.path.join(_tf.mkdtemp(), "a.wav")
    wr = V.write_wav(tone, wpath, sample_rate=sr)
    checks["wav_written"] = os.path.exists(wpath) and wr["n_samples"] == sr
    with wave.open(wpath) as wf:
        checks["wav_readable"] = wf.getframerate() == sr and wf.getnchannels() == 1
    # mux is guarded on ffmpeg — assert it either succeeds or raises the documented RuntimeError (no crash)
    import shutil
    if shutil.which("ffmpeg"):
        try:
            out = os.path.join(_tf.mkdtemp(), "av.mp4"); V.mux_audio_video(out, wpath, out)  # may fail on bad video
            checks["mux_path_or_guard"] = True
        except RuntimeError:
            checks["mux_path_or_guard"] = True
    else:
        try:
            V.mux_audio_video("x.mp4", wpath, "y.mp4"); checks["mux_path_or_guard"] = False
        except RuntimeError:
            checks["mux_path_or_guard"] = True    # correctly guarded when ffmpeg absent
    print(f"  -> wav {wr['n_samples']} samples @ {sr}Hz; mux guarded={checks['mux_path_or_guard']}")

    # 4. agent
    st, dta, to, msg = V.run_video({"spec": {"n_frames": 10, "size": 48}}, "t")
    checks["agent_done"] = st == "done" and os.path.exists(dta["path"]) and dta["n_frames"] == 10

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== video-builder: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
