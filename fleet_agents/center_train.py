"""center-train — train the pilkwang full-frame 3D-UNet center-heatmap detector.

Wraps ``research/pilkwang_support_pack/source_scripts/train_full_frame_center_detector.py``
VERBATIM via subprocess (the positive-unlabelled center detector; supports ``--resume``).
No training logic is reimplemented here — this agent only builds a well-formed argv, sets the
PYTHONPATH, runs the script, captures output, parses the best score / checkpoint, and logs.

GPU-gated like the other trainers (refuses while gpu_train_hold.flag exists).

Spec: {data_dir, output_dir, epochs, learning_rate, batch_size, pool_factor, base_channels,
       movie_limit, frames_per_movie, resume, overwrite, cpu, timeout}.
A BaseAgent subclass with its own data-wise test (test_fleet_agents/center_train_test.py).
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
_SCRIPT = COMP / "research" / "pilkwang_support_pack" / "source_scripts" / "train_full_frame_center_detector.py"
_REPO = COMP / "research" / "pilkwang_support_pack" / "repo"
_OUT_DEFAULT = COMP / "research" / "pilkwang_deepcenter" / "weights" / "full_frame_center"


def _env():
    env = dict(os.environ)
    pp = os.pathsep.join([str(_REPO / "src"), str(COMP)])
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def _py():
    v = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(v) if v.exists() else sys.executable


class CenterTrain(BaseAgent):
    name = "center-train"
    thread = "B"
    kind = "verdict"

    def build_argv(self, spec) -> list[str]:
        data_dir = spec.get("data_dir") or str(COMP / "input" / "biohub-cell-tracking-during-development" / "train")
        out_dir = spec.get("output_dir") or str(_OUT_DEFAULT)
        argv = [_py(), str(_SCRIPT), "--data-dir", str(data_dir), "--output-dir", str(out_dir),
                "--epochs", str(int(spec.get("epochs", 50)))]
        if spec.get("learning_rate") is not None:
            argv += ["--learning-rate", str(spec["learning_rate"])]
        if spec.get("batch_size") is not None:
            argv += ["--batch-size", str(int(spec["batch_size"]))]
        if spec.get("pool_factor") is not None:
            argv += ["--pool-factor", str(int(spec["pool_factor"]))]
        if spec.get("base_channels") is not None:
            argv += ["--base-channels", str(int(spec["base_channels"]))]
        if spec.get("movie_limit") is not None:
            argv += ["--movie-limit", str(int(spec["movie_limit"]))]
        if spec.get("frames_per_movie") is not None:
            argv += ["--frames-per-movie", str(int(spec["frames_per_movie"]))]
        if spec.get("resume"):
            argv += ["--resume"]
        if spec.get("overwrite"):
            argv += ["--overwrite"]
        if spec.get("cpu") or spec.get("device") == "cpu":   # `device`="cpu" is an alias for the --cpu flag
            argv += ["--cpu"]
        return argv

    def run(self, q, worker):
        from .base import gpu_train_held
        if gpu_train_held():
            return self.escalate(worker, "leader",
                                 f"[{worker}] center-train HELD — GPU training parked (5090 power-cap gate). "
                                 f"Remove config/_auto/gpu_train_hold.flag (human GO) before training.")
        spec = self.spec(q)
        if not _SCRIPT.exists():
            return self.escalate(worker, "researcher", f"[{worker}] center-train: script missing at {_SCRIPT}")
        argv = self.build_argv(spec)
        out_dir = spec.get("output_dir") or str(_OUT_DEFAULT)
        timeout = int(spec.get("timeout", 60 * 60 * 11))
        try:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=str(COMP),
                               env=_env(), timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] center-train: timed out after {timeout}s.")
        except Exception as e:  # noqa: BLE001 — OSError / env failure → escalate cleanly
            return self.escalate(worker, "researcher", f"[{worker}] center-train: subprocess failed ({str(e)[:80]}).")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()[-25:]
        best = None
        for m in re.finditer(r"best[^0-9]{0,20}(\d+\.\d+)", out, re.I):
            best = float(m.group(1))
        ckpt = Path(out_dir) / "checkpoint_last.pt"
        ok = r.returncode == 0
        result = {"status": "ok" if ok else "failed", "returncode": r.returncode,
                  "best_score": best, "output_dir": out_dir,
                  "checkpoint": str(ckpt) if ckpt.exists() else None,
                  "resumed": bool(spec.get("resume")), "argv": " ".join(argv),
                  "stdout_tail": "\n".join(tail)}
        if not ok:
            self.log(summary=f"center-train FAILED (rc={r.returncode})", detail="\n".join(tail),
                     kind="finding", recommendation="check --data-dir / --output-dir (empty dir needs --overwrite or --resume)")
            return self.escalate(worker, "researcher",
                                 f"[{worker}] center-train FAILED rc={r.returncode}: {tail[-1] if tail else ''}")
        self.save_state({"best_score": best, "output_dir": out_dir})
        self.log(summary=f"center-train: full-frame center detector trained best={best} → {out_dir}",
                 detail=result["stdout_tail"], kind="verdict",
                 recommendation="use as center-prior fusion in tracker-postproc (fuse_full_frame_nodes)")
        msg = (f"[{worker}] **CENTER-TRAIN** · full-frame 3D-UNet center detector"
               f"{' (resumed)' if spec.get('resume') else ''}\n"
               f"best_score **{best}** · checkpoint → `{out_dir}`")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(result, msg, to="leader")


_AGENT = CenterTrain()


def run(q, worker):
    return _AGENT.run(q, worker)
