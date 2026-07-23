"""tracker-train — train the pilkwang UNet detector + edge transformer.

Wraps ``research/pilkwang_support_pack/repo/scripts/train_unet_transformer.py`` VERBATIM
via subprocess (no training logic is reimplemented here). It builds a well-formed argv,
sets the PYTHONPATH the script needs (repo/src + repo/scripts + comp root), runs it, captures
stdout/stderr, parses the best score / checkpoint path, and logs the outcome to the ledger.

GPU-gated: refuses to run while ``config/_auto/gpu_train_hold.flag`` exists (5090 power-cap
human gate), escalating instead — matching combined-train.

Spec: {data_dir, splits, split, epochs, lr, batch_size, num_workers, unet_weights (warm-start),
       method, max_iters, debug_video, out_dir, timeout}.
A BaseAgent subclass with its own data-wise test (test_fleet_agents/tracker_train_test.py).
"""
from __future__ import annotations
import os
import re
import subprocess
import sys
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
_REPO = COMP / "research" / "pilkwang_support_pack" / "repo"
_SCRIPT = _REPO / "scripts" / "train_unet_transformer.py"
_WEIGHTS_DEFAULT = _REPO / "weights"


def _env(spec=None):
    """`device`: optional device hint; 'cpu' forces a CUDA-free run. `seed`: exported as TRACKER_SEED."""
    env = dict(os.environ)
    pp = os.pathsep.join([str(_REPO / "src"), str(_REPO / "scripts"), str(COMP)])
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    spec = spec or {}
    if str(spec.get("device", "") or "").lower() in ("cpu", "-1"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    if spec.get("seed") is not None:
        env["TRACKER_SEED"] = str(spec["seed"])
    return env


def _py():
    v = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(v) if v.exists() else sys.executable


class TrackerTrain(BaseAgent):
    name = "tracker-train"
    thread = "B"
    kind = "verdict"

    def build_argv(self, spec) -> list[str]:
        method = spec.get("method", "unet_transformer")
        argv = [_py(), str(_SCRIPT), "--method", method, "--split", str(spec.get("split", "0")),
                "--epochs", str(int(spec.get("epochs", 50))), "--lr", str(spec.get("lr", 1e-4)),
                "--batch-size", str(int(spec.get("batch_size", 16)))]
        if spec.get("data_dir"):
            argv += ["--data-dir", str(spec["data_dir"])]
        if spec.get("splits"):
            argv += ["--splits", str(spec["splits"])]
        if spec.get("num_workers") is not None:
            argv += ["--num-workers", str(int(spec["num_workers"]))]
        if spec.get("unet_weights"):
            argv += ["--unet-weights", str(spec["unet_weights"])]
        if spec.get("max_iters") is not None:
            argv += ["--max-iters", str(int(spec["max_iters"]))]
        if spec.get("debug_video"):
            argv += ["--debug-video", str(spec["debug_video"])]
        if spec.get("single_gpu"):        # force single-GPU (skip DataParallel)
            argv += ["--single-gpu"]
        if spec.get("data_parallel"):     # opt-in multi-GPU DataParallel
            argv += ["--data-parallel"]
        return argv

    def run(self, q, worker):
        from .base import gpu_train_held
        if gpu_train_held():
            return self.escalate(worker, "leader",
                                 f"[{worker}] tracker-train HELD — GPU training parked (5090 power-cap gate). "
                                 f"Remove config/_auto/gpu_train_hold.flag (human GO) before training.")
        spec = self.spec(q)
        if not _SCRIPT.exists():
            return self.escalate(worker, "researcher", f"[{worker}] tracker-train: script missing at {_SCRIPT}")
        argv = self.build_argv(spec)
        method = spec.get("method", "unet_transformer")
        split = str(spec.get("split", "0"))
        out_dir = spec.get("out_dir") or str(_WEIGHTS_DEFAULT / method / f"split_{split}")
        try:
            timeout = int(spec.get("timeout", 60 * 60 * 11))
        except (TypeError, ValueError):
            timeout = 60 * 60 * 11
        env = _env(spec)
        # FULL-MODEL continue-training: spec {"resume_checkpoint": "<path>" | "pilkwang"} → load whole model+optimizer
        rck = spec.get("resume_checkpoint")
        if rck:
            if str(rck) == "pilkwang":   # shortcut → the real shipped pilkwang checkpoint (support-pack, not repo/)
                rck = COMP / "research" / "pilkwang_support_pack" / "weights" / "unet_transformer" / "split_0" / "checkpoint_last.pth"
            env["TRACKER_RESUME_CKPT"] = str(rck)
        try:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=str(_REPO),
                               env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] tracker-train: timed out after {timeout}s.")
        except Exception as e:  # noqa: BLE001 — launch failure (bad python/OSError) → clean escalate
            return self.escalate(worker, "researcher", f"[{worker}] tracker-train: could not launch — {type(e).__name__}: {str(e)[:120]}")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()[-25:]
        best = None
        for m in re.finditer(r"best[^0-9]{0,20}(\d+\.\d+)", out, re.I):
            best = float(m.group(1))
        ckpt = Path(out_dir) / "edge_predictor_best.pth"
        ok = r.returncode == 0
        result = {"status": "ok" if ok else "failed", "returncode": r.returncode,
                  "best_score": best, "weights_dir": out_dir,
                  "checkpoint": str(ckpt) if ckpt.exists() else None,
                  "argv": " ".join(argv), "stdout_tail": "\n".join(tail)}
        if not ok:
            self.log(summary=f"tracker-train FAILED (rc={r.returncode})",
                     detail="\n".join(tail), kind="finding",
                     recommendation="inspect argv/data-dir; check PYTHONPATH + weights")
            return self.escalate(worker, "researcher",
                                 f"[{worker}] tracker-train FAILED rc={r.returncode}: {tail[-1] if tail else ''}")
        self.save_state({"best_score": best, "weights_dir": out_dir, "method": method, "split": split})
        self.log(summary=f"tracker-train: UNet+edge-transformer trained (method={method} split={split}) "
                         f"best={best} → {ckpt.name if ckpt.exists() else out_dir}",
                 detail=result["stdout_tail"], kind="verdict",
                 recommendation="run tracker-predict with these weights, then tracker-postproc → submission")
        msg = (f"[{worker}] **TRACKER-TRAIN** · UNet detector + edge transformer\n"
               f"method=`{method}` split={split} · best_score **{best}**\n"
               f"weights → `{out_dir}`")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(result, msg, to="leader")


_AGENT = TrackerTrain()


def run(q, worker):
    return _AGENT.run(q, worker)
