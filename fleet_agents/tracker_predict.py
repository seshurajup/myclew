"""tracker-predict — run pilkwang UNet+transformer detection + edge prediction + ILP linking
to produce per-dataset ``.geff`` predictions.

Wraps ``research/pilkwang_support_pack/repo/scripts/predict_unet_transformer.py`` VERBATIM
through the thin ``research/tracker_pipeline/predict_runner.py`` (which reuses the script's own
load_model / predict_video / build_graph / ILP / save_graph and only adds an optional
``--max-frames`` cap for cheap smoke runs). No inference logic is reimplemented here.

Predictions are written to the script's own layout:
``research/pilkwang_support_pack/repo/predictions/$USER/<method>/split_<split>/*.geff``.

Spec: {data_dir, splits, split, weights, det_threshold, use_ilp, method, debug_video,
       max_frames, unet_batch_size, out_dir (informational), timeout}.
A BaseAgent subclass with its own data-wise test (test_fleet_agents/tracker_predict_test.py)
that MUST actually run inference on one dataset / few frames and confirm nodes>0.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
_REPO = COMP / "research" / "pilkwang_support_pack" / "repo"
_RUNNER = COMP / "research" / "tracker_pipeline" / "predict_runner.py"
_WEIGHTS_DEFAULT = COMP / "research" / "pilkwang_support_pack" / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
_DATA_DEFAULT = COMP / "input" / "biohub-cell-tracking-during-development" / "train"


def _env(spec=None):
    """`device`: optional device hint; 'cpu' forces a CUDA-free inference run (CPU fallback)."""
    env = dict(os.environ)
    pp = os.pathsep.join([str(_REPO / "src"), str(_REPO / "scripts"), str(COMP)])
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if str((spec or {}).get("device", "") or "").lower() in ("cpu", "-1"):
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _py():
    v = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(v) if v.exists() else sys.executable


def _pred_dir(method, split):
    user = os.environ.get("USER", os.environ.get("USERNAME", "unknown"))
    return _REPO / "predictions" / user / method / f"split_{split}"


class TrackerPredict(BaseAgent):
    name = "tracker-predict"
    thread = "B"
    kind = "verdict"

    def build_argv(self, spec) -> list[str]:
        method = spec.get("method", "unet_transformer")
        split = str(spec.get("split", "0"))
        weights = spec.get("weights") or str(_WEIGHTS_DEFAULT)
        data_dir = spec.get("data_dir") or str(_DATA_DEFAULT)
        argv = [_py(), str(_RUNNER), "--method", method, "--split", split,
                "--weights", str(weights), "--data-dir", str(data_dir),
                "--det-threshold", str(spec.get("det_threshold", 0.5))]
        argv += ["--use-ilp"] if spec.get("use_ilp", True) else ["--no-ilp"]
        if spec.get("splits"):
            argv += ["--splits", str(spec["splits"])]
        if spec.get("debug_video"):
            argv += ["--debug-video", str(spec["debug_video"])]
        if spec.get("max_frames") is not None:
            argv += ["--max-frames", str(int(spec["max_frames"]))]
        if spec.get("unet_batch_size") is not None:
            argv += ["--unet-batch-size", str(int(spec["unet_batch_size"]))]
        return argv

    def _count_nodes(self, pred_dir: Path):
        """Sum node counts across produced geffs (via the offline zarr reader)."""
        sys.path.insert(0, str(COMP)); sys.path.insert(0, str(COMP / "src"))
        from src import io as SIO
        total = 0; per = {}
        for g in sorted(pred_dir.glob("*.geff")):
            try:
                n, _ = SIO.read_geff(g); per[g.name] = len(n); total += len(n)
            except Exception as e:  # noqa: BLE001
                per[g.name] = f"err:{type(e).__name__}"
        return total, per

    def run(self, q, worker):
        spec = self.spec(q)
        if not _RUNNER.exists():
            return self.escalate(worker, "researcher", f"[{worker}] tracker-predict: runner missing at {_RUNNER}")
        argv = self.build_argv(spec)
        method = spec.get("method", "unet_transformer")
        split = str(spec.get("split", "0"))
        try:
            timeout = int(spec.get("timeout", 60 * 60 * 11))
        except (TypeError, ValueError):
            timeout = 60 * 60 * 11
        try:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=str(_REPO),
                               env=_env(spec), timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] tracker-predict: timed out after {timeout}s.")
        except Exception as e:  # noqa: BLE001 — launch failure (e.g. bad python/OSError) → clean escalate
            return self.escalate(worker, "researcher", f"[{worker}] tracker-predict: could not launch — {type(e).__name__}: {str(e)[:120]}")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()[-25:]
        if r.returncode != 0:
            self.log(summary=f"tracker-predict FAILED (rc={r.returncode})", detail="\n".join(tail),
                     kind="finding", recommendation="check weights/data-dir + PYTHONPATH")
            return self.escalate(worker, "researcher",
                                 f"[{worker}] tracker-predict FAILED rc={r.returncode}: {tail[-1] if tail else ''}")
        pred_dir = _pred_dir(method, split)
        total_nodes, per = self._count_nodes(pred_dir)
        result = {"status": "ok", "returncode": 0, "predictions_dir": str(pred_dir),
                  "n_geffs": len(per), "total_nodes": total_nodes, "per_dataset_nodes": per,
                  "argv": " ".join(argv)}
        if total_nodes <= 0:
            return self.escalate(worker, "researcher",
                                 f"[{worker}] tracker-predict: ran OK but produced 0 nodes in {pred_dir}.")
        self.save_state({"predictions_dir": str(pred_dir), "total_nodes": total_nodes, "n_geffs": len(per)})
        self.log(summary=f"tracker-predict: {len(per)} geff(s), {total_nodes:,} nodes → {pred_dir}",
                 detail=str(per), kind="verdict",
                 recommendation="feed predictions_dir to tracker-postproc → submission.csv")
        msg = (f"[{worker}] **TRACKER-PREDICT** · UNet+transformer detect+edge+ILP\n"
               f"{len(per)} dataset(s) · **{total_nodes:,} nodes** → `{pred_dir}`")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(result, msg, to="leader")


_AGENT = TrackerPredict()


def run(q, worker):
    return _AGENT.run(q, worker)
