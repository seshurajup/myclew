"""detect-quality — measure the detector's DETECTION QUALITY on the EXTERNAL densely-labeled Zebrahub crops.

This is the ONLY honest place to measure detection PRECISION: competition GT is ~1% sparse-lineage (an
apparent false-positive may be a real unlabeled cell), whereas external Zebrahub is ~100% dense (every
unmatched prediction IS a false positive). For each external embryo it runs pilkwang detection and reports
node RECALL (does it detect all cells?) + PRECISION (are detections real?) + F1, per embryo, via the
byte-identical 7µm metric matching. Wraps ``research/detect_quality/eval_detect_external.py`` — no metric
or inference logic reimplemented here.

Spec: {embryos (csv), max_crops (per-embryo, 0=all), det_threshold (default 0.99), max_frames (0=all),
       out_dir, timeout}.
A BaseAgent subclass with a data-wise test (test_fleet_agents/detect_quality_test.py) that runs the real
detector on a 1-crop / few-frame slice and confirms a real recall+precision come back.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
_REPO = COMP / "research" / "pilkwang_support_pack" / "repo"
_SCRIPT = COMP / "research" / "detect_quality" / "eval_detect_external.py"
_DEFAULT_OUT = COMP / "research" / "detect_quality" / "out"


def _env(spec):
    env = dict(os.environ)
    pp = os.pathsep.join([str(_REPO / "src"), str(_REPO / "scripts"), str(COMP)])
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if spec.get("embryos"):
        env["CELLMOT_EMBRYOS"] = spec["embryos"] if isinstance(spec["embryos"], str) else ",".join(spec["embryos"])
    if spec.get("max_crops") is not None:
        env["CELLMOT_MAX_CROPS"] = str(int(spec["max_crops"]))
    if spec.get("det_threshold") is not None:
        env["CELLMOT_DET_THRESHOLD"] = str(spec["det_threshold"])
    if spec.get("max_frames") is not None:
        env["CELLMOT_MAX_FRAMES"] = str(int(spec["max_frames"]))
    if spec.get("out_dir"):
        env["CELLMOT_DETQ_OUT"] = str(spec["out_dir"])
    if spec.get("selection"):   # explicit crop list from the select agent (sample-match/box-sample)
        env["CELLMOT_SELECTION"] = spec["selection"] if isinstance(spec["selection"], str) else ",".join(spec["selection"])
    return env


def _py():
    v = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(v) if v.exists() else sys.executable


def _read(spec):
    out_dir = Path(spec.get("out_dir") or _DEFAULT_OUT)
    f = out_dir / "detect_quality.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


class DetectQuality(BaseAgent):
    name = "detect-quality"
    thread = "S"

    def run(self, q, worker):
        spec = self.spec(q)
        if not _SCRIPT.exists():
            return self.escalate(worker, "researcher", f"[{worker}] detect-quality: script missing at {_SCRIPT}")
        timeout = int(spec.get("timeout", 60 * 60 * 6))
        argv = [_py(), "-u", str(_SCRIPT)]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=str(COMP),
                               env=_env(spec), timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] detect-quality: timed out after {timeout}s.")
        except Exception as e:  # noqa: BLE001 — OSError / env failure → escalate cleanly
            return self.escalate(worker, "researcher", f"[{worker}] detect-quality: subprocess failed ({str(e)[:80]}).")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()[-25:]
        if r.returncode != 0:
            self.log(summary=f"detect-quality FAILED (rc={r.returncode})", detail="\n".join(tail),
                     kind="finding", recommendation="check PYTHONPATH + external geff_trainset + weights")
            return self.escalate(worker, "researcher",
                                 f"[{worker}] detect-quality FAILED rc={r.returncode}: {tail[-1] if tail else ''}")
        d = _read(spec)
        if not d or not d.get("per_embryo"):
            return self.escalate(worker, "researcher",
                                 f"[{worker}] detect-quality ran OK but no results parsed. tail={tail[-1] if tail else ''}")
        per = d["per_embryo"]; ov = d.get("overall", {})
        worst = min(per.items(), key=lambda kv: kv[1].get("precision", 1.0)) if per else (None, {})
        result = {"status": "ok", "returncode": 0, "per_embryo": per, "overall": ov,
                  "det_threshold": d.get("det_threshold"),
                  "worst_precision_embryo": worst[0], "out_dir": str(spec.get("out_dir") or _DEFAULT_OUT)}
        # canonical flow keys: use overall recall as "cv" surrogate is misleading → expose recall/precision explicitly
        result["recall"] = ov.get("recall"); result["precision"] = ov.get("precision"); result["f1"] = ov.get("f1")
        self.save_state({"overall": ov, "worst_precision_embryo": worst[0]})
        tbl = " · ".join(f"{e}: R{v['recall']:.3f}/P{v['precision']:.3f}" for e, v in per.items())
        self.log(summary=f"detect-quality (external dense): overall R{ov.get('recall')}/P{ov.get('precision')} F{ov.get('f1')}",
                 detail=tbl, kind="verdict",
                 recommendation=f"lever = PRECISION (over-detection); worst embryo={worst[0]}")
        msg = (f"[{worker}] **DETECT-QUALITY** · external 100%-dense · thr={d.get('det_threshold')}\n"
               f"overall **recall {ov.get('recall')} / precision {ov.get('precision')}** (F1 {ov.get('f1')})\n"
               f"{tbl}\nworst precision: **{worst[0]}** → over-detection is the classification lever")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(result, msg, to="leader")


_AGENT = DetectQuality()


def run(q, worker):
    return _AGENT.run(q, worker)
