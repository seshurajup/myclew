"""full-cv-baseline — the HONEST baseline every LoRA/detector experiment must beat.

Predicts + ILP-links the pilkwang base UNet+transformer on ALL 199 training datasets (NOT the
leaky golden-12), scores with the byte-identical official metric, and reports the full-CV score
plus per-embryo (44b6 / 6bba) breakdown. Wraps ``research/lora_finetune/full_cv_baseline.py``
VERBATIM through env vars — no scoring or inference logic is reimplemented here.

Spec: {max_datasets (0=all 199), out_dir, deadline_h (default 11), timeout}.
The script resumes (skips already-scored datasets) so a killed run continues cheaply.
A BaseAgent subclass with a data-wise test (test_fleet_agents/full_cv_baseline_test.py) that
runs the real predict+ILP+score on a 1-dataset slice and confirms a real (non-degenerate) score.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
_REPO = COMP / "research" / "pilkwang_support_pack" / "repo"
_SCRIPT = COMP / "research" / "lora_finetune" / "full_cv_baseline.py"
_DEFAULT_OUT = COMP / "research" / "lora_finetune" / "full_cv"


def _env(spec):
    env = dict(os.environ)
    pp = os.pathsep.join([str(_REPO / "src"), str(_REPO / "scripts"), str(COMP)])
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if int(spec.get("max_datasets", 0)):
        env["CELLMOT_MAX_DATASETS"] = str(int(spec["max_datasets"]))
    if spec.get("out_dir"):
        env["CELLMOT_FULLCV_OUT"] = str(spec["out_dir"])
    if spec.get("deadline_h") is not None:
        env["CELLMOT_DEADLINE_H"] = str(spec["deadline_h"])
    return env


def _py():
    v = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(v) if v.exists() else sys.executable


def _read_score(spec, out):
    """Prefer the structured results.json; fall back to parsing the DONE stdout line."""
    out_dir = Path(spec.get("out_dir") or _DEFAULT_OUT)
    res = out_dir / "full_cv_results.json"
    if res.exists():
        try:
            d = json.loads(res.read_text())
            fc = d.get("full_cv") or {}
            if fc.get("score") is not None:
                return {"score": float(fc["score"]), "edge": float(fc.get("adj_edge_jaccard", 0.0)),
                        "div": float(fc.get("division_jaccard", 0.0)), "n": int(d.get("n", 0)),
                        "done": bool(d.get("done"))}
        except Exception:  # noqa: BLE001
            pass
    m = re.search(r"DONE:\s*(\d+)\s*datasets,\s*FULL-CV score=([0-9.]+)\s*edge=([0-9.]+)", out)
    if m:
        return {"score": float(m.group(2)), "edge": float(m.group(3)), "div": None,
                "n": int(m.group(1)), "done": True}
    m = re.search(r"running FULL-CV score=([0-9.]+)\s*edge=([0-9.]+)", out.strip().splitlines()[-40:][-1]
                  if out.strip() else "")
    return None


class FullCvBaseline(BaseAgent):
    name = "full-cv-baseline"
    thread = "S"

    def run(self, q, worker):
        spec = self.spec(q)
        if not _SCRIPT.exists():
            return self.escalate(worker, "researcher", f"[{worker}] full-cv-baseline: script missing at {_SCRIPT}")
        timeout = int(spec.get("timeout", 60 * 60 * 11))
        argv = [_py(), "-u", str(_SCRIPT)]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=str(COMP),
                               env=_env(spec), timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] full-cv-baseline: timed out after {timeout}s.")
        except Exception as e:  # noqa: BLE001 — OSError / env failure → escalate cleanly
            return self.escalate(worker, "researcher", f"[{worker}] full-cv-baseline: subprocess failed ({str(e)[:80]}).")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()[-25:]
        if r.returncode != 0:
            self.log(summary=f"full-cv-baseline FAILED (rc={r.returncode})", detail="\n".join(tail),
                     kind="finding", recommendation="check PYTHONPATH + weights + train data dir")
            return self.escalate(worker, "researcher",
                                 f"[{worker}] full-cv-baseline FAILED rc={r.returncode}: {tail[-1] if tail else ''}")
        sc = _read_score(spec, out)
        if not sc or sc["score"] <= 0:
            return self.escalate(worker, "researcher",
                                 f"[{worker}] full-cv-baseline ran OK but no score parsed. tail={tail[-1] if tail else ''}")
        result = {"status": "ok", "returncode": 0, "cv": sc["score"], "score": sc["score"],
                  "edge": sc["edge"], "div": sc["div"], "n_datasets": sc["n"], "done": sc["done"],
                  "out_dir": str(spec.get("out_dir") or _DEFAULT_OUT), "argv": " ".join(argv)}
        self.save_state({"baseline_cv": sc["score"], "n": sc["n"], "done": sc["done"]})
        self.log(summary=f"full-cv-baseline: {sc['n']} ds, FULL-CV={sc['score']:.4f} edge={sc['edge']:.4f}",
                 detail=str(sc), kind="verdict",
                 recommendation="this is the honest baseline; lora-validate must beat it on the SAME 199")
        if sc["done"] and sc["n"] >= 150:
            self.record(change="full-CV baseline (pilkwang base, all 199)", cv=sc["score"],
                        script="full-cv-baseline", train_set="full199", kept=None,
                        observation=f"edge={sc['edge']:.4f} div={sc['div']} n={sc['n']}")
        msg = (f"[{worker}] **FULL-CV-BASELINE** · pilkwang base, {sc['n']} datasets\n"
               f"**score {sc['score']:.4f}** (edge {sc['edge']:.4f}) — the honest bar for every LoRA/detector run")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(result, msg, to="leader")


_AGENT = FullCvBaseline()


def run(q, worker):
    return _AGENT.run(q, worker)
