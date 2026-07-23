"""official-score — score predicted .geff graphs with the ORGANIZERS' byte-identical metric (edge + division),
NOT our src/metric.py proxy. Use this for ANY division_jaccard number so it matches Kaggle exactly.

Wraps ``research/official_score/score_preds.py`` which calls the host `tracking_cellmot.metrics.evaluate`
(division scoring = official ±1-frame-tolerance, daughter-lineage-coverage, connected-component, bipartite).
Reports edge_jaccard + division_jaccard + TP/FP/FN, per embryo.

Spec: {method, pred_dir, split, max (cap, 0=all), out_dir, timeout}.
Data-wise test (test_fleet_agents/official_score_test.py) scores a real predictions dir and confirms official
edge/division jaccard + counts come back.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
_REPO = COMP / "research" / "official_repo"
_SCRIPT = COMP / "research" / "official_score" / "score_preds.py"
_DEFAULT_OUT = COMP / "research" / "official_score" / "out"


def _env(spec):
    env = dict(os.environ)
    pp = os.pathsep.join([str(_REPO / "src"), str(_REPO / "scripts"), str(COMP)])
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if spec.get("method"):
        env["CELLMOT_METHOD"] = str(spec["method"])
    if spec.get("pred_dir"):
        env["CELLMOT_PRED_DIR"] = str(spec["pred_dir"])
    if spec.get("split") is not None:
        env["CELLMOT_SPLIT"] = str(spec["split"])
    if spec.get("max") is not None:
        env["CELLMOT_MAX"] = str(int(spec["max"]))
    if spec.get("out_dir"):
        env["CELLMOT_OS_OUT"] = str(spec["out_dir"])
    if spec.get("datasets"):
        env["CELLMOT_DATASETS"] = spec["datasets"] if isinstance(spec["datasets"], str) else ",".join(spec["datasets"])
    if spec.get("gt_dir"):
        env["CELLMOT_GT_DIR"] = str(spec["gt_dir"])
    for k, v in (spec.get("extra_env") or {}).items():   # extra_env: optional dict of env vars for the scorer subprocess
        env[str(k)] = str(v)
    return env


def _py():
    v = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(v) if v.exists() else sys.executable


def _read(spec):
    f = Path(spec.get("out_dir") or _DEFAULT_OUT) / "official_score.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


class OfficialScore(BaseAgent):
    name = "official-score"
    thread = "S"

    def run(self, q, worker):
        spec = self.spec(q)
        if not _SCRIPT.exists():
            return self.escalate(worker, "researcher", f"[{worker}] official-score: script missing at {_SCRIPT}")
        timeout = int(spec.get("timeout", 60 * 60 * 2))
        argv = [_py(), "-u", str(_SCRIPT)]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=str(COMP), env=_env(spec), timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] official-score: timed out after {timeout}s.")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()[-25:]
        if r.returncode != 0:
            self.log(summary=f"official-score FAILED (rc={r.returncode})", detail="\n".join(tail),
                     kind="finding", recommendation="check tracksdata + predictions dir + PYTHONPATH")
            return self.escalate(worker, "researcher",
                                 f"[{worker}] official-score FAILED rc={r.returncode}: {tail[-1] if tail else ''}")
        d = _read(spec)
        if not d or not d.get("overall"):
            return self.escalate(worker, "researcher",
                                 f"[{worker}] official-score ran OK but no results parsed. tail={tail[-1] if tail else ''}")
        ov = d["overall"] if isinstance(d.get("overall"), dict) else {}
        per = d.get("per_embryo", {}) if isinstance(d.get("per_embryo"), dict) else {}
        result = {"status": "ok", "returncode": 0, "method": d.get("method"), "n_scored": d.get("n_scored"),
                  "edge_jaccard": ov.get("edge_jaccard"), "division_jaccard": ov.get("division_jaccard"),
                  "cv": ov.get("edge_jaccard"), "overall": ov, "per_embryo": per,
                  "out_dir": str(spec.get("out_dir") or _DEFAULT_OUT)}
        self.save_state({"method": d.get("method"), "division_jaccard": ov.get("division_jaccard"),
                         "edge_jaccard": ov.get("edge_jaccard")})
        pe = " · ".join(f"{e}: edgeJ={s.get('edge_jaccard')}/divJ={s.get('division_jaccard')}"
                        for e, s in per.items() if isinstance(s, dict))
        self.log(summary=f"official-score {d.get('method')}: edgeJ={ov.get('edge_jaccard')} divJ={ov.get('division_jaccard')} (OFFICIAL metric)",
                 detail=pe, kind="verdict",
                 recommendation="this is the ORGANIZER metric — use for every division_jaccard number (not the proxy)")
        msg = (f"[{worker}] **OFFICIAL-SCORE** · `{d.get('method')}` · {d.get('n_scored')} geffs · organizer metric\n"
               f"edge_jaccard **{ov.get('edge_jaccard')}** · division_jaccard **{ov.get('division_jaccard')}** "
               f"(divTP={ov.get('division_tp')} FP={ov.get('division_fp')} FN={ov.get('division_fn')})\n{pe}")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(result, msg, to="leader")


_AGENT = OfficialScore()


def run(q, worker):
    return _AGENT.run(q, worker)
