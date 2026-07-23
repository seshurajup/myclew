"""stage-dynamics — profile per-developmental-STAGE cell dynamics (motion + division rate) from GT and use
math-master two-sample tests to decide whether STAGE-ADAPTIVE ILP priors are justified.

Domain-grounded (user 2026-07-14): developmental stage is readable from cell density, and each embryo moves
through stages at its own speed — so a single global linking gate is likely miscalibrated per stage. This
agent measures the per-stage inter-frame motion distribution + division rate from GT, runs Wasserstein /
energy-permutation / KS / Cliff's-δ between adjacent stages, and emits self-calibrating per-stage ILP priors
(motion gate + gap-close + division prior) for mh-ilp / tracker-predict to consume. Wraps
``research/stage_dynamics/profile.py`` — no stats or dynamics logic reimplemented here.

Spec: {max_ds (per-stage dataset cap, 0=all), out_dir, timeout}.
Data-wise test (test_fleet_agents/stage_dynamics_test.py) runs the real GT profiling on a small per-stage
cap and confirms per-stage motion percentiles + a significance verdict come back.
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
_SCRIPT = COMP / "research" / "stage_dynamics" / "profile.py"
_DEFAULT_OUT = COMP / "research" / "stage_dynamics" / "out"


def _env(spec):
    env = dict(os.environ)
    pp = os.pathsep.join([str(_REPO / "src"), str(COMP), str(COMP / "tools" / "researchpapers")])
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if spec.get("max_ds") is not None:
        env["CELLMOT_MAX_DS"] = str(int(spec["max_ds"]))
    if spec.get("out_dir"):
        env["CELLMOT_SD_OUT"] = str(spec["out_dir"])
    if spec.get("external"):
        env["CELLMOT_SD_EXTERNAL"] = "1"
    if spec.get("ext_embryos"):
        env["CELLMOT_EXT_EMBRYOS"] = spec["ext_embryos"] if isinstance(spec["ext_embryos"], str) else ",".join(spec["ext_embryos"])
    return env


def _py():
    v = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(v) if v.exists() else sys.executable


def _read(spec):
    fname = "stage_dynamics_external.json" if spec.get("external") else "stage_dynamics.json"
    f = Path(spec.get("out_dir") or _DEFAULT_OUT) / fname
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


class StageDynamics(BaseAgent):
    name = "stage-dynamics"
    thread = "S"

    def run(self, q, worker):
        spec = self.spec(q)
        if not _SCRIPT.exists():
            return self.escalate(worker, "researcher", f"[{worker}] stage-dynamics: script missing at {_SCRIPT}")
        timeout = int(spec.get("timeout", 60 * 60 * 2))
        argv = [_py(), "-u", str(_SCRIPT)]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=str(COMP),
                               env=_env(spec), timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] stage-dynamics: timed out after {timeout}s.")
        except Exception as e:  # noqa: BLE001 — OSError / env failure → escalate cleanly
            return self.escalate(worker, "researcher", f"[{worker}] stage-dynamics: subprocess failed ({str(e)[:80]}).")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()[-25:]
        if r.returncode != 0:
            self.log(summary=f"stage-dynamics FAILED (rc={r.returncode})", detail="\n".join(tail),
                     kind="finding", recommendation="check stage parquet + PYTHONPATH + math_master import")
            return self.escalate(worker, "researcher",
                                 f"[{worker}] stage-dynamics FAILED rc={r.returncode}: {tail[-1] if tail else ''}")
        d = _read(spec)
        if spec.get("external"):
            if not d or not d.get("per_embryo"):
                return self.escalate(worker, "researcher",
                                     f"[{worker}] stage-dynamics external ran OK but no results. tail={tail[-1] if tail else ''}")
            result = {"status": "ok", "source": "external_dense", "per_embryo": d["per_embryo"],
                      "invariance": d.get("invariance", {}), "cross_check_vs_comp_gt": d.get("cross_check_vs_comp_gt", {}),
                      "n_divisions": d.get("n_divisions"), "verdict": d.get("verdict"),
                      "out_dir": str(spec.get("out_dir") or _DEFAULT_OUT)}
            self.log(summary=f"stage-dynamics EXTERNAL: {d.get('verdict')}",
                     detail=json.dumps(d.get("cross_check_vs_comp_gt", {})), kind="verdict",
                     recommendation="if signature holds on external dense → wire the generic division prior")
            msg = (f"[{worker}] **STAGE-DYNAMICS (external dense)** · {d.get('n_divisions')} real divisions\n"
                   f"**verdict: {d.get('verdict')}**")
            self.post(worker, "leader", msg, routine=False, kind="verdict")
            return self.done(result, msg, to="leader")
        if not d or not d.get("per_embryo_stage"):
            return self.escalate(worker, "researcher",
                                 f"[{worker}] stage-dynamics ran OK but no results parsed. tail={tail[-1] if tail else ''}")
        result = {"status": "ok", "returncode": 0, "per_embryo_stage": d["per_embryo_stage"],
                  "motion_tests_within_embryo": d.get("motion_tests_within_embryo", {}),
                  "n_motion_significant": d.get("n_motion_significant"),
                  "division_within_embryo": d.get("division_within_embryo", {}),
                  "verdict": d.get("verdict"), "ilp_priors_per_embryo": d.get("ilp_priors_per_embryo", {}),
                  "confound": d.get("embryo_stage_confound"), "paper_ref": d.get("paper_ref"),
                  "out_dir": str(spec.get("out_dir") or _DEFAULT_OUT)}
        self.save_state({"verdict": d.get("verdict"), "n_motion_significant": d.get("n_motion_significant")})
        rows = []
        for emb, st in d["per_embryo_stage"].items():
            rows.append(f"{emb}: " + " ".join(f"S{s.lstrip('S')}(p50={v['motion_p50']},dr={v['div_rate_gt']})"
                                               for s, v in st.items() if v.get("n_ds")))
        detail = " || ".join(rows)
        self.log(summary=f"stage-dynamics (per-embryo, confound-controlled): {d.get('verdict')}",
                 detail=detail + " | " + (d.get("paper_ref") or ""), kind="verdict",
                 recommendation="both metrics: motion→edge, div-rate→division; confirm div-rate on external dense before wiring")
        msg = (f"[{worker}] **STAGE-DYNAMICS** · per-embryo (confound-controlled), BOTH CV metrics\n"
               f"{detail}\n**verdict: {d.get('verdict')}**")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(result, msg, to="leader")


_AGENT = StageDynamics()


def run(q, worker):
    return _AGENT.run(q, worker)
