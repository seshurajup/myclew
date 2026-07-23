"""division-rescue — add geometry-consistent SECOND-CHILD forks to predicted graphs using the proven-generic,
embryo-invariant division signature (angle≈137°, asym≈1.6, sib/near≈2.4 — self-calibrated by each movie's own
median motion; verified 2026-07-14 that neither Ultrack nor the host baseline uses ANY division geometry).

Rate-capped (top-K by geometry fit, K≈0.0015·n_nodes = biological division rate) so it does NOT flood false
positives; only links UNCLAIMED t+1 nodes (protects edge_jaccard). Writes rescued .geff per dataset; score the
output with `official-score` (division_jaccard, per embryo) — before (0.000) vs after, without hurting edges.
Wraps ``research/division_rescue/rescue.py``.

Spec: {in_dir, out_dir, in_method, out_method, rate, ang_lo/ang_hi, asym_lo/asym_hi, sib_lo/sib_hi,
       rad_xmot, max (cap, 0=all), timeout}.
Data-wise test (test_fleet_agents/division_rescue_test.py) runs on a few real predictions and confirms forks
are added at ~biological rate and rescued geffs are written.
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
_SCRIPT = COMP / "research" / "division_rescue" / "rescue.py"
_PREDROOT = COMP / "research" / "pilkwang_support_pack" / "repo" / "predictions"


def _predsdir(method, split="0"):
    user = os.environ.get("USER", "seshu")
    return str(_PREDROOT / user / method / f"split_{split}")


def _env(spec):
    env = dict(os.environ)
    pp = os.pathsep.join([str(_REPO / "src"), str(_REPO / "scripts"), str(COMP)])
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    in_dir = spec.get("in_dir") or (_predsdir(spec["in_method"]) if spec.get("in_method") else None)
    out_dir = spec.get("out_dir") or (_predsdir(spec["out_method"]) if spec.get("out_method") else None)
    if in_dir:
        env["CELLMOT_DR_INDIR"] = str(in_dir)
    if out_dir:
        env["CELLMOT_DR_OUTDIR"] = str(out_dir)
    for k, ev in (("rate", "CELLMOT_DR_RATE"), ("rad_xmot", "CELLMOT_DR_RAD_XMOT"),
                  ("ang_lo", "CELLMOT_DR_ANG_LO"), ("ang_hi", "CELLMOT_DR_ANG_HI"),
                  ("asym_lo", "CELLMOT_DR_ASYM_LO"), ("asym_hi", "CELLMOT_DR_ASYM_HI"),
                  ("sib_lo", "CELLMOT_DR_SIB_LO"), ("sib_hi", "CELLMOT_DR_SIB_HI"),
                  ("max", "CELLMOT_MAX")):
        if spec.get(k) is not None:
            env[ev] = str(spec[k])
    if spec.get("reassign"):
        env["CELLMOT_DR_REASSIGN"] = "1"
    return env


def _py():
    v = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(v) if v.exists() else sys.executable


def _out_dir(spec):
    return spec.get("out_dir") or (_predsdir(spec["out_method"]) if spec.get("out_method")
                                   else _predsdir("div_rescue"))


class DivisionRescue(BaseAgent):
    name = "division-rescue"
    thread = "B"

    def run(self, q, worker):
        spec = self.spec(q)
        if not _SCRIPT.exists():
            return self.escalate(worker, "researcher", f"[{worker}] division-rescue: script missing at {_SCRIPT}")
        timeout = int(spec.get("timeout", 60 * 60 * 3))
        argv = [_py(), "-u", str(_SCRIPT)]
        try:
            r = subprocess.run(argv, capture_output=True, text=True, cwd=str(COMP), env=_env(spec), timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] division-rescue: timed out after {timeout}s.")
        except Exception as e:  # noqa: BLE001 — OSError / env failure → escalate cleanly
            return self.escalate(worker, "researcher", f"[{worker}] division-rescue: subprocess failed ({str(e)[:80]}).")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()[-25:]
        if r.returncode != 0:
            self.log(summary=f"division-rescue FAILED (rc={r.returncode})", detail="\n".join(tail),
                     kind="finding", recommendation="check in_dir geffs + tracksdata + PYTHONPATH")
            return self.escalate(worker, "researcher",
                                 f"[{worker}] division-rescue FAILED rc={r.returncode}: {tail[-1] if tail else ''}")
        out_dir = Path(_out_dir(spec))
        summ = out_dir / "rescue_summary.json"
        try:
            d = json.loads(summ.read_text()) if summ.exists() else {}
        except Exception:  # noqa: BLE001 — corrupt summary → proceed with empty stats
            d = {}
        forks = d.get("forks_added"); nds = d.get("n_datasets")
        result = {"status": "ok", "returncode": 0, "out_dir": str(out_dir), "forks_added": forks,
                  "n_datasets": nds, "out_method": spec.get("out_method", "div_rescue")}
        self.save_state({"out_dir": str(out_dir), "forks_added": forks})
        self.log(summary=f"division-rescue: +{forks} forks across {nds} datasets → {out_dir.name}",
                 detail=str(d), kind="verdict",
                 recommendation="score the out_method with official-score → compare division_jaccard vs 0.000, edge must hold")
        msg = (f"[{worker}] **DIVISION-RESCUE** · +{forks} geometry forks · {nds} datasets → `{out_dir.name}`\n"
               f"next: official-score `{spec.get('out_method','div_rescue')}` (div_jaccard vs 0.000, edge must hold)")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(result, msg, to="leader")


_AGENT = DivisionRescue()


def run(q, worker):
    return _AGENT.run(q, worker)
