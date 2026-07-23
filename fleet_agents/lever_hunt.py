"""lever-hunt — automate the metric-driven mini-experiment loop: XAI picks the SUBSET + GOAL, a small
LEVER runs on that subset, the OFFICIAL patched metric VERIFIES before-vs-after, and the agent REPORTS
"SOLID CLUE" or "DEAD" with the numbers. Reusable across competitions via CompConfig (nothing biohub is
hardcoded beyond the spec defaults, which point at this comp's official scorer + predictions).

The four phases (all in research/lever_hunt/lever_hunt_run.py, scored with the byte-identical host metric
`tracking_cellmot.metrics.evaluate`/`per_sample_metrics`/`summarise`):

  1. XAI-DECIDE   node-recall decomposition of missed GT nodes → GAP-recoverable (a predicted node sits
                  within 7µm at t±1) vs SCATTERED (detector-blind) + the gap-length histogram → chooses the
                  goal ("node-recall via gap_fill") and the subset (datasets with the most headroom).
  2. RUN a LEVER  a pluggable library (starts with `gap_fill`) run ONLY on the chosen subset.
  3. VERIFY       edge_jaccard / adj_edge_jaccard (over-prediction-penalised) / division / score before vs
                  after, per embryo, + math_master paired significance across datasets.
  4. REPORT       SOLID CLUE (adj AND combined score both up) or DEAD, with per-embryo Δ.

gap_fill (first lever): bridge a track-END@t to a track-START@t+(k+1) with k interpolated nodes + consecutive
edges (the patched metric keeps only t_target==t_source+1 edges → recovers the node AND the gap edges). Every
knob is math_master-governed + metric-gated: data-derived per-movie q95 motion scale, per-k distance caps that
never imply >7µm/frame (the absolute match gate), velocity motion-consistency filter, density/stage-adaptive +
per-embryo self-calibration. The reusable pipeline stage is research/lever_hunt/gap_fill_postproc.gap_fill_graph
(wired behind CELLMOT_GAPFILL_MAXK in predict_unet_transformer.py).

Spec: {pred_dir, gt_dir, datasets (csv/list), mode (xai|gapfill|both), lever (gap_fill), thresholds (csv µm),
       max_k, max_distance, out_dir, timeout}. Data-wise test: test_fleet_agents/lever_hunt_test.py.
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
_SCRIPT = COMP / "research" / "lever_hunt" / "lever_hunt_run.py"
_DEFAULT_OUT = COMP / "research" / "lever_hunt" / "out"
_DEFAULT_PRED = COMP / "research" / "official_repo" / "predictions" / "seshu" / "cv_flip" / "split_0"
_DEFAULT_GT = COMP / "input" / "biohub-cell-tracking-during-development" / "train"


def _py():
    v = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(v) if v.exists() else sys.executable


def _env(spec):
    env = dict(os.environ)
    pp = os.pathsep.join([str(_REPO / "src"), str(_REPO / "scripts"), str(COMP / "tools" / "researchpapers"),
                          str(COMP)])
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    m = {"pred_dir": "CELLMOT_LH_PRED_DIR", "gt_dir": "CELLMOT_LH_GT_DIR", "mode": "CELLMOT_LH_MODE",
         "lever": "CELLMOT_LH_LEVER", "max_k": "CELLMOT_LH_MAXK", "max_distance": "CELLMOT_LH_MAXD",
         "out_dir": "CELLMOT_LH_OUT"}
    for k, ev in m.items():
        if spec.get(k) is not None:
            env[ev] = str(spec[k])
    if spec.get("datasets"):
        env["CELLMOT_LH_DATASETS"] = spec["datasets"] if isinstance(spec["datasets"], str) else ",".join(spec["datasets"])
    if spec.get("thresholds"):
        env["CELLMOT_LH_THRESHOLDS"] = spec["thresholds"] if isinstance(spec["thresholds"], str) \
            else ",".join(str(x) for x in spec["thresholds"])
    return env


def _read(out_dir):
    f = Path(out_dir or _DEFAULT_OUT) / "lever_hunt.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            return None
    return None


class LeverHunt(BaseAgent):
    name = "lever-hunt"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        if not _SCRIPT.exists():
            return self.escalate(worker, "researcher", f"[{worker}] lever-hunt: engine missing at {_SCRIPT}")
        spec.setdefault("pred_dir", str(_DEFAULT_PRED))
        spec.setdefault("gt_dir", str(_DEFAULT_GT))
        spec.setdefault("mode", "both")
        spec.setdefault("lever", "gap_fill")
        out_dir = spec.get("out_dir") or str(_DEFAULT_OUT)
        timeout = int(spec.get("timeout", 60 * 30))
        try:
            r = subprocess.run([_py(), "-u", str(_SCRIPT)], capture_output=True, text=True,
                               cwd=str(COMP), env=_env(spec), timeout=timeout)
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] lever-hunt: timed out after {timeout}s.")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] lever-hunt: subprocess failed ({str(e)[:80]}).")
        out = (r.stdout or "") + "\n" + (r.stderr or "")
        tail = out.strip().splitlines()[-25:]
        d = _read(out_dir)
        if r.returncode != 0 or not d:
            self.log(summary=f"lever-hunt FAILED (rc={r.returncode})", detail="\n".join(tail), kind="finding",
                     recommendation="check cellmot_venv + predictions dir + PYTHONPATH")
            return self.escalate(worker, "researcher",
                                 f"[{worker}] lever-hunt FAILED rc={r.returncode}: {tail[-1] if tail else ''}")

        xai = d.get("xai", {}); gf = d.get("gapfill", {}); verdict = d.get("verdict", {})
        dec = xai.get("decision", {})
        rec = gf.get("recommended", {}); rov = rec.get("overall", {})
        result = {"status": "ok", "returncode": 0, "mode": d.get("mode"), "lever": d.get("lever"),
                  "goal": dec.get("goal"), "subset": dec.get("subset"),
                  "gap_length_hist": (xai.get("overall", {}) or {}).get("gap_length_hist"),
                  "chosen_max_k": gf.get("chosen_max_k"), "recommended_mode": gf.get("recommended_mode"),
                  "d_edge_jaccard": rec.get("d_edge_jaccard"), "d_adj_edge_jaccard": rec.get("d_adj_edge_jaccard"),
                  "d_primary": rec.get("d_primary"), "verdict": verdict.get("status"),
                  "cv": rov.get("edge_jaccard"), "out_dir": out_dir}
        self.save_state({"verdict": verdict.get("status"), "d_adj_edge_jaccard": rec.get("d_adj_edge_jaccard"),
                         "chosen_max_k": gf.get("chosen_max_k")})
        pe = rec.get("per_embryo", {}) or {}
        pe_str = " · ".join(f"{e}: edgeJ={v.get('edge_jaccard')}/adjJ={v.get('adj_edge_jaccard')}" for e, v in pe.items())
        emoji = "✅" if verdict.get("status") == "SOLID CLUE" else "⛔"
        summary = (f"lever-hunt {d.get('lever')}: {verdict.get('status')} — Δedge={rec.get('d_edge_jaccard')} "
                   f"Δadj={rec.get('d_adj_edge_jaccard')} Δscore={rec.get('d_primary')} @k≤{gf.get('chosen_max_k')} "
                   f"({gf.get('recommended_mode')})")
        self.log(summary=summary, detail=f"goal={dec.get('goal')} subset={dec.get('subset')} | {pe_str}",
                 kind="verdict", recommendation=verdict.get("note", ""))
        msg = (f"[{worker}] {emoji} **LEVER-HUNT** · `{d.get('lever')}` · **{verdict.get('status')}**\n"
               f"XAI goal: {dec.get('goal')}\nsubset: {dec.get('subset')}\n"
               f"gap-length hist: {(xai.get('overall', {}) or {}).get('gap_length_hist')}\n"
               f"recommended: {gf.get('recommended_mode')} @ k≤{gf.get('chosen_max_k')} — "
               f"Δedge_jaccard **{rec.get('d_edge_jaccard')}** · Δadj **{rec.get('d_adj_edge_jaccard')}** · "
               f"Δscore **{rec.get('d_primary')}** (paired wilcoxon_p={(rec.get('paired') or {}).get('wilcoxon_p')})\n"
               f"{pe_str}\n{verdict.get('note', '')}")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(result, msg, to="leader")


_AGENT = LeverHunt()


def run(q, worker):
    return _AGENT.run(q, worker)
