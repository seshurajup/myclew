"""div-temporal-feas — the GO/NO-GO gate for a TEMPORAL division-link head.

Question it answers (grounded, not re-derived): a single-frame appearance model tops out at ~0.671 TP-vs-FP
separability because a real division and two touching/crossing cells look identical in ONE frame. The proven
missing signal is TEMPORAL (before->after split dynamics). This agent MEASURES whether a temporal appearance
model beats the single-frame ceiling — and, critically, whether it does so for EVERY cell regime (both
embryos, every developmental/density STAGE, and the geometrically-invisible hard cases) — so a GO is proof of
coverage, not an average that hides a failing stage.

Design is MINI-FIRST (mode='mini' -> a few stage-spanning datasets, one fast fold) then scale (mode='full').
Every knob is tied to a MEASURED number (XAI proof), never eyeballed — see the _why fields in the report.

Reusable / spec-driven:
  {mode:'mini'|'full', n_datasets, epochs, seeds, max_neg, crop, temporal_window}
Runs its compute under cellmot_venv (geff/zarr + torch/cuda) via a subprocess worker so the fleet's env is
untouched. Emits a stratified separability table + a GO/NO-GO verdict to the ledger.
"""
from __future__ import annotations
import json, os, subprocess
from pathlib import Path
from .base import BaseAgent, COMP

PY_ENV = COMP / "research" / "cellmot_venv" / "bin" / "python"
WORKER = COMP / "fleet_agents" / "_div_temporal_feas_worker.py"


class DivTemporalFeas(BaseAgent):
    name = "div-temporal-feas"
    thread = "B"
    kind = "verdict"

    # GO threshold: temporal must clearly clear the 0.671 single-frame ceiling AND not regress any stage.
    GO_AUC = 0.80
    CEILING = 0.671

    def run(self, q, worker="w"):
        spec = self.spec(q)
        if not PY_ENV.exists() or not WORKER.exists():
            return ("escalated", {}, "researcher", f"[{worker}] div-temporal-feas: venv/worker missing.")
        payload = json.dumps({
            "mode": spec.get("mode", "mini"),
            "n_datasets": int(spec.get("n_datasets", 16)),
            "epochs": int(spec.get("epochs", 25)),
            "seeds": list(spec.get("seeds", [0])),
            "max_neg": int(spec.get("max_neg", 600)),
            "crop": spec.get("crop"),                       # override; else data-derived
            "temporal_window": spec.get("temporal_window"), # override; else data-derived
            "out": str(COMP / "results" / "div_temporal_feas.json"),
        })
        timeout = int(spec.get("timeout", 1800))
        try:
            r = subprocess.run([str(PY_ENV), str(WORKER), payload], capture_output=True, text=True,
                               timeout=timeout, cwd=str(COMP))
        except Exception as e:  # noqa: BLE001
            return ("escalated", {}, "researcher", f"[{worker}] div-temporal-feas subprocess failed: {str(e)[:120]}")
        line = [l for l in (r.stdout or "").splitlines() if l.startswith("{")]
        if not line:
            return ("escalated", {}, "researcher",
                    f"[{worker}] div-temporal-feas: no result. stderr={(r.stderr or '')[-200:]}")
        res = json.loads(line[-1])
        verdict = self._verdict(res)
        res["verdict"] = verdict
        try:
            from . import ledger
            ledger.log(self.name, kind="verdict",
                       summary=f"TEMPORAL div-head feasibility [{res['mode']}]: {verdict['decision']} "
                               f"(temporal AUC={res['overall']['temporal']} vs single={res['overall']['single']} "
                               f"vs ceiling {self.CEILING}; min-stage temporal={verdict['min_stage_temporal']})",
                       detail=json.dumps(res)[:1500],
                       recommendation=verdict["recommendation"])
        except Exception:  # noqa: BLE001
            pass
        return ("done", res, "leader",
                f"[{worker}] div-temporal-feas {verdict['decision']}: {verdict['recommendation']}")

    def _verdict(self, res):
        ov = res["overall"]; stages = res.get("by_stage", {}); emb = res.get("by_embryo", {})
        hard = res.get("hard_invisible", {})
        stage_temps = [v["temporal"] for v in stages.values() if v.get("temporal") is not None]
        emb_temps = [v["temporal"] for v in emb.values() if v.get("temporal") is not None]
        min_stage = min(stage_temps) if stage_temps else None
        min_emb = min(emb_temps) if emb_temps else None
        delta = round(ov["temporal"] - ov["single"], 3)
        # GO requires: overall temporal clears the ceiling+GO bar, beats single-frame, AND every stage &
        # embryo also clears the single-frame ceiling (coverage proof — no regime left behind), incl. hard cases.
        go = (ov["temporal"] >= self.GO_AUC and delta >= 0.05
              and (min_stage is None or min_stage >= self.CEILING)
              and (min_emb is None or min_emb >= self.CEILING)
              and (hard.get("temporal") is None or hard.get("temporal", 0) >= self.CEILING))
        if go:
            dec, rec = "GO", ("temporal beats single-frame across ALL stages/embryos and hard cases -> "
                              "design + train the temporal division-link head (Step 2)")
        elif ov["temporal"] - ov["single"] >= 0.05:
            dec, rec = "WEAK-GO", ("temporal helps on average but a stage/embryo/hard-case still < ceiling -> "
                                   "scale the run (mode=full) before committing; check the failing regime")
        else:
            dec, rec = "NO-GO", ("temporal ~= single-frame -> divisions are not recoverable from appearance "
                                 "dynamics in this data; do NOT train — report honestly and stop")
        return {"decision": dec, "delta_temporal_vs_single": delta, "min_stage_temporal": min_stage,
                "min_embryo_temporal": min_emb, "recommendation": rec}

def run(q, worker="w"):
    return DivTemporalFeas().run(q, worker)
