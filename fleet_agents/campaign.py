"""campaign — the top-level WORKFLOW that marshals the WHOLE fleet (not 5-6 agents) into a grandmaster
campaign, in ordered phases with gates. Every registered agent is classified into a phase; any agent NOT
explicitly listed is swept into its phase automatically, so NOTHING in the arsenal goes unused (the
run() asserts full coverage and reports it).

Phases (each runs MANY agents; sequential with gates between):
  1. understand — data integrity + diagnosis (data-audit, eda-stats, adversarial-val, cv-build, analysis…)
                  GATE: if adversarial-val reports the CV is leaky → STOP (a leaky CV makes everything moot).
  2. mine       — learn from public/prior (notebook-sync, reproduce-score, prior-art, trick-extractor…)
  3. detect     — detection & data levers (det-sweep, arch-probe, box-sample, combined-train, aug-find…)
  4. linkpost   — linking / post-proc / config levers (recipe-adopt, config-ablate, fullconfig-search…)
  5. validate   — measure + explain + verify (scorer, xai, verify-cv, decision-audit, trick-gate…)
  6. decide     — calibrate + submission gate + report (cv-lb-calibrate, submit-guard, insights, ledger…)
  7. infra      — fleet health + composers (smoke, train-monitor, heal, pipeline, beat-bar, improve-loop…)

Default mode is PLAN (list what each phase would run + coverage) — executing all ~76 agents is huge and
GPU-bound; pass spec.execute=True (optionally spec.only=[phases]) to actually run. Runs agents within a
phase concurrently and gates between phases. A BaseAgent subclass with its own data-wise test.
"""
from __future__ import annotations
from .base import BaseAgent

# explicit role → agents; the catch-all in run() sweeps any unlisted registered agent into a phase so
# coverage is TOTAL. Keep names in sync with the registry (run() reports any that were auto-swept).
PHASES = {
    "understand": ["data-audit", "eda-stats", "ext-label-stats", "adversarial-val", "cv-build",
                   "split-build", "analysis", "pre-analysis", "perf-choice", "gpu-best-practices",
                   "journey-status", "gnn-probe", "arch-builder", "temporal-audit"],
    "mine": ["notebook-sync", "kaggle-scout", "reproduce-score", "public-config", "prior-art",
             "trick-extractor", "paper-research", "notes-sync", "plan-ingest",
             "research-search", "lit-search", "deep-research", "lb-sync"],
    "detect": ["det-sweep", "arch-probe", "arch-search", "aug-ablation", "aug-find", "box-sample",
               "sample-match", "combined-train", "deep-sister", "layer-grow", "flow-gt-build",
               "gnn-link-train", "block-synth", "baseline",
               "mh-ilp", "detector-select", "detector-arch-search", "compress-select",
               "frozen-exploit", "saliency-detect", "ext-transfer", "pattern-tune",
               "distill", "component-graft", "keyframe", "quantize"],
    "linkpost": ["recipe-adopt", "config-ablate", "config-gen", "fullconfig-search", "combo-search",
                 "linking", "post-proc", "single-model-tune", "division", "div-model", "stage-1-div",
                 "tracker-consensus", "ensemble", "combine-winners", "best-config", "pipeline-run",
                 "tracker-select", "link-tune"],
    "validate": ["scorer", "score", "metrics-report", "xai", "decision-audit", "trick-gate", "verify-cv",
                 "scoreboard", "guard", "ablate-best", "post-analysis", "math-master"],
    "decide": ["cv-lb-calibrate", "submit-verify", "nb-preflight", "submit-guard", "submission-build",
               "insights", "ledger", "learn", "orchestrate"],
    "infra": ["smoke", "train-monitor", "heal", "pipeline", "beat-bar", "improve-loop"],
}
_ORDER = ["understand", "mine", "detect", "linkpost", "validate", "decide", "infra"]


class Campaign(BaseAgent):
    name = "campaign"
    thread = "B"
    kind = "verdict"

    def _registered(self):
        from . import _RAW_HANDLERS
        return _RAW_HANDLERS

    def _plan(self):
        """Return (phase→[agents]) with EVERY registered agent placed exactly once (unlisted → swept into a
        best-guess phase, default 'linkpost'). Guarantees total coverage of the fleet."""
        reg = set(self._registered().keys()) - {"campaign"}
        placed = {}
        seen = set()
        for ph in _ORDER:
            placed[ph] = [a for a in PHASES.get(ph, []) if a in reg and a not in seen]
            seen.update(placed[ph])
        missing = sorted(reg - seen)                          # any registered agent not yet placed
        for a in missing:                                     # sweep into 'linkpost' (improvement bucket) so none is unused
            placed["linkpost"].append(a)
        return placed, missing

    def run(self, q, worker):
        spec = self.spec(q)
        reg = self._registered()
        plan, swept = self._plan()
        # OPTIONAL: 'phases' aliases 'only' (which phases to run); 'dry_run' forces PLAN mode even if execute set;
        # 'continue_on_error' (default True) — when False a failed/escalated agent halts the campaign like the gate.
        only = set(spec.get("only") or spec.get("phases") or _ORDER)
        execute = bool(spec.get("execute", False)) and not spec.get("dry_run")
        continue_on_error = bool(spec.get("continue_on_error", True))
        covered = sorted({a for ags in plan.values() for a in ags})
        total = sorted(set(reg.keys()) - {"campaign"})
        coverage_ok = covered == total

        results, ran = {}, 0
        gate_stop = None
        if execute:
            def call(kind):
                if kind not in reg:
                    return {"status": "missing"}
                try:                                          # a crashing agent must not kill the campaign
                    out = reg[kind]({"question": f"campaign:{kind}", "spec": spec.get("agent_spec", {})}, worker)
                except Exception as e:  # noqa: BLE001
                    return {"status": "failed", "data": {"error": str(e)[:200]}}
                st = out[0] if isinstance(out, (list, tuple)) and out else "done"
                d = out[1] if isinstance(out, (list, tuple)) and len(out) > 1 and isinstance(out[1], dict) else {}
                return {"status": st, "data": d}
            for ph in _ORDER:
                if ph not in only:
                    continue
                results[ph] = {a: call(a) for a in plan[ph]}
                ran += len(plan[ph])
                if not continue_on_error:                     # stop on the first hard failure/escalation
                    bad = [a for a, r in results[ph].items() if r.get("status") in ("failed", "escalated")]
                    if bad:
                        gate_stop = f"{ph}: agent(s) {bad[:3]} failed and continue_on_error=False — stopping"
                        break
                if ph == "understand":                        # GATE: leaky CV → stop the whole campaign
                    av = results[ph].get("adversarial-val", {}).get("data", {})
                    if av.get("leaky") is True or av.get("leak") is True:
                        gate_stop = "understand: adversarial-val flags a LEAKY CV — stopping (fix the split first)"
                        break

        phase_summary = {ph: {"n": len(plan[ph]), "agents": plan[ph]} for ph in _ORDER}
        self.save_state({"coverage_ok": coverage_ok, "n_agents": len(covered), "auto_swept": swept,
                         "phases": {ph: len(plan[ph]) for ph in _ORDER}, "executed": execute,
                         "ran": ran, "gate_stop": gate_stop})
        lines = "\n".join(f"| {i+1}. **{ph}** | {len(plan[ph])} | {', '.join('`'+a+'`' for a in plan[ph][:8])}"
                          + ("…" if len(plan[ph]) > 8 else "") + " |" for i, ph in enumerate(_ORDER))
        self.log(summary=f"campaign {'EXECUTED' if execute else 'PLAN'}: {len(covered)}/{len(total)} agents across {len(_ORDER)} phases"
                         + (f" · GATE {gate_stop}" if gate_stop else ""),
                 detail=f"coverage_ok={coverage_ok}; auto-swept {swept or 'none'}", kind="verdict",
                 recommendation="run with execute=True (optionally only=[phases]) to march the whole fleet")
        msg = (f"[{worker}] **CAMPAIGN** ({'executed' if execute else 'plan'}) · "
               f"**{len(covered)}/{len(total)}** agents used across {len(_ORDER)} phases "
               f"{'✅ full coverage' if coverage_ok else '⚠️ coverage gap'}"
               + (f" · 🛑 {gate_stop}" if gate_stop else "") + "\n"
               f"| phase | # | agents |\n|---|--:|---|\n{lines}\n"
               + (f"auto-swept (unlisted→linkpost): {', '.join(swept)}\n" if swept else ""))
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"coverage_ok": coverage_ok, "n_agents": len(covered), "total": len(total),
                          "phases": phase_summary, "auto_swept": swept, "executed": execute,
                          "ran": ran, "gate_stop": gate_stop}, msg, to="leader")


_AGENT = Campaign()


def run(q, worker):
    return _AGENT.run(q, worker)
