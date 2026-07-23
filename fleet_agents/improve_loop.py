"""improve-loop — a SMART WORKFLOW agent: the grandmaster "attack the weakest link, then re-diagnose"
loop, composing a diagnosis agent + the matching lever agent each round until CV stops improving.

Each round:
  1. DIAGNOSE the weakest metric bucket (call `pre-analysis`/`analysis` → returns the limiting lever, e.g.
     "node_recall" or "edge_precision"/"postproc").
  2. ROUTE to the agent that owns that lever:
        node_recall  → det-sweep      (recover missed nuclei at calibrated count)
        postproc/edge→ recipe-adopt   (graft proven post-proc knobs)  or  config-ablate
        linking      → linking / motion-relink
  3. APPLY it, read the new canonical CV.
  4. If CV improved by > eps → keep it and re-diagnose (the weakest link may have moved); else STOP (converged).

This is smarter than a fixed pipeline: the NEXT agent is chosen by the CURRENT weakest link, so the fleet
never wastes a round tuning a lever that isn't limiting ([[feedback_metric_driven_research_loop]]).
Never trains blindly — respects the GPU hold + provenance gate through the agents it calls.

Reusable / spec-driven: {rounds: 4, eps: 0.001, routing: {lever: kind}, agents: injected {kind: fn} for test}.
A BaseAgent subclass with its own data-wise test (scripted diagnose→improve→converge).
"""
from __future__ import annotations
from .base import BaseAgent

_DEFAULT_ROUTING = {
    "node_recall": "det-sweep",
    "recall": "det-sweep",
    "detection": "det-sweep",
    "postproc": "recipe-adopt",
    "edge": "recipe-adopt",
    "edge_precision": "recipe-adopt",
    "linking": "linking",
}


class ImproveLoop(BaseAgent):
    name = "improve-loop"
    thread = "B"
    kind = "verdict"

    def _agents(self):
        from . import _RAW_HANDLERS
        return _RAW_HANDLERS

    @staticmethod
    def _data(out):
        return out[1] if isinstance(out, (list, tuple)) and len(out) > 1 and isinstance(out[1], dict) else {}

    @staticmethod
    def _cv_of(d):
        for k in ("cv", "merged_cv", "score", "golden_cv", "canonical", "best_cv"):
            v = d.get(k) if isinstance(d, dict) else None
            if isinstance(v, (int, float)):
                return float(v)
        pick = d.get("pick") if isinstance(d, dict) else None            # det-sweep returns {pick:{cv}}
        if isinstance(pick, dict) and isinstance(pick.get("cv"), (int, float)):
            return float(pick["cv"])
        return None

    def run(self, q, worker):
        spec = self.spec(q)
        A = spec.get("agents") or self._agents()
        routing = {**_DEFAULT_ROUTING, **(spec.get("routing") or {})}
        # OPTIONAL: 'max_iters' aliases 'rounds' (hard-capped at 100 so the loop can never run unbounded).
        # 'patience' = consecutive non-improving rounds tolerated before stopping (default 0 = old behaviour).
        try:
            rounds = min(100, max(1, int(spec.get("rounds", spec.get("max_iters", 4)))))
        except Exception:  # noqa: BLE001
            rounds = 4
        try:
            eps = float(spec.get("eps", 0.001))
        except Exception:  # noqa: BLE001
            eps = 0.001
        try:
            patience = max(0, int(spec.get("patience", 0)))
        except Exception:  # noqa: BLE001
            patience = 0

        def call(kind, sub):
            if kind not in A:
                return "missing", {}
            try:                                            # a crashing sub-agent must not kill the loop
                out = A[kind]({"question": f"improve-loop:{kind}", "spec": sub}, worker)
            except Exception as e:  # noqa: BLE001
                return f"error:{type(e).__name__}", {}
            st = out[0] if isinstance(out, (list, tuple)) and out else "done"
            return st, self._data(out)

        best_cv = spec.get("start_cv")
        trace = []
        stopped = "max_rounds"
        stale = 0
        for r in range(rounds):
            _, diag = call("pre-analysis", {"round": r})
            lever = (diag.get("weakest") or diag.get("lever") or diag.get("bucket") or "").lower()
            kind = routing.get(lever)
            if not kind:
                stopped = f"no agent routes lever '{lever or 'unknown'}'"; break
            st, res = call(kind, spec.get("lever_spec", {}))
            cv = self._cv_of(res)
            improved = cv is not None and (best_cv is None or cv > best_cv + eps)
            trace.append({"round": r + 1, "lever": lever, "agent": kind, "cv": cv,
                          "improved": bool(improved)})
            if improved:
                best_cv = cv; stale = 0
            else:
                stale += 1
                if stale > patience:                        # exhausted patience → converged
                    stopped = "converged (no improvement)"; break

        self.save_state({"best_cv": best_cv, "rounds_run": len(trace), "stopped": stopped, "trace": trace})
        chain = " → ".join(f"R{t['round']}:{t['lever']}→`{t['agent']}`({t['cv']}{'↑' if t['improved'] else '·'})" for t in trace)
        self.log(summary=f"improve-loop: {len(trace)} rounds, best CV {best_cv}, stop: {stopped}",
                 detail=chain, kind="verdict",
                 recommendation="adopt the improving config; if converged, the current weakest link is exhausted — change modality")
        msg = (f"[{worker}] **IMPROVE-LOOP** · best canonical CV **{best_cv}** over {len(trace)} round(s) · stop: _{stopped}_\n"
               f"{chain or '(no rounds run)'}")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"best_cv": best_cv, "rounds_run": len(trace), "stopped": stopped, "trace": trace}, msg, to="leader")


_AGENT = ImproveLoop()


def run(q, worker):
    return _AGENT.run(q, worker)
