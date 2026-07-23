"""beat-bar — a SMART WORKFLOW agent: composes several Python agents into the grandmaster "beat the bar"
playbook, with real branching (not a flat pipeline). One dispatch → an end-to-end, evidence-gated decision.

The play (each step is another fleet agent; results feed the next + gate the decision):
  1. cv-lb-calibrate   → learn CV→LB from journal anchors (+ the DECOUPLING flag).
  2. recipe-adopt      → graft a reproduced public recipe onto ours; keep only measured-positive knobs → merged CV.
  3. det-sweep         → find the best detection operating point (recall at calibrated count).
  4. take best canonical CV achieved (merged vs current base).
  5. submit-guard      → map best CV → predicted LB; recommend a submit only if it beats the bar + budget.

BRANCHING (the "smart" part — why this isn't just pipeline templates):
  • If calibrate says LB is DECOUPLED from CV at the top → HOLD regardless of CV wins (canonical gains won't
    move LB); recommend diversify / real-submit verification instead. This is the grandmaster guard against
    over-fitting a proxy ([[feedback_local_lb_kaggle_lb_correlation]]).
  • Else defer to submit-guard's predicted-LB-vs-bar decision.
Never auto-submits — a positive decision ESCALATES to the human.

Reusable / spec-driven: {recipe, base, det_grid, pool_grid, candidate_cv (if you already scored a merge),
   agents: optional injected {kind: fn} for testing}. A BaseAgent subclass with its own data-wise test.
"""
from __future__ import annotations
from .base import BaseAgent


class BeatBar(BaseAgent):
    name = "beat-bar"
    thread = "B"
    kind = "verdict"

    def _agents(self):
        from . import _RAW_HANDLERS
        return _RAW_HANDLERS

    @staticmethod
    def _data(out):
        return out[1] if isinstance(out, (list, tuple)) and len(out) > 1 and isinstance(out[1], dict) else {}

    def run(self, q, worker):
        spec = self.spec(q)
        A = spec.get("agents") or self._agents()
        steps = []

        def call(kind, sub):
            if kind not in A:
                steps.append({"kind": kind, "status": "missing"}); return {}
            try:                                       # a crashing sub-agent must not kill the whole workflow
                out = A[kind]({"question": f"beat-bar:{kind}", "spec": sub}, worker)
            except Exception as e:  # noqa: BLE001
                steps.append({"kind": kind, "status": f"error:{type(e).__name__}"}); return {}
            st = out[0] if isinstance(out, (list, tuple)) and out else "done"
            d = self._data(out); steps.append({"kind": kind, "status": st}); return d

        # 1. calibration (know the bar + decoupling)
        cal = call("cv-lb-calibrate", {"predict": [spec.get("candidate_cv")] if spec.get("candidate_cv") else []})
        decoupled = bool(cal.get("decoupled"))

        # 2. recipe-adopt (graft winning knobs) — optional, only if a recipe is given
        merged_cv = spec.get("candidate_cv")
        adopt = {}
        if spec.get("recipe"):
            adopt = call("recipe-adopt", {"base": spec.get("base") or {}, "recipe": spec["recipe"],
                                          **({"score_fn": spec["score_fn"]} if spec.get("score_fn") else {})})
            if isinstance(adopt.get("merged_cv"), (int, float)):
                merged_cv = adopt["merged_cv"]

        # 3. det-sweep (detection lever) — optional if grids given
        det = {}
        if spec.get("det_grid"):
            det = call("det-sweep", {"det_grid": spec["det_grid"], "pool_grid": spec.get("pool_grid", [3.0, 5.0]),
                                     **({"eval_fn": spec["eval_fn"]} if spec.get("eval_fn") else {})})
            pcv = (det.get("pick") or {}).get("cv")
            if isinstance(pcv, (int, float)):
                merged_cv = pcv if merged_cv is None else max(merged_cv, pcv)

        # 4/5. gate: DECOUPLED short-circuits to HOLD; else submit-guard decides on predicted LB
        if merged_cv is None:
            decision, guard = "HOLD", {"recommend": False, "reason": "no candidate CV produced"}
        elif decoupled:
            decision, guard = "HOLD", {"recommend": False,
                                       "reason": "LB decoupled from canonical CV — a CV win won't move LB; diversify / verify by real submit"}
        else:
            guard = call("submit-guard", {"candidate_cv": merged_cv,
                                          "candidate_desc": spec.get("desc", "beat-bar merged config"),
                                          **({"calib": {"slope": cal.get("slope"), "intercept": cal.get("intercept"),
                                                        "confidence": cal.get("confidence")}} if cal.get("slope") is not None else {}),
                                          **({"current_best_lb": spec["current_best_lb"]} if spec.get("current_best_lb") is not None else {})})
            decision = "SUBMIT" if guard.get("recommend") else "HOLD"

        result = {"decision": decision, "merged_cv": merged_cv, "decoupled": decoupled,
                  "predicted_lb": guard.get("predicted_lb"), "kept_knobs": adopt.get("kept"),
                  "det_pick": det.get("pick"), "chain": [f"{s['kind']}:{s['status']}" for s in steps]}
        self.save_state(result)
        self.log(summary=f"beat-bar → {decision} (merged CV {merged_cv}, predicted LB {guard.get('predicted_lb')}, decoupled={decoupled})",
                 detail=" → ".join(result["chain"]), kind="verdict",
                 recommendation=("escalate to human for submit" if decision == "SUBMIT"
                                 else ("diversify — CV decoupled from LB" if decoupled else "keep improving; don't burn a slot")))
        to = "human" if decision == "SUBMIT" else "leader"
        msg = (f"[{worker}] **BEAT-BAR** workflow → **{decision}**\n"
               f"• chain: {' → '.join(f'`{c}`' for c in result['chain'])}\n"
               f"• merged canonical CV **{merged_cv}** · predicted LB **{guard.get('predicted_lb')}** · decoupled={decoupled}\n"
               f"• adopt kept: {adopt.get('kept') or '—'} · det pick: {det.get('pick') or '—'}\n"
               f"→ {guard.get('reason') or ('human review + submit' if decision=='SUBMIT' else 'hold')}")
        self.post(worker, to, msg, routine=False, kind="verdict")
        return ("escalated" if decision == "SUBMIT" else "done", result, to, msg)


_AGENT = BeatBar()


def run(q, worker):
    return _AGENT.run(q, worker)
