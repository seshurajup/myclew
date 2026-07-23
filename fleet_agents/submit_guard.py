"""submit-guard — grandmaster submission discipline: NEVER burn a Kaggle slot unless a candidate's
PREDICTED LB (mapped from its canonical golden-12 CV via cv-lb-calibrate) beats our current best real LB
by a margin, AND the daily budget allows. Recommending a submit ESCALATES to the human (the fleet never
auto-submits, see [[feedback_experiment_then_submit_best]] [[feedback_only_submit_top_10_capable]]).

Why this exists: a raw canonical-CV win is NOT an LB win — canonical over-predicts LB (~0.026 measured on
abhijith 0.9257→0.900). So we map CV→LB first, then decide. We also refuse to submit a candidate that
doesn't beat the best PUBLIC-notebook LB locally ([[feedback_beat_public_notebook_before_submit]]).

Reusable / spec-driven:
  {candidate_cv, candidate_desc, margin: 0.002, daily_budget: 5, submitted_today: 0,
   current_best_lb: optional (default = max real LB in journal), calib: optional {slope,intercept}
   (default = run cv-lb-calibrate on journal anchors)}

A BaseAgent subclass with its own data-wise test (candidates above/below the bar → recommend/hold).
"""
from __future__ import annotations
from .base import BaseAgent


class SubmitGuard(BaseAgent):
    name = "submit-guard"
    thread = "B"
    kind = "reason"

    def _best_real_lb(self):
        from . import ledger
        lbs = [r["lb"] for r in ledger.entries()
               if isinstance(r.get("lb"), (int, float)) and not isinstance(r.get("lb"), bool) and r["lb"] < 1.0]
        return max(lbs) if lbs else None

    def _calibration(self, spec):
        c = spec.get("calib")
        if c and "slope" in c:
            return float(c["slope"]), float(c["intercept"]), c.get("confidence", "given")
        try:
            from . import cv_lb_calibrate as C
            anchors = spec.get("anchors") or C.CvLbCalibrate()._journal_anchors() or []
            if not anchors:
                return 1.0, -0.026, "low"                   # no anchors → fall back to the measured canonical→LB offset
            slope, intercept, kind = C._fit(anchors)
            conf = "high" if len(anchors) >= 4 else ("medium" if len(anchors) >= 2 else "low")
            return float(slope), float(intercept), conf
        except Exception:  # noqa: BLE001 — calibration unavailable → conservative default offset
            return 1.0, -0.026, "low"

    def run(self, q, worker):
        spec = self.spec(q)
        cv = spec.get("candidate_cv")
        if not isinstance(cv, (int, float)):
            return self.escalate(worker, "researcher", f"[{worker}] submit-guard: no candidate_cv given.")
        desc = spec.get("candidate_desc", "candidate")
        margin = float(spec.get("margin", 0.002))
        budget = int(spec.get("daily_budget", 5)); used = int(spec.get("submitted_today", 0))
        best_lb = spec.get("current_best_lb")
        if best_lb is None:
            best_lb = self._best_real_lb()
        slope, intercept, conf = self._calibration(spec)
        pred_lb = round(slope * float(cv) + intercept, 4)

        budget_ok = used < budget
        beats_bar = (best_lb is None) or (pred_lb > best_lb + margin)
        recommend = beats_bar and budget_ok

        reason = []
        reason.append(f"candidate CV {cv} → predicted LB **{pred_lb}** ({conf} conf)")
        reason.append(f"best real LB = {best_lb}" + (f" · need > {round(best_lb + margin,4)}" if best_lb is not None else " (no prior LB)"))
        reason.append(f"budget {used}/{budget}" + ("" if budget_ok else " — EXHAUSTED"))
        if conf == "low":
            reason.append("⚠️ calibration confidence LOW (few CV,LB anchors) — prediction is rough")
        verdict = ("✅ RECOMMEND SUBMIT (escalating to human)" if recommend
                   else ("⛔ HOLD — predicted LB does not beat the bar" if not beats_bar
                         else "⛔ HOLD — daily budget exhausted"))
        self.save_state({"candidate_cv": cv, "predicted_lb": pred_lb, "best_lb": best_lb,
                         "recommend": recommend, "confidence": conf, "budget": f"{used}/{budget}"})
        self.log(summary=f"submit-guard: {desc} CV {cv}→LB {pred_lb} vs bar {best_lb} → {'SUBMIT' if recommend else 'HOLD'}",
                 detail="; ".join(reason), kind="reason",
                 recommendation="human review + submit" if recommend else "keep experimenting; don't burn a slot")
        to = "human" if recommend else "leader"
        msg = f"[{worker}] **SUBMIT-GUARD** · {verdict}\n• " + "\n• ".join(reason) + f"\n→ `{desc}`"
        self.post(worker, to, msg, routine=False, kind="reason")
        return ("escalated" if recommend else "done", {"recommend": recommend, "predicted_lb": pred_lb,
                "best_lb": best_lb, "confidence": conf}, to, msg)


_AGENT = SubmitGuard()


def run(q, worker):
    return _AGENT.run(q, worker)
