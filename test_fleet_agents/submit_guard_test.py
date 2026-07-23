"""submit_guard_test — with a KNOWN calibration and bar, assert: a candidate whose PREDICTED LB clears
the bar (+ budget) is recommended (escalates to human); one below the bar is held; and an out-of-budget
candidate is held even if it would clear the bar. No journal dependency — calib+bar injected."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import submit_guard


def _call(cv, best_lb, used=0, budget=5):
    # calibration lb = 1.0*cv - 0.026  (the measured canonical→LB offset)
    spec = {"candidate_cv": cv, "candidate_desc": f"cand{cv}", "current_best_lb": best_lb,
            "calib": {"slope": 1.0, "intercept": -0.026, "confidence": "high"},
            "submitted_today": used, "daily_budget": budget, "margin": 0.002}
    return submit_guard.SubmitGuard().run({"question": "guard", "spec": spec}, "test")


def _run():
    print("=== SUBMIT-GUARD DATA-WISE VERIFIER ===")
    # bar = best real LB 0.900. Candidate cv 0.95 → pred LB 0.924 > 0.902 → RECOMMEND.
    s1, d1, to1, _ = _call(0.95, 0.900)
    # candidate cv 0.92 → pred LB 0.894 < 0.902 → HOLD.
    s2, d2, to2, _ = _call(0.92, 0.900)
    # winning candidate but budget exhausted → HOLD.
    s3, d3, to3, _ = _call(0.95, 0.900, used=5, budget=5)
    checks = {
        "recommends_winner": d1["recommend"] is True and to1 == "human" and s1 == "escalated",
        "winner_pred_lb": abs(d1["predicted_lb"] - 0.924) < 1e-6,
        "holds_below_bar": d2["recommend"] is False and to2 == "leader" and s2 == "done",
        "holds_out_of_budget": d3["recommend"] is False,
    }
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== submit-guard: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
