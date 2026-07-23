"""beat_bar_test — inject MOCK sub-agents and assert the workflow's grandmaster branching:
(1) recipe wins + not decoupled + predicted LB beats bar → SUBMIT (escalates to human);
(2) DECOUPLED short-circuits to HOLD even though CV wins (submit-guard must NOT be consulted);
(3) predicted LB below bar → HOLD. No GPU, no real agents."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import beat_bar


def _mock_agents(decoupled, merged_cv, guard_recommend, guard_lb, calls):
    def cal(q, w): calls.append("cv-lb-calibrate"); return ("done", {"slope": 1.0, "intercept": -0.026, "confidence": "high", "decoupled": decoupled}, "all", "")
    def adopt(q, w): calls.append("recipe-adopt"); return ("done", {"merged_cv": merged_cv, "kept": ["BIOHUB_DET"]}, "all", "")
    def guard(q, w): calls.append("submit-guard"); return (("escalated" if guard_recommend else "done"), {"recommend": guard_recommend, "predicted_lb": guard_lb}, "human" if guard_recommend else "leader", "")
    return {"cv-lb-calibrate": cal, "recipe-adopt": adopt, "submit-guard": guard}


def _call(decoupled, merged_cv, guard_recommend, guard_lb):
    calls = []
    spec = {"recipe": {"BIOHUB_DET": 0.97}, "base": {"BIOHUB_DET": 0.90},
            "agents": _mock_agents(decoupled, merged_cv, guard_recommend, guard_lb, calls)}
    s, d, to, msg = beat_bar.BeatBar().run({"question": "beat", "spec": spec}, "test")
    return s, d, to, calls


def _run():
    print("=== BEAT-BAR WORKFLOW VERIFIER ===")
    # 1) winner → SUBMIT
    s1, d1, to1, c1 = _call(decoupled=False, merged_cv=0.95, guard_recommend=True, guard_lb=0.924)
    # 2) decoupled → HOLD, submit-guard NOT called
    s2, d2, to2, c2 = _call(decoupled=True, merged_cv=0.95, guard_recommend=True, guard_lb=0.924)
    # 3) below bar → HOLD
    s3, d3, to3, c3 = _call(decoupled=False, merged_cv=0.90, guard_recommend=False, guard_lb=0.874)
    checks = {
        "winner_submits": d1["decision"] == "SUBMIT" and to1 == "human" and s1 == "escalated",
        "winner_ran_full_chain": c1 == ["cv-lb-calibrate", "recipe-adopt", "submit-guard"],
        "decoupled_holds": d2["decision"] == "HOLD" and d2["decoupled"] is True,
        "decoupled_skips_guard": "submit-guard" not in c2,     # short-circuit — the smart branch
        "below_bar_holds": d3["decision"] == "HOLD" and s3 == "done",
        "below_bar_consulted_guard": "submit-guard" in c3,
    }
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== beat-bar: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
