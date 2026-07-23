"""decision_audit_test — DATA-WISE verifier for the decision-audit agent.

Ground truth: build a ledger with KNOWN violations — one impossible CV (1.12), one kept-without-CV row,
and clean rows. A correct decision-audit must flag exactly those. We assert the counts.
"""
import json
import os
import sys
import tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import decision_audit


def test_decision_audit_flags_known_violations():
    with tempfile.TemporaryDirectory() as d:
        led = os.path.join(d, "ledger.jsonl")
        rows = [
            {"exp": "E1", "cv": 0.88, "kept": True},          # clean
            {"exp": "E2", "cv": 1.12, "kept": True},          # impossible CV (planted)
            {"exp": "E3", "cv": None, "kept": True},          # kept-without-CV (planted)
            {"exp": "E4", "cv": 0.85, "recommendation": "adopt this"},  # clean rec (has cv)
        ]
        with open(led, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        status, res, to, msg = decision_audit.report({"question": "t", "spec": {"ledger_path": led}}, "test")
        assert status == "done", f"agent errored: {msg}"
        assert res["impossible_cv"] >= 1, f"missed the planted 1.12 impossible CV: {res}"
        assert res["kept_no_cv"] >= 1, f"missed the planted kept-without-CV row: {res}"
        return {"caught_impossible_cv": res["impossible_cv"] >= 1, "caught_kept_no_cv": res["kept_no_cv"] >= 1}


def _run():
    print("=== DECISION-AUDIT DATA-WISE VERIFIER ===")
    try:
        r = test_decision_audit_flags_known_violations()
        for k, v in r.items():
            print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== decision-audit: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
