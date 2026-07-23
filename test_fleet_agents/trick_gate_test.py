"""trick_gate_test — stub the (slow) golden-12 scorer with KNOWN deltas; assert the evidence-gate verdicts:
ADOPT only if measured Δ≥threshold, REJECT if Δ<0, NEEDS-GPU for prediction-time tricks (never on popularity)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import trick_gate


def test_trick_gate_adopts_only_proven():
    BASE = 0.880
    def fake_score(env, n):                       # stub: score depends on a marker in env
        if env.get("MARK") == "good":  return {"score": 0.885, "adjE": 0.885, "n": 12}   # +0.005 → ADOPT
        if env.get("MARK") == "bad":   return {"score": 0.876, "adjE": 0.876, "n": 12}   # -0.004 → REJECT
        return {"score": BASE, "adjE": BASE, "n": 12}                                     # base
    orig = trick_gate._score; trick_gate._score = fake_score
    try:
        cands = [{"name": "good_trick", "env": {"MARK": "good"}},
                 {"name": "bad_trick", "env": {"MARK": "bad"}},
                 {"name": "tta", "env": {}, "prediction_time": True}]
        s, res, to, msg = trick_gate.run(
            {"question": "t", "spec": {"candidates": cands, "threshold": 0.001, "base_cv": BASE}}, "test")
    finally:
        trick_gate._score = orig
    assert s == "done", msg
    v = res["verdicts"]
    assert v["good_trick"] == "ADOPT", f"proven-positive should ADOPT: {v}"
    assert v["bad_trick"] == "REJECT", f"proven-negative should REJECT: {v}"
    assert v["tta"] == "NEEDS-GPU", f"prediction-time trick must be NEEDS-GPU, not adopted on popularity: {v}"
    return {"adopt_proven": True, "reject_negative": True, "tta_needs_gpu": True}


def _run():
    print("=== TRICK-GATE DATA-WISE VERIFIER ===")
    try:
        r = test_trick_gate_adopts_only_proven()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== trick-gate: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
