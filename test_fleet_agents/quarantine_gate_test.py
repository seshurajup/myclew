"""quarantine_gate_test — verify agents RESPECT their tests: a red-tested (quarantined) agent must NOT run
its logic; it escalates to be fixed. A green agent runs normally. This makes the tests governance, not decor."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import fleet_agents as fa
from fleet_agents import base


def test_quarantined_agent_does_not_run():
    name = "heal"
    base.QUARANTINE.discard(name)
    # green: runs normally
    s0, d0, _, _ = fa.AGENTS[name].run({"question": "x", "spec": {}}, "t")
    assert s0 in {"done", "escalated"} and "quarantined" not in (d0 or {}), f"green agent should run: {s0} {d0}"
    # red: quarantined → escalates, does NOT run its logic
    base.QUARANTINE.add(name)
    try:
        s1, d1, to1, _ = fa.AGENTS[name].run({"question": "x", "spec": {}}, "t")
        assert s1 == "escalated" and d1.get("quarantined") == name and to1 == "leader", f"gate not respected: {s1} {d1}"
    finally:
        base.QUARANTINE.discard(name)
    return {"green_runs": True, "red_gated": True}


def _run():
    print("=== QUARANTINE-GATE VERIFIER (agents respect their tests) ===")
    try:
        r = test_quarantined_agent_does_not_run()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== quarantine-gate: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
