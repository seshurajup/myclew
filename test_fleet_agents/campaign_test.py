"""campaign_test — the KEY guarantee: the campaign workflow places EVERY registered agent into a phase
(total coverage — nothing unused), phases are ordered, and execute-mode honors the leaky-CV gate.
Coverage is checked against the REAL registry; execution/gate uses an injected mini-registry."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import campaign
import fleet_agents


def _run():
    print("=== CAMPAIGN WORKFLOW VERIFIER ===")
    # 1) TOTAL COVERAGE against the real registry (the 'use ALL agents' guarantee)
    s, d, to, msg = campaign.Campaign().run({"question": "plan", "spec": {"execute": False}}, "test")
    reg = set(fleet_agents._RAW_HANDLERS.keys()) - {"campaign"}
    covered = {a for ph in d["phases"].values() for a in ph["agents"]}
    checks = {}
    checks["covers_ALL_agents"] = covered == reg and d["coverage_ok"]
    checks["uses_many_not_few"] = d["n_agents"] >= 60          # whole fleet, not 5-6
    checks["seven_phases"] = len(d["phases"]) == 7
    checks["each_agent_once"] = len([a for ph in d["phases"].values() for a in ph["agents"]]) == len(covered)

    # 2) EXECUTE + GATE with a mini injected registry (adversarial-val reports leaky → stop after understand)
    ran = []
    def ok(q, w): ran.append(q["question"]); return ("done", {}, "all", "")
    def leaky(q, w): ran.append(q["question"]); return ("done", {"leaky": True}, "all", "")
    mini = {"data-audit": ok, "adversarial-val": leaky, "recipe-adopt": ok, "scorer": ok}
    C = campaign.Campaign(); C._registered = lambda: mini
    s2, d2, _, _ = C.run({"question": "exec", "spec": {"execute": True}}, "test")
    checks["gate_stopped_on_leak"] = d2["gate_stop"] is not None and "leaky" in d2["gate_stop"].lower()
    checks["stopped_before_linkpost"] = not any("recipe-adopt" in r for r in ran)   # never reached linkpost phase
    checks["ran_understand_agents"] = any("data-audit" in r for r in ran)

    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok_all = all(checks.values())
    print(f"\n  coverage: {d['n_agents']}/{len(reg)} agents; auto-swept: {d['auto_swept'] or 'none'}")
    print(f"=== campaign: {'PASS' if ok_all else 'FAIL'} ===")
    return ok_all


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
