"""task_spec_test — data-wise verifier for the MindOS spec-driven task contract.

Core properties:
  1. new_spec_template has every required section; an empty one is incomplete.
  2. validate_spec flags missing sections and the data-flow readers/writers lint.
  3. acceptance_status counts pass/fail; a checklist with any failure is not done.
  4. gate() allows completion only when spec is complete AND all acceptance criteria pass.
  5. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import task_spec as T


def _run():
    print("=== TASK-SPEC VERIFIER ===")
    checks = {}

    # 1. template completeness
    tpl = T.new_spec_template("x")
    checks["template_has_sections"] = all(k in tpl for k in T.REQUIRED_SECTIONS)
    ok, missing, _ = T.validate_spec(tpl)
    checks["empty_incomplete"] = (not ok) and "goal" in missing

    # 2. data-flow lint: filled everything but readers/writers empty → data_flow missing
    spec = T.new_spec_template("x")
    spec.update(goal="g", current_state="c", plan="p", impact=["f"],
                edge_cases=["a", "b", "c"], acceptance=[{"check": "x", "passed": True}],
                data_flow={"readers": [], "writers": []})
    ok2, missing2, warns2 = T.validate_spec(spec)
    checks["dataflow_lint"] = (not ok2) and "data_flow" in missing2

    # 3. acceptance status
    acc = T.acceptance_status([{"check": "a", "passed": True}, {"check": "b", "passed": False}])
    checks["acceptance_counts"] = acc == {"passed": 1, "total": 2, "done": False, "failing": ["b"]}
    checks["acceptance_all_pass"] = T.acceptance_status([{"check": "a", "passed": True}])["done"] is True

    # 4. gate: complete + accepted → allowed; flip one acceptance → blocked
    good = T.new_spec_template("x")
    good.update(goal="g", current_state="c", plan="p", impact=["f"],
                data_flow={"readers": ["detector"], "writers": ["gapfill"]},
                edge_cases=["a", "b", "c"],
                acceptance=[{"check": "cv up", "passed": True}, {"check": "p<0.05", "passed": True}])
    g = T.gate(good)
    checks["gate_allows_complete"] = g["allowed"] is True
    good2 = {**good, "acceptance": [{"check": "cv up", "passed": False}]}
    checks["gate_blocks_failed_acceptance"] = T.gate(good2)["allowed"] is False
    incomplete = T.new_spec_template("x"); incomplete["goal"] = "g"
    checks["gate_blocks_incomplete"] = T.gate(incomplete)["allowed"] is False
    print(f"  -> gate(complete)={g['allowed']} reason={g['reason']!r}")

    # 5. agent
    st, dta, to, msg = T.run_taskspec({"spec": {}}, "t")
    checks["agent_done"] = st == "done" and dta["allowed"] is True

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok_all = all(checks.values())
    print(f"=== task-spec: {'PASS' if ok_all else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
