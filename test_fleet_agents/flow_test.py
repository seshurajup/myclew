"""flow_test — the composition contract: canon() pulls a CV from any naming; carry_spec() re-emits it
under every input alias; BaseAgent.done() FORCES canonical keys onto every agent's output; and a real
pipeline chains an agent that outputs `merged_cv` into one that reads `candidate_cv` — proving ANY agent
loops with ANY agent."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import flow, pipeline
from fleet_agents.base import BaseAgent


def _run():
    print("=== FLOW COMPOSITION VERIFIER ===")
    checks = {}
    # 1) canon pulls cv from different names + nested pick.cv
    checks["canon_merged_cv"] = flow.canon({"merged_cv": 0.91}).get("cv") == 0.91
    checks["canon_score"] = flow.canon({"score": 0.92}).get("cv") == 0.92
    checks["canon_pick_cv"] = flow.canon({"pick": {"cv": 0.93}}).get("cv") == 0.93
    # 2) carry_spec re-emits cv under the aliases downstream agents read
    cs = flow.carry_spec({}, {"merged_cv": 0.9})
    checks["carry_to_candidate_cv"] = cs.get("candidate_cv") == 0.9 and cs.get("start_cv") == 0.9
    checks["explicit_spec_wins"] = flow.carry_spec({"candidate_cv": 0.5}, {"merged_cv": 0.9})["candidate_cv"] == 0.5

    # 3) BaseAgent.done() FORCES canonical keys onto output
    class A(BaseAgent):
        name = "tmp-a"
        def run(self, q, w): return self.done({"merged_cv": 0.88}, "a")
    st, data, to, msg = A().run({}, "t")
    checks["done_forces_canonical_cv"] = data.get("cv") == 0.88   # 'cv' auto-added from 'merged_cv'

    # 4) INTEGRATION: pipeline chains producer(merged_cv) → consumer(reads candidate_cv)
    seen = {}
    def producer(q, w): return ("done", {"merged_cv": 0.87}, "all", "p")
    def consumer(q, w):
        seen["candidate_cv"] = q["spec"].get("candidate_cv"); return ("done", {"ok": True}, "all", "c")
    pipeline.Pipeline._handlers = lambda self: {"prod": producer, "cons": consumer}
    pipeline.Pipeline().run({"question": "chain", "spec": {"steps": [{"kind": "prod"}, {"kind": "cons"}], "carry": True}}, "t")
    checks["pipeline_wires_producer_to_consumer"] = seen.get("candidate_cv") == 0.87

    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== flow: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
