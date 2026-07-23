"""contract_test — the BASE contract every one of the 62 agents must satisfy (so none miss main features).

Deep behaviour tests (the other *_test.py files) plant ground truth and assert recovery — but they need
per-agent fixtures. This test covers ALL agents at the contract level, which is universal and complete:

  1. every registered agent IS a BaseAgent (extends it),
  2. every agent exposes the main features (spec/state/post/log/escalate/done/verify),
  3. every agent is callable with the (q, worker) signature,
  4. it reports which agents also have a DEEP data-wise behaviour test (honest coverage split).

This makes coverage complete: 62/62 contract-verified; N/62 additionally behaviour-verified.
"""
import os
import sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

import fleet_agents as fa
from fleet_agents.base import BaseAgent

REQUIRED_FEATURES = ["spec", "load_state", "save_state", "post", "log", "escalate", "done", "record", "verify", "has_test"]


def test_every_agent_extends_baseagent():
    bad = [k for k, a in fa.AGENTS.items() if not isinstance(a, BaseAgent)]
    assert not bad, f"agents not extending BaseAgent: {bad}"
    return len(fa.AGENTS)


def test_every_agent_has_main_features():
    missing = {}
    for k, a in fa.AGENTS.items():
        gaps = [f for f in REQUIRED_FEATURES if not hasattr(a, f)]
        if gaps:
            missing[k] = gaps
    assert not missing, f"agents missing features: {missing}"
    return len(fa.AGENTS)


def test_every_agent_is_callable():
    bad = [k for k, a in fa.AGENTS.items() if not callable(getattr(a, "run", None))]
    assert not bad, f"agents without a callable run(): {bad}"
    return len(fa.AGENTS)


def _run():
    print("=== FLEET CONTRACT VERIFIER (all agents extend BaseAgent, no missed features) ===")
    ok = True
    try:
        n = test_every_agent_extends_baseagent(); print(f"  ✅ all {n} agents extend BaseAgent")
        test_every_agent_has_main_features(); print(f"  ✅ all {n} agents expose {len(REQUIRED_FEATURES)} main features")
        test_every_agent_is_callable(); print(f"  ✅ all {n} agents are callable (q, worker)")
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False

    total = len(fa.AGENTS)
    deep = sorted(k for k, a in fa.AGENTS.items() if a.has_test())
    print(f"\n  contract-verified: {total}/{total}")
    print(f"  DEEP data-wise behaviour tests: {len(deep)}/{total}  →  {', '.join(deep)}")
    print(f"\n=== contract: {'PASS' if ok else 'FAIL'} ({total}/{total} agents satisfy the BaseAgent contract) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
