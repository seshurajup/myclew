"""harness_opt_gate_test — data-wise verifier for the harness-opt-gate agent.

Ground-truth properties (the deepagents better-harness acceptance rule):
  1. ACCEPT iff COMBINED (train+holdout) pass-count STRICTLY improves over baseline.
  2. A train-only gain that REGRESSES the blind holdout is REJECTED (the whole point of the blind gate).
  3. A flat/negative combined delta is REJECTED (strict improvement, not >=).
  4. Input coercion: int / list(of bools/0-1) / dict({"passed","total"}|{"results"}) all count correctly.
  5. Same-strata guard: a per-split TOTAL mismatch between baseline and candidate → valid=False, accept=False.
  6. Agent contract: run(q, worker) returns a valid (status, data, to, message) 4-tuple, incl. empty-spec smoke.
"""
import os
import sys

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import harness_opt_gate as H

VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}


def _run():
    print("=== HARNESS-OPT-GATE VERIFIER ===")
    checks = {}

    # 1. combined improvement → ACCEPT (train +1, holdout flat)
    g = H.gate(baseline_train=[1, 1, 0], baseline_holdout=[1, 0],
               cand_train=[1, 1, 1], cand_holdout=[1, 0])
    checks["accept_combined_improves"] = g["accept"] and g["delta_combined"] == 1 and g["valid"]

    # 2. train gains but holdout REGRESSES so combined is flat → REJECT (blind gate does its job)
    g2 = H.gate(baseline_train=[1, 0, 0], baseline_holdout=[1, 1],
                cand_train=[1, 1, 0], cand_holdout=[1, 0])   # train +1, holdout -1 → combined +0
    checks["reject_holdout_regression"] = (not g2["accept"]) and g2["train_delta"] == 1 and g2["holdout_delta"] == -1

    # 3. strict improvement (flat combined) → REJECT
    g3 = H.gate(2, 2, 2, 2)
    checks["reject_flat"] = (not g3["accept"]) and g3["delta_combined"] == 0

    # 3b. combined worse → REJECT
    g3b = H.gate(3, 3, 2, 3)
    checks["reject_worse"] = (not g3b["accept"]) and g3b["delta_combined"] == -1

    # 4. input coercion: int / list / dict all yield the same combined counts
    gi = H.gate(2, 1, 3, 1)                                              # ints
    gl = H.gate([1, 1, 0], [1, 0], [1, 1, 1], [1, 0])                    # lists → same passes
    gd = H.gate({"passed": 2, "total": 3}, {"passed": 1, "total": 2},
                {"passed": 3, "total": 3}, {"passed": 1, "total": 2})   # dicts
    gr = H.gate({"results": [1, 1, 0]}, {"results": [1, 0]},
                {"results": [1, 1, 1]}, {"results": [1, 0]})            # dict-results
    checks["coerce_agree"] = (gi["candidate_combined"] == gl["candidate_combined"]
                              == gd["candidate_combined"] == gr["candidate_combined"] == 4
                              and gl["accept"] and gd["accept"] and gr["accept"])

    # 5. same-strata guard: candidate train total differs from baseline → invalid, rejected
    gm = H.gate({"passed": 2, "total": 3}, {"passed": 1, "total": 2},
                {"passed": 4, "total": 5}, {"passed": 1, "total": 2})   # train total 3 != 5
    checks["strata_guard"] = (not gm["valid"]) and (not gm["accept"])

    print(f"  -> accept(+1 combined)={g['accept']}  reject(holdout-regress)={not g2['accept']}  "
          f"reject(flat)={not g3['accept']}  strata_guard_invalid={not gm['valid']}")

    # 6. agent contract — populated spec + empty-spec smoke
    out = H.run({"question": "q", "spec": {"baseline_train": 1, "baseline_holdout": 1,
                                           "cand_train": 2, "cand_holdout": 1}}, "tester")
    checks["contract_spec"] = (isinstance(out, tuple) and len(out) == 4 and out[0] in VALID
                               and out[1]["accept"] is True)
    sm = H.run({"question": "smoke", "spec": {}}, "tester")
    checks["contract_smoke"] = isinstance(sm, tuple) and len(sm) == 4 and sm[0] in VALID and "accept" in sm[1]
    print(f"  -> contract: spec_accept={checks['contract_spec']} smoke={checks['contract_smoke']}")

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"=== harness-opt-gate: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
