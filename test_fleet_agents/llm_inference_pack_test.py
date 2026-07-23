"""llm_inference_pack_test — verifier for the LLM inference-orchestration agents (offline, no model)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import llm_inference_pack as L


def _run():
    print("=== LLM INFERENCE PACK VERIFIER ===")
    checks = {}

    # self-consistency: majority answer wins; a single high-confidence wrong outlier doesn't flip it
    ans = ["42", "42", "42", "7"]; conf = [0.5, 0.5, 0.5, 0.99]
    a, share = L.aggregate_answers(ans, conf)
    checks["self_consistency_majority"] = a == "42"  # 1.5 vote vs 0.99

    # consensus-early-stop: 4 agree with agree_k=4 → stop; uncatchable lead → stop; else continue
    checks["stop_on_agree"] = L.should_stop({"a": 4, "b": 1}, remaining=5, agree_k=4) is True
    checks["stop_uncatchable"] = L.should_stop({"a": 5, "b": 1}, remaining=2, agree_k=99) is True
    checks["continue_close"] = L.should_stop({"a": 3, "b": 2}, remaining=10, agree_k=8) is False

    # risk-abstain: confident → submit; uncertain with heavy penalty → skip
    sub_hi, ev_hi = L.decide_submit(0.9, reward_correct=1, penalty_wrong=1)
    sub_lo, ev_lo = L.decide_submit(0.4, reward_correct=1, penalty_wrong=3)  # EV = 0.4 - 0.6*3 < 0
    checks["abstain_submit_confident"] = sub_hi is True
    checks["abstain_skip_risky"] = sub_lo is False

    # budget scheduler: even share, harder problem gets more, clipped to remaining
    b_even = L.allocate_budget(1000, 10, difficulty=1.0)
    b_hard = L.allocate_budget(1000, 10, difficulty=2.0)
    checks["budget_even"] = 30 <= b_even <= 300
    checks["budget_harder_more"] = b_hard > b_even

    # sample-pool-simulator: a pool where problems are mostly solvable → higher k → higher est accuracy
    rng = np.random.RandomState(0)
    pool = [rng.rand(20) < p for p in rng.uniform(0.3, 0.8, 30)]  # per-problem sample correctness
    acc_k1 = L.simulate_config(pool, k=1); acc_k9 = L.simulate_config(pool, k=9)
    checks["poolsim_more_samples_help"] = acc_k9 >= acc_k1
    print(f"  -> pool-sim acc k=1 {acc_k1:.3f} → k=9 {acc_k9:.3f}")

    # agent contracts
    st, d, to, msg = L.run_abstain({"spec": {"confidence": 0.4, "penalty_wrong": 3}}, "t")
    checks["abstain_agent"] = st == "done" and d["submit"] is False
    st, d, to, msg = L.run_selfconsistency({"spec": {"answers": ans, "confidences": conf}}, "t")
    checks["sc_agent"] = st == "done" and d["answer"] == "42"

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== llm-inference-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
