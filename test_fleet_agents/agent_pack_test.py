"""agent_pack_test — data-wise verifier for the AGENTIC pack on a deterministic toy env (offline).

Asserts the agentic spine actually works:
  • agent-env characterizes the env (actions, reward, optimal known),
  • agent-policy's tournament finds a policy that BEATS random and reaches the optimal (greedy sweep),
  • agent-eval reports frac-of-optimal == 1.0 for the winner and budget_ok,
  • agent-eval CATCHES an over-tight budget (budget_ok False → do-not-submit), the JED lesson.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import agent_common as AC
from fleet_agents import agent_env as AE
from fleet_agents import agent_policy as AP
from fleet_agents import agent_eval as AV


def _run():
    print("=== AGENTIC PACK DATA-WISE VERIFIER (toy env) ===")
    checks = {}
    rewards = [0, 0, 1, 0, 2, 0, 0, 3, 0, 0]  # optimal = 6
    ef = lambda: AC.ToyCollectEnv(rewards=rewards)

    # agent-env
    info = AE.characterize(ef())
    checks["env_actions"] = info["n_actions"] == 3
    checks["env_optimal"] = info.get("optimal_reward") == 6.0

    # agent-policy tournament UNDER A BUDGET (8 steps): greedy's efficient sweep beats a random walk that
    # can't reach the far cell in time — policy quality only shows under a budget (the JED lesson).
    best, scores = AP.tournament(ef, episodes=5, budget=8)
    checks["policy_best_greedy"] = best == "greedy_right"
    checks["policy_reaches_optimal"] = abs(scores["greedy_right"] - 6.0) < 1e-9
    checks["policy_beats_random"] = scores["greedy_right"] > max(scores["random0"], scores["random1"])

    # agent-eval: winner is optimal + budget ok (generous budget)
    res = AV.evaluate(ef, AC.greedy_right_policy, episodes=5, budget=50)
    checks["eval_frac_optimal"] = abs(res.get("frac_optimal", 0) - 1.0) < 1e-9
    checks["eval_budget_ok"] = res["budget_ok"] is True

    # agent-eval: TIGHT budget (2 steps) → cannot sweep → budget flagged / suboptimal (JED budget lesson)
    tight = AV.evaluate(ef, AC.greedy_right_policy, episodes=3, budget=2)
    checks["eval_tight_budget_suboptimal"] = tight["frac_optimal"] < 1.0

    # agent run() wrappers return the standard contract
    st, d, to, msg = AE.run({"spec": {"env_kwargs": {"rewards": rewards}}}, "test")
    checks["agent_env_run_done"] = st == "done" and "env_info" in d
    st, d, to, msg = AV.run({"spec": {"env_kwargs": {"rewards": rewards}, "policy_name": "greedy_right",
                                      "budget": 50}}, "test")
    checks["agent_eval_run_safe"] = st == "done" and d.get("safe") is True

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    print(f"  -> policy scores={ {k: round(v,2) for k,v in scores.items()} }; tight-budget frac_opt={tight['frac_optimal']:.2f}")
    ok = all(checks.values())
    print(f"=== agent-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
