"""skill_optimizer_test — data-wise verifier for the SkillOpt SGD-analogy skill optimizer.

Core properties:
  1. rank_and_select returns the top-k by score.
  2. gate_accept only accepts a real improvement (held-out gate).
  3. optimize() monotonically improves the held-out value and converges toward the optimum (the SGD analogy).
  4. reflect_patch_llm works offline via the dummy provider (returns candidate rewrites).
  5. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import skill_optimizer as SO


def _run():
    print("=== SKILL-OPTIMIZER VERIFIER ===")
    import random
    checks = {}

    # 1. rank_and_select
    checks["rank_topk"] = SO.rank_and_select(["a", "b", "c"], [0.1, 0.9, 0.5], k=2) == ["b", "c"]
    # 2. gate
    checks["gate_accepts_better"] = SO.gate_accept(0.0, 0.1, 0.0) is True
    checks["gate_rejects_worse"] = SO.gate_accept(0.0, -0.1, 0.0) is False
    checks["gate_min_delta"] = SO.gate_accept(0.0, 0.005, 0.01) is False

    # 3. optimize converges + monotone held-out value
    rnd = random.Random(0); target = 7.0
    def score(x): return -abs(x - target)
    def val(x): return -abs(x - target)
    def propose(x, tr):
        step = 0.5 * (1 + abs(tr))
        return [x + rnd.uniform(-step, step) for _ in range(4)]
    best, bv, hist = SO.optimize(0.0, propose, score, val, epochs=60, patience=12)
    monotone = all(hist[i + 1] >= hist[i] - 1e-9 for i in range(len(hist) - 1))
    print(f"  -> converged 0.0->{best:.3f} (target {target}); val {hist[0]:.2f}->{bv:.2f}; monotone={monotone}")
    checks["converges"] = abs(best - target) < 0.5
    checks["held_out_monotone"] = monotone

    # 4. llm reflect offline (dummy provider echoes)
    cands = SO.reflect_patch_llm("be concise", "failed on task 3", model="dummy/echo", n=3)
    checks["reflect_offline"] = len(cands) == 3 and all(isinstance(c, str) for c in cands)

    # 4b. Aggregate — failure patches take priority, edits deduped
    agg = SO.aggregate_patches([{"edits": ["b", "c"], "source": "success"},
                                {"edits": ["a", "b"], "source": "failure"}])
    checks["aggregate_failure_first"] = agg["edits"][0] == "a" and agg["edits"] == ["a", "b", "c"]
    checks["aggregate_counts"] = agg["n_failure_sources"] == 1 and agg["n_success_sources"] == 1

    # 4c. 3-way best-tracking gate
    checks["gate_new_best"] = SO.evaluate_gate(0.9, 0.5, 0.8, 0, 5)[0] == "accept_new_best"
    checks["gate_accept_local"] = SO.evaluate_gate(0.7, 0.5, 0.9, 3, 5)[0] == "accept"
    checks["gate_reject"] = SO.evaluate_gate(0.4, 0.5, 0.9, 3, 5)[0] == "reject"

    # 4d. mine retry chains → labels
    tasks = SO.mine_retry_chains([{"intent": "fix bug", "feedback": ["neg"]},
                                  {"intent": "add test", "feedback": ["pos"]},
                                  {"intent": "unclear", "feedback": []}])
    labels = {t["intent"]: t["label"] for t in tasks}
    checks["mine_labels"] = labels == {"fix bug": "fail", "add test": "success", "unclear": "unknown"}

    # 4e. sleep cycle adopts an improving candidate through the gate (offline)
    def sc_score(x): return -abs(x - 7.0)
    def sc_propose(x, failing): return [7.0, x - 1]        # 7.0 is the improving candidate
    adopted, action, rep = SO.sleep_cycle(0.0, [{"intent": "hit target", "feedback": ["neg"]}],
                                          sc_propose, sc_score)
    print(f"  -> sleep_cycle: mined {rep['n_failing']} failing, action={action}, adopted={adopted}")
    checks["sleep_adopts"] = action in ("accept", "accept_new_best") and abs(adopted - 7.0) < 1e-9
    checks["sleep_mined_failure"] = rep["n_failing"] == 1

    # 5. agent contract
    st, dta, to, msg = SO.run_skillopt({"spec": {"target": 7.0, "epochs": 40}}, "t")
    checks["agent_done"] = st == "done" and dta["err"] < 0.6

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== skill-optimizer: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
