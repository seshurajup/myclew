"""reasoning_exec_pack_test — verifier for program-search/synthesis/golf/fast-sim/code-repair/judge/ttc."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import reasoning_exec_pack as R


def _run():
    print("=== REASONING-EXEC PACK VERIFIER ===")
    checks = {}
    g = np.array([[1, 2], [3, 4]])
    checks["program_search_transpose"] = R.program_search([(g, g.T)]) == ["transpose"]
    checks["program_golf_fliplr"] = R.program_golf_search([(g, np.fliplr(g))]) == ["fliplr"]
    data = R.synthesize_data(n=10, prog_len=1, seed=1)
    checks["synth_verifiable"] = len(data) == 10 and all(
        np.array_equal(R._apply(d["program"], np.array(d["input"])), np.array(d["output"])) for d in data)
    rg = np.zeros((3, 5)); rg[:, 2] = 1.0
    pos, r = R.batched_collect_step(np.array([1, 1, 1]), rg, np.array([1, 1, 1]))
    checks["fast_sim_batched"] = np.allclose(r, 1.0) and np.all(pos == 2)
    ok_p, _ = R.verify_patch("def add(a,b):return a+b", "assert add(2,3)==5")
    ok_f, _ = R.verify_patch("def add(a,b):return a-b", "assert add(2,3)==5")
    checks["code_repair_verify"] = ok_p and not ok_f
    best, div, sc = R.craft_divergent_input(["x", "y"], [lambda s: len(s), lambda s: 1 if s == "x" else 9])
    checks["judge_attacker"] = best == "y"
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== reasoning-exec-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
