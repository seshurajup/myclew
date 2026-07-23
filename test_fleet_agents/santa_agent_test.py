"""santa_agent_test — data-wise verifier for the Santa optimization toolkit.

Core properties:
  1. two_opt shortens (or keeps) a random TSP tour and never lengthens it.
  2. or_opt does not lengthen a tour.
  3. simulated_annealing finds a near-optimal solution on a tiny known problem (sort-by-SA), monotone best.
  4. beam_search reaches a goal on a toy search.
  5. santa_solutions KB covers the last 5+ years with task+winner+tool.
  6. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import santa_agent as S


def _run():
    print("=== SANTA-AGENT VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}

    # 1-2. TSP local search
    n = 25; pts = rng.rand(n, 2)
    dist = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    init = list(range(n)); L0 = S.tour_length(init, dist)
    t2 = S.two_opt(init, dist); L2 = S.tour_length(t2, dist)
    checks["two_opt_shortens"] = L2 <= L0 and L2 < 0.95 * L0
    checks["two_opt_valid_perm"] = sorted(t2) == list(range(n))
    to = S.or_opt(t2, dist, seg_len=1); checks["or_opt_no_worse"] = S.tour_length(to, dist) <= L2 + 1e-6
    print(f"  -> TSP {L0:.2f} → 2-opt {L2:.2f} ({(1-L2/L0)*100:.0f}% shorter)")

    # 3. SA on a toy: minimize sum of |x_i - target_i| by swapping a permutation into sorted order
    target = np.arange(10)
    def energy(state): return float(np.abs(np.array(state) - target).sum())
    def neigh(state, r):
        a, b = r.randint(0, 10, 2); c = list(state); c[a], c[b] = c[b], c[a]; return c
    start = list(np.random.RandomState(1).permutation(10))
    best, be, hist = S.simulated_annealing(start, energy, neigh, steps=4000, seed=0)
    checks["sa_finds_optimum"] = be < energy(start) and be <= 4          # near-sorted
    checks["sa_best_monotone"] = all(hist[i + 1] <= hist[i] + 1e-9 for i in range(len(hist) - 1))
    print(f"  -> SA energy {energy(start):.0f} → {be:.0f}")

    # 4. beam search toy: climb from 0 to >=20 by +1/+3 steps, goal=exactly 20, minimize steps
    def expand(node): return [(node[0] + 1, node[1] + [1]), (node[0] + 3, node[1] + [3])]
    def score(node): return abs(20 - node[0]) + len(node[1]) * 0.01
    def goal(node): return node[0] == 20
    res = S.beam_search((0, []), expand, score, goal, beam=8, max_depth=30)
    checks["beam_reaches_goal"] = res[0] == 20
    print(f"  -> beam search reached {res[0]} in {len(res[1])} steps")

    # 5. KB
    kb = S.santa_solutions()
    years = [k for k in kb if not k.startswith("_")]
    checks["kb_covers_5plus_years"] = len(years) >= 5
    checks["kb_entries_complete"] = all(all(f in kb[y] for f in ("task", "winner", "tool")) for y in years)
    checks["kb_has_sa_and_tsp"] = any("SA" in kb[y]["tool"] for y in years) and any("2-opt" in kb[y]["tool"] for y in years)
    print(f"  -> KB covers {len(years)} Santas: {years}")

    # 6. agent
    st, dta, to2, msg = S.run_santa({"spec": {"n_cities": 30}}, "t")
    checks["agent_done"] = st == "done" and dta["tsp_2opt"] < dta["tsp_init"]

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== santa-agent: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
