"""optimization_pack_test — verifier for the black-box/combinatorial optimizers on a small TSP (offline).

A random 12-city TSP: local search, GA, and island-harness must each beat a random tour by a wide margin,
and the oracle harness must show cache hits (memoization working)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import optimization_pack as O


def _run():
    print("=== OPTIMIZATION PACK VERIFIER (12-city TSP) ===")
    rng = np.random.RandomState(0); checks = {}
    cities = rng.rand(12, 2)
    D = np.linalg.norm(cities[:, None] - cities[None, :], axis=2)
    n = len(D)
    def tour_len(p): return float(sum(D[p[i], p[(i + 1) % n]] for i in range(n)))
    # random baseline (mean of many random tours)
    rand_len = np.mean([tour_len(rng.permutation(n)) for _ in range(200)])

    # local search
    best, best_s = O.local_search(np.arange(n), tour_len, maximize=False, iters=4000)
    checks["localsearch_beats_random"] = best_s < rand_len * 0.72
    checks["localsearch_valid_perm"] = sorted(best.tolist()) == list(range(n))
    print(f"  -> local-search {best_s:.3f} vs random {rand_len:.3f}")

    # GA
    pop = [rng.permutation(n) for _ in range(30)]
    gbest, gs = O.evolve(pop, tour_len, maximize=False, generations=150)
    checks["ga_beats_random"] = gs < rand_len * 0.7
    checks["ga_valid_perm"] = sorted(gbest.tolist()) == list(range(n))
    print(f"  -> GA {gs:.3f}")

    # island harness + cache
    ibest, iss, info = O.island_search(lambda sd: np.random.RandomState(sd).permutation(n), tour_len,
                                       maximize=False, n_islands=4, iters=1500)
    checks["island_beats_random"] = iss < rand_len * 0.72
    checks["oracle_cache_hits"] = info["cache_hits"] > 0        # memoization actually triggered
    print(f"  -> island {iss:.3f}, oracle_calls={info['oracle_calls']} cache_hits={info['cache_hits']}")

    # agent contracts
    st, d, to, msg = O.run_localsearch({"spec": {"dist_matrix": D.tolist(), "iters": 2000}}, "t")
    checks["localsearch_agent"] = st == "done" and d["best_objective"] < rand_len * 0.7
    st, d, to, msg = O.run_oracle({"spec": {"dist_matrix": D.tolist(), "islands": 3, "iters": 500}}, "t")
    checks["oracle_agent"] = st == "done" and "cache_hits" in d

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== optimization-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
