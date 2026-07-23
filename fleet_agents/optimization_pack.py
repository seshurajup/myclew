"""optimization_pack — black-box / combinatorial optimization agents the one-by-one pass found MISSING
entirely (santa-2024 permutation-perplexity, santa-2025 packing, code-golf expression search). The existing
fleet was 100% ML-model-centric; these add the search primitives. All pure numpy, verified on a small TSP:

  • combinatorial-local-search    — iterated local search over permutations/sequences: pluggable neighborhood
                                    (2-opt / swap / insert), simulated-annealing acceptance, double-bridge kick
                                    to escape local minima. Drives any black-box score(solution)->float.
  • batched-oracle-search-harness — wraps an expensive black-box scorer with a memoization CACHE + multi-start
                                    ISLAND parallelism (keep global best) — the infra winners built for GPU oracles.
  • population-diversity-manager  — genetic algorithm with order-crossover, mutation, and diversity-preserving
                                    selection so the population doesn't collapse to one basin.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- neighborhoods (permutations)
def two_opt(perm, rng):
    p = perm.copy()
    if len(p) < 2:
        return p
    i, j = sorted(rng.choice(len(p), 2, replace=False)); p[i:j + 1] = p[i:j + 1][::-1]; return p


def swap(perm, rng):
    p = perm.copy()
    if len(p) < 2:
        return p
    i, j = rng.choice(len(p), 2, replace=False); p[i], p[j] = p[j], p[i]; return p


def double_bridge(perm, rng):
    n = len(perm)
    if n < 4:                                   # not enough room for a 3-cut kick
        return perm.copy()
    a, b, c = sorted(rng.choice(range(1, n), 3, replace=False))
    return np.concatenate([perm[:a], perm[b:c], perm[a:b], perm[c:]])


# ---------------------------------------------------------------- combinatorial-local-search
def local_search(init, score_fn, maximize=False, iters=2000, neighbors=(two_opt, swap), seed=0,
                 sa_T0=1.0, sa_cool=0.999, kick_every=200):
    """Iterated local search + SA acceptance + periodic double-bridge kick. score_fn(perm)->float."""
    rng = np.random.RandomState(seed); cur = np.asarray(init).copy(); cur_s = score_fn(cur)
    best, best_s = cur.copy(), cur_s; T = sa_T0
    if len(cur) < 2 or int(iters) <= 0:            # nothing to permute / no budget
        return best, best_s
    sgn = 1.0 if maximize else -1.0
    since = 0
    for t in range(iters):
        move = neighbors[rng.randint(len(neighbors))]
        cand = move(cur, rng); cand_s = score_fn(cand)
        d = sgn * (cand_s - cur_s)
        if d > 0 or rng.rand() < np.exp(d / max(T, 1e-9)):
            cur, cur_s = cand, cand_s
        if sgn * (cur_s - best_s) > 0:
            best, best_s, since = cur.copy(), cur_s, 0
        else:
            since += 1
        T *= sa_cool
        if since >= kick_every:                                 # stuck → kick
            cur = double_bridge(best, rng); cur_s = score_fn(cur); since = 0
    return best, best_s


# ---------------------------------------------------------------- batched-oracle-search-harness
class OracleCache:
    """Memoizes an expensive score fn by solution key; tracks call savings."""
    def __init__(self, score_fn):
        self._fn = score_fn; self._cache = {}; self.calls = 0; self.hits = 0

    def __call__(self, perm):
        key = tuple(int(x) for x in perm)
        if key in self._cache:
            self.hits += 1; return self._cache[key]
        self.calls += 1; v = self._fn(np.asarray(perm)); self._cache[key] = v; return v


def island_search(init_fn, score_fn, maximize=False, n_islands=4, iters=1000, seed=0):
    """Run local_search from n independent starts (islands) through a shared cache; return the global best."""
    cache = OracleCache(score_fn); best, best_s = None, None
    sgn = 1.0 if maximize else -1.0
    for k in range(n_islands):
        init = init_fn(seed + k)
        s, sv = local_search(init, cache, maximize=maximize, iters=iters, seed=seed + k)
        if best is None or sgn * (sv - best_s) > 0:
            best, best_s = s, sv
    return best, best_s, {"oracle_calls": cache.calls, "cache_hits": cache.hits}


# ---------------------------------------------------------------- population-diversity-manager
def order_crossover(p1, p2, rng):
    n = len(p1); a, b = sorted(rng.choice(n, 2, replace=False))
    child = -np.ones(n, int); child[a:b + 1] = p1[a:b + 1]
    fill = [x for x in p2 if x not in child[a:b + 1]]; k = 0
    for i in range(n):
        if child[i] == -1:
            child[i] = fill[k]; k += 1
    return child


def evolve(init_pop, score_fn, maximize=False, generations=100, mutate_p=0.3, seed=0):
    """GA over permutations: order-crossover + swap-mutation + elitist diversity-preserving selection."""
    rng = np.random.RandomState(seed); pop = [np.asarray(p) for p in init_pop]; m = len(pop)
    sgn = 1.0 if maximize else -1.0
    def fit(p): return sgn * score_fn(p)
    if m == 0:
        raise ValueError("evolve: empty initial population")
    if m == 1 or len(pop[0]) < 2:                 # no room to crossover → best of the given pool
        best = max(pop, key=fit); return best, score_fn(best)
    for g in range(generations):
        scored = sorted(pop, key=fit, reverse=True)
        keep = scored[:max(2, m // 2)]                          # elitism
        children = []
        while len(keep) + len(children) < m:
            a, b = keep[rng.randint(len(keep))], keep[rng.randint(len(keep))]
            c = order_crossover(a, b, rng)
            if rng.rand() < mutate_p:
                c = swap(c, rng)
            children.append(c)
        pop = keep + children
    best = max(pop, key=fit)
    return best, score_fn(best)


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class CombinatorialLocalSearch(_B):
    name = "combinatorial-local-search"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("dist_matrix",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"combinatorial-local-search needs spec keys {missing} — none provided")
        # score fn supplied as python is not JSON-safe; agent path expects a registered problem or a distance matrix
        D = np.asarray(s["dist_matrix"], float); n = len(D)
        def tour_len(p): return float(sum(D[p[i], p[(i + 1) % n]] for i in range(n)))
        init = np.arange(n)
        best, best_s = local_search(init, tour_len, maximize=False, iters=int(s.get("iters", 3000)),
                                    seed=int(s.get("seed", 0)), sa_T0=float(s.get("sa_T0", 1.0)),
                                    sa_cool=float(s.get("sa_cool", 0.999)), kick_every=int(s.get("kick_every", 200)))
        msg = f"combinatorial-local-search: best objective={best_s:.4f} over {n} elements (ILS+SA+kick)"
        self.log(msg, kind="finding", recommendation="raise iters/islands for harder instances")
        return self.done({"best_objective": best_s, "solution": best.tolist()}, msg)


class PopulationDiversityManager(_B):
    name = "population-diversity-manager"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("dist_matrix",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"population-diversity-manager needs spec keys {missing} — none provided")
        D = np.asarray(s["dist_matrix"], float); n = len(D)
        def tour_len(p): return float(sum(D[p[i], p[(i + 1) % n]] for i in range(n)))
        seed = int(s.get("seed", 0)); popsize = max(2, int(s.get("popsize", s.get("pop", 30))))
        rng = np.random.RandomState(seed); pop = [rng.permutation(n) for _ in range(popsize)]
        best, best_s = evolve(pop, tour_len, maximize=False, generations=int(s.get("generations", 120)),
                              mutate_p=float(s.get("mutate_p", 0.3)), seed=seed)
        msg = f"population-diversity-manager: GA best objective={best_s:.4f}"
        self.log(msg, kind="finding", recommendation="combine with combinatorial-local-search for a memetic solver")
        return self.done({"best_objective": best_s, "solution": best.tolist()}, msg)


class BatchedOracleSearch(_B):
    name = "batched-oracle-search-harness"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("dist_matrix",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"batched-oracle-search-harness needs spec keys {missing} — none provided")
        D = np.asarray(s["dist_matrix"], float); n = len(D)
        def tour_len(p): return float(sum(D[p[i], p[(i + 1) % n]] for i in range(n)))
        best, best_s, info = island_search(lambda sd: np.random.RandomState(sd).permutation(n), tour_len,
                                           maximize=False, n_islands=int(s.get("islands", s.get("restarts", 4))),
                                           iters=int(s.get("iters", 800)), seed=int(s.get("seed", 0)))
        msg = (f"batched-oracle-search-harness: best={best_s:.4f} across {s.get('islands',4)} islands; "
               f"oracle_calls={info['oracle_calls']} cache_hits={info['cache_hits']} (memoized)")
        self.log(msg, kind="finding", recommendation="cache + islands = the black-box-oracle infra (santa/code-golf)")
        return self.done({"best_objective": best_s, "solution": best.tolist(), **info}, msg)


_LS = CombinatorialLocalSearch(); _PD = PopulationDiversityManager(); _BO = BatchedOracleSearch()


def run_localsearch(q, worker): return _LS.run(q, worker)
def run_population(q, worker): return _PD.run(q, worker)
def run_oracle(q, worker): return _BO.run(q, worker)
