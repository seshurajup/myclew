"""santa_agent — the recurring TOOLKIT + solution knowledge base for Kaggle's annual Santa optimization
competitions (2019→2025). Every Santa is a COMBINATORIAL OPTIMIZATION puzzle, and across the years the winning
meta-recipe is remarkably stable (Kaggle blog, multiple gold write-ups): "use an Integer Linear Program where a
clean formulation exists; otherwise LOCAL SEARCH + SIMULATED ANNEALING, and the single most important design
decision is the shape of the neighborhood (which moves are allowed)." TSP-flavored years add Lin-Kernighan /
2-opt / Or-opt; permutation-puzzle years (Rubik-like) add IDA*/beam search over move sequences.

This agent ports the reusable, offline-testable SOLVERS behind those wins, plus a per-year KB so a Santa entry
starts from the right tool:
  • two_opt(tour, dist)              — TSP local search (the LKH-lite core of the routing years).
  • or_opt(tour, dist, seg)          — move a segment of 1-3 cities (complements 2-opt).
  • simulated_annealing(...)         — general SA for any (state, energy, neighbor) — THE Santa workhorse.
  • beam_search(...)                 — sequence/permutation search (the puzzle years).
  • santa_solutions()               — KB: per-year task + winning technique.
"""
from __future__ import annotations
import math
from .base import BaseAgent
import numpy as np


# ---------------------------------------------------------------- TSP local search (routing years)
def tour_length(tour, dist):
    return float(sum(dist[tour[i], tour[(i + 1) % len(tour)]] for i in range(len(tour))))


def two_opt(tour, dist, max_pass=20):
    """2-opt local search: repeatedly reverse a segment if it shortens the closed tour. Returns improved tour.
    The workhorse of every TSP-flavored Santa (2022 Christmas Card, stolen-sleigh routing)."""
    tour = list(tour); n = len(tour); improved = True; passes = 0
    while improved and passes < max_pass:
        improved = False; passes += 1
        for i in range(n - 1):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                a, b = tour[i], tour[i + 1]; c, d = tour[j], tour[(j + 1) % n]
                if dist[a, c] + dist[b, d] < dist[a, b] + dist[c, d] - 1e-12:
                    tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]; improved = True
    return tour


def or_opt(tour, dist, seg_len=1, max_pass=10):
    """Or-opt: relocate a contiguous segment of `seg_len` cities to a better position. Complements 2-opt."""
    tour = list(tour); n = len(tour); improved = True; passes = 0
    while improved and passes < max_pass:
        improved = False; passes += 1
        for i in range(n):
            seg = tour[i:i + seg_len]
            if len(seg) < seg_len:
                continue
            rest = tour[:i] + tour[i + seg_len:]
            base = tour_length(tour, dist)
            for k in range(len(rest) + 1):
                cand = rest[:k] + seg + rest[k:]
                if tour_length(cand, dist) < base - 1e-9:
                    tour = cand; improved = True; break
            if improved:
                break
    return tour


# ---------------------------------------------------------------- simulated annealing (THE Santa workhorse)
def simulated_annealing(init_state, energy_fn, neighbor_fn, *, steps=5000, t0=1.0, t1=1e-3, seed=0):
    """General SA minimizer. init_state; energy_fn(state)->float; neighbor_fn(state, rng)->new_state. Geometric
    cooling t0→t1. Accepts uphill moves with prob exp(-Δ/T). Returns (best_state, best_energy, history).
    The recurring Santa winner — the art is the neighbor_fn (neighborhood shape)."""
    rng = np.random.RandomState(seed)
    state = init_state; e = energy_fn(state); best, best_e = state, e
    ratio = (t1 / t0) ** (1.0 / max(steps, 1))
    T = t0; hist = [e]
    for _ in range(steps):
        cand = neighbor_fn(state, rng); ce = energy_fn(cand)
        if ce < e or rng.rand() < math.exp(-(ce - e) / max(T, 1e-12)):
            state, e = cand, ce
            if e < best_e:
                best, best_e = state, e
        T *= ratio; hist.append(best_e)
    return best, best_e, hist


# ---------------------------------------------------------------- beam search (permutation-puzzle years)
def beam_search(start, expand_fn, score_fn, is_goal_fn, *, beam=32, max_depth=100):
    """Beam search over sequences/moves. expand_fn(node)->[children]; score_fn(node)->lower-is-better;
    is_goal_fn(node)->bool. Returns the best goal node found (or best-scoring node). Used for the Rubik-like
    permutation puzzles (2023 Polytope, 2024 Perplexity ordering)."""
    frontier = [start]
    best = start
    for _ in range(max_depth):
        cand = []
        for node in frontier:
            if is_goal_fn(node):
                return node
            cand.extend(expand_fn(node))
        if not cand:
            break
        cand.sort(key=score_fn)
        frontier = cand[:beam]
        if score_fn(frontier[0]) < score_fn(best):
            best = frontier[0]
        for node in frontier:
            if is_goal_fn(node):
                return node
    return best


# ---------------------------------------------------------------- per-year solution knowledge base
def santa_solutions():
    """What each Santa was, and the technique that won (top-5 recurring approach). KB for starting an entry."""
    return {
        "2019 Workshop Tour": {"task": "assign families to visit days under occupancy limits + accounting cost",
                               "winner": "Integer/Mixed-ILP (exact) + local search polish", "tool": "MIP/ILP"},
        "2020 Candy Cane": {"task": "multi-armed bandit contest (repeated game vs opponents)",
                            "winner": "adaptive bandit policies (UCB/Thompson) + opponent modeling", "tool": "bandit-RL"},
        "2021 Movie Montage": {"task": "shortest string containing all permutations (superpermutation-like)",
                               "winner": "greedy overlap + local-search repair on the string", "tool": "local-search"},
        "2022 Christmas Card": {"task": "TSP-like path over an image with color-change penalty",
                               "winner": "LKH / 2-opt / Or-opt on a custom distance", "tool": "TSP(2-opt/LKH)"},
        "2023 Polytope Puzzle": {"task": "restore Rubik-like polytope puzzles in fewest moves",
                                "winner": "IDA*/beam search + macro moves; some ML solvers", "tool": "beam/IDA*"},
        "2024 Perplexity Puzzle": {"task": "reorder words to minimize an LLM's perplexity",
                                  "winner": "simulated annealing over permutations, LLM-perplexity energy", "tool": "SA"},
        "2025 (optimization)": {"task": "combinatorial optimization puzzle",
                               "winner": "custom simulated annealing with problem-specific moves", "tool": "SA"},
        "_meta": "ILP where a clean formulation exists; else local search + SA. Neighborhood shape = the lever.",
    }


# ---------------------------------------------------------------- agent
class SantaAgent(BaseAgent):
    name = "santa-agent"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        rng = np.random.RandomState(int(s.get("seed", 0)))
        n = int(s.get("n_cities", 30))
        pts = rng.rand(n, 2)
        dist = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
        init = list(range(n))
        L0 = tour_length(init, dist)
        t = two_opt(init, dist); L2 = tour_length(t, dist)
        # SA on the same TSP with a 2-swap neighborhood, to show the general workhorse
        def energy(tr): return tour_length(tr, dist)
        def neigh(tr, r):
            a, b = r.randint(0, n, 2); c = list(tr); c[a], c[b] = c[b], c[a]; return c
        _, Lsa, _ = simulated_annealing(init, energy, neigh, steps=3000, seed=0)
        kb = santa_solutions()
        msg = (f"santa-agent: TSP({n}) tour {L0:.2f}→2-opt {L2:.2f} ({(1-L2/L0)*100:.0f}% shorter), "
               f"SA {Lsa:.2f}. Toolkit = 2-opt/Or-opt (routing yrs) + simulated_annealing (THE workhorse) + "
               f"beam_search (puzzle yrs) + ILP note. KB covers {len([k for k in kb if not k.startswith('_')])} "
               f"Santas 2019-2025 (meta: ILP if clean else SA, neighborhood shape is the lever)")
        self.log(msg, kind="finding",
                 recommendation="start a Santa entry from santa_solutions()[year]['tool']; for most, custom "
                                "simulated_annealing with a problem-specific neighbor_fn is the winning path")
        return self.done({"tsp_init": L0, "tsp_2opt": L2, "tsp_sa": Lsa, "n_santas": 7}, msg)


_AGENT = SantaAgent()


def run_santa(q, worker):
    return _AGENT.run(q, worker)
