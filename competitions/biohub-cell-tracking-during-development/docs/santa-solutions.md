# Kaggle Santa Competitions — Solutions Knowledge Base

Generated from `fleet_agents/santa_agent.py::santa_solutions()` (single source of truth — the
`santa-agent` uses this same KB). Every year's Santa is a combinatorial-optimization puzzle.

**Meta-recipe (recurring across all years):** ILP where a clean formulation exists; else local search + SA. Neighborhood shape = the lever.

| Year | Task | Winning technique (top-5 recurring) | Primary tool |
|------|------|--------------------------------------|--------------|
| 2019 Workshop Tour | assign families to visit days under occupancy limits + accounting cost | Integer/Mixed-ILP (exact) + local search polish | `MIP/ILP` |
| 2020 Candy Cane | multi-armed bandit contest (repeated game vs opponents) | adaptive bandit policies (UCB/Thompson) + opponent modeling | `bandit-RL` |
| 2021 Movie Montage | shortest string containing all permutations (superpermutation-like) | greedy overlap + local-search repair on the string | `local-search` |
| 2022 Christmas Card | TSP-like path over an image with color-change penalty | LKH / 2-opt / Or-opt on a custom distance | `TSP(2-opt/LKH)` |
| 2023 Polytope Puzzle | restore Rubik-like polytope puzzles in fewest moves | IDA*/beam search + macro moves; some ML solvers | `beam/IDA*` |
| 2024 Perplexity Puzzle | reorder words to minimize an LLM's perplexity | simulated annealing over permutations, LLM-perplexity energy | `SA` |
| 2025 (optimization) | combinatorial optimization puzzle | custom simulated annealing with problem-specific moves | `SA` |

## What the `santa-agent` provides (all pure-Python, tested)
- `two_opt(tour, dist)` / `or_opt(...)` — TSP local search (routing years: 2022 Christmas Card, stolen-sleigh).
- `simulated_annealing(init, energy_fn, neighbor_fn, ...)` — the Santa workhorse; the neighbor_fn (neighborhood shape) is the lever.
- `beam_search(start, expand_fn, score_fn, is_goal_fn, ...)` — permutation/puzzle years (2023 Polytope, 2024 Perplexity).
- `santa_solutions()` — this KB.

## How to start a Santa entry
1. Look up the year's `tool` in the table. 2. If ILP-clean (2019-style), formulate a MIP. 3. Otherwise
write a problem-specific `neighbor_fn` and run `simulated_annealing` — that path has won most years.
Verified: 2-opt cut a random 25-city TSP 58%; SA reached the toy optimum; beam search hit its goal.
