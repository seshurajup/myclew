# 2nd place: simulated annealing, iterated local search, and beam search

We would like to thank the organizers for this year's santa competition.

Especially the diversity between top solutions was fascinating.

## Links
- Solution csv: https://www.kaggle.com/datasets/cnumber/santa-2025-kirare
- Solution visualization: https://www.kaggle.com/code/cnumber/santa-2025-2nd-place?scriptVersionId=296899077
- Source code: https://github.com/terry-u16/santa2025

## TL;DR

- Final score: **68.800801749630**
- Key ideas: a high-density initial solution + large neighborhood search

---

## 1. Key Points

Key points are

1. **Strong initial solution**: build an initial solution with high density and a periodic structure
2. **Large neighborhood**: slightly loosen the whole layout, re-place multiple trees, then tighten the whole layout again

The large neighborhood is combined with two search frameworks: Beam Search (BS) and Iterated Local Search (ILS).
Within the large-neighborhood “tighten” phase (and also ILS “long-SA”), we use Simulated Annealing (SA).

---

## 2. Initial solution

### 2.1 Goal

Since \\(score_N = L_N^2/N\\), especially for large \\(N\\) the following structure tends to be strong:

- a high-density periodic pattern near the center
- small local adjustments near the boundary to fill gaps

So we first construct a periodic initial solution **with a fixed height**.

### 2.2 Tiling strategy

Compute \\(L_N\\) from an existing solution and set:

$$
height = L_N - ε \quad (ε = 1e-2)
$$

In other words, we try to “re-pack the same \\(N\\) trees into a square that is 0.01 shorter than the current one”.

Below is an example crop from this tiled initial solution (\\(N=100\\)).

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6624777%2Fe9a69c01e395c4664dc499e78ccc23b4%2Ftile_N100.png?generation=1770711401939741&alt=media)

---

## 3. Standard SA

SA is used for large-neighborhood `tighten` and ILS `long-SA`.

### 3.1 State representation

- The state of \\(N\\) trees is stored as \\(((x_i, y_i, \theta_i))_{i=1}^N\\).

### 3.2 Neighborhood

At each step, pick one random tree \\(i\\) and apply:
Here, \\(T\\) is the current temperature; smaller \\(T\\) means smaller moves.

$$
\begin{cases}
x_i' = x_i + \Delta x \newline
y_i' = y_i + \Delta y \newline
\theta_i' = \theta_i + \Delta \theta
\end{cases}
$$

$$
\Delta x,\Delta y \sim \mathcal{N}(0,\sigma^2),\quad
\Delta \theta \sim \mathcal{N}(0,(\pi\sigma)^2),\quad
\sigma = 30T
$$

- Reject if the proposed center is outside \\([-100, 100]^2\\).
- Reject if it collides with existing trees (collision detection details are in Section 7).

### 3.3 Acceptance rule

The temperature is decreased by geometric interpolation over progress \\(p \in [0,1]\\):

$$
T(p)=T_{\mathrm{high}}^{1-p}T_{\mathrm{low}}^p
$$

Let \\(\Delta = score' - score\\) be the score difference. We use the standard Metropolis rule:

$$
\text{accept if } \Delta \le 0,\quad
\text{otherwise accept with probability } \exp\left(-\frac{\Delta}{T}\right)
$$

---

## 4. Large neighborhood (destroy → loosen → place → tighten)

This is the main mechanism to escape local minima.
One application consists of these 4 steps:

1. **remove**: delete \\(k\\) trees locally
2. **loose**: create space around a target point (push the remaining trees away)
3. **place**: re-place \\(k\\) trees near the target
4. **tighten**: tighten the whole layout with standard SA

Initial state

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6624777%2F50483c180ac1620a4caba95dc056e578%2Fneigh_step1.png?generation=1770711231341258&alt=media)

Remove trees

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6624777%2Fc42acda954b4a924b606e4dc4416ebd9%2Fneigh_step2.png?generation=1770711238311723&alt=media)

Make space

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6624777%2F12692c55657aea513275ef86402fa762%2Fneigh_step3.png?generation=1770711246466452&alt=media)

Re-place trees

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6624777%2Fa7ba086d81d5ef2e4611a96f6ccdfe8c%2Fneigh_step4.png?generation=1770711253103384&alt=media)

Tighten the whole layout again

![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F6624777%2Fc6ef55077193ac98b364f87d95d005b8%2Fneigh_step5.png?generation=1770711259142856&alt=media)

### 4.1 remove: choosing the destruction center and which trees to remove

The number of removed trees \\(k\\) is sampled each time:

$$
k \sim \mathrm{Uniform}\left(1, \ldots, \min\left(\frac{N+1}{2}, 4\right)\right)
$$

To decide “where to destroy”, we sample a target point.
In the current implementation, regions with more disordered tree orientations are sampled with higher probability.
Additionally, with 10% probability we directly sample a corner as well, mixing in boundary-structure destruction.

For the deletion set, we do not simply take “the \\(k\\) nearest trees to the target”.
Instead, we select and remove \\(k\\) trees via a BFS-like randomized walk on a proximity graph.
This makes it easier to create a compact local empty region.

### 4.2 loose: making room

To make space for newly placed trees, we briefly optimize the remaining \\(N-k\\) trees after removal:

1. Slightly expand the current bounding square
2. Disallow transitions that go outside the expanded square
3. Run SA to push each tree away from the target

The objective is:

$$
\max \sum_i \mathrm{dist}\left(\mathrm{centroid}(tree_i), \text{target}\right)
$$

This is not a “final-solution” phase but a “make-space” phase, so we run it relatively briefly and roughly.

### 4.3 place: re-placing trees

Re-place \\(k\\) trees into the opened space.
Naive re-sampling often fails due to collisions, so we do it in two stages:

1. Place 1 tree near the target (with a maximum number of attempts)
2. Using that tree as an initial state, sample many pose candidates via MCMC

The first tree is generated approximately by:

$$
\begin{cases}
x \sim \mathcal{N}(x^\ast, \sigma^2) \newline
y \sim \mathcal{N}(y^\ast, \sigma^2) \newline
\theta \sim \mathrm{Uniform}(0, 2\pi)
\end{cases}
$$

From the resulting candidate pool, we select “\\(k\\) trees that do not collide with each other” using a clique-search algorithm.

### 4.4 tighten: global re-shrinking

Finally, apply standard SA to re-shrink the whole (loosened) layout.
The evaluation is the usual bounding-square side length \\(L_N\\).

---

## 5. BS (Beam Search)

Large neighborhoods are powerful, but highly hit-or-miss.
So we use Beam Search to “keep multiple good states and extend them”.

### 5.1 One turn

- Generate multiple large-neighborhood candidates from each node
- Merge all candidates and sort by score
- Drop overly similar solutions and keep the top \\(K\\)

Pseudocode (\\(K=width, D=depth, F=branchfactor\\)):

```text
beam = dedup(sort(seed_nodes))[:K]

for turn in 1..=D:
  cands = beam * F
  cands = large_neighborhood(cands)
  beam  = dedup(sort(beam + cands))[:K]

return best(beam)
```

### 5.2 `dedup` (maintaining diversity)

If we keep near-duplicates, the beam effectively collapses to width 1.
So we detect similarity using the **bottleneck distance** between two tree sets and drop close ones.

- Treat translation, 90-degree rotation, and mirror reflection as equivalent
- Test “whether there exists a perfect matching within a distance threshold” via a max-flow check

---

## 6. ILS (Iterated Local Search)

If Beam Search “keeps many good states”, ILS “digs deep into one state”.

### 6.1 One iteration

- kick: apply `large_neighborhood` once to the current `ils` state. If it returns `Err`, retry. Also, if the result is “too similar” to `ils` by bottleneck distance, resample.
- hill-climb: repeatedly apply a short `large_neighborhood` to the kicked state; accept improvements; stop after `no_improve_limit` non-improving steps.
- long-SA: run one longer standard SA; **accept only if it improves**.
- Then update `best`, apply the ILS acceptance rule, and reset if needed.

Pseudocode:

```text
best_trees = init
best_score = score(init)
ils_trees  = init
ils_score  = best_score
iter = 0

while True:
  # kick (retry on failure / too-similar)
  while True:
    kicked = large_neighborhood(ils_trees, duration_mul=1.0)
    if kicked is Err:
      continue
    if not similar_by_bottleneck(ils_trees, kicked):
      break

  current_trees = kicked
  current_score = score(current_trees)

  # hill-climb (stop after no_improve_limit non-improvements)
  not_improved_count = 0
  while True:
    cand = large_neighborhood(current_trees, duration_mul=hc_duration_mul)
    if cand is Err:
      continue

    cand_score = score(cand)
    if cand_score < current_score:
      current_trees = cand
      current_score = cand_score
      not_improved_count = 0
    else:
      not_improved_count += 1
      if not_improved_count >= no_improve_limit:
        break

  # long-SA (accept only if improved)
  sa_trees  = long_sa_single(current_trees, duration=N*100ms)
  sa_score  = score(sa_trees)
  if sa_score < current_score:
    current_trees = sa_trees
    current_score = sa_score

  # update best
  if current_score < best_score:
    best_trees = current_trees
    best_score = current_score

  # ILS acceptance
  T_ils = cooler_temperature(iter)
  Δ = current_score - ils_score
  accepted = (Δ <= 0) or (rand() < exp(-Δ / T_ils))
  if accepted:
    ils_trees = current_trees
    ils_score = current_score

  # reset (rules differ for constant/cooling)
  if should_reset(ils_score, best_score, iter):
    ils_trees = best_trees
    ils_score = best_score

  iter += 1
```

### 6.2 Acceptance and reset

The acceptance rule is SA-like: worsening moves are allowed depending on temperature \\(T_{\mathrm{ils}}\\).

$$
\Delta = score_{\mathrm{current}} - score_{\mathrm{ILS}}
$$

$$
\text{accept if } \Delta \le 0,\quad
\text{otherwise accept with probability } \exp\left(-\frac{\Delta}{T_{\mathrm{ils}}}\right)
$$

- `constant`: keep \\(T_{\mathrm{ils}}\\) constant. When \\(\Delta_{\mathrm{reset}} = score_{\mathrm{ILS}} - score_{\mathrm{best}}\ (>0)\\), reset to `best` with the probability below (if \\(\Delta_{\mathrm{reset}} \le 0\\), do not reset).

$$
p_{\mathrm{reset}}
= \min\left(1,\ prob_\text{reset}\cdot\left(\frac{\Delta_{\mathrm{reset}}}{T_{\mathrm{ils}}}\right)^2\right)
$$

- `cooling`: with \\(S = cooling\_steps\\), cool \\(T_{\mathrm{ils}}\\) periodically by:

$$
T_{\mathrm{ils}}(iter) = T_{\mathrm{high}} \left(\frac{T_{\mathrm{low}}}{T_{\mathrm{high}}}\right)^{\frac{iter \bmod S}{S-1}}
$$

Additionally, if \\(reset\_solution = true\\), reset to `best` at the end of each period where \\((iter + 1) \bmod S = 0\\).

---

## 7. Collision detection

Since collision checks are often the main bottleneck inside SA, the implementation uses multi-stage AABB pruning and only performs exact segment-intersection tests as the final step.

### 7.1 Data representation

- Treat one tree as 12 line segments, and store each segment’s \\((x_0, y_0, x_1, y_1)\\) in an SoA layout of `f64`.
- Precompute and store the tree-level AABB (`min_x, max_x, min_y, max_y`) and also each-edge AABB.
- For the tree array as well, store AABBs in chunks so we can test 4 trees at once with AVX2.

### 7.2 Three-stage pruning

Checks proceed in this order:

1. **tree AABB vs tree AABB**  
   First compare the moved tree’s AABB with the AABBs of existing trees, and discard all non-overlapping trees at once.
2. **edge AABB vs tree AABB**  
   For remaining trees, check whether each edge-AABB of the moved tree overlaps the other tree’s AABB.
3. **edge AABB vs edge AABB**  
   For further remaining candidates, compare edge AABBs to keep only edge pairs that could intersect.

Candidates eliminated in these stages never reach the exact test, so most cases are filtered out cheaply.

### 7.3 Final exact test

Only for the remaining edge pairs, run a line-segment intersection test.
We use the standard cross-product sign conditions:

$$
\mathrm{cross}(\overrightarrow{ab}, \overrightarrow{ac}) \cdot \mathrm{cross}(\overrightarrow{ab}, \overrightarrow{ad}) \le 0,\quad
\mathrm{cross}(\overrightarrow{cd}, \overrightarrow{ca}) \cdot \mathrm{cross}(\overrightarrow{cd}, \overrightarrow{cb}) \le 0
$$

This is also evaluated in batches of 4 using AVX2.
If any intersection is found, we treat it as a collision and reject the move.

---

## 8. Others

- Within the large neighborhood, `loose/tighten` may be run with a constraint that the layout height never exceeds the initial-solution height.
- We also created initial solutions for \\(N\pm 1\\) by adding/removing trees from an \\(N\\) solution and used them as BS/ILS seeds.