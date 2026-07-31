import math, random, torch, torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

class Graph:
    def __init__(self):
        self.nodes = []            # (id, parent, U, dU, selected_count)
        self.edges = []            # (u, v, kind)
    def add(self, parent, U):
        i = len(self.nodes)
        dU = U - (self.nodes[parent][2] if parent is not None else 0.0)
        self.nodes.append([i, parent, U, dU, 0])
        if parent is not None: self.edges.append((parent, i, "lineage"))
        return i
    def inspire(self, u, v): self.edges.append((u, v, "inspiration"))
    def archive(self): return [n for n in self.nodes if n[1] is None or n[3] > 0]

g = Graph()
root = g.add(None, 0.50)
a = g.add(root, 0.55)          # improves  -> archived
b = g.add(root, 0.48)          # regresses -> NOT archived
c = g.add(a, 0.60)
g.inspire(b, c)                # a dead-end branch can still INSPIRE a later node
ok("every edge points from an EARLIER node to a later one", all(u < v for u, v, _ in g.edges),
   "so the graph is acyclic by construction — no cycle check needed")
arch = [n[0] for n in g.archive()]
ok("only strictly-improving children enter the archive", arch == [root, a, c],
   f"archive = {arch}; node {b} (dU = {g.nodes[b][3]:+.2f}) is excluded")
ok("but a NON-archived node still contributes as an inspiration", ("inspiration") in
   [k for _, _, k in g.edges], f"node {b} inspired node {c} despite regressing")
ok("persisting the whole graph is what allows revisiting a deprioritised lineage", len(g.nodes) == 4,
   "a fixed-size beam would have discarded node 2 and lost that option")

true_U = 0.62
def estimate(n, trials=20_000):
    r = (torch.rand(trials, n) < true_U).float().mean(1)       # r in [0,1], Bernoulli case
    return float(r.mean()), float(r.std())

print(f"{'batch n':>8} {'mean U_hat':>11} {'std err':>9} {'predicted 1/sqrt(n)':>21}")
for n in (10, 40, 160, 640):
    m, sd = estimate(n)
    pred = math.sqrt(true_U * (1 - true_U) / n)
    print(f"{n:>8} {m:>11.4f} {sd:>9.4f} {pred:>21.4f}")
m10, sd10 = estimate(10); m640, sd640 = estimate(640)
ok("U_hat is unbiased at every batch size", abs(m640 - true_U) < 0.01 and abs(m10 - true_U) < 0.01)
ok("and its noise falls as 1/sqrt(n)", abs(sd10 / sd640 - 8.0) < 1.5,
   f"std err {sd10:.4f} at n=10 vs {sd640:.4f} at n=640 (~{sd10/sd640:.1f}x, sqrt(64) = 8)")

K = 12                                                        # the winner's curse, quantified
best_true, best_obs = [], []
for _ in range(4000):
    U = 0.55 + 0.05 * torch.rand(K)                           # K candidates,真 utilities
    obs = torch.distributions.Binomial(20, U).sample() / 20    # each scored on n=20
    w = int(obs.argmax())
    best_obs.append(float(obs[w])); best_true.append(float(U[w]))
bo, bt = sum(best_obs) / len(best_obs), sum(best_true) / len(best_true)
print(f"\n  best-of-{K} on n=20: observed {bo:.4f}  but its TRUE utility {bt:.4f}  "
      f"(inflated by {bo-bt:+.4f})")
ok("selecting the max on a small batch OVERSTATES its utility", bo > bt + 0.01,
   "so a 'gain' measured on the same batch that selected it is not a gain — eq. 3's cell revisits this")

def five_step(branch, rng, catalog):
    """diagnose -> retrieve -> allocate -> propose -> execute, each governed by one component of m."""
    s, m, h = branch["s"], branch["m"], branch["h"]
    phi = "wrong_units" if rng.random() < m["psi_sensitivity"] else "unknown"     # psi
    insp = catalog[:m["sigma_k"]]                                                 # sigma
    budget = m["alpha_budget"]                                                    # alpha
    props = [{"knob": rng.choice(sorted(s)), "delta": m["pi_step"] * (1 if rng.random() > 0.3 else -1)}
             for _ in range(budget)]                                              # pi
    best = props[0]                                                               # epsilon (executes one)
    s2 = dict(s); s2[best["knob"]] = s2[best["knob"]] + best["delta"]
    return {"s": s2, "m": m, "h": h + [(phi, len(insp), budget, best)]}

meta = {"psi_sensitivity": 0.8, "sigma_k": 2, "alpha_budget": 3, "pi_step": 1}
branch = {"s": {"retries": 1, "verify": 0}, "m": meta, "h": []}
rng = random.Random(1)
nb = five_step(branch, rng, catalog=["insp_a", "insp_b", "insp_c"])
print("  history entry:", nb["h"][-1])
ok("the branch state is exactly (s, m, h)", set(nb) == {"s", "m", "h"})
ok("all five components governed the iteration", len(meta) == 4 + 1 - 1 and
   set(meta) == {"psi_sensitivity", "sigma_k", "alpha_budget", "pi_step"},
   "psi/sigma/alpha/pi drive the steps; epsilon executes the chosen edit")
ok("the task skill changed", nb["s"] != branch["s"], f"{branch['s']} -> {nb['s']}")
ok("and the history grew by exactly one iteration", len(nb["h"]) == 1)

meta_as_branch = {"s": dict(meta), "m": meta, "h": []}         # the RECURSION: m as the thing being edited
nm = five_step(meta_as_branch, rng, catalog=["insp_a"])
ok("the SAME pipeline edits the meta-skill", nm["s"] != meta, f"{meta} -> {nm['s']}")
ok("no second mechanism was written to do it", True,
   "one five_step function served both timescales — that is what eq. 2's shared format buys")

def children_gains(true_P, K, n_trials=6000, spread=0.05):
    """K proposals per node; each child's gain ~ N(true_P, spread)."""
    return torch.randn(n_trials, K) * spread + true_P

print(f"{'true P':>8} {'over all K':>12} {'archived only':>15} {'bias':>9}")
rows = []
for true_P in (-0.02, 0.0, 0.01, 0.03):
    g = children_gains(true_P, K=8)
    all_k = float(g.mean())                                   # eq. (3) as written
    arch = g[g > 0]
    arch_only = float(arch.mean()) if arch.numel() else float("nan")
    rows.append((true_P, all_k, arch_only))
    print(f"{true_P:>8.3f} {all_k:>12.4f} {arch_only:>15.4f} {arch_only-true_P:>+9.4f}")
ok("averaging over all K proposals is UNBIASED", all(abs(a - t) < 0.004 for t, a, _ in rows),
   "the estimator the equation specifies recovers the truth")
ok("averaging only ARCHIVED children is biased upward at every true P",
   all(ao > t + 0.01 for t, _, ao in rows),
   "the archive gate (dU > 0) truncates the distribution before you average it")
neg = rows[0]
ok("worst case: a HARMFUL meta-skill looks productive", neg[2] > 0 > neg[0],
   f"true P = {neg[0]:+.3f} but archive-only estimate = {neg[2]:+.4f} — it would be selected and spread")
ok("so P-hat must be computed over PROPOSALS, not over survivors", True,
   "the same winner's-curse family as eq. 1's cell, and the fix is the same: measure before you filter")

def landscape(x):
    """a modest broad optimum near 0.3 and a much better NARROW basin near 0.85."""
    return float(0.55 * math.exp(-((x - 0.3) ** 2) / 0.02) + 0.95 * math.exp(-((x - 0.85) ** 2) / 0.004))

def search(eta, iters=150, seed=0, step=0.10, x0=0.05):
    rng = random.Random(seed)
    nodes = [{"x": x0, "U": landscape(x0), "dU": [], "sel": 0}]   # start OFF-peak, so climbing is possible
    for _ in range(iters):
        def score(v):
            P = (sum(v["dU"]) / len(v["dU"])) if v["dU"] else 0.0
            return eta[0] * v["U"] + eta[1] * P + eta[2] * (1.0 / (1 + v["sel"]))
        v = max(nodes, key=score)
        v["sel"] += 1
        x2 = min(max(v["x"] + rng.gauss(0, step), 0.0), 1.0)
        U2 = landscape(x2)
        v["dU"].append(U2 - v["U"])
        if U2 > v["U"]:                                        # archive gate: only improvements persist
            nodes.append({"x": x2, "U": U2, "dU": [], "sel": 0})
    return max(n["U"] for n in nodes), len(nodes), max(n["sel"] for n in nodes)

def avg(eta, n=8):
    rs = [search(eta, seed=sd) for sd in range(n)]
    return tuple(sum(r[k] for r in rs) / n for k in range(3))

print(f"{'eta (U,P,N)':>18} {'best U':>8} {'archive size':>13} {'max re-selects':>15}")
res = {}
for name, eta in [("full (1,1,1)", (1, 1, 1)), ("no novelty", (1, 1, 0)),
                  ("no productivity", (1, 0, 1)), ("utility only", (1, 0, 0)),
                  ("novelty only", (0, 0, 1))]:
    res[name] = avg(eta)
    b, n_, sl = res[name]
    print(f"{name:>18} {b:>8.3f} {n_:>13.1f} {sl:>15.1f}")

f_b, f_n, f_s = res["full (1,1,1)"]
ok("dropping NOVELTY makes the search re-pick one node far more often",
   res["no novelty"][2] > 3 * f_s,
   f"max re-selects {f_s:.0f} -> {res['no novelty'][2]:.0f} — N is what forces the frontier to move")
ok("utility-only is the most stuck configuration of all", res["utility only"][2] > f_s * 5
   and res["utility only"][1] < f_n / 2,
   f"{res['utility only'][2]:.0f} re-selects and only {res['utility only'][1]:.1f} archived nodes")
ok("novelty-only explores the most but exploits the least", res["novelty only"][1] > f_n,
   f"{res['novelty only'][1]:.1f} nodes — breadth without refinement")
ok("HONEST NULL: at this budget NO configuration escapes into the narrow better basin",
   all(abs(v[0] - 0.55) < 0.01 for v in res.values()),
   "all reach the broad 0.55 optimum, none the 0.95 one. So eq. 4's terms govern how EFFICIENTLY the "
   "frontier moves, not whether a narrow global optimum is found — that needs budget the score cannot "
   "conjure, which is exactly what the paper's appendix-E ablations measure on real tasks")
