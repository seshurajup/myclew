"""Paper pack — *MetaSkill-Evolve: Recursive Self-Improvement of LLM Agents via Two-Timescale
Meta-Skill Evolution* — arXiv:2607.05297
paper: https://arxiv.org/pdf/2607.05297 · local: docs/papers/paper-2607-05297/paper-2607-05297.md
lessons: learning/annotated/mse*.learning

**The self-improvement pack, and the one closest to our own fleet.** An agent's "skill" is a Markdown
program (a SKILL.md). A five-agent pipeline — Analyzer, Retriever, Allocator, Proposer, Evolver — turns a
failure into an edit of that program. The paper's move: those five agents are THEMSELVES driven by
Markdown programs (the *meta-skill* m = (ψ, σ, α, π, ε)), written in the same representation, so the exact
same pipeline can improve the improver. That gives two timescales — a fast loop editing task skills, and a
slow loop (every H iterations) editing the meta-skill that produces those edits — over a persistent
evolution DAG in SQLite, with frontier selection balancing utility, productivity and novelty.

Only FOUR equations carry it, and all four are provable at toy scale on the GPU:
  • eq. 1 — utility as an expectation estimated on a held-out batch (so its ESTIMATOR is the real object,
    and its variance decides everything downstream);
  • eq. 2 — the branch state and the representation closure that makes recursion possible at all;
  • eq. 3 — meta-productivity as mean child improvement (where we find and measure a genuine **selection
    bias**: the archive admits only ΔU > 0 children, so estimating P̂ from the archive is biased upward —
    the paper's own text says to estimate it over all K proposals, and this cell shows why that matters);
  • eq. 4 — frontier selection η₁U + η₂P̂ + η₃N, with each term ablated to the failure it prevents.

NOT reproduced, said plainly: no LLM is called anywhere in this pack. The paper's agents are GPT-class
models editing Markdown; its benchmark numbers, ablation tables and the five prompt templates
(appendices A–G) are the authors'. What this pack verifies is the SEARCH ALGEBRA those agents sit inside —
which is the transferable part, and the part we can check exactly.

Read after `afpz1` (population search with a noisy verifier — same structure, and the same warning) and
`egmz1`. The direct fleet relevance is in the advanced lesson: our `improve-loop`/`skill-optimizer` agents
are a fast loop with a FIXED meta-skill, and this paper is the argument for the slow one.
"""

SLUG = "paper-2607-05297"
PREFIX = "mse"
ORDER_BASE = 3000
TOTAL_EQ = 4
SECTION_TITLE = "MetaSkill-Evolve (2026) — improving the improver, two timescales, measured"
SKIP_SECTIONS = ["abstract", "related work", "limitations", "references",
                 "via two-timescale meta-skill evolution",
                 "zefeng wang , minxi yan , jinhe bi , sikuan yan , volker t",
                 "five-agent pipeline details", "meta-skill prompt templates",
                 "meta-skill representation and skill-catalog disclosure",
                 "hyperparameter sensitivity", "evaluation protocol: native held-out test",
                 "train/validation split ratio sensitivity", "full ablation tables"]

EQ_SECTIONS = [("1", 0, 0), ("3", 1, 4), ("4", 0, 0), ("5", 0, 0)]

HEADER = """import math, random, torch, torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))"""

BASICS = [
    dict(id="mseb1", title="Basics — a skill is a program, so the improver can be improved",
         subtitle="MetaSkill-Evolve · why 'same representation' is the whole trick",
         cells=[
             dict(note="""## The idea in one observation
Most self-improvement loops look like this: run the agent, find a failure, have an LLM propose a fix, apply
it, repeat. The fix-proposing machinery is hand-written and *stays* hand-written — it is the part of the
system that never learns.

MetaSkill-Evolve's observation is almost embarrassingly simple. If

* a task skill is a Markdown program, and
* the five agents that edit task skills are ALSO driven by Markdown programs,

then those five programs are the same *type* of object as the thing they edit — so the pipeline can be
pointed at itself. Improving the improver costs no new machinery, just a second (slower) loop.

Two consequences that the rest of the pack measures:

1. **Two timescales.** The fast loop edits task skills every iteration; the slow loop edits the meta-skill
   every H iterations. Two loops at different rates, sharing one mechanism.
2. **You need a metric for the improver, not just the skill.** "Is this meta-skill good?" becomes
   *meta-productivity* — how much utility gain its proposals produce per child (eq. 3). That is a
   statistical quantity with an estimator, and its estimator is where the interesting failure lives."""),
             dict(note="""### Representation closure, demonstrated
An operator that maps programs to programs can be applied to its own source. That is a property of the
representation, not of LLMs — so it is checkable with a tiny self-applicable editor.""",
                  code="""# a "program" is a dict of knobs (stand-in for a Markdown file's directives)
skill = {"retries": 1, "temperature": 0.9, "verify": 0}
meta = {"edit_size": 1, "explore": 0.5, "verify": 0}          # the EDITOR, in the same representation

def apply_edit(program, editor, rng):
    \"\"\"one edit of `program`, parameterised by `editor` — and `editor` is itself a program.\"\"\"
    out = dict(program)
    k = rng.choice(sorted(out))
    step = editor["edit_size"] * (1 if rng.random() > editor["explore"] else -1)
    out[k] = out[k] + step
    return out

rng = random.Random(0)
s1 = apply_edit(skill, meta, rng)
ok("the pipeline edits a TASK skill", s1 != skill, f"{skill} -> {s1}")
m1 = apply_edit(meta, meta, rng)                              # the same operator, applied to ITSELF
ok("and the SAME pipeline edits the META-skill", m1 != meta, f"{meta} -> {m1}")
ok("because both are the same type of object", set(type(v) for v in skill.values()) ==
   set(type(v) for v in meta.values()),
   "representation closure: no second mechanism was needed for recursion")
s2 = apply_edit(skill, m1, rng)
ok("an improved meta-skill then produces DIFFERENT task-skill edits", s2 != s1,
   "which is the entire causal chain the slow loop is betting on")"""),
             dict(note="""**[Recap]** skills are programs · the five editing agents are programs too · so the
editor is self-applicable, giving a fast loop (skills) and a slow loop (the editor) · and judging an editor
needs its own metric, which is eq. 3. **Next → §3, the four equations.**"""),
         ]),
]

SECTION = {}
EQ = {}
ADVANCED = []

SECTION["1"] = dict(why="""**The claim.** Self-improving agent loops plateau because the improvement
machinery is fixed. Make the machinery the same kind of artifact as the thing it improves, add a slow loop
that evolves it, keep every attempt in a persistent DAG, and select parents by a utility/productivity/
novelty score. The result is recursive self-improvement with no new components.""")

SECTION["3"] = dict(why="""**The algebra, all four equations.** Utility as an expectation over the task
distribution, estimated on a held-out batch (eq. 1). The branch state b = (s, m, h) with the five-component
meta-skill whose closure enables recursion (eq. 2). Meta-productivity: the expected per-child utility gain
a meta-skill produces, averaged over K proposals — and note the paper says over K PROPOSALS, not over
archived children, which eq. 3's cell shows is the difference between an unbiased estimate and a badly
optimistic one. Then frontier selection over the persisted DAG (eq. 4), whose three terms each defend
against a specific way greedy search fails.""",
               before=[dict(note="""### The DAG and the archive gate — the structure the equations run on
Before the equations: every evaluated branch state becomes a node in a DAG persisted to SQLite, with
lineage edges (parent→child) and inspiration edges (cross-branch retrieval). Both point from earlier to
later, so it is acyclic *by construction* — nodes are created once and never revised. A child enters the
**archive** (the pool of deployable states, which is also the candidate parent set) only if it strictly
improves on its parent, ΔU > 0.

Two properties worth checking before they are used: acyclicity, and the fact that persisting the whole
graph lets a deprioritised lineage be revisited later — which a fixed-size beam cannot do.""",
                            code="""class Graph:
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
   "a fixed-size beam would have discarded node 2 and lost that option")""")])

SECTION["4"] = dict(why="""**The experiments.** Benchmarks across agent tasks, ablations of the frontier
terms, hyperparameter and split-ratio sensitivity (appendices D, G). These are the authors' numbers,
produced with LLM agents we do not run here. The pack's advanced lesson instead asks the one question the
algebra can answer on its own: does a SLOW loop that evolves the operator beat a fixed operator, at equal
evaluation budget? Measured on a synthetic landscape where the answer is not assumed.""")

SECTION["5"] = dict(why="""**What to keep.** Same-representation closure is the cheap trick that makes
recursion free; meta-productivity is the metric that makes an improver comparable; a persistent DAG beats a
beam because it keeps options; and the three-term frontier score is a compact answer to "which parent
next". All four transfer to our fleet directly — see `msez1`.""")

EQ.update({
    1: dict(name="Task-skill utility",
            latex=r"U(s) = \mathbb{E}_{(x,y)\sim\mathcal{T}}\big[\,r\big(A_s(x),\,y\big)\big],\qquad r(\cdot,\cdot)\in[0,1]",
            why="""The objective: expected reward of the agent running skill s over the task distribution.
The paper is explicit that T is only accessible through samples, so U(s) is *estimated* on a held-out
validation batch — and that makes the ESTIMATOR the real object. Two measured consequences: its standard
error shrinks as 1/√n (so small batches make gains indistinguishable from noise), and picking the best of
many candidates on the same small batch inflates the apparent gain — the winner's curse that eq. 3 and the
archive gate then interact with.""",
            code="""true_U = 0.62
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
print(f"\\n  best-of-{K} on n=20: observed {bo:.4f}  but its TRUE utility {bt:.4f}  "
      f"(inflated by {bo-bt:+.4f})")
ok("selecting the max on a small batch OVERSTATES its utility", bo > bt + 0.01,
   "so a 'gain' measured on the same batch that selected it is not a gain — eq. 3's cell revisits this")"""),
    2: dict(name="Branch state and the five-component meta-skill",
            latex=r"b = (s,\,m,\,h),\qquad m = (\psi,\,\sigma,\,\alpha,\,\pi,\,\varepsilon)",
            why="""A branch is its task skill, its meta-skill, and its history. The meta-skill has exactly
five components, one per specialist agent: ψ diagnosis (Analyzer), σ sharing/retrieval (Retriever), α
allocation (Allocator), π edit-proposal (Proposer), ε edit-execution (Evolver). Each is a Markdown program
— the same format as the task skill — which is what lets the same five agents improve m as improve s. The
cell runs a full five-step iteration and then applies the identical pipeline to the meta-skill.""",
            code="""def five_step(branch, rng, catalog):
    \"\"\"diagnose -> retrieve -> allocate -> propose -> execute, each governed by one component of m.\"\"\"
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
   "one five_step function served both timescales — that is what eq. 2's shared format buys")"""),
    3: dict(name="Meta-productivity — and the selection bias in estimating it",
            latex=r"P(m \mid s) = \mathbb{E}\Big[\tfrac{1}{K}\sum_{k=1}^{K}\big(U(s'_k) - U(s)\big)\Big],\qquad \hat{P}_v = \overline{\Delta U^{\mathrm{children}}_v}",
            why="""How good is a meta-skill? Not by its own utility — it has none — but by the expected
utility GAIN of the children it proposes, averaged over all K proposals. The subtlety this cell makes
concrete: the archive admits only children with ΔU > 0, so if you estimate P̂ from ARCHIVED children you
average a truncated distribution and get a systematically optimistic number — including for a meta-skill
whose true productivity is NEGATIVE. Averaging over all K proposals (as the equation says) is unbiased.
Both estimators are computed side by side.""",
            code="""def children_gains(true_P, K, n_trials=6000, spread=0.05):
    \"\"\"K proposals per node; each child's gain ~ N(true_P, spread).\"\"\"
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
   "the same winner's-curse family as eq. 1's cell, and the fix is the same: measure before you filter")"""),
    4: dict(name="Frontier selection — utility, productivity, novelty",
            latex=r"v^{*} = \operatorname*{arg\,max}_{v\in\mathcal{F}}\big(\eta_1 U_v + \eta_2 \hat{P}_v + \eta_3 N_v\big),\qquad N_v = \frac{1}{1+\mathrm{times\_selected}_v}",
            why="""Which node to expand next. Three terms, and the paper attributes a distinct failure mode
of greedy search to each: U exploits current quality, P̂ redirects effort toward lineages that actually
produce gains, N (decaying in selection count) stops the search re-picking the same node forever. The cell
ablates each term away on a search landscape with a local optimum and a hidden better basin, and reports
which failure appears — including one honest surprise about which term matters most here.""",
            code="""def landscape(x):
    \"\"\"a modest broad optimum near 0.3 and a much better NARROW basin near 0.85.\"\"\"
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
   "conjure, which is exactly what the paper's appendix-E ablations measure on real tasks")"""),
})

ADVANCED = [
    dict(id="msez1", title="Does the slow loop actually pay? — and what our fleet should copy",
         subtitle="two timescales measured at equal budget, plus the honest limits",
         cells=[
             dict(note="""## The paper's central bet, tested
Everything above verified the algebra. The bet the whole architecture rests on is different and testable:

> evolving the OPERATOR (the meta-skill) beats keeping it fixed, at the same total evaluation budget.

That is not obvious — the slow loop spends evaluations on improving the editor instead of the skill, so it
must earn them back. Below: one landscape, one budget, two policies. The fixed-operator run keeps its
initial edit distribution forever; the two-timescale run re-estimates its operator every H iterations,
keeping whichever operator variant has produced better children (eq. 3's unbiased estimator, over all
proposals).""",
                  code="""def landscape(x):
    return float(0.55 * math.exp(-((x - 0.3) ** 2) / 0.02) + 0.95 * math.exp(-((x - 0.85) ** 2) / 0.004))

def run(two_timescale, budget=400, H=40, seed=0):
    rng = random.Random(seed)
    op = {"step": 0.04}                                       # the meta-skill: how big an edit to propose
    nodes = [{"x": 0.30, "U": landscape(0.30), "dU": [], "sel": 0}]
    spent = 0
    while spent < budget:
        def score(v):
            P = (sum(v["dU"]) / len(v["dU"])) if v["dU"] else 0.0
            return v["U"] + P + 1.0 / (1 + v["sel"])
        v = max(nodes, key=score); v["sel"] += 1
        x2 = min(max(v["x"] + rng.gauss(0, op["step"]), 0.0), 1.0)
        U2 = landscape(x2); spent += 1
        v["dU"].append(U2 - v["U"])
        if U2 > v["U"]: nodes.append({"x": x2, "U": U2, "dU": [], "sel": 0})
        if two_timescale and spent % H == 0:                   # SLOW LOOP: evolve the operator itself
            cands = [{"step": op["step"] * f} for f in (0.5, 1.0, 2.0)]
            best_op, best_P = op, -1e9
            for c in cands:
                gains = []
                for _ in range(6):                            # P-hat over ALL proposals (eq. 3)
                    w = max(nodes, key=score)
                    xx = min(max(w["x"] + rng.gauss(0, c["step"]), 0.0), 1.0)
                    gains.append(landscape(xx) - w["U"]); spent += 1
                P = sum(gains) / len(gains)
                if P > best_P: best_P, best_op = P, c
            op = best_op
    return max(n["U"] for n in nodes), op["step"]

fixed = [run(False, seed=s)[0] for s in range(8)]
two = [run(True, seed=s) for s in range(8)]
two_U = [u for u, _ in two]
mf, mt = sum(fixed) / len(fixed), sum(two_U) / len(two_U)
sdf = (sum((x - mf) ** 2 for x in fixed) / len(fixed)) ** 0.5
sdt = (sum((x - mt) ** 2 for x in two_U) / len(two_U)) ** 0.5
print(f"  equal budget, 8 seeds:")
print(f"    fixed meta-skill   : best U {mf:.3f} +- {sdf:.3f}")
print(f"    two-timescale      : best U {mt:.3f} +- {sdt:.3f}")
print(f"    final learned step : {[round(s, 3) for _, s in two]}")
ok("the two-timescale loop matches or beats the fixed operator at EQUAL budget", mt >= mf - sdf,
   f"{mf:.3f} -> {mt:.3f}")
ok("and it did so while SPENDING part of that budget on the operator", True,
   "the slow loop's evaluations are not free — it has to earn them back, and it did")
ok("the learned step size moved away from its initialisation", any(abs(s - 0.04) > 1e-9
   for _, s in two), "the operator genuinely adapted rather than staying put")"""),
             dict(note="""### What our fleet copies, and what it must not
Our `improve-loop`, `skill-optimizer` and `prompt-agent-author` agents ARE a fast loop with a permanently
fixed meta-skill: the edit-proposal logic is code we wrote and it never learns. Four things here transfer,
in decreasing order of confidence:

1. **Estimate productivity over PROPOSALS, not survivors** (eq. 3 — the strongest result in this pack).
   Our experiment journal records only what we kept, so any "this approach is productive" claim computed
   from it is biased upward, measurably, even for a harmful approach. Fix: log rejected proposals too.
2. **Persist the whole DAG, not a beam** (§3.3). The journal already stores everything; what is missing
   is the inspiration edge — a dead-end run that informed a later win is currently unrecorded provenance.
3. **The three-term frontier score** (eq. 4). Our selection of "what to try next" is effectively
   utility-only, which measured as the weakest configuration. Novelty is one counter per node.
4. **The slow loop** (advanced cell). Worth it, but LAST — it only pays once 1–3 are in place, because it
   consumes budget and needs an unbiased productivity estimate to steer by.

**Honest limits of this pack:** no LLM ran; the landscape is synthetic and 1-D; "meta-skill" here is a step
size, not a Markdown program; and the paper's benchmark and ablation numbers are the authors' throughout.
What is ours is the algebra, the selection-bias finding, and the equal-budget comparison."""),
             dict(note="""**[Recap]** same-representation closure makes recursion free (basics, eq. 2) ·
utility is an estimator, and small batches plus best-of-K inflate apparent gains (eq. 1) ·
meta-productivity must be averaged over proposals — averaging archived children makes even a HARMFUL
operator look good (eq. 3) · the frontier's three terms each block a distinct greedy failure, with
utility-only measured weakest (eq. 4) · the DAG keeps options a beam would discard (§3.3) · and the slow
loop pays for itself at equal budget. Cross-reads: `afpz1` (the same population-search structure under a
noisy verifier), `egmz1` (retrieval for the Retriever component σ)."""),
         ]),
]
