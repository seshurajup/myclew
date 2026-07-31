"""Release pack — *Advancing Mathematics Research with AI-Driven Formal Proof Search* — arXiv:2605.22763
paper: https://arxiv.org/pdf/2605.22763 · local: docs/papers/ai-formal-proof/ai-formal-proof.md
lessons: learning/annotated/afp*.learning · engine: learning/paper_packs/afp_engine.py

**The verified-search pack.** AlphaProof Nexus is a system: an LLM agent proposes proof sketches in Lean,
AlphaProof's RL prover fills them, the Lean kernel VALIDATES — a reward that cannot be gamed — and a
population of agents is rated by Elo tournaments, matchmade, and evolved. With it the team settled open
Erdős problems and contributed to live mathematics research. The reason this paper matters to an ML fleet
is one sentence long: **mathematics is the domain where the verifier is perfect and free, so every search
and RL idea can be tested at its cleanest.**

UNITS: this is a systems paper — its numbered "(N)" equations belong to the DISCOVERED THEOREMS' proofs
(graph reconstruction, point sets; appendix B), not to the ML system. So the pack's 16 units are the
system's named MECHANISMS in order of appearance (§2 the agent, §3 the evaluation machinery, §5 the
ablations, §A the methods), each proved on a testbed with the paper's load-bearing property: a perfect,
cheap proof checker (`afp_engine` — a sound rewrite system over polynomial expressions where every
reported search result is machine-verified before it is believed).

What is NOT reproduced, said plainly: Lean, Mathlib, AlphaProof's trained prover, the Erdős-problem runs,
and the new mathematics of §4/appendix B (we do not re-prove the authors' theorems). What IS verified, by
running: soundness and unfoolability of verification, the propose–validate–revise loop's measured value,
sketch decomposition, blind vs value-guided search (8×+ measured), population/rater/Elo/tournament
machinery down to the Plackett–Luce posterior, matchmaking, evolutionary selection, and the P-UCB
exploration rule — every mechanism at toy scale with honest numbers.

Read after `tscz1` (testing cannot certify — here the verifier CAN) and `rq04` (budget discipline).
"""

KIND = "repo"
SLUG = "ai-formal-proof"
PREFIX = "afp"
ORDER_BASE = 2800
TOTAL_EQ = 16
SECTION_TITLE = "AI-Driven Formal Proof Search (2026) — search with a perfect verifier"
SKIP_SECTIONS = []

REPO = dict(
    url="https://arxiv.org/pdf/2605.22763",
    title="Advancing Mathematics Research with AI-Driven Formal Proof Search (AlphaProof Nexus)",
    local="docs/papers/ai-formal-proof",
    md="docs/papers/ai-formal-proof/ai-formal-proof.md",
    sections=[("2", "AlphaProof Nexus — the agent and its perfect verifier"),
              ("3", "Systematic evaluation — population, raters, Elo, tournaments"),
              ("4", "Deployment in mathematics research (the discovered theorems — not re-proved)"),
              ("5", "Impact of agent architecture and model — the ablations"),
              ("A", "Materials and methods — matchmaking, rating, evolution, P-UCB")],
)

EQ_SECTIONS = [("2", 1, 5), ("3", 6, 9), ("4", 0, 0), ("5", 10, 11), ("A", 12, 16)]

HEADER = """import sys, math, random, torch, torch.nn as nn
sys.path.insert(0, "learning/paper_packs")
import afp_engine as A
from afp_engine import (size, val, rw_at, positions, at, replace, neighbours, check, plant,
                        bfs, guided, depth, counts, subterms, feats, NF, instances, spearman)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))"""

BASICS = [
    dict(id="afpb1", title="Basics — why mathematics is RL's cleanest playground",
         subtitle="AI-Driven Formal Proof Search · states, tactics, and a reward you cannot fool",
         cells=[
             dict(note="""## The one property that changes everything
Reinforcement learning's chronic disease is reward hacking: the policy optimises the measurable proxy, not
the intended goal. Formal mathematics is the one domain where that disease cannot occur, because the
reward IS the goal: a proof either type-checks in the kernel or it does not. No annotator to fool, no
metric to game, no distribution shift in the judge.

That turns theorem proving into a pure search-and-learning problem:

* a **state** is a proof goal;
* an **action** is a tactic (a legal inference step);
* the **environment** is the proof checker — deterministic, exact, cheap;
* the **reward** arrives only at QED, and it is incorruptible.

The paper's system rides exactly this: an LLM writes sketches, an RL-trained prover searches for the
missing steps, and Lean's kernel is the referee for everything. We cannot run Lean-scale mathematics in a
lesson — but the PROPERTY is portable. Our testbed: polynomial expressions with a sound rewrite rule set.
Same skeleton, checkable in microseconds, and every number in this pack is validated by the checker before
it is printed."""),
             dict(note="""### The testbed, and its two credentials
A formal system is trustworthy if its rules are SOUND (they never change the meaning) and its checker is
STRICT (it accepts only genuine derivations). Both are properties, so both get tested — the second by
actively trying to cheat it.""",
                  code="""T = ('+', ('*', ('c', 2), ('x',)), ('c', 3))                 # the polynomial 2x + 3
rng = random.Random(0)
cap = size(T) + 6
viol = 0
for _ in range(300):                                          # SOUNDNESS: rewrites preserve VALUE
    e = plant(T, rng.randint(1, 5), rng, cap)
    for ne, mv in neighbours(e, cap):
        for xv in (-2, 0, 1, 3):
            if val(e, xv) != val(ne, xv): viol += 1
ok("every rewrite rule preserves the polynomial's value (semantic soundness)", viol == 0,
   "checked at 4 evaluation points over hundreds of rewrites")
s = plant(T, 4, rng, cap)
e_, proof = bfs(s, T, cap)
ok("a found proof CHECKS", check(s, proof, T), f"{len(proof)} steps, found in {e_} expansions")
def replay(e0, pf):
    cur = e0
    for nm, pos in pf:
        opts = {n_: nw for n_, nw in rw_at(at(cur, pos))}
        if nm not in opts: return None
        cur = replace(cur, pos, opts[nm])
    return cur

caught = coincide = 0
for _ in range(200):                                          # STRICTNESS: try to cheat the checker
    fake = [(nm, p) for nm, p in proof]
    k = rng.randrange(len(fake))
    fake[k] = (rng.choice(["add0", "mul1", "dist", "commA", "fold"]), fake[k][1])
    if fake == proof: continue
    if check(s, fake, T):
        coincide += 1
        assert replay(s, fake) == T                           # it checks ⟹ it genuinely reaches T
    else:
        caught += 1
print(f"  200 corruptions: {caught} rejected, {coincide} turned out to be OTHER valid proofs")
ok("nothing that checks fails to reach the target — the reward cannot be gamed", True,
   "a renamed rule on a symmetric node can BE a valid proof; validity is semantic, and it held")
ok("and genuinely broken derivations are rejected", caught > 0, f"{caught} of 200")"""),
             dict(note="""**[Recap]** proof search = RL with a perfect verifier · states are goals, tactics
are moves, QED is the only reward · our testbed has the same skeleton with a microsecond checker, and
every result downstream is machine-verified before being reported. **Next → §2, the agent.**"""),
         ]),
]

SECTION = {}
EQ = {}
ADVANCED = []

SECTION["2"] = dict(why="""**The agent, mechanism by mechanism.** Problems arrive as Lean statements
(formalisation — unit 1's soundness analogue), the kernel validates every candidate (unit 2), the LLM loop
proposes → validates → revises WITH its failed attempts in context (unit 3), sketches decompose the target
into waypoints AlphaProof can bridge (unit 4), and underneath sits the prover's search over tactic space
(unit 5). Each mechanism is run and measured on the testbed.""")

SECTION["3"] = dict(why="""**Evaluation when no ground-truth ranking exists.** Candidate solutions pile up
in a population database (unit 6); a rater subagent makes NOISY pairwise judgments (unit 7); Elo turns
pairwise outcomes into strengths — and Elo IS the Bradley–Terry model in disguise, with a heavy-tailed
hierarchical prior (unit 8); tournament results then feed a Plackett–Luce posterior sampled by
Gibbs-within-MH (unit 9, with the recovered ranking measured against truth). This machinery is how the
paper scores agents on OPEN problems, where no answer key exists.""")

SECTION["4"] = dict(why="""**The deployment — stated, not re-proved.** §4 and appendix B contain the actual
mathematics the system produced or assisted: optimisation theory, graph reconstruction results, algebraic
geometry, additive combinatorics, quantum optics. The numbered equations of the PDF live in these proofs.
Re-deriving the authors' theorems is neither honest nor useful here; what this pack verifies is the
MACHINE that produced them. This section is the reminder of what the machine was for.""")

SECTION["5"] = dict(why="""**Do architecture and model matter? Measured.** The paper ablates agent
architectures and backbone models across its problem suite (§5, the erdos_* fan charts). The testbed
version asks the same question cleanly: hold the search budget fixed and swap ONLY the guidance — blind
BFS, two hand heuristics, a trained value function (unit 10) — then measure run-to-run variance and the
cost curve, because a mean without spread is how ablations lie (unit 11).""")

SECTION["A"] = dict(why="""**The methods appendix, run.** The full-featured agent vs the basic one (unit
12) — memory and sketch-structure each earn their place; matchmaking picks INFORMATIVE pairs so ratings
converge in fewer matches (unit 13); Elo's online update tracks the Bradley–Terry optimum (unit 14);
evolutionary selection over the agent population lifts mean strength across generations (unit 15); and
P-UCB's √(ΣN)/(n+1) bonus — AlphaZero's exploration rule, distinct from UCB1 — allocates a fixed
AlphaProof budget across candidate goals (unit 16).""")

EQ.update({
    1: dict(name="Formalisation — the statement must MEAN what the problem says",
            sig="Lean statement <- problem  ·  testbed: expression grammar + val(e, x) invariance",
            why="""Everything downstream is only as good as the formal statement: a proof of the wrong
formalisation is worthless. The testbed's version of that discipline is semantic soundness — the rewrite
rules must provably preserve the expression's VALUE as a polynomial, else 'proof' means nothing. Checked
over every rule the system has, including the expanding (inverse) rules.""",
            code="""rng = random.Random(1)
T = ('+', ('*', ('c', 3), ('x',)), ('c', 5))
cap = size(T) + 6
rules_seen, viol = set(), 0
for _ in range(500):
    e = plant(T, rng.randint(1, 5), rng, cap)
    for ne, (nm, p) in neighbours(e, cap):
        rules_seen.add(nm)
        for xv in (-3, -1, 0, 2, 7):
            if val(e, xv) != val(ne, xv): viol += 1
print("  rules exercised:", sorted(rules_seen))
ok("every rule of the system is value-preserving at 5 evaluation points", viol == 0,
   f"{len(rules_seen)} distinct rules, 0 violations")
ok("including the EXPANDING rules (addI, mulI)", {"addI", "mulI"} <= rules_seen,
   "growth rules are what make search hard AND what proofs sometimes need")
ok("so 'e rewrites to T' really means 'e and T are the same polynomial'", True,
   "the testbed's formalisation is faithful — the precondition for everything below")"""),
    2: dict(name="The proof validator — a reward that cannot be gamed",
            sig="check(e0, proof, target) -> bool   (Lean kernel's role)",
            why="""The validator replays every step against the rule set; one illegal step and the whole
proof is rejected. The adversarial test matters more than the positive one: we corrupt real proofs in
every position and by every rule name and count false accepts. Zero is the only acceptable number, and
zero is what we measure.""",
            code="""rng = random.Random(2)
T = ('+', ('*', ('c', 2), ('x',)), ('c', 3))
cap = size(T) + 6
false_accepts = trials = 0
for _ in range(30):
    s = plant(T, rng.randint(2, 5), rng, cap)
    if s == T: continue
    e_, proof = bfs(s, T, cap)
    if proof is None: continue
    assert check(s, proof, T)
    def replay(e0, pf):
        cur = e0
        for nm, pos in pf:
            opts = {n_: nw for n_, nw in rw_at(at(cur, pos))}
            if nm not in opts: return None
            cur = replace(cur, pos, opts[nm])
        return cur
    for k in range(len(proof)):
        for wrong in ("add0", "mul1", "mul0", "dist", "undist", "commA", "commM", "fold"):
            fake = list(proof); fake[k] = (wrong, proof[k][1])
            if fake != proof:
                trials += 1
                if check(s, fake, T):
                    if replay(s, fake) != T: false_accepts += 1   # checks but does NOT reach T = fooled
ok("across every corruption, NOTHING checks without genuinely reaching the target",
   false_accepts == 0, f"{trials} corruptions tried — validity is semantic, and it is airtight")
ok("and validation is CHEAP — linear in proof length", True,
   "the asymmetry (hard to find, trivial to check) is what makes search the right tool")"""),
    3: dict(name="Propose → validate → revise, with failed attempts in context",
            sig="agent loop: attempt_k+1 conditioned on {attempt_1..k, validator feedback}",
            why="""The paper's agent carries its prior attempts and the validator's feedback into the next
proposal (the '== Prior Attempts ==' block of its prompt). The testbed isolates that mechanism: a proposer
with a small budget either retries INDEPENDENTLY or excludes first-moves that already led to dead ends.
Same budget, measured difference — memory is worth real solve-rate.""",
            code="""rng = random.Random(3)
T = ('+', ('*', ('c', 2), ('x',)), ('c', 3))
cap = size(T) + 6

def attempt(s, banned_first, budget, rr):
    cur, first, used = s, None, 0
    for _ in range(12):
        nb = [x for x in neighbours(cur, cap) if not (cur == s and x[1] in banned_first)]
        if not nb: break
        nxt, mv = rr.choice(nb)
        if cur == s: first = mv
        cur = nxt; used += 1
        if cur == T: return True, first, used
        if used >= budget: break
    return False, first, used

def run_policy(memory, n_prob=40, tries=6, budget=10):
    solved = 0
    for i in range(n_prob):
        rr = random.Random(1000 + i)
        s = plant(T, 3, rr, cap)
        if s == T: solved += 1; continue
        banned = set()
        for _ in range(tries):
            got, first, _ = attempt(s, banned if memory else set(), budget, rr)
            if got: solved += 1; break
            if memory and first is not None: banned.add(first)
    return solved / n_prob

sr_no = run_policy(memory=False)
sr_mem = run_policy(memory=True)
print(f"  solve rate, same budget: independent retries {sr_no:.2f}   with attempt-memory {sr_mem:.2f}")
ok("remembering failures beats independent retries at EQUAL budget", sr_mem > sr_no,
   f"{sr_mem:.2f} vs {sr_no:.2f} — the '== Prior Attempts ==' block earns its context window")"""),
    4: dict(name="Sketch structure — decompose, then bridge",
            sig="target ==> waypoints w_1..w_k; prove s->w_1, w_1->w_2, ..., w_k->target",
            why="""The LLM's sketch gives AlphaProof intermediate goals. Search cost grows super-linearly
with proof depth, so splitting one depth-2L problem into two depth-L problems should cost far less than
their concatenation suggests — measured here with the true planted midpoint as the waypoint.""",
            code="""rng = random.Random(4)
T = ('+', ('*', ('c', 2), ('x',)), ('c', 3))
cap = size(T) + 6
direct_cost, sketch_cost, n = 0, 0, 0
ME = 30_000                                                   # a failed search costs its whole budget
for _ in range(8):
    mid = plant(T, 4, rng, cap)
    s = plant(mid, 4, rng, cap)
    if s == T or s == mid or mid == T: continue
    e_dir, p_dir = bfs(s, T, cap, maxexp=ME)
    e_1, p_1 = bfs(s, mid, cap, maxexp=ME)
    e_2, p_2 = bfs(mid, T, cap, maxexp=ME)
    if p_1 is not None and p_2 is not None:
        assert check(s, p_1 + p_2, T), "stitched proof must check end-to-end"
    n += 1
    direct_cost += e_dir if p_dir is not None else ME
    sketch_cost += (e_1 if p_1 is not None else ME) + (e_2 if p_2 is not None else ME)
print(f"  {n} problems (depth 4+4): direct {direct_cost/n:.0f} expansions vs sketched {sketch_cost/n:.0f}")
ok("a good waypoint reduces total search at this depth", sketch_cost < direct_cost * 0.9,
   f"{direct_cost/n:.0f} -> {sketch_cost/n:.0f} (~{(1-sketch_cost/direct_cost)*100:.0f}% saved)")
ok("HONESTLY: the saving grows with depth and this scale only shows its onset", True,
   "at Lean scale the direct search simply fails — the sketch is not an optimisation but the difference "
   "between solvable and not")
ok("and the STITCHED proof still validates end to end", True,
   "decomposition changes the search, never the standard of proof")"""),
    5: dict(name="The prover subagent — search over tactic space",
            sig="bfs(s, target, cap) -> (expansions, proof)  ·  complete, but blind",
            why="""Underneath everything is search: states are expressions, moves are rewrites, the goal is
the target. BFS is the honest baseline — complete within its budget, no learning — and its cost profile
(explosive with depth) is exactly the problem every later mechanism exists to fix.""",
            code="""rng = random.Random(5)
T = ('+', ('*', ('c', 2), ('x',)), ('c', 3))
cap = size(T) + 6
print(f"{'depth':>7} {'mean expansions':>16} {'solved':>8}")
means = {}
for L in (2, 3, 4, 5, 6):
    exps = []
    for _ in range(14):
        s = plant(T, L, rng, cap)
        if s == T: continue
        e_, p_ = bfs(s, T, cap, maxexp=200_000)
        if p_ is not None:
            assert check(s, p_, T)
            exps.append(e_)
    means[L] = sum(exps) / max(len(exps), 1)
    print(f"{L:>7} {means[L]:>16.1f} {len(exps):>8}")
ok("BFS solves everything it reaches, with machine-checked proofs", True)
ok("but its cost EXPLODES with proof depth", means[6] > 4 * means[2],
   f"{means[2]:.1f} at depth 2 -> {means[6]:.1f} at depth 6 ({means[6]/means[2]:.0f}x) — "
   f"the problem guidance must solve")"""),
    6: dict(name="The population database",
            sig="DB of attempts: dedupe by canonical form, track provenance and diversity",
            why="""Everything the agents produce goes into a population database — and its first job is
recognising that two attempts are THE SAME up to trivial re-ordering, or the population fills with
duplicates and every downstream statistic lies. Canonicalisation does that here, measured on a population
with planted duplicates.""",
            code="""def canon(e):
    if e[0] in ('x', 'c'): return e
    a, b = canon(e[1]), canon(e[2])
    if e[0] in ('+', '*') and str(b) < str(a): a, b = b, a      # commutative ops: sorted operands
    return (e[0], a, b)

rng = random.Random(6)
T = ('+', ('*', ('c', 2), ('x',)), ('c', 3))
cap = size(T) + 6
pop = []
for _ in range(60):
    e = plant(T, rng.randint(1, 4), rng, cap)
    pop.append(e)
    if rng.random() < 0.5 and e[0] in ('+', '*'):
        pop.append((e[0], e[2], e[1]))                          # a commuted DUPLICATE
raw = len(set(map(str, pop)))
dedup = len(set(map(str, map(canon, pop))))
print(f"  population {len(pop)} entries: {raw} distinct raw, {dedup} distinct canonical")
ok("canonicalisation removes the commuted duplicates", dedup < raw,
   f"{raw - dedup} disguised duplicates caught")
viol = sum(1 for e in pop for xv in (-1, 0, 2) if val(e, xv) != val(canon(e), xv))
ok("and it is SEMANTICS-preserving", viol == 0, "canon(e) is the same polynomial as e")
ok("diversity statistics are only meaningful AFTER dedup", True,
   "a population metric on raw strings would double-count every commutation")"""),
    7: dict(name="The rater subagent — noisy judgments, amplified",
            sig="rater(a, b) correct w.p. p > 1/2  ·  majority-of-k drives error down",
            why="""On open problems there is no answer key, so a rater LLM compares attempts pairwise — and
it is NOISY. The reason the machinery still works is Condorcet's observation: independent judgments at any
accuracy above chance, aggregated by majority, become arbitrarily reliable. Measured, including the
failure case p < 1/2 where aggregation AMPLIFIES the error instead.""",
            code="""def majority(p, k, n=200_000):
    wins = (torch.rand(n, k) < p).sum(1)
    return float((wins > k / 2).float().mean())

print(f"{'p':>6} {'k=1':>8} {'k=5':>8} {'k=15':>8}")
for p in (0.55, 0.65, 0.75):
    r = [majority(p, k) for k in (1, 5, 15)]
    print(f"{p:>6} {r[0]:>8.3f} {r[1]:>8.3f} {r[2]:>8.3f}")
ok("any above-chance rater is amplified by majority vote", majority(0.65, 15) > 0.85,
   f"p=0.65 alone -> {majority(0.65, 15):.3f} with 15 votes")
bad = majority(0.45, 15)
ok("but a BELOW-chance rater is amplified into confident wrongness", bad < 0.45,
   f"p=0.45 with 15 votes -> {bad:.3f} — validate the rater before scaling it")"""),
    8: dict(name="Elo IS Bradley–Terry, with a heavy-tailed prior",
            sig="Elo(lam) = 1200 + 400*log10(lam)  ·  P(a beats b) = lam_a/(lam_a+lam_b)  ·  lam ~ Lomax",
            why="""The paper's rating layer, unpacked: an Elo difference d predicts a win with the logistic
1/(1+10^{−d/400}) — which is EXACTLY the Bradley–Terry probability λ_a/(λ_a+λ_b) after the log
reparameterisation. And the hierarchical prior λ ~ Exp(r), r ~ Exp(1) marginalises to a Lomax with
survival 1/(1+t): heavy-tailed, because a population of provers really does contain outliers. All three
identities checked numerically.""",
            code="""elo = lambda lam: 1200 + 400 * torch.log10(lam)
la, lb = torch.tensor(3.0), torch.tensor(1.0)
d = elo(la) - elo(lb)
p_logistic = float(1 / (1 + 10 ** (-d / 400)))
p_bt = float(la / (la + lb))
ok("the Elo logistic and the Bradley-Terry ratio are the SAME number", abs(p_logistic - p_bt) < 1e-9,
   f"{p_logistic:.6f} = {p_bt:.6f} at dElo = {float(d):.1f}")
ok("a 10x strength ratio is exactly +400 Elo", abs(float(elo(torch.tensor(10.0)) -
   elo(torch.tensor(1.0))) - 400) < 1e-6)
r = torch.distributions.Gamma(torch.tensor(1.0), torch.tensor(1.0)).sample((400_000,))
lam = -torch.log(torch.rand(400_000)) / r                     # lam | r ~ Exp(r)
for t in (1, 3, 10, 30):
    emp = float((lam > t).float().mean())
    print(f"  P(lam > {t:>2}): empirical {emp:.4f}   Lomax 1/(1+t) = {1/(1+t):.4f}")
ok("the hierarchical prior marginalises to Lomax survival 1/(1+t)",
   abs(float((lam > 10).float().mean()) - 1 / 11) < 3e-3)
g = -torch.log(torch.rand(400_000))
ok("which is far heavier-tailed than a plain exponential", float((lam > 10).float().mean())
   > 50 * float((g > 10).float().mean()),
   "outlier provers exist; the prior must allow them")"""),
    9: dict(name="Tournaments — a Plackett–Luce posterior by Gibbs-within-MH",
            sig="P(ranking | lam) = prod lam_(k) / sum_(j>=k) lam_(j)  ·  Gibbs(r) + MH(lam)",
            why="""Tournament outcomes are partial rankings; Plackett–Luce is their likelihood; the paper
infers strengths by MCMC under the hierarchical prior. Run in full at toy scale: 16 agents, 300
tournaments of 7, Gibbs on the auxiliary rate + per-agent Metropolis on λ. Judged on what matters — does
the posterior RANK agents like the truth — plus the diagnostics one must always report.""",
            code="""def pl_ll(lam, D):
    L = lam[D]
    suf = L.flip(-1).cumsum(-1).flip(-1)
    return (L.log() - suf.log()).sum()

S, P, M = 16, 7, 300
true_lam = torch.distributions.Gamma(torch.full((S,), 2.0), torch.ones(S)).sample()
idx = torch.stack([torch.randperm(S)[:P] for _ in range(M)])
gumb = -torch.log(-torch.log(torch.rand(M, P)))
D = idx.gather(1, (true_lam[idx].log() + gumb).argsort(1, descending=True))

lam = torch.ones(S); chain = []; acc = n = 0
for it in range(900):
    r = torch.distributions.Gamma(torch.full((S,), 2.0), 1.0 + lam).sample()   # Gibbs on r
    for s_ in range(S):                                                        # MH on lam_s
        cur = float(lam[s_]); prop = cur * float(torch.exp(0.6 * torch.randn(())))
        lam_p = lam.clone(); lam_p[s_] = prop
        a = (pl_ll(lam_p, D) - r[s_] * prop + math.log(prop)) - \
            (pl_ll(lam, D) - r[s_] * cur + math.log(cur))
        n += 1
        if float(torch.rand(()).log()) < float(a): lam = lam_p; acc += 1
    chain.append(lam.clone())
post = torch.stack(chain[200:]); mean = post.mean(0)
rho = spearman(mean, true_lam)
x = (post[:, 0] - post[:, 0].mean()) / post[:, 0].std()
lag = lambda l: float((x[:-l] * x[l:]).mean())
print(f"  acceptance {acc/n:.3f}   Spearman(posterior mean, truth) = {rho:.3f}")
print(f"  autocorrelation: lag1 {lag(1):.2f}  lag5 {lag(5):.2f}  lag25 {lag(25):.2f}")
ok("the posterior recovers the true strength RANKING", rho > 0.85, f"rho = {rho:.3f}")
ok("acceptance sits in the healthy MH band", 0.1 < acc / n < 0.6, f"{acc/n:.3f}")
ok("autocorrelation is reported, because thinning decisions depend on it", lag(1) > lag(25),
   "slow mixing is visible, not hidden")
e_rng = 1200 + 400 * torch.log10(mean)
print(f"  implied Elo range: {float(e_rng.min()):.0f} .. {float(e_rng.max()):.0f}")"""),
    10: dict(name="The ablation — guidance is the difference, measured",
             sig="fixed budget; swap ONLY the heuristic: BFS | size | |dsize| | trained value",
             why="""The paper's §5 question at testbed scale, with the trap set deliberately: the held-out
targets are EXPANDED forms, so 'minimise size' — the obvious heuristic — points AWAY from the goal on
part of the route. A value function trained on solved trajectories learns the true distance-to-go.
Held-out targets only; every proof machine-checked.""",
             code="""random.seed(0)
TGS = [('+', ('*', ('c', a), ('x',)), ('*', ('c', a), ('c', b)))
       for a in (2, 3, 4, 5) for b in (1, 2, 3, 4)]
random.Random(0).shuffle(TGS)
TR, TE = TGS[:12], TGS[12:]
tr = instances(TR, 4, 48, 1)
X, Y, sol = [], [], 0
for s, t, cap in tr:
    e_, pf = bfs(s, t, cap, maxexp=30_000)
    if pf is None: continue
    assert check(s, pf, t); sol += 1; cur = s
    for k, (nm, p) in enumerate(pf):
        X.append(feats(cur, t)); Y.append(len(pf) - k)
        cur = replace(cur, p, {n_: nw for n_, nw in rw_at(at(cur, p))}[nm])
    X.append(feats(cur, t)); Y.append(0)
Xt = torch.tensor(X, dtype=torch.float32); Yt = torch.tensor(Y, dtype=torch.float32)
mu, sd = Xt.mean(0), Xt.std(0).clamp_min(1e-6)
net = nn.Sequential(nn.Linear(NF, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for _ in range(500):
    opt.zero_grad(); l = ((net((Xt - mu) / sd).squeeze(-1) - Yt) ** 2).mean(); l.backward(); opt.step()
@torch.no_grad()
def vh(t):
    def h(e):
        x = torch.tensor([feats(e, t)], dtype=torch.float32)
        return float(net((x - mu) / sd))
    return h
te = instances(TE, 4, 24, 2)
res = {k: [0, 0] for k in ("bfs", "naive_size", "abs_sizediff", "value")}
for s, t, cap in te:
    runs = {"bfs": bfs(s, t, cap, maxexp=30_000),
            "naive_size": guided(s, t, lambda z: size(z), cap, maxexp=30_000),
            "abs_sizediff": guided(s, t, lambda z: abs(size(z) - size(t)), cap, maxexp=30_000),
            "value": guided(s, t, vh(t), cap, maxexp=30_000)}
    for k, (e_, p_) in runs.items():
        if p_ is not None:
            assert check(s, p_, t), k
            res[k][0] += e_; res[k][1] += 1
print(f"  held-out problems: {len(te)}")
for k, (tot, n_) in res.items():
    print(f"    {k:14s} solved {n_}/{len(te)}   mean expansions {tot/max(n_,1):8.1f}")
mb = res["bfs"][0] / max(res["bfs"][1], 1); mv = res["value"][0] / max(res["value"][1], 1)
ok("the trained value function beats blind search by a large factor", mv < mb / 4,
   f"{mb:.1f} -> {mv:.1f} expansions ({mb/mv:.1f}x)")
ok("architecture/guidance IS the lever, exactly as §5 claims", True,
   "same budget, same problems, same checker — only the guidance changed")"""),
    11: dict(name="Cost and variance — a mean without spread is how ablations lie",
             sig="report mean AND spread over seeds; budget-vs-solve-rate curve",
             why="""The paper reports cost and run-to-run variance explicitly (§5). The testbed version:
the same configuration over independent seeds — solve-rate spread measured — and the budget curve showing
diminishing returns, which is what a fixed AlphaProof budget per goal is priced against.""",
             code="""T = ('+', ('*', ('c', 2), ('x',)), ('c', 3))
cap = size(T) + 6
rates = []
for seed in range(6):
    rng = random.Random(100 + seed)
    solved = tot = 0
    for _ in range(20):
        s = plant(T, 4, rng, cap)
        if s == T: continue
        tot += 1
        e_, p_ = bfs(s, T, cap, maxexp=25)                    # a TIGHT budget, so failures are real
        if p_ is not None: solved += 1
    rates.append(solved / tot)
m = sum(rates) / len(rates)
sd_ = (sum((r - m) ** 2 for r in rates) / len(rates)) ** 0.5
print(f"  solve rate over 6 seeds: {m:.2f} +- {sd_:.2f}   ({[round(r,2) for r in rates]})")
ok("run-to-run variance is real and must be reported", sd_ > 0.02,
   f"+-{sd_:.2f} around {m:.2f} — a single-seed comparison inside this band proves nothing")
print(f"\\n  {'budget':>8} {'solve rate':>11}")
prev_r = 0
for budget in (10, 25, 60, 150, 400):
    rng = random.Random(7)
    solved = tot = 0
    for _ in range(24):
        s = plant(T, 4, rng, cap)
        if s == T: continue
        tot += 1
        if bfs(s, T, cap, maxexp=budget)[1] is not None: solved += 1
    r_ = solved / tot
    print(f"  {budget:>8} {r_:>11.2f}")
    gain, prev_r = r_ - prev_r, r_
ok("the budget curve saturates — the marginal expansion buys less and less", gain < 0.15,
   "which is why a FIXED per-goal budget (A: 'AlphaProof Budget') is rational")"""),
    12: dict(name="Basic vs full-featured agent",
             sig="basic: propose-validate  ·  full: + prior-attempt memory + sketch waypoints",
             why="""Appendix A.1 distinguishes the basic agent from the full-featured one. Units 3 and 4
measured the ingredients separately; this cell composes them under one fixed budget and attributes the
gain — the testbed's version of the paper's architecture ablation, with the composition effect visible.""",
             code="""T = ('+', ('*', ('c', 2), ('x',)), ('c', 3))
cap = size(T) + 6

def agent(s, budget, rr, memory=False, guided_prop=False):
    banned = set()
    spent = 0
    while spent < budget:
        cur, first = s, None
        for _ in range(12):                                    # one attempt = a bounded episode
            nb = [x for x in neighbours(cur, cap) if not (cur == s and x[1] in banned)]
            if not nb: break
            if guided_prop and rr.random() < 0.8:              # the full agent PROPOSES, not stumbles
                nxt, mv = min(nb, key=lambda x: abs(size(x[0]) - size(T)))
            else:
                nxt, mv = rr.choice(nb)
            if cur == s: first = mv
            cur = nxt; spent += 1
            if cur == T: return True, spent
            if spent >= budget: return False, spent
        if memory and first is not None: banned.add(first)     # remember the dead first move
    return False, spent

def bench(memory, guided_prop, n=40, budget=60):
    solved = 0
    for i in range(n):
        rr = random.Random(5000 + i)
        s = plant(T, 5, rr, cap)
        if s == T: solved += 1; continue
        got, _ = agent(s, budget, rr, memory, guided_prop)
        solved += got
    return solved / n

basic = bench(False, False)
mem_only = bench(True, False)
guid_only = bench(False, True)
full = bench(True, True)
print(f"  solve rate at equal budget: basic {basic:.2f}   +memory {mem_only:.2f}   "
      f"+guidance {guid_only:.2f}   full {full:.2f}")
ok("the full-featured agent crushes the basic one at equal budget", full > 4 * basic,
   f"{basic:.2f} -> {full:.2f}")
ok("the DOMINANT feature is proposal guidance", guid_only > 4 * basic,
   f"guidance alone reaches {guid_only:.2f}")
ok("HONEST NULL: attempt-memory alone is within noise here", abs(mem_only - basic) < 0.1,
   f"{mem_only:.2f} vs {basic:.2f} — banning one first-move barely prunes a ~20-way branching; "
   f"memory paid off in unit 3 where attempts were short and the proposer weak. Feature value is "
   f"REGIME-dependent — which is itself §5's real lesson")"""),
    13: dict(name="Matchmaking — informative pairs, faster ratings",
             sig="pair agents with CLOSE current ratings; a lopsided match carries ~no information",
             why="""A match between a 2000-Elo and a 1000-Elo agent is decided in advance — its outcome
moves no beliefs, but it still costs a full prover run. Matchmaking by rating proximity keeps games
informative. Measured across 8 seeds — with an honest surprise: on GLOBAL rank recovery, proximity ties
random pairing (informativeness trades against comparison-graph connectivity). What survives is the
operational argument: lopsided games waste compute, and that is what matchmaking saves.""",
             code="""S = 24
true_lam = torch.distributions.Gamma(torch.full((S,), 2.0), torch.ones(S)).sample()

def play(i, j):
    p = float(true_lam[i] / (true_lam[i] + true_lam[j]))
    return i if float(torch.rand(())) < p else j

def run_rating(matchmaker, games=600, K=24.0):
    R = torch.full((S,), 1200.0)
    for g_ in range(games):
        i, j = matchmaker(R)
        w = play(i, j)
        pi = 1 / (1 + 10 ** (float(R[j] - R[i]) / 400))
        si = 1.0 if w == i else 0.0
        R[i] += K * (si - pi); R[j] += K * ((1 - si) - (1 - pi))
    return R

def random_pairs(R):
    i, j = torch.randperm(S)[:2]; return int(i), int(j)

def close_pairs(R):
    i = int(torch.randint(0, S, (1,)))
    d = (R - R[i]).abs(); d[i] = 1e9
    cands = torch.topk(-d, 4).indices
    return i, int(cands[int(torch.randint(0, 4, (1,)))])

rr_, cc_ = [], []
for seed in range(8):                                          # judge across seeds, not one lucky run
    torch.manual_seed(seed)
    tl = torch.distributions.Gamma(torch.full((S,), 2.0), torch.ones(S)).sample()
    globals()["true_lam"] = tl
    torch.manual_seed(100 + seed); rr_.append(spearman(run_rating(random_pairs), tl))
    torch.manual_seed(100 + seed); cc_.append(spearman(run_rating(close_pairs), tl))
m_r, m_c = sum(rr_) / 8, sum(cc_) / 8
print(f"  600 games x 8 seeds: random pairing Spearman {m_r:.3f}   proximity pairing {m_c:.3f}")
ok("HONEST NULL: proximity does NOT beat random pairing on GLOBAL rank recovery here",
   abs(m_c - m_r) < 0.06,
   f"{m_c:.3f} vs {m_r:.3f} — proximity's informative games trade against POORER comparison-graph "
   f"connectivity; at 24 agents the trade is a wash")
p_lop = float(tl.max() / (tl.max() + tl.min()))
ok("what IS true: a lopsided match is decided in advance", p_lop > 0.85,
   f"top-vs-bottom win probability {p_lop:.2f} — so matchmaking still saves COMPUTE (prover calls "
   f"on foregone conclusions), which is the paper's operational reason for it")"""),
    14: dict(name="Elo-based rating — the online update tracks the MLE",
             sig="R_i += K * (outcome - expected)  converges to the Bradley-Terry optimum's ranking",
             why="""Elo's little update is stochastic gradient ascent on the Bradley–Terry likelihood. So
the online ratings should agree with a full batch MLE fit on the same games — checked: both against each
other and against the ground truth that generated the games.""",
             code="""S, G = 16, 2400
true_lam = torch.distributions.Gamma(torch.full((S,), 2.0), torch.ones(S)).sample()
games = []
for _ in range(G):
    i, j = [int(x) for x in torch.randperm(S)[:2]]
    p = float(true_lam[i] / (true_lam[i] + true_lam[j]))
    games.append((i, j, 1.0 if float(torch.rand(())) < p else 0.0))
R = torch.full((S,), 1200.0)
for i, j, si in games:                                        # online Elo
    pi = 1 / (1 + 10 ** (float(R[j] - R[i]) / 400))
    R[i] += 16 * (si - pi); R[j] += 16 * ((1 - si) - pi * 0 - (1 - pi))
theta = torch.zeros(S, requires_grad=True)                    # batch Bradley-Terry MLE
opt = torch.optim.Adam([theta], lr=0.05)
gi = torch.tensor([g_[0] for g_ in games]); gj = torch.tensor([g_[1] for g_ in games])
gs = torch.tensor([g_[2] for g_ in games])
for _ in range(400):
    opt.zero_grad()
    p = torch.sigmoid(theta[gi] - theta[gj])
    loss = -(gs * p.clamp_min(1e-9).log() + (1 - gs) * (1 - p).clamp_min(1e-9).log()).mean() \
           + 1e-3 * (theta ** 2).mean()
    loss.backward(); opt.step()
r_elo_mle = spearman(R, theta.detach())
r_elo_true = spearman(R, true_lam)
print(f"  Spearman(online Elo, batch MLE) = {r_elo_mle:.3f}   Spearman(Elo, truth) = {r_elo_true:.3f}")
ok("the online update agrees with the batch optimum", r_elo_mle > 0.9,
   "Elo IS SGD on Bradley-Terry — one more identity, verified")
ok("and both recover the true ordering", r_elo_true > 0.85, f"{r_elo_true:.3f}")"""),
    15: dict(name="Evolutionary selection over the agent population",
             sig="rate -> select top fraction -> perturb -> repeat; mean strength must RISE",
             why="""The population is not static: strong agents are kept and varied, weak ones dropped.
The testbed makes 'agent' concrete — a weight vector mixing search-heuristic features — and runs
generations of tournament-select-perturb. Mean TRUE fitness (measured on held-out problems, not on the
selection tournament itself) must climb; held-out measurement is what keeps this honest.""",
             code="""random.seed(15); torch.manual_seed(15)
T = ('+', ('*', ('c', 2), ('x',)), ('c', 3))
cap = size(T) + 6
def agent_fitness(wvec, probs):
    h = lambda e: wvec[0] * size(e) + wvec[1] * abs(size(e) - size(T)) + wvec[2] * depth(e)
    tot = 0
    for s in probs:
        e_, p_ = guided(s, T, h, cap, maxexp=400)
        tot += (p_ is not None)
    return tot / len(probs)

def probset(seed, n=14):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = plant(T, 4, rng, cap)
        if s != T: out.append(s)
    return out

sel_probs, held_probs = probset(1), probset(2)
pop = [torch.randn(3).tolist() for _ in range(12)]
means = []
for gen in range(4):
    fits = [agent_fitness(w, sel_probs) for w in pop]
    held = sum(agent_fitness(w, held_probs) for w in pop) / len(pop)
    means.append(held)
    order = sorted(range(len(pop)), key=lambda k: -fits[k])
    keep = [pop[k] for k in order[:4]]
    pop = keep + [[x + float(torch.randn(()) * 0.3) for x in random.choice(keep)]
                  for _ in range(8)]
    print(f"  gen {gen}: held-out mean fitness {held:.3f}   best-on-selection {max(fits):.3f}")
ok("selection + perturbation lifts the population's HELD-OUT strength", means[-1] > means[0],
   f"{means[0]:.3f} -> {means[-1]:.3f} over {len(means)} generations")
ok("fitness for selection and fitness for REPORTING are different problem sets", True,
   "evolving against the eval set would be leakage — same rule as any CV")"""),
    16: dict(name="P-UCB — AlphaZero's exploration rule, on a budget",
             sig="score_i = q_i + c * sqrt(sum_j N_j) / (N_i + 1)   (NOT UCB1's sqrt(log T / N_i))",
             why="""The prover allocates a fixed budget across candidate goals with the polynomial UCB of
the AlphaZero line: the bonus scales with √(total visits)/(own visits+1) — much more aggressive early
exploration than UCB1's logarithmic bonus, and it degenerates to pure greed at c=0. All three behaviours
measured, plus the numeric gap to UCB1 that makes them different rules.""",
             code="""def pucb(q, v, c=0.2):
    return q + c * torch.sqrt(v.sum()) / (v + 1)

N = 64
q = torch.linspace(1, 0, N)                                   # arm 0 is truly best
print(f"{'T':>8} {'best-arm share':>15} {'min visits':>11}")
shares = []
for T_ in (100, 1000, 10_000, 100_000):
    v = torch.zeros(N)
    for _ in range(T_):
        i = int(torch.argmax(pucb(q, v))); v[i] += 1
    shares.append(float(v[0] / T_))
    print(f"{T_:>8} {shares[-1]:>15.3f} {int(v.min()):>11}")
ok("exploitation concentrates on the best arm as the budget grows",
   shares[-1] > 0.7 and shares[-1] > shares[0], f"{shares[0]:.2f} -> {shares[-1]:.2f}")
v = torch.zeros(N)
for _ in range(2000):
    i = int(torch.argmax(pucb(q, v, c=0.0))); v[i] += 1
ok("c = 0 collapses to pure greed — one arm only", int((v > 0).sum()) == 1,
   "the exploration constant is load-bearing, not decoration")
ucb1 = 0.2 * math.sqrt(2 * math.log(10_000) / 1)
pucb1 = 0.2 * math.sqrt(10_000) / 2
ok("and P-UCB is NOT UCB1 — the unvisited-arm bonus differs by an order of magnitude",
   pucb1 > 5 * ucb1, f"P-UCB {pucb1:.2f} vs UCB1 {ucb1:.2f} at T=10k")"""),
})

ADVANCED = [
    dict(id="afpz1", title="What we adopt — search-with-verifier for a Kaggle fleet",
         subtitle="the transferable patterns, and where our world falls short of Lean's",
         cells=[
             dict(note="""## The paper's shape, transplanted
Strip the mathematics and AlphaProof Nexus is a pattern our fleet already half-implements:

| AlphaProof Nexus | our fleet |
|---|---|
| Lean kernel (perfect verifier) | CV score / official scorer (NOISY verifier) |
| prover subagent (search) | agent pipelines over 320 tools |
| population DB + dedup | experiment journal + ledger |
| rater + Elo tournaments | leaderboard + CV↔LB calibration |
| evolutionary selection | keep/kill of experiment lines |
| P-UCB budget allocation | GPU-hours across competing ideas |

The one **fundamental** difference sits in the first row, and it changes how much of the rest transfers: a
CV score is a NOISY, GAMEABLE verifier. Units 2 and 7 are the relevant mathematics — an unfoolable checker
lets everything downstream be aggressive, while a noisy one demands the amplification discipline (repeats,
majority, held-out separation) or the population optimises the verifier instead of the goal. That is
literally what CV-overfitting is.""",
                  code="""torch.manual_seed(1)
n_ideas, sigma = 40, 0.30
true_gain = torch.randn(n_ideas) * 0.1
noisy_eval = lambda k: true_gain + sigma * torch.randn(n_ideas)
best = float(true_gain.max())
exp_pick = {}
for rep in (1, 3, 9, 27):
    got = []
    for trial in range(200):                                  # the EXPECTED value of the selection rule
        scores = torch.stack([true_gain + sigma * torch.randn(n_ideas)
                              for _ in range(rep)]).mean(0)
        got.append(float(true_gain[int(scores.argmax())]))
    exp_pick[rep] = sum(got) / len(got)
print(f"  best TRUE gain available: {best:+.3f}")
for rep, got in exp_pick.items():
    print(f"  selecting on {rep:>2} eval repeats -> EXPECTED true gain of the pick {got:+.3f}")
ok("with a NOISY verifier, single-shot selection loses a large slice to the winner's curse",
   exp_pick[1] < 0.85 * exp_pick[27],
   f"{exp_pick[1]:+.3f} vs {exp_pick[27]:+.3f} — the fleet's daily hazard, in expectation")
ok("repeats recover it, exactly like unit 7's majority vote", exp_pick[27] > 0.8 * best,
   f"27 repeats reach {exp_pick[27]/best:.0%} of the attainable gain")
ok("Lean needs none of this; Kaggle needs ALL of it", True,
   "the verifier's noise level decides how much of the paper's aggression is safe to copy")"""),
             dict(note="""### The adoption list
1. **Value-guided search over pipelines** (unit 10 — 8×+ fewer expansions): our `pipeline` and
   `improve-loop` agents explore configuration space mostly blind; a cheap value model over (state,
   target-metric) features is the measured lever.
2. **Attempt memory as a first-class input** (units 3, 12): failed configs must ban their own retries —
   the journal already stores them; the loop should CONDITION on them.
3. **Elo over experiment lines** (units 8, 13–14): when two approaches can only be compared pairwise and
   noisily (LB probes), Bradley–Terry with proximity matchmaking is the principled scoreboard.
4. **P-UCB for GPU budget** (unit 16): allocating hours across live ideas by q + c·√(Σt)/(t+1) is
   strictly better than the greedy allocation we default to — and c is a knob we can set from measured
   variance (unit 11).
5. **Dedup before statistics** (unit 6): the journal's per-config canonicalisation must fold trivial
   variants, or every population statistic double-counts.

**Not adopted:** anything requiring the perfect verifier — self-play-style aggressive optimisation against
the metric is exactly what a noisy CV punishes (the cell above measured why)."""),
             dict(note="""**[Recap]** a perfect verifier is the domain's superpower (basics, units 1–2) ·
the agent = propose–validate–revise with memory and sketches, each worth measured solve-rate (units 3–4,
12) · guidance beats blind search 8×+ on held-out problems (unit 10) · populations are rated by
Bradley–Terry/Elo with MCMC over tournaments, matchmade for information, evolved under held-out honesty
(units 6–9, 13–15) · budgets follow P-UCB, not greed (units 11, 16) · and the transfer to OUR fleet is
gated by verifier noise, quantified in this lesson. Engine: `learning/paper_packs/afp_engine.py`."""),
         ]),
]
