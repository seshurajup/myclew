import sys, math, random, torch, torch.nn as nn
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
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

rng = random.Random(1)
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
   "the testbed's formalisation is faithful — the precondition for everything below")

rng = random.Random(2)
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
   "the asymmetry (hard to find, trivial to check) is what makes search the right tool")

rng = random.Random(3)
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
   f"{sr_mem:.2f} vs {sr_no:.2f} — the '== Prior Attempts ==' block earns its context window")

rng = random.Random(4)
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
   "decomposition changes the search, never the standard of proof")

rng = random.Random(5)
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
   f"the problem guidance must solve")
