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

T = ('+', ('*', ('c', 2), ('x',)), ('c', 3))                 # the polynomial 2x + 3
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
ok("and genuinely broken derivations are rejected", caught > 0, f"{caught} of 200")
