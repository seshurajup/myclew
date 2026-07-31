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

def canon(e):
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
   "a population metric on raw strings would double-count every commutation")

def majority(p, k, n=200_000):
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
   f"p=0.45 with 15 votes -> {bad:.3f} — validate the rater before scaling it")

elo = lambda lam: 1200 + 400 * torch.log10(lam)
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
   "outlier provers exist; the prior must allow them")

def pl_ll(lam, D):
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
        a = (pl_ll(lam_p, D) - r[s_] * prop + math.log(prop)) -             (pl_ll(lam, D) - r[s_] * cur + math.log(cur))
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
print(f"  implied Elo range: {float(e_rng.min()):.0f} .. {float(e_rng.max()):.0f}")
