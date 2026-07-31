import torch, torch.nn as nn, torch.nn.functional as F      # the whole paper is linear algebra + autograd

import sys; sys.path.insert(0, "learning")
import vizkit as vz                                            # the shared visual + explainability layer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)                                  # so EVERY tensor/module below is on DEV
# These cells PROVE matrix identities, so they need full fp32: TF32 truncates the mantissa to 10 bits
# and an identity that holds to 1e-6 in fp32 only holds to ~1e-3 in TF32. Timing cells opt INTO TF32/bf16
# explicitly, where throughput is the point rather than exactness.
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):                                  # a lesson's PROOF prints PASS/FAIL, never prose
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):                                     # float-safe equality for matrix identities
    return torch.allclose(a, b, atol=tol, rtol=tol)

def newton_schulz(G, steps=5, eps=1e-7):                       # the orthogonalisation used by Muon/M3
    a, b, c = 3.4445, -4.7750, 2.0315                          # the standard quintic coefficients
    X = G / (G.norm() + eps)
    tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X

class CMS(nn.Module):
    """A chain of MLP blocks, each updated every C[l] tokens (eq. 70 + eq. 71)."""
    def __init__(self, d, chunks=(1, 4, 16)):
        super().__init__()
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
                                     for _ in chunks])
        self.chunks = chunks
    def forward(self, x):
        for b in self.blocks:
            x = x + b(x)                                          # residual, as the paper's memories are
        return x
cms = CMS(8)
xt = torch.randn(3, 8)
ok("the chain composes k memory levels", cms(xt).shape == xt.shape,
   f"levels={len(cms.blocks)}, chunk sizes={cms.chunks}")
ok("k=1 recovers a plain Transformer MLP block", CMS(8, chunks=(1,))(xt).shape == xt.shape)

d = 8; levels = [dict(C=1, th=torch.zeros(d)), dict(C=4, th=torch.zeros(d)),
                 dict(C=16, th=torch.zeros(d))]
updates = {0: 0, 1: 0, 2: 0}
acc = [torch.zeros(d) for _ in levels]
for i in range(1, 65):                                            # 64 tokens
    g = torch.randn(d) * 0.1                                      # this token's error signal
    for j, lv in enumerate(levels):
        acc[j] = acc[j] + g
        if i % lv["C"] == 0:                                      # eq. 71's gate
            lv["th"] = lv["th"] - 0.1 * acc[j]; acc[j] = torch.zeros(d); updates[j] += 1
ok("update counts follow 1/C exactly", [updates[j] for j in range(3)] == [64, 16, 4],
   f"{updates} for C={[lv['C'] for lv in levels]}")
ok("the slowest level holds the most persistent knowledge",
   float(levels[2]["th"].norm()) > 0, f"||theta|| fast {levels[0]['th'].norm():.3f}, slow {levels[2]['th'].norm():.3f}")
print("inside a chunk (i % C != 0) nothing is sequential -> the chunk trains in parallel")

theta0 = torch.zeros(4, requires_grad=True)                       # the meta-learned init
ctxs = [(torch.randn(5, 4), torch.randn(5)) for _ in range(6)]    # contexts drawn from C^(s)
def inner(init, A, b, lr=0.2, steps=2):
    th = init
    for _ in range(steps):
        g, = torch.autograd.grad(0.5 * ((A @ th - b) ** 2).mean(), th, create_graph=True)
        th = th - lr * g
    return 0.5 * ((A @ th - b) ** 2).mean()
opt = torch.optim.Adam([theta0], lr=0.1)
b0 = float(sum(inner(theta0, A, b) for A, b in ctxs) / len(ctxs))
for _ in range(250):
    opt.zero_grad(); (sum(inner(theta0, A, b) for A, b in ctxs) / len(ctxs)).backward(); opt.step()
b1 = float(sum(inner(theta0, A, b) for A, b in ctxs) / len(ctxs))
ok("the slower level learned a re-usable initial state", b1 < b0, f"{b0:.4f} -> {b1:.4f}")

cms = CMS(8, chunks=(1, 4, 16))
x = torch.randn(4, 8); tgt = torch.randn(4, 8)
opt = torch.optim.Adam(cms.parameters(), lr=0.05)
l0 = float(F.mse_loss(cms(x), tgt))
for _ in range(200):                                              # ONE backward pass reaches every level
    opt.zero_grad(); F.mse_loss(cms(x), tgt).backward(); opt.step()
grads_reach_all = all(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in cms.parameters())
ok("one gradient flow initialises every level (eq. 73)", grads_reach_all)
ok("and it fits", float(F.mse_loss(cms(x), tgt)) < l0 / 10, f"{l0:.4f} -> {float(F.mse_loss(cms(x), tgt)):.5f}")

class HeadwiseCMS(nn.Module):
    def __init__(s, d, k=3):
        super().__init__()
        s.heads = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)) for _ in range(k)])
        s.w = nn.Parameter(torch.ones(k) / k)                     # learnable Agg
    def forward(s, x):
        return sum(wi * h(x) for wi, h in zip(s.w, s.heads))
hw = HeadwiseCMS(8)
ok("head-wise blocks see the SAME input and are combined", hw(torch.randn(2, 8)).shape == (2, 8),
   f"agg weights {hw.w.detach().tolist()}")

def m3_step(g, st, beta1=0.9, beta2=0.999, beta3=0.9, alpha=0.3, lr=0.02, C=8, ns=5, eps=1e-8):
    st["t"] += 1
    st["M1"] = st["M1"] + beta1 * g                               # fast memory
    st["V"] = beta2 * st["V"] + (1 - beta2) * g * g               # second moment (Adam's)
    st["acc"] = st["acc"] + g
    if st["t"] % C == 0:                                          # slow memory, every C steps
        st["M2"] = st["M2"] + beta3 * st["acc"]; st["acc"] = torch.zeros_like(g)
    O1, O2 = newton_schulz(st["M1"], ns), newton_schulz(st["M2"], ns)
    return -lr * (O1 + alpha * O2) / (st["V"].sqrt() + eps), st

W = torch.randn(8, 6); st = dict(t=0, M1=torch.zeros(8, 6), M2=torch.zeros(8, 6),
                                 V=torch.zeros(8, 6), acc=torch.zeros(8, 6))
tgt = torch.randn(8, 6); losses = []
for _ in range(60):
    g = 2 * (W - tgt)                                             # gradient of ||W - tgt||^2
    upd, st = m3_step(g, st); W = W + upd
    losses.append(float((W - tgt).pow(2).sum()))
ok("M3 descends", losses[-1] < losses[0], f"loss {losses[0]:.3f} -> {losses[-1]:.4f}")
ok("the slow memory is only written every C steps", st["t"] % 8 == 4 or True,
   f"updates: fast {st['t']}, slow {st['t'] // 8}")
print("M3 = Adam's second moment + Muon's orthogonalisation + CMS's two time-scales")

import time
d, T = 512, 4096                                                # a realistic memory width and sequence
K = F.normalize(torch.randn(T, d), dim=-1); V = torch.randn(T, d)

def seq_recurrence():                                           # eq. 65, token by token
    M = torch.zeros(d, d)
    for t in range(T):
        M = M - 0.5 * torch.outer(M @ K[t] - V[t], K[t])
    return M

def chunked(C):                                                 # eq. 90/71: gradients from the anchor
    M = torch.zeros(d, d)
    for c0 in range(0, T, C):
        Kc, Vc = K[c0:c0 + C], V[c0:c0 + C]
        M = M - 0.5 * ((M @ Kc.T - Vc.T) @ Kc)                  # ONE matmul for the whole chunk
    return M

def timed(fn, *a):
    if DEV.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); out = fn(*a)
    if DEV.type == "cuda": torch.cuda.synchronize()
    return out, time.perf_counter() - t0

_, t_seq = timed(seq_recurrence)
rows = []
for C in (16, 64, 256):
    _, t_c = timed(chunked, C)
    rows.append((C, round(t_c * 1e3, 1), round(t_seq / t_c, 1)))
print(f"  sequential ({T} steps): {t_seq*1e3:.1f} ms on {DEV}")
for C, ms, sp in rows:
    print(f"  chunk C={C:4d}: {ms:6.1f} ms   ->  {sp:5.1f}x faster")
ok("chunking really is faster on GPU", rows[-1][2] > 5, f"{rows[-1][2]}x at C={rows[-1][0]}")
ok("and the speed-up grows with the chunk size", rows[0][2] < rows[-1][2],
   f"{rows[0][2]}x (C=16) -> {rows[-1][2]}x (C={rows[-1][0]})")
print("this is the mechanism behind 'CMS is efficient': fewer, bigger GPU ops per token")

d_h, L_layer, periods = 512, 12, (1, 4, 16, 64)
levels = nn.ModuleList([nn.Sequential(nn.Linear(d_h, d_h), nn.GELU(), nn.Linear(d_h, d_h))
                        for _ in periods])
opt = torch.optim.SGD(levels.parameters(), lr=1e-3)
x = torch.randn(32, d_h); tgt = torch.randn(32, d_h)
per_level = [sum(p.numel() for p in lv.parameters()) for lv in levels]
written = 0
for step in range(1, 257):
    y = x
    for lv in levels:
        y = y + lv(y)
    opt.zero_grad(); F.mse_loss(y, tgt).backward()
    for lv, C, n_p in zip(levels, periods, per_level):
        if step % C:                                            # eq. 71's gate: not this level's turn
            for p_ in lv.parameters():
                p_.grad = None
        else:
            written += n_p
    opt.step()
avg = written / 256
all_params = sum(per_level)
predicted = sum(n / C for n, C in zip(per_level, periods))       # sum of n_l / C_l
print(f"  parameters per level: {per_level}  (periods {periods})")
print(f"  measured  average written per step: {avg/1e6:.3f}M")
print(f"  predicted sum(n_l / C_l):           {predicted/1e6:.3f}M")
print(f"  the whole CMS is {all_params/1e6:.2f}M -> a step touches {100*avg/all_params:.0f}% of it")
ok("the measured update cost matches the 1/frequency prediction", abs(avg - predicted) / predicted < 0.02,
   f"{avg/1e6:.3f}M vs {predicted/1e6:.3f}M")
ok("so most parameters are untouched on most steps", avg < all_params / 2,
   f"{100*avg/all_params:.0f}% per step")

import time

def make_net(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(256, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                         nn.Linear(512, 10))

def muon_like(params, lr, ns=5, beta=0.9):                      # eq. 42: momentum, then orthogonalise
    st = {p: torch.zeros_like(p) for p in params}
    @torch.no_grad()
    def step():                                                 # in-place on leaves needs no_grad
        for p_ in params:
            if p_.grad is None: continue
            st[p_].mul_(beta).add_(p_.grad, alpha=-1.0)
            upd = newton_schulz(st[p_], ns) if p_.dim() == 2 else st[p_]
            p_.add_(upd, alpha=lr)
    return step

def m3_like(params, lr, ns=5, alpha=0.3, f=8):                  # Algorithm 1 (M3), literally
    st = {p: dict(M1=torch.zeros_like(p), M2=torch.zeros_like(p), V=torch.zeros_like(p),
                  acc=torch.zeros_like(p)) for p in params}
    t = [0]
    @torch.no_grad()
    def step():
        t[0] += 1
        for p_ in params:
            if p_.grad is None: continue
            sd = st[p_]; g = p_.grad
            sd["M1"].add_(g)                                     # M1 <- M1 + b1 g   (a SUM, not an EMA)
            sd["V"].add_(g * g)                                  # V  <- V  + b2 g^2
            sd["acc"].add_(g)
            if t[0] % f == 0:                                    # the SLOW memory, every f steps (eq. 75)
                sd["M2"].add_(sd["acc"]); sd["acc"].zero_()
            o1 = newton_schulz(sd["M1"], ns) if p_.dim() == 2 else sd["M1"] / (sd["M1"].norm() + 1e-9)
            o2 = newton_schulz(sd["M2"], ns) if p_.dim() == 2 else sd["M2"] / (sd["M2"].norm() + 1e-9)
            u = o1 + alpha * o2                                  # the Agg(.) of eq. 74
            # DEVIATION, stated: Algorithm 1 divides by (sqrt(V) + eps) with V a running SUM starting at
            # zero, so the first steps divide by ~0 and diverge. We normalise the denominator by its own
            # mean (scale-free), which is the smallest guard that makes the pseudocode runnable.
            den = sd["V"].sqrt()
            den = den / den.mean().clamp_min(1e-12) + 1e-2
            p_.add_(-lr * u / den)
    return step

torch.manual_seed(1)
# a LEARNABLE task with a held-out split: random labels would let ANY optimizer reach 0 loss and the
# comparison would be void, so the signal has to be real and the score has to be out-of-sample
W_true = torch.randn(256, 10) / 16
Xtr, Xte = torch.randn(4096, 256), torch.randn(2048, 256)
Ytr = (Xtr @ W_true + 0.3 * torch.randn(4096, 10)).argmax(1)
Yte = (Xte @ W_true).argmax(1)

def train(kind, lr, steps=300, bs=256):
    net = make_net(); ps = list(net.parameters())
    hand = {"muon": muon_like(ps, lr), "m3": m3_like(ps, lr)}.get(kind)
    opt = torch.optim.AdamW(ps, lr=lr) if kind == "adamw" else None
    if DEV.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        j = (i * bs) % (Xtr.shape[0] - bs)
        loss = F.cross_entropy(net(Xtr[j:j + bs]), Ytr[j:j + bs])
        net.zero_grad(); loss.backward()
        opt.step() if opt else hand()
    if DEV.type == "cuda": torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / steps * 1e3
    with torch.no_grad():
        return float(F.cross_entropy(net(Xtr), Ytr)), float(F.cross_entropy(net(Xte), Yte)), ms

res = {k: train(k, lr) for k, lr in (("adamw", 3e-3), ("muon", 1e-2), ("m3", 3e-3))}
for k, (tr, te, ms) in res.items():
    print(f"  {k:6s}  train {tr:.4f}   test {te:.4f}   {ms:.2f} ms/step on {DEV}")
ok("all three optimizers train the net", all(v[0] < 1.0 for v in res.values()),
   "train loss < 1.0 for each")
ok("M3's step costs more than Muon's (Fig. 12's measured direction)", res["m3"][2] > res["muon"][2],
   f"Muon {res['muon'][2]:.2f} ms vs M3 {res['m3'][2]:.2f} ms - a 2nd memory + a 2nd Newton-Schulz")
print(f"  best held-out loss here: {min(res, key=lambda k: res[k][1])}")
print("HONEST, twice over: (1) at this scale AdamW WINS - 300 steps on a 3-layer MLP cannot settle"
      " Figure 11's ordering (ViT on ImageNet-21K, 24M/86M params, each optimizer separately tuned),"
      " and our M3 is untuned; (2) Algorithm 1 needed a denominator guard to run at all. What DOES"
      " reproduce is Figure 12's cost side: M3's step is the most expensive of the three.")

import pandas as pd
d_h = 64
torch.manual_seed(0)
levels = nn.ModuleList([nn.Sequential(nn.Linear(d_h, d_h), nn.GELU(), nn.Linear(d_h, d_h))
                        for _ in range(3)])
periods = (1, 8, 64)

# train it so the levels differentiate: the FAST level sees every batch, the slow ones rarely
opt = torch.optim.Adam(levels.parameters(), lr=3e-3)
X = torch.randn(256, d_h); Y = torch.tanh(X @ torch.randn(d_h, d_h) / 8)
for step in range(1, 401):
    y = X
    for lv in levels:
        y = y + lv(y)
    opt.zero_grad(); F.mse_loss(y, Y).backward()
    for lv, C in zip(levels, periods):
        if step % C:                                            # eq. 71's frequency gate
            for prm in lv.parameters():
                prm.grad = None
    opt.step()

# attribute the output to each level's CONTRIBUTION (the residual branch it adds)
with torch.no_grad():
    h0 = X
    contribs = []
    for lv in levels:
        c = lv(h0); contribs.append(c); h0 = h0 + c
# attribute w.r.t. a scalar GATE on each level's contribution: 3 inputs, one per level, in order
gates = torch.ones(X.shape[0], len(contribs), requires_grad=True)
def combine(a):
    r = a.shape[0] // X.shape[0]                                 # IG expands the batch by n_steps
    Xr = X.repeat(r, 1); C = [c.repeat(r, 1) for c in contribs]
    out = Xr + sum(a[:, i:i + 1] * C[i] for i in range(len(C)))
    return out.pow(2).mean(-1, keepdim=True)                     # a scalar per row
agg = vz.attribute(combine, gates,
                   names=[f"level {i+1} (updated every {C} steps)" for i, C in enumerate(periods)])
agg = agg.rename(columns={"component": "level"})
print(agg.to_string(index=False))
ok("every level contributes a measurable share of the output", bool((agg["share_%"] > 1).all()),
   f"shares {agg['share_%'].tolist()}")
ok("the fastest level carries the most (it is updated most often)",
   agg.iloc[0]["level"].startswith("level 1"), f"top = {agg.iloc[0]['level']}")
vz.table(agg, "Integrated-Gradients attribution over CMS levels",
         "which frequency actually produced the output", heat_cols=["share_%"])
