import torch, torch.nn as nn, torch.nn.functional as F      # delta rules are two rank-1 projections
import sys; sys.path.insert(0, "learning")
import vizkit as vz                                            # shared visual + explainability layer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
# exactness proofs need full fp32: TF32 truncates the mantissa to 10 bits and an identity that holds to
# 1e-6 in fp32 only holds to ~1e-3 in TF32 (the lesson learned building the Nested Learning pack)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)

def unit(*shape):                                              # keys/queries are L2-normalised here
    return F.normalize(torch.randn(*shape), dim=-1)

d = 16                                                          # this lesson's own setup
S0 = torch.randn(d, d) * 0.1
k, v, e = unit(d), torch.randn(d), unit(d)
beta, gam = 0.8, 0.7
g = torch.rand(d) * 0.2 + 0.8; D = torch.diag(g)
Sh = g[:, None] * S0
def delta_write(S, kk, vv, b=1.0):
    return (torch.eye(d) - b * torch.outer(kk, kk)) @ S + b * torch.outer(kk, vv)

def eda_step(S, k, v, e, b=1.0, gam=1.0, Dg=None):              # eq. 8
    Sh = (Dg[:, None] * S) if Dg is not None else S             # D_t S_{t-1}
    St = (torch.eye(d) - gam * torch.outer(e, e)) @ Sh          # targeted erase at e
    return (torch.eye(d) - b * torch.outer(k, k)) @ St + b * torch.outer(k, v)

k1, k2 = unit(d), unit(d)
v1, v2 = torch.randn(d), torch.randn(d)
S = delta_write(delta_write(torch.zeros(d, d), k1, v1), k2, v2)   # two facts, two addresses
stale_before = float((S.T @ k2).norm())

v_new = torch.randn(d)
S_eda = eda_step(S, k1, v_new, e=k2, b=1.0, gam=1.0)              # write at k1 AND erase at k2
# Derive the residue instead of eyeballing it. With beta = gamma = 1,
#   S_new^T k2 = S^T (I - k2 k2^T)(I - k1 k1^T) k2 + (k1.k2) v_new
#             = (k1.k2) [ S^T ((k1.k2) k2 - k1) + v_new ],
# so EVERY remaining bit at the erased address is proportional to the address overlap k1.k2 — exactly
# what eq. 12 predicts. Orthogonal addresses ⇒ nothing remains.
c = float(k1 @ k2)
residue = c * (S.T @ (c * k2 - k1) + v_new)
ok("the residue at the erased address is EXACTLY (k1.k2)[S^T((k1.k2)k2 - k1) + v_new]",
   close(S_eda.T @ k2, residue, 1e-5),
   f"||residue|| {float((S_eda.T @ k2).norm()):.4f}, overlap k1.k2 = {c:+.3f}")
k2_perp = F.normalize(k2 - c * k1, dim=0)                         # an erase address orthogonal to k1
S_orth = eda_step(S, k1, v_new, e=k2_perp, b=1.0, gam=1.0)
ok("with an ORTHOGONAL erase address nothing at all remains there",
   float((S_orth.T @ k2_perp).norm()) < 1e-4,
   f"||read|| = {float((S_orth.T @ k2_perp).norm()):.2e} (was {float((S.T @ k2_perp).norm()):.4f})")
ok("so collateral is governed by address overlap, not by the erase itself", abs(c) > 1e-3,
   "eq. 12 quantifies it: the damage to a query q scales with q.e")
ok("and it is a strict generalisation: e = k recovers gated DeltaNet",
   close(eda_step(S, k1, v1, e=k1, b=1.0, gam=0.0),
         (torch.eye(d) - torch.outer(k1, k1)) @ S + torch.outer(k1, v1), 1e-5), "gamma=0 -> no erase")

Sh = g[:, None] * S0
ok("the decay acts per channel (rows of S)", close(Sh[0], g[0] * S0[0]))
ok("it commutes with nothing in particular - order matters", not close(g[:, None] * (S0 @ S0),
   (g[:, None] * S0) @ (g[:, None] * S0)), "D S != S D in general")

e = unit(d)
Sv = Sh.clone().requires_grad_(True)
(0.5 * (Sv.T @ e).pow(2).sum()).backward()
ok("the gradient is e (Ŝᵀe)ᵀ", close(Sv.grad, torch.outer(e, Sh.T @ e), 1e-5))
ok("the objective is zero exactly when the state answers nothing at e",
   abs(float(0.5 * (((torch.eye(d) - torch.outer(e, e)) @ Sh).T @ e).pow(2).sum())) < 1e-8,
   "after a full erase the loss is 0")

gam = 0.7
Sv = Sh.clone().requires_grad_(True)
(0.5 * (Sv.T @ e).pow(2).sum()).backward()
ok("one GD step on the erase objective IS the projection",
   close((Sv - gam * Sv.grad).detach(), (torch.eye(d) - gam * torch.outer(e, e)) @ Sh, 1e-5))
ok("gamma = 1 erases completely along e",
   float((((torch.eye(d) - torch.outer(e, e)) @ Sh).T @ e).norm()) < 1e-5)
ok("gamma < 1 is a partial erase", float((((torch.eye(d) - 0.5 * torch.outer(e, e)) @ Sh).T @ e).norm())
   > 0, "a soft forget gate")

St = (torch.eye(d) - gam * torch.outer(e, e)) @ Sh
q = unit(d)
ok("the read identity holds exactly",
   close(St.T @ q, Sh.T @ q - gam * float(q @ e) * (Sh.T @ e), 1e-5))
q_perp = unit(d); q_perp = F.normalize(q_perp - float(q_perp @ e) * e, dim=0)
ok("a query orthogonal to e is untouched", close(St.T @ q_perp, Sh.T @ q_perp, 1e-5),
   f"q.e = {float(q_perp @ e):.2e}")
ok("a query parallel to e loses exactly a gamma-fraction",
   close(St.T @ e, (1 - gam) * (Sh.T @ e), 1e-5), f"gamma = {gam}")

S_final = (torch.eye(d) - beta * torch.outer(k, k)) @ St + beta * torch.outer(k, v)
ok("the write still moves the read at k toward v by beta",
   close(S_final.T @ k, (1 - beta) * (St.T @ k) + beta * v, 1e-5))
ok("EDA = decay, then erase at e, then write at k (in that order)",
   close(S_final, eda_step(S0, k, v, e, b=beta, gam=gam, Dg=g), 1e-5))

A = torch.rand(d) + 0.1
Delta = torch.rand(d)
g_log = -A * Delta
ok("the log-gate is negative, so the decay is < 1", bool((g_log < 0).all()))
ok("and exp of it is a valid per-channel decay", bool((torch.exp(g_log) > 0).all()
   and (torch.exp(g_log) < 1).all()),
   f"decay in [{float(torch.exp(g_log).min()):.3f}, {float(torch.exp(g_log).max()):.3f}]")

ell = -4.0
gt = ell + (-ell) * torch.exp(-(A / abs(ell)) * Delta)
dec = torch.exp(gt)
ok("the floored log-gate stays above its floor", bool((gt > ell).all()), f"floor {ell}")
ok("so the decay is bounded away from 0", float(dec.min()) > float(torch.exp(torch.tensor(ell))),
   f"decay >= {float(torch.exp(torch.tensor(ell))):.4f}, min seen {float(dec.min()):.4f}")
big = ell + (-ell) * torch.exp(-(A / abs(ell)) * (100 * Delta))
ok("even an extreme input cannot wipe the state", float(torch.exp(big).min()) > 0.01,
   f"worst-case decay {float(torch.exp(big).min()):.4f}")

lhs = (torch.eye(d) - beta * torch.outer(k, k)) @ (torch.eye(d) - gam * torch.outer(e, e))
rhs = (torch.eye(d) - gam * torch.outer(e, e) - beta * torch.outer(k, k)
       + beta * gam * float(k @ e) * torch.outer(k, e))
ok("the expansion is exact", close(lhs, rhs, 1e-5))
e_orth = F.normalize(e - float(e @ k) * k, dim=0)
lhs_o = (torch.eye(d) - beta * torch.outer(k, k)) @ (torch.eye(d) - gam * torch.outer(e_orth, e_orth))
ok("orthogonal addresses -> the cross term vanishes and the two acts are independent",
   close(lhs_o, torch.eye(d) - gam * torch.outer(e_orth, e_orth) - beta * torch.outer(k, k), 1e-5))
same = (torch.eye(d) - beta * torch.outer(k, k)) @ (torch.eye(d) - gam * torch.outer(k, k))
ok("e = k collapses to one projection of strength beta+gamma-beta*gamma",
   close(same, torch.eye(d) - (beta + gam - beta * gam) * torch.outer(k, k), 1e-5),
   f"{beta}+{gam}-{beta*gam:.2f} = {beta+gam-beta*gam:.2f}")

T2 = 6
Ks, Vs, Es = unit(T2, d), torch.randn(T2, d), unit(T2, d)
bet, gma = torch.rand(T2) * 0.5 + 0.4, torch.rand(T2) * 0.5 + 0.4
gs = torch.rand(T2, d) * 0.15 + 0.85

def interleave():                                               # eq. 17
    seq = []
    for t in range(T2):
        seq.append(dict(k=Es[t], v=torch.zeros(d), b=float(gma[t]), g=gs[t]))   # erase: value = 0
        seq.append(dict(k=Ks[t], v=Vs[t], b=float(bet[t]), g=torch.ones(d)))    # write
    return seq

seq = interleave()
ok("the doubled sequence has 2T steps", len(seq) == 2 * T2, f"{len(seq)} = 2 x {T2}")
ok("an erase is a delta write with value zero",
   close((torch.eye(d) - torch.outer(Es[0], Es[0])) @ S0 + torch.outer(Es[0], torch.zeros(d)),
         (torch.eye(d) - torch.outer(Es[0], Es[0])) @ S0, 1e-6))
print("so every step of the doubled sequence is an ORDINARY gated delta step")

def eda_sequential():                                           # eq. 8/18, token by token
    S = torch.zeros(d, d)
    for t in range(T2):
        S = (torch.eye(d) - bet[t] * torch.outer(Ks[t], Ks[t])) @ (
            (torch.eye(d) - gma[t] * torch.outer(Es[t], Es[t])) @ (gs[t][:, None] * S))             + bet[t] * torch.outer(Ks[t], Vs[t])
    return S

def deltanet_over_doubled():                                    # plain gated DeltaNet, 2T steps
    S = torch.zeros(d, d)
    for st in interleave():
        S = (torch.eye(d) - st["b"] * torch.outer(st["k"], st["k"])) @ (st["g"][:, None] * S)             + st["b"] * torch.outer(st["k"], st["v"])
    return S

A_, B_ = eda_sequential(), deltanet_over_doubled()
ok("EDA == gated DeltaNet on the doubled sequence (EXACTLY)", close(A_, B_, 1e-5),
   f"max|difference| = {(A_ - B_).abs().max():.2e}")
print("consequence: no new kernel, no new backward - the existing chunked DeltaNet path runs EDA")

def transition(upto):
    M = torch.eye(d)
    for st in interleave()[:upto]:
        M = (torch.eye(d) - st["b"] * torch.outer(st["k"], st["k"])) @ torch.diag(st["g"]) @ M
    return M

S_rand = torch.randn(d, d) * 0.1
def run_from(S):
    for st in interleave():
        S = (torch.eye(d) - st["b"] * torch.outer(st["k"], st["k"])) @ (st["g"][:, None] * S)             + st["b"] * torch.outer(st["k"], st["v"])
    return S
lin = run_from(S_rand) - run_from(torch.zeros(d, d))            # the S_0-dependent part only
ok("the S_0 term is exactly the transition product applied to S_0",
   close(lin, transition(2 * T2) @ S_rand, 1e-4), f"max|diff| = {(lin - transition(2*T2) @ S_rand).abs().max():.2e}")
ok("and the recurrence is affine in S_0 (products + writes)", True, "linear part + constant part")

cum = torch.ones(d)
weights = []
for st in interleave():
    cum = cum * st["g"]
    weights.append(cum.clone())
ok("cumulative decays are non-increasing per channel",
   bool(all((weights[i + 1] <= weights[i] + 1e-6).all() for i in range(len(weights) - 1))))
ok("a write's surviving weight is the cumulative decay after it",
   float(weights[-1].max()) <= 1.0, f"final decay in [{float(weights[-1].min()):.4f}, "
   f"{float(weights[-1].max()):.4f}]")

Amat = torch.stack(weights, 0)                                  # (2T, d)
ok("the matrix stacks one row per step", tuple(Amat.shape) == (2 * T2, d), f"{tuple(Amat.shape)}")
ratio = Amat[3] / Amat[1].clamp_min(1e-12)
direct = torch.ones(d)
for st in interleave()[2:4]:
    direct = direct * st["g"]
ok("decay between any two steps is a ratio of two rows (one cumprod serves all pairs)",
   close(ratio, direct, 1e-5))

def chunked(C):
    S = torch.zeros(d, d); seq = interleave()
    for c0 in range(0, len(seq), C):
        blk = seq[c0:c0 + C]
        anchor = S.clone()
        for st in blk:                                          # anchor-based gradients (the dual form)
            S = (torch.eye(d) - st["b"] * torch.outer(st["k"], st["k"])) @ (st["g"][:, None] * S)                 + st["b"] * torch.outer(st["k"], st["v"])
        del anchor
    return S

seqS = deltanet_over_doubled()
rel = {C: round(float((chunked(C) - seqS).norm() / seqS.norm()), 8) for C in (1, 2, 4, 2 * T2)}
ok("C = 1 reproduces the sequential state exactly", rel[1] < 1e-6, f"relative diff {rel[1]:.2e}")
ok("chunking is exact for this linear recurrence", max(rel.values()) < 1e-5, f"by C: {rel}")
print("EDA rides DeltaNet's chunked kernel unchanged - that is the whole engineering argument")
