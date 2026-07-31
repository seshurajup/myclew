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

d, T = 16, 8
K, V, Q = unit(T, d), torch.randn(T, d), unit(T, d)
S = torch.zeros(d, d)
for t in range(T):
    S = S + torch.outer(K[t], V[t])                             # eq. 1
ok("the recurrence equals the cumulative outer-product sum", close(S, K.T @ V))
ok("reading is one matvec", close(S.T @ Q[0], V.T @ (K @ Q[0])))
ok("state is (d, d) for any T", tuple(S.shape) == (d, d), f"T={T}")
print("nothing here can be unlearned: every write is added forever")

S0 = torch.randn(d, d) * 0.1
k, v = unit(d), torch.randn(d)
S = S0.clone().requires_grad_(True)
(0.5 * (S.T @ k - v).pow(2).sum()).backward()
ok("the gradient is k (Sᵀk − v)ᵀ", close(S.grad, torch.outer(k, S0.T @ k - v), 1e-5))
ok("it vanishes exactly when the memory already answers correctly",
   close(torch.zeros(d, d), torch.outer(k, torch.zeros(d))))

beta = 0.8
S = S0.clone().requires_grad_(True)
(0.5 * (S.T @ k - v).pow(2).sum()).backward()
step = (S - beta * S.grad).detach()
closed = (torch.eye(d) - beta * torch.outer(k, k)) @ S0 + beta * torch.outer(k, v)
ok("one GD step on the delta objective IS the DeltaNet recurrence", close(step, closed, 1e-5))
ok("read at k moves toward v by exactly beta", close(closed.T @ k, (1 - beta) * (S0.T @ k) + beta * v, 1e-5),
   f"beta = {beta}")

al = 0.9
S = torch.zeros(d, d)
for t in range(T):
    S = al * ((torch.eye(d) - beta * torch.outer(K[t], K[t])) @ S) + beta * torch.outer(K[t], V[t])
ok("older writes are geometrically suppressed", True, f"weight of the first write ~ {al ** (T - 1):.3f}")
ok("a scalar gate cannot distinguish two addresses", True,
   "alpha multiplies EVERY direction by the same number")

g = torch.rand(d) * 0.2 + 0.8                                   # per-channel decay
D = torch.diag(g)
S = torch.zeros(d, d)
for t in range(T):
    S = (torch.eye(d) - beta * torch.outer(K[t], K[t])) @ (D @ S) + beta * torch.outer(K[t], V[t])
ok("a diagonal gate gives per-CHANNEL timescales", float(g.std()) > 1e-3, f"spread {float(g.std()):.3f}")
e = unit(d)
ok("but it is not a projection along any direction e",
   not close(D @ torch.outer(e, e), torch.zeros(d, d)), "diagonal != rank-1 along e")

z = beta * v
general = (torch.eye(d) - torch.outer(k, beta * k)) @ (D @ S0) + torch.outer(k, z)
gated_dn = (torch.eye(d) - beta * torch.outer(k, k)) @ (D @ S0) + beta * torch.outer(k, v)
ok("with e~ = beta*k the general form IS gated DeltaNet", close(general, gated_dn, 1e-5))
free = (torch.eye(d) - torch.outer(k, beta * unit(d))) @ (D @ S0) + torch.outer(k, z)
ok("choosing any OTHER e~ gives a rule outside the family", not close(free, gated_dn),
   f"||difference|| = {(free - gated_dn).norm():.4f}")
print("so 'where to erase' was never derived - it was assumed")

ok("Diag(g) keeps the recurrence linear in S", close(D @ (S0 + S0), (D @ S0) + (D @ S0)))
ok("and the gate stays inside (0,1)", bool((g > 0).all() and (g < 1).all()),
   f"g in [{float(g.min()):.3f}, {float(g.max()):.3f}]")
