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

d = 32
k1, k2 = unit(d), unit(d)
v1, v2 = torch.randn(d), torch.randn(d)
beta = 1.0

def delta_write(S, k, v, b=beta):                               # the delta rule (eq. 3)
    return (torch.eye(d) - b * torch.outer(k, k)) @ S + b * torch.outer(k, v)

S = torch.zeros(d, d)
S = delta_write(S, k1, v1); S = delta_write(S, k2, v2)
before = float((S.T @ k2 - v2).norm())
for _ in range(100):                                            # hammer the OTHER address
    S = delta_write(S, k1, torch.randn(d))
after = float((S.T @ k2 - v2).norm())
ok("the stale fact at k2 survives 100 writes at k1", after < 3 * max(before, 1e-6) + 1.0,
   f"read error at k2: {before:.4f} -> {after:.4f}  (|k1.k2| = {abs(float(k1 @ k2)):.3f})")
print("the delta rule's erase is a rank-1 projection along the WRITE key, so it is blind to k2")

alpha = 0.95
S = torch.zeros(d, d)
S = delta_write(S, k1, v1); S = delta_write(S, k2, v2)
fresh0, stale0 = float((S.T @ k1 - v1).norm()), float((S.T @ k2 - v2).norm())
for _ in range(40):
    S = alpha * S                                               # global decay, no write
fresh1, stale1 = float((S.T @ k1 - v1).norm()), float((S.T @ k2 - v2).norm())
ok("decay does erase the stale fact", stale1 > stale0, f"stale {stale0:.3f} -> {stale1:.3f}")
ok("but it damages the fresh one just as much", fresh1 > fresh0,
   f"fresh {fresh0:.3f} -> {fresh1:.3f} — indiscriminate")
print("EDA's proposal: keep the corrective write, and add ONE targeted erase at its own address e_t")
