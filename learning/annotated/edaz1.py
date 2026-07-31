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

d, n = 64, 12
Kf, Vf = unit(n, d), torch.randn(n, d)
S = torch.zeros(d, d)
for i in range(n):                                              # store n facts with the delta rule
    S = (torch.eye(d) - torch.outer(Kf[i], Kf[i])) @ S + torch.outer(Kf[i], Vf[i])
err0 = torch.tensor([float((S.T @ Kf[i] - Vf[i]).norm()) for i in range(n)])

target = 3                                                      # fact 3 is stale: remove exactly it
S2 = (torch.eye(d) - torch.outer(Kf[target], Kf[target])) @ S    # eq. 11 with gamma = 1
err1 = torch.tensor([float((S2.T @ Kf[i] - Vf[i]).norm()) for i in range(n)])
others = [i for i in range(n) if i != target]
d_target = float(err1[target] - err0[target])
d_others = float((err1[others] - err0[others]).abs().max())
ok("the targeted fact is degraded the most, by a wide margin", d_target > 3 * d_others,
   f"target +{d_target:.3f} vs worst other +{d_others:.3f} ({d_target/max(d_others,1e-9):.1f}x)")
ok("nothing is read from the erase direction any more", float((S2.T @ Kf[target]).norm()) < 1e-4,
   f"||read at the erased address|| = {float((S2.T @ Kf[target]).norm()):.2e}")
ok("the change to the state is rank ONE", int(torch.linalg.matrix_rank(S2 - S, tol=1e-4)) == 1,
   f"rank = {int(torch.linalg.matrix_rank(S2 - S, tol=1e-4))}")
print("compare a global decay of the same magnitude, which would damage all", n, "facts at once")
vz.heat(S2 - S, "learning/assets/eda-delta-attention/xai_erase_delta.png",
        "what a targeted erase changes")
