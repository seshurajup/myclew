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

d = 5
theta_t = torch.randn(d); x = F.normalize(torch.randn(d), dim=0); eta = 0.2
def prox_argmin(Lfn, theta_t, iters=200):
    th = theta_t.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([th], max_iter=iters)
    def closure():
        opt.zero_grad(); obj = Lfn(th) + (th - theta_t).pow(2).sum() / (2 * eta); obj.backward(); return obj
    opt.step(closure); return th.detach()
g = torch.randn(d)
lin = prox_argmin(lambda th: (th * g).sum(), theta_t)             # linear objective -> GD step
ok("linear inner objective recovers Definition 3 / gradient descent", close(lin, theta_t - eta * g, 1e-4))
quad = prox_argmin(lambda th: 0.5 * ((th @ x) - 1.0) ** 2, theta_t)
ok("a different objective gives a different rule, same framework",
   not close(quad, theta_t - eta * g), f"||diff||={(quad-lin).norm():.4f}")

D = 4
M_t = torch.randn(D, D); k = F.normalize(torch.randn(D), dim=0); v = torch.randn(D); eta = 0.3
def mem_argmin(Lfn, iters=250):
    M = M_t.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([M], max_iter=iters)
    def closure():
        opt.zero_grad(); obj = Lfn(M) + (M - M_t).pow(2).sum() / (2 * eta); obj.backward(); return obj
    opt.step(closure); return M.detach()
l2 = mem_argmin(lambda M: 0.5 * (M @ k - v).pow(2).sum())
l1 = mem_argmin(lambda M: (M @ k - v).abs().sum())                # an L_p objective (p = 1)
ok("L2 gives the delta-rule solution",
   close(l2 @ k, (M_t @ k + eta * v) / (1 + eta), 1e-3), "shrink-and-write")
ok("L_p (p=1) gives a different, robust solution in the same framework",
   not close(l1, l2), f"||L1 - L2|| = {(l1 - l2).norm():.4f}")
