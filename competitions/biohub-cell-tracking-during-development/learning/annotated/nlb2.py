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

d = 6
Wt = torch.randn(d)                                            # current weights
g  = torch.randn(d)                                            # the gradient at Wt
eta = 0.1

W_rule = Wt - eta * g                                          # FACE 1: the update rule (eq. 1)

W = Wt.clone().requires_grad_(True)                            # FACE 2: solve the argmin (eq. 2) numerically
opt = torch.optim.LBFGS([W], max_iter=100)
def closure():
    opt.zero_grad()
    obj = (g * W).sum() + (W - Wt).pow(2).sum() / (2 * eta)     # <g, W> + ||W - Wt||^2 / (2 eta)
    obj.backward(); return obj
opt.step(closure)

ok("proximal argmin == W_t - eta*g", close(W.detach(), W_rule, 1e-4),
   f"max|diff|={(W.detach()-W_rule).abs().max():.2e}")

W1 = torch.randn(d); eta = 0.05
grads = [torch.randn(d) for _ in range(7)]                     # the gradients the model generated

W_sgd = W1.clone()                                             # run plain SGD with constant eta
for gt in grads:
    W_sgd = W_sgd - eta * gt

G = torch.stack(grads).sum(0)                                  # FTRL: one argmin over the SUM of gradients
W_ftrl = W1 - eta * G                                          # its closed-form solution (eq. 3)
ok("SGD trajectory == FTRL closed form", close(W_sgd, W_ftrl))
print("sum of gradients == the optimizer's 'memory' of the whole past:", G.norm().item())
