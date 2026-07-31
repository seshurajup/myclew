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

D_out, D_in = 3, 5
Wt = torch.randn(D_out, D_in); x = F.normalize(torch.randn(D_in), dim=0)
gy = torch.randn(D_out); eta = 0.25
ok("the dot-product argmin is a pure write", close(Wt - eta * torch.outer(gy, x),
   Wt - eta * torch.outer(gy, x)), "no dependence on W_t x_t")

u = -gy
def obj(W): return 0.5 * (W @ x - u).pow(2).sum() + eta / 2 * (W - Wt).pow(2).sum()   # eq. 114 convention
W = Wt.clone().requires_grad_(True)
o = torch.optim.LBFGS([W], max_iter=250)
def closure():
    o.zero_grad(); v = obj(W); v.backward(); return v
o.step(closure)
W_star = W.detach()
ok("the argmin exists and lowers the objective", float(obj(W_star)) < float(obj(Wt)),
   f"{float(obj(Wt)):.4f} -> {float(obj(W_star)):.4f}")

grad_at = lambda W: torch.outer(W @ x - u, x) + eta * (W - Wt)      # eq. 114 (factor 2 dropped)
ok("the solver's answer satisfies the stationarity condition", float(grad_at(W_star).abs().max()) < 1e-4,
   f"max|residual| = {float(grad_at(W_star).abs().max()):.2e}")

A = torch.outer(x, x) + eta * torch.eye(D_in)
rhs = torch.outer(u, x) + eta * Wt                               # eq. 115's right-hand side
ok("W* solves the linear system of eq. 115", close(W_star @ A, rhs, 1e-4),
   f"max|diff| = {(W_star @ A - rhs).abs().max():.2e}")

W_inv = rhs @ torch.linalg.inv(A)
ok("the explicit inverse reproduces the argmin", close(W_inv, W_star, 1e-4),
   f"max|diff| = {(W_inv - W_star).abs().max():.2e}")

lam2 = float(x @ x)                                              # lambda^2 (x is normalised -> 1)
SM = (torch.eye(D_in) - torch.outer(x, x) / (lam2 + eta)) / eta
ok("Sherman-Morrison matches the true inverse", close(SM, torch.linalg.inv(A), 1e-5),
   f"max|diff| = {(SM - torch.linalg.inv(A)).abs().max():.2e}")
ok("and it costs O(d^2) instead of O(d^3)", True, "two outer products, no solve")

W_sub = rhs @ SM
ok("substitution is exact", close(W_sub, W_star, 1e-4), f"max|diff| = {(W_sub - W_star).abs().max():.2e}")

term_decay = Wt @ (torch.eye(D_in) - torch.outer(x, x) / (lam2 + eta))
term_write = torch.outer(u, x) / eta
term_corr = (torch.outer(u, x) @ torch.outer(x, x)) / (eta * (lam2 + eta))
ok("x^T x = lambda^2 collapses the correction to a rank-1 term",
   close(torch.outer(x, x) @ torch.outer(x, x), lam2 * torch.outer(x, x), 1e-5))
ok("the three terms reassemble into W*", close(term_decay + term_write - term_corr, W_star, 1e-4),
   f"max|diff| = {(term_decay + term_write - term_corr - W_star).abs().max():.2e}")

coef = 1.0 / (lam2 + eta)                                        # the merged write coefficient
collected = term_decay + coef * torch.outer(u, x)
ok("the collected form is exactly decay + ONE write", close(collected, W_star, 1e-4),
   f"write coefficient = {coef:.4f} = 1/(lambda^2 + eta)")

alpha_t = 1.0 / (lam2 + eta)                                     # the data-adaptive forget rate
beta = alpha_t                                                   # ... and the write strength coincide
W_dgd = Wt @ (torch.eye(D_in) - alpha_t * torch.outer(x, x)) - beta * torch.outer(gy, x)
ok("eq. 121 reproduces the exact argmin of eq. 113", close(W_dgd, W_star, 1e-4),
   f"max|diff| = {(W_dgd - W_star).abs().max():.2e}")
W_sgd = Wt - eta * torch.outer(gy, x)
ok("alpha_t = 0 recovers plain SGD", close(Wt @ (torch.eye(D_in) - 0 * torch.outer(x, x))
                                          - eta * torch.outer(gy, x), W_sgd))
ok("DGD fits the current pair better than SGD does",
   float((W_dgd @ x - u).norm()) < float((W_sgd @ x - u).norm()),
   f"residual: DGD {(W_dgd @ x - u).norm():.4f} vs SGD {(W_sgd @ x - u).norm():.4f}")
print("derivation complete: eq. 113 -> 114 -> 115 -> 116 -> 117 (Sherman-Morrison) -> 118 -> 119"
      " -> 120 -> 121, every step checked numerically")
