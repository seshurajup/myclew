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
W = torch.randn(d, requires_grad=True); x = torch.randn(d); y = torch.tensor(1.7)
eta = 0.05
loss0 = 0.5 * ((W @ x) - y) ** 2                                # L(W_t; x_t)
loss0.backward()
W1 = (W - eta * W.grad).detach()                                # eq. 1, one step
loss1 = 0.5 * ((W1 @ x) - y) ** 2
ok("one SGD step decreases the loss", loss1 < loss0.detach(), f"{loss0.item():.5f} -> {loss1.item():.5f}")
ok("the step is exactly -eta * surprise", close(W1 - W.detach(), -eta * W.grad))

g = torch.randn(d); Wt = torch.randn(d); eta = 0.1
grid = Wt + torch.linspace(-1.5, 1.5, 3001)[:, None] * (-g / g.norm())   # search along -g
obj = (grid @ g) + (grid - Wt).pow(2).sum(-1) / (2 * eta)                # <g, W> + prox
W_argmin = grid[obj.argmin()]
ok("argmin of the proximal objective == W_t - eta*g", close(W_argmin, Wt - eta * g, 2e-3),
   f"max|diff|={(W_argmin-(Wt-eta*g)).abs().max():.2e}")

W1 = torch.randn(d); eta = 0.05
gs = [torch.randn(d) for _ in range(9)]
W_sgd = W1.clone()
for gt in gs:
    W_sgd = W_sgd - eta * gt                                    # per-step form
ok("SGD trajectory == FTRL closed form", close(W_sgd, W1 - eta * torch.stack(gs).sum(0)))

# inner: one gradient step on a task; outer: choose the shared init Phi that makes that step best
def task_loss(theta, A, b):
    return 0.5 * ((A @ theta - b) ** 2).mean()
tasks = [(torch.randn(4, 3), torch.randn(4)) for _ in range(6)]  # p(T): 6 linear tasks
def outer(Phi, inner_lr=0.3):
    tot = 0.
    for A, b in tasks:
        g, = torch.autograd.grad(task_loss(Phi, A, b), Phi, create_graph=True)   # INNER: one step
        tot = tot + task_loss(Phi - inner_lr * g, A, b)           # loss AFTER adaptation
    return tot / len(tasks)
Phi = torch.zeros(3, requires_grad=True)
before = outer(Phi).item()
opt = torch.optim.Adam([Phi], lr=0.1)
for _ in range(300):                                             # OUTER level: eq. 4
    opt.zero_grad(); l = outer(Phi); l.backward(); opt.step()
ok("outer loop lowers post-adaptation loss across tasks", outer(Phi).item() < before,
   f"{before:.4f} -> {outer(Phi).item():.4f}")

d_k = d_v = 4; T = 6; alpha = 0.9
K = F.normalize(torch.randn(T, d_k), dim=-1); V = torch.randn(T, d_v)
M = torch.zeros(d_v, d_k)
for t in range(T):                                              # the recurrence (eq. 5)
    M = alpha * M + torch.outer(V[t], K[t])
decayed = sum(alpha ** (T - 1 - t) * torch.outer(V[t], K[t]) for t in range(T))
ok("FWP state == decayed sum of rank-1 writes", close(M, decayed))
q = K[3]; ok("read is a matrix-vector product", close(M @ q, decayed @ q))
print("state size is CONSTANT in T:", tuple(M.shape), "for T =", T)
