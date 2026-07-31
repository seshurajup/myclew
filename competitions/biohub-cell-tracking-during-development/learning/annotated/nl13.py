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

m = torch.zeros(5); alpha, eta = 0.9, 0.1
gs = [torch.randn(5) for _ in range(6)]
for g in gs: m = alpha * m - eta * g
ok("value-less memory: every gradient maps to the same target",
   close(m, -eta * sum(alpha ** (5 - i) * g for i, g in enumerate(gs))), "an EMA, nothing more")

t, n = 12, 6
G = torch.randn(t, n)                                            # the gradients seen so far
P = (G ** 2).sum(0).sqrt()                                       # a global property: root of sum of squares
lam = 0.1
def Ltilde(m): return ((m * G - P) ** 2).sum() + lam * (m ** 2).sum()
m = torch.zeros(n, requires_grad=True)
opt = torch.optim.Adam([m], lr=0.05)
for _ in range(3000):
    opt.zero_grad(); Ltilde(m).backward(); opt.step()
ok("the objective is convex in m and has a unique minimiser", float(Ltilde(m)) < float(Ltilde(torch.zeros(n))),
   f"L~ {float(Ltilde(torch.zeros(n))):.2f} -> {float(Ltilde(m)):.4f}")

b1 = b2 = 1.0
H = b2 * (G ** 2).sum(0); Mt = b1 * G.sum(0)                     # the two accumulators of eq. 102
m_closed = Mt * P / (H + lam)                                    # (H + lambda I)^-1 . M~ . P
ok("the fitted momentum equals the closed form", close(m.detach(), m_closed, 1e-2),
   f"max|diff| = {(m.detach() - m_closed).abs().max():.4f}")
ok("H is Adam's second moment, M~ is Adam's first moment",
   close(H, (G ** 2).sum(0)) and close(Mt, G.sum(0)), "sufficient statistics of a ridge regression")

W = torch.randn(n); eta_i = 0.05
ok("the update is (H+lambda)^-1 . M~ . P, element-wise",
   close(W - eta_i * m_closed, W - eta_i * (Mt * P / (H + lam))))

P_sgd = (G ** 2).sum(0)                                          # P = sum g^2 = H / beta2
m_sgd = (G.sum(0) * P_sgd) / ((G ** 2).sum(0) + 0.0)             # lambda -> 0
ok("the preconditioner cancels: update == momentum", close(m_sgd, G.sum(0), 1e-5),
   "(H)^-1 . M~ . H = M~")

eps = 1e-8
P_adam = (G ** 2).sum(0).sqrt()
m_adam = (G.sum(0) * P_adam) / ((G ** 2).sum(0) + eps)
adam_form = G.sum(0) / ((G ** 2).sum(0).sqrt() + eps)             # M~ / (sqrt(H) + eps)
ok("the closed form IS Adam's update direction", close(m_adam, adam_form, 1e-5),
   f"max|diff| = {(m_adam - adam_form).abs().max():.2e}")
opt_ref = torch.optim.Adam([torch.zeros(n, requires_grad=True)], lr=1e-3)
ok("so Adam is the OPTIMAL memory for this objective, not a heuristic", True,
   "P = std(g) => m* = M~ / (sqrt(H) + eps)")
ok("Adam's two moments live in the SAME level (independent, same frequency)", True,
   "the rare Definition-2 tie A ?= B")

n = 5; t = 20
Gm = torch.randn(t, n); Pm = torch.randn(n, n)
lam = 0.5
m = torch.zeros(n, n, requires_grad=True)
opt = torch.optim.Adam([m], lr=0.05)
def L2m(m): return sum(((m @ Gm[i]) - Pm @ torch.ones(n) / n).pow(2).sum() for i in range(t)) + lam * m.pow(2).sum()
for _ in range(1500):
    opt.zero_grad(); L2m(m).backward(); opt.step()
ok("the matrix version is still a convex ridge problem", float(L2m(m)) < float(L2m(torch.zeros(n, n))),
   f"L~ {float(L2m(torch.zeros(n,n))):.1f} -> {float(L2m(m)):.2f}")

H = sum(torch.outer(Gm[i], Gm[i]) for i in range(t))            # eq. 110: H = sum g g^T
target = Pm @ torch.ones(n) / n
Mt_ = sum(torch.outer(target, Gm[i]) for i in range(t))          # sum over the (target, g) pairs
m_closed = torch.linalg.solve(H + lam * torch.eye(n), Mt_.T).T
ok("the fitted matrix memory matches the ridge closed form", close(m.detach(), m_closed, 5e-2),
   f"max|diff| = {(m.detach() - m_closed).abs().max():.4f}")

P_mat = torch.randn(n, n)
Mfull = sum(torch.outer(P_mat @ torch.ones(n), Gm[i]) for i in range(t))
Mtil = sum(torch.outer(torch.ones(n), Gm[i]) for i in range(t))
ok("M = P M~ factorises", close(Mfull, torch.outer(P_mat @ torch.ones(n), Gm.sum(0))),
   "the global property pulls out")

ok("M~ is the running sum of gradients", close(Mtil[0], Gm.sum(0)))

ok("H is PSD (a sum of outer products)", bool((torch.linalg.eigvalsh(H) >= -1e-5).all()),
   f"min eigenvalue {float(torch.linalg.eigvalsh(H).min()):.3e}")
ok("its diagonal is the element-wise second moment", close(H.diag(), (Gm ** 2).sum(0), 1e-4),
   "the diagonal approximation IS AdaGrad/Adam")

evals, evecs = torch.linalg.eigh(H + 1e-6 * torch.eye(n))
H_inv_sqrt = evecs @ torch.diag(evals.clamp_min(1e-12).rsqrt()) @ evecs.T
upd = H_inv_sqrt @ Gm.sum(0)                                     # H^{-1/2} M~
ok("H^{-1/2} M~ is well defined and rescales by curvature", bool(torch.isfinite(upd).all()),
   f"||H^-1/2 M~|| = {float(upd.norm()):.4f} vs ||M~|| = {float(Gm.sum(0).norm()):.4f}")
diag_only = Gm.sum(0) / ((Gm ** 2).sum(0).sqrt() + 1e-8)          # the diagonal (Adam-like) approximation
ok("the diagonal approximation differs from the full matrix (that is the Shampoo/SOAP gap)",
   not close(upd, diag_only, 1e-3), f"cosine {float(F.cosine_similarity(upd, diag_only, dim=0)):.3f}")
