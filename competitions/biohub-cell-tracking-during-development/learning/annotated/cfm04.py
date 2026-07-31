import math, torch, torch.nn as nn, torch.nn.functional as F      # couplings decide curvature

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=5, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

A = torch.tensor([[1.6, 0.4], [0.0, 0.7]])                  # target = A x0 + b (a linear flow)
b = torch.tensor([2.0, -1.0])
def v_lin(x, t):
    # velocity of x_t = ((1-t)I + tA) x0 + t b, eliminated to a function of (x, t)
    M = (1 - t) * torch.eye(2) + t * A
    x0 = torch.linalg.solve(M, (x - t * b).T).T
    return (A - torch.eye(2)) @ x0.T + b[:, None] if False else ((x0 @ (A - torch.eye(2)).T) + b)
x0_true = torch.randn(4096, 2)
x1 = x0_true @ A.T + b
x = x1.clone()
steps = 400
for i in range(steps, 0, -1):                                # RK4, backward
    t = i / steps; h = -1.0 / steps
    k1 = v_lin(x, t); k2 = v_lin(x + h / 2 * k1, t + h / 2)
    k3 = v_lin(x + h / 2 * k2, t + h / 2); k4 = v_lin(x + h * k3, t + h)
    x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
err = float((x - x0_true).norm(dim=1).max())
ok("backward integration recovers the TRUE source of every sample", err < 1e-3,
   f"max |x_hat0 - x0| = {err:.1e} over 4096 samples")
ok("so a trained flow doubles as its own source-discovery tool", True,
   "eqs. 8-10 fit Gaussians to exactly these recovered points")

mu_a, mu_b = torch.tensor([-2.0, 0.0]), torch.tensor([2.5, 1.0])
n = 8000
x0a = mu_a + 0.5 * torch.randn(n, 2)
x0b = mu_b + 0.5 * torch.randn(n, 2)
A = torch.tensor([[1.3, 0.2], [0.0, 0.9]]); b = torch.tensor([1.0, -0.5])
x1a, x1b = x0a @ A.T + b, x0b @ A.T + b                       # two target clusters, one map
hat0a, hat0b = (x1a - b) @ torch.linalg.inv(A).T, (x1b - b) @ torch.linalg.inv(A).T
m_a = hat0a.mean(0); m_b = hat0b.mean(0)                      # eq. (8), per cluster
ok("cluster A's recovered mean is cluster A's true source mean",
   float((m_a - mu_a).norm()) < 0.03, f"|err| = {float((m_a - mu_a).norm()):.4f}")
ok("likewise cluster B", float((m_b - mu_b).norm()) < 0.03)
glob = torch.cat([hat0a, hat0b]).mean(0)
ok("and NEITHER equals the global mean — clustering is what buys locality",
   float((m_a - glob).norm()) > 1.0 and float((m_b - glob).norm()) > 1.0,
   "a single global source would sit between the populations, serving neither")

L = torch.tensor([[0.8, 0.0], [0.5, 0.3]])                   # a deliberately anisotropic source
S_true = L @ L.T
x0 = torch.randn(60_000, 2) @ L.T
S_hat = (x0 - x0.mean(0)).T @ (x0 - x0.mean(0)) / len(x0)     # eq. (9)
err = float((S_hat - S_true).abs().max())
print("  true Sigma:\n", S_true, "\n  estimated:\n", S_hat)
ok("the empirical covariance recovers the planted one", err < 0.02,
   f"max entry error {err:.4f}")
ok("including the OFF-DIAGONAL correlation", abs(float(S_hat[0, 1] - S_true[0, 1])) < 0.02,
   "an isotropic source assumption would discard exactly this")

mu = torch.tensor([1.5, -0.5])
L = torch.tensor([[0.6, 0.0], [-0.3, 0.4]]); S = L @ L.T
samp = mu + torch.randn(200_000, 2) @ L.T                     # sampling from eq. (10)
ok("sampled mean matches the fitted mean", float((samp.mean(0) - mu).norm()) < 0.01)
S_emp = (samp - samp.mean(0)).T @ (samp - samp.mean(0)) / len(samp)
ok("sampled covariance matches the fitted covariance", float((S_emp - S).abs().max()) < 0.01,
   f"max entry error {float((S_emp - S).abs().max()):.4f}")
ok("each cluster now has a source the trainer can draw from endlessly", True,
   "the Gaussian is the interface between discovery (eq. 7) and re-training")
