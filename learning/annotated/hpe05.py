import math, torch, torch.nn as nn, torch.nn.functional as F     # neurons as FUNCTIONS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=5, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

SQ2 = math.sqrt(2.0)

def Phi(z):                                          # standard normal CDF
    return 0.5 * (1.0 + torch.erf(z / SQ2))

def phi(z):                                          # standard normal PDF
    return torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

def K_self(gamma, beta):                             # eq (3): ||f||^2 per unit output norm
    r = beta / gamma.abs()
    return (gamma ** 2 + beta ** 2) * Phi(r) + beta * gamma.abs() * phi(r)

def mc_E(fn, n=10_000_000, chunk=2_000_000):         # big-sample Monte-Carlo expectation
    tot = 0.0
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        tot += float(fn(m).sum())
    return tot / n

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

print(f"{'beta':>7} {'gamma':>7} {'closed form':>13} {'Monte-Carlo':>13} {'z-score':>9}")
worst_z = 0.0
N_MC = 10_000_000
for b_, g_ in [(0.0, 1.0), (0.8, 1.3), (-0.9, 0.7), (2.5, 0.5), (-2.0, 1.0)]:
    beta, gamma = torch.tensor(b_), torch.tensor(g_)
    cf = float(K_self(gamma, beta))
    y = F.relu(beta + gamma * torch.randn(N_MC)) ** 2
    mc, se = float(y.mean()), float(y.std()) / math.sqrt(N_MC)   # the estimator's OWN standard error
    z = abs(cf - mc) / max(se, 1e-12)
    worst_z = max(worst_z, z)
    print(f"{b_:>7.1f} {g_:>7.1f} {cf:>13.6f} {mc:>13.6f} {z:>9.2f}")
ok("every case agrees within 4 standard errors of ITS OWN estimator", worst_z < 4.0,
   f"worst z = {worst_z:.2f} — the right yardstick for a tiny K (beta=-2) is MC noise, not a blanket %")
ok("beta=0 recovers the textbook gamma^2/2", abs(float(K_self(torch.tensor(1.0), torch.tensor(0.0))) - 0.5) < 1e-6,
   "half the second moment survives rectification")
ok("a strongly negative beta drives capacity toward 0",
   float(K_self(torch.tensor(1.0), torch.tensor(-4.0))) < 5e-3,
   "a neuron that almost never fires almost has no function — the score agrees")

kappa = torch.linspace(-30, 30, 20001)
rho_hat = 2 * kappa / (1 + torch.sqrt(1 + 4 * kappa ** 2))          # eq. (4)
back = rho_hat / (1 - rho_hat ** 2)
ok("rho_hat INVERTS the map rho -> rho/(1-rho^2)", torch.allclose(back, kappa, atol=1e-3),
   f"max |rho/(1-rho^2) - kappa| = {float((back-kappa).abs().max()):.1e}")
ok("|rho_hat| < 1 for every kappa — it IS a correlation", float(rho_hat.abs().max()) < 1.0,
   f"sup |rho_hat| = {float(rho_hat.abs().max()):.6f}")
ok("it is odd and monotone (order of similarity preserved)",
   torch.allclose(rho_hat.flip(0), -rho_hat, atol=1e-6) and bool((rho_hat.diff() > 0).all()))
w_i, w_j = torch.randn(64), torch.randn(64)
rho_eff = float(F.cosine_similarity(w_i, w_j, dim=0))
x = torch.randn(2_000_000, 64)
emp = float(torch.corrcoef(torch.stack([x @ w_i, x @ w_j]))[0, 1])
ok("rho_eff is exactly the pre-activation correlation under the surrogate",
   abs(rho_eff - emp) < 2e-3, f"cosine {rho_eff:+.4f} vs measured {emp:+.4f}")

def cross_cf(rho, Kii, Kjj):
    rho = torch.clamp(torch.as_tensor(rho), -1 + 1e-7, 1 - 1e-7)
    return (torch.sqrt(1 - rho ** 2) + (math.pi - torch.arccos(rho)) * rho) / math.pi            * math.sqrt(Kii * Kjj)

def mc_cross(rho, bi, bj, gi, gj, n=10_000_000):
    z1 = torch.randn(n)
    z2 = rho * z1 + math.sqrt(1 - rho ** 2) * torch.randn(n)
    return float((F.relu(bi + gi * z1) * F.relu(bj + gj * z2)).mean())

print("zero bias — the regime the formula claims exactly:")
worst = 0.0
for rho in (-0.8, -0.3, 0.0, 0.5, 0.95):
    cf = float(cross_cf(rho, 0.5, 0.5))                 # K_self(1,0) = 1/2
    mc = mc_cross(rho, 0.0, 0.0, 1.0, 1.0)
    rel = abs(cf - mc) / max(mc, 1e-9); worst = max(worst, rel)
    print(f"  rho={rho:+.2f}: closed {cf:.6f}  MC {mc:.6f}  rel {rel:.1e}")
ok("zero-bias arc-cosine is EXACT (to MC noise)", worst < 5e-3, f"worst {worst:.1e}")

print("\nnonzero bias — the approximation's real cost:")
cf = float(cross_cf(0.5, float(K_self(torch.tensor(1.0), torch.tensor(0.8))),
                    float(K_self(torch.tensor(1.0), torch.tensor(0.8)))))
mc = mc_cross(0.5, 0.8, 0.8, 1.0, 1.0)
err = abs(cf - mc) / mc
print(f"  beta=0.8: closed {cf:.5f} vs true {mc:.5f}  ({err:.1%} off)")
ok("with real biases it is an APPROXIMATION, and we measured how much", 0.02 < err < 0.40,
   f"{err:.1%} at beta=0.8 — tens of percent, traded for avoiding a bivariate CDF per neuron pair")
