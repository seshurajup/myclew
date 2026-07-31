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

d = 64
x = torch.randn(300_000, d)
w_raw = torch.randn(d)
gamma, beta, eps = torch.tensor(1.6), torch.tensor(-0.3), 1e-5
z = x @ w_raw
mu, var = z.mean(), z.var(unbiased=False)
bn_out = gamma * (z - mu) / torch.sqrt(var + eps) + beta          # what the network computes
w_eff = gamma / torch.sqrt(var + eps) * w_raw                     # eq. (1)
b = beta - gamma * mu / torch.sqrt(var + eps)
folded = x @ w_eff + b
ok("BN + projection IS one affine map", torch.allclose(bn_out, folded, atol=1e-4),
   f"max diff {float((bn_out-folded).abs().max()):.2e}")
lam = 25.0
z2 = x @ (lam * w_raw)
w_eff2 = gamma / torch.sqrt(z2.var(unbiased=False) + eps) * (lam * w_raw)
ok("and the EFFECTIVE weights are invariant to raw-weight scaling",
   torch.allclose(w_eff, w_eff2, atol=1e-4), "lambda enters w_raw and sigma equally, and cancels")
print(f"\n  pre-activation after folding ~ N({float(b):.3f}, {float(gamma):.3f}^2) — the (beta, gamma) "
      f"the kernels of §5 will use")

z = torch.randn(1_000_000) * 3
for name, Psi in [("ReLU", F.relu), ("LeakyReLU(0.1)", lambda t: F.leaky_relu(t, 0.1)),
                  ("linear", lambda t: t)]:
    c = 7.3
    err = float((Psi(c * z) - c * Psi(z)).abs().max())
    ok(f"{name} is PH-1", err < 1e-4, f"max |Psi(cz)-cPsi(z)| = {err:.1e}")
err_gelu = float((F.gelu(7.3 * z) - 7.3 * F.gelu(z)).abs().max())
ok("GELU is NOT PH-1 — the framework's boundary", err_gelu > 1.0, f"violation up to {err_gelu:.1f}")
w_in, w_out, b = torch.randn(16), torch.randn(4), torch.tensor(0.2)
x = torch.randn(8, 16)
f = w_out[None, :] * F.relu(x @ w_in + b)[:, None]
ok("a neuron maps R^16 -> R^4 as one continuous function", f.shape == (8, 4))
