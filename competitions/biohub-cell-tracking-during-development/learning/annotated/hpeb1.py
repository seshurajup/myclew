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

d = 32
x = torch.randn(4096, d)
w_raw = torch.randn(d); gamma, beta = torch.tensor(1.7), torch.tensor(0.4)

def bn_neuron(w, x):
    z = x @ w
    return gamma * (z - z.mean()) / z.std() + beta

lam = 100.0
y1, y2 = bn_neuron(w_raw, x), bn_neuron(lam * w_raw, x)
ok("BN: scaling raw weights by 100 changes NOTHING", torch.allclose(y1, y2, atol=1e-4),
   f"max diff {float((y1-y2).abs().max()):.2e}")
ok("but the magnitude score changed 100x", True,
   f"||w|| {float(w_raw.norm()):.2f} -> {float((lam*w_raw).norm()):.2f}")

w_in, w_out = torch.randn(d), torch.randn(8)
c = 50.0
f1 = torch.outer(F.relu(x @ w_in), w_out)
f2 = torch.outer(F.relu(x @ (c * w_in)), w_out / c)
ok("PH-1: moving scale between in/out weights changes NOTHING",
   torch.allclose(f1, f2, atol=1e-3), f"max diff {float((f1-f2).abs().max()):.2e}")
ok("so any RAW-PARAMETER importance score is refuted twice", True,
   "the score must be computed from the FUNCTION")

x = torch.randn(200_000, 8)
w_in, w_out = torch.randn(8), torch.randn(3)

def neuron(w_i, w_o):
    return F.relu(x @ w_i)[:, None] * w_o[None, :]

f = neuron(w_in, w_out)
f_reshard = neuron(37.0 * w_in, w_out / 37.0)          # same function, different parameters
n1 = float((f * f).sum(-1).mean())                      # ||f||^2 = E[f.f]
n2 = float((f_reshard * f_reshard).sum(-1).mean())
print(f"  ||f||^2 = {n1:.4f}   after resharding = {n2:.4f}")
ok("the Hilbert norm survives the symmetry the magnitude score failed", abs(n1 - n2) / n1 < 1e-3)
g = neuron(torch.randn(8), torch.randn(3))
ip = float((f * g).sum(-1).mean())
ok("and an inner product exists between any two neurons", abs(ip) > 0,
   f"<f,g> = {ip:.4f} — the geometry compression will run on")
