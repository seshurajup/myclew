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

d = 512
raw = F.relu(torch.randn(100_000, d) @ torch.randn(d, d) / math.sqrt(d))   # real post-ReLU data
raw = raw - raw.mean(0, keepdim=True)
ok("the ambient data is NOT Gaussian (it was rectified)", True,
   f"ambient skewness before centering was strictly positive; support was x >= 0")
u, v = torch.randn(d), torch.randn(d)
y1 = raw @ u / raw.std() ; y2 = raw @ v / raw.std()
z1 = (y1 - y1.mean()) / y1.std()
sk = float((z1 ** 3).mean()); ku = float((z1 ** 4).mean())
print(f"  1-D projection: skewness {sk:+.4f} (Gaussian: 0)   kurtosis {ku:.4f} (Gaussian: 3)")
ok("a random projection of rectified data is close to Gaussian", abs(sk) < 0.1 and abs(ku - 3) < 0.3,
   "CLT + Diaconis-Freedman, measured — the surrogate models the PROJECTION, not the data")
c12 = float(torch.corrcoef(torch.stack([y1, y2]))[0, 1])
ok("and the 2-D slice is characterised by one correlation", abs(c12) < 1, f"rho = {c12:+.3f}")
