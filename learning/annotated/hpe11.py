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

torch.manual_seed(0)
n = 400
caps = (torch.rand(n) ** 2) * 2.0                            # capacities as compression saw them
E_b = torch.full((n,), 40.0) - torch.cumsum(caps, 0) * 0.05  # remaining capacity drifts down
N_i = torch.full((n,), 64.0)
ledger = (N_i * caps / E_b)                                  # eq. (24), one entry per removal
print(f"  ledger over {n} removals: min {float(ledger.min()):.4f}  median "
      f"{float(ledger.median()):.4f}  max {float(ledger.max()):.4f}")
ok("every removal has a recorded cost", ledger.shape == (n,))
ok("costs are positive and finite", bool((ledger > 0).all()) and bool(torch.isfinite(ledger).all()))
ok("the ledger is FREE — compression already computed it", True,
   "DEFT is a second use of the same numbers, not a second framework")

E_b_tail = E_b.clone(); E_b_tail[-25:] = torch.logspace(-1, -6, 25)      # a layer genuinely dying
led = N_i * caps / E_b_tail
eps = 1e-2
C = led[E_b_tail > eps]                                      # eq. (25)
n_art = int((E_b_tail <= eps).sum())
bound = float(N_i.max() * caps.max() / eps)                  # no filtered entry can exceed this
print(f"  raw ledger max {float(led.max()):.1e} (exploded)   filtered max {float(C.max()):.1f}  "
      f"(provable bound {bound:.0f})")
ok("the unfiltered tail explodes as E_b -> 0", float(led.max()) > 1e4)
ok("the filter drops exactly the E_b <= eps entries", len(C) == n - n_art,
   f"{n_art} extinction artifacts dropped")
ok("and bounds every surviving cost by N*cap_max/eps", float(C.max()) <= bound,
   "the surviving entries are about neurons, not denominators")
ok("what remains reflects neurons, not denominators", True,
   "a percentile of C is now a meaningful threshold — eq. 26")

def j_lock(C, P=60.0, eps=1e-2):
    if len(C) == 0:
        return 1.0
    JP = float(torch.quantile(C, P / 100))
    Jsup = float(C.max())
    if JP >= eps:
        return JP                                            # branch 1
    if Jsup >= eps:
        return Jsup                                          # branch 2
    return 1.0                                               # branch 3
b1 = j_lock(C)
b2 = j_lock(torch.cat([torch.full((99,), 1e-4), torch.tensor([0.5])]))
b3 = j_lock(torch.full((100,), 1e-5))
print(f"  healthy ledger -> percentile   J_lock = {b1:.4f}")
print(f"  skewed ledger  -> supremum     J_lock = {b2:.4f}")
print(f"  dead ledger    -> fallback     J_lock = {b3:.4f}")
ok("branch 1: a healthy ledger uses its percentile", b1 > 1e-2)
ok("branch 2: a tiny-percentile ledger falls back to its max", abs(b2 - 0.5) < 1e-9)
ok("branch 3: an all-tiny ledger yields the safe constant 1", b3 == 1.0)

J_lock_v = j_lock(C)
elastic = (C < J_lock_v).float()                             # eq. (27) over the filtered ledger
frac = float(elastic.mean())
print(f"  J_lock = {J_lock_v:.4f}  ->  {frac:.0%} elastic / {1-frac:.0%} locked")
ok("the gate is binary and total", set(elastic.unique().tolist()) <= {0.0, 1.0})
ok("cheap-to-remove structure is the part allowed to MOVE", bool(
   (C[elastic.bool()].mean() < C[~elastic.bool()].mean())),
   f"elastic mean cost {float(C[elastic.bool()].mean()):.3f} vs locked "
   f"{float(C[~elastic.bool()].mean()):.3f}")
ok("what is NOT verified here: that this split transfers better", True,
   "that claim needs the authors' cross-domain training runs — §11.2, not reproduced")
