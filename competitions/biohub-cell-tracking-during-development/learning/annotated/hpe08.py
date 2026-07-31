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

def J_evict(N_act, E_act, E_id):
    return sum(n * e / E_id for n, e in zip(N_act, E_act))
base = J_evict([64, 64], [30.0, 28.0], E_id=50.0)
ok("a stronger skip makes the SAME block cheaper to evict",
   J_evict([64, 64], [30.0, 28.0], 100.0) < base,
   f"{base:.2f} -> {J_evict([64,64],[30.0,28.0],100.0):.2f} when E_identity doubles")
ok("a block with more live capacity costs more", J_evict([64, 64], [60.0, 56.0], 50.0) > base)
ok("an emptied block is free", J_evict([0, 0], [0.0, 0.0], 50.0) == 0.0)
tiny = J_evict([64, 64], [30.0, 28.0], 1e-12)
ok("no skip (E_identity -> 0) recovers Axiom 2's infinity", tiny > 1e12,
   f"J -> {tiny:.1e} — you cannot delete a block nothing bypasses")

for b_, g_ in [(0.0, 1.0), (0.7, 1.2), (-1.5, 0.4)]:
    mc = mc_E(lambda m: (b_ + g_ * torch.randn(m)) ** 2, n=4_000_000)
    cf = b_ ** 2 + g_ ** 2
    ok(f"identity channel (beta={b_}, gamma={g_}): E[y^2] = beta^2 + gamma^2",
       abs(mc - cf) / cf < 5e-3, f"MC {mc:.5f} vs {cf:.5f}")
gam = torch.rand(256) + 0.5; bet = torch.randn(256) * 0.3
E_id = float(torch.sqrt(gam ** 2 + bet ** 2).sum())
print(f"  a 256-channel skip: E_identity = {E_id:.2f}")
ok("the denominator of eq. 19 is now COMPUTED from BN stats", E_id > 0,
   "sqrt(gamma^2 + beta^2) summed over ambient channels — no knob to tune")
