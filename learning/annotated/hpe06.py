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

def J_prune(norm_f, E_a, N):
    return N * norm_f / (E_a - norm_f)

caps = torch.tensor([0.001, 1.0, 1.0, 1.0, 1.0])          # one nearly-dead neuron among peers
E_a, N = float(caps.sum()), len(caps)
costs = torch.tensor([J_prune(float(c), E_a, N) for c in caps])
print("  capacities:", caps.tolist(), "\n  prune costs:", [f"{c:.4f}" for c in costs])
ok("a near-dead neuron is near-free to prune", float(costs[0]) < 0.01)
ok("equal-capacity peers cost equally", torch.allclose(costs[1:], costs[1]))
rich = J_prune(1.0, 10.0, 10)
poor = J_prune(1.0, 2.0, 2)
ok("the SAME neuron costs more in a depleted layer", poor > rich,
   f"{rich:.3f} in a rich layer vs {poor:.3f} when only one peer remains")
last = J_prune(1.0, 1.0 + 1e-9, 1)
ok("removing the last capacity costs ~infinity (Axiom 2)", last > 1e8,
   f"J = {last:.2e} — greedy compression cannot disconnect the graph")
for n_ in (4, 64, 1024):
    jm = J_prune(1.0, float(n_), n_)                       # mean-field: every neuron has capacity 1
    print(f"  mean-field width {n_:>5}: J = {jm:.4f}")
ok("mean-field cost ~ 1 at any width — costs are comparable ACROSS layers",
   abs(J_prune(1.0, 1024.0, 1024) - 1.0) < 0.01)
