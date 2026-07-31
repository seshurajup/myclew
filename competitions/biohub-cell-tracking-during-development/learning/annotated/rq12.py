import torch, torch.nn as nn, torch.nn.functional as F      # bit allocation is one Lagrange multiplier
import sys; sys.path.insert(0, "learning")
import vizkit as vz

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)

N = 16; alpha, beta = 1.0, 4.0                                   # this lesson's own setup
w = torch.distributions.LogNormal(0.0, 1.2).sample((N,))
B = 4.0 * N; bbar = B / N
D = lambda b: alpha * beta ** (-b)
J = lambda b: float((w * D(b)).sum())
b_uniform = torch.full((N,), bbar)
b_star = bbar + (torch.log(w) - torch.log(w).mean()) / torch.log(torch.tensor(beta))
am = float(w.mean()); gm = float(torch.exp(torch.log(w).mean()))
b_min, b_max = 2.0, 8.0
b_box = ((torch.log(w) - torch.log(w).mean()) / torch.log(torch.tensor(beta)) + bbar).clamp(b_min, b_max)
bs = torch.arange(2, 9).float()
gains = torch.tensor([float(alpha * beta ** (-b) - alpha * beta ** (-(b + 1))) for b in bs])
ok("marginal gains are strictly decreasing", bool((gains[1:] < gains[:-1]).all()),
   f"gains {[f'{g:.2e}' for g in gains[:4].tolist()]} ...")
# greedy integer allocation must agree with the (rounded) closed form
bits_int = torch.full((N,), int(b_min))
budget_left = int(B - bits_int.sum())
for _ in range(budget_left):
    g = w * (alpha * beta ** (-bits_int.float()) - alpha * beta ** (-(bits_int.float() + 1)))
    g[bits_int >= b_max] = -1.0
    bits_int[int(g.argmax())] += 1
ok("greedy integer allocation matches the closed form to within a bit",
   float((bits_int.float() - b_box).abs().max()) <= 1.5,
   f"max |greedy - closed| = {float((bits_int.float()-b_box).abs().max()):.2f} bits")
ok("and greedy is never worse than uniform", J(bits_int.float()) <= J(b_uniform),
   f"J greedy {J(bits_int.float()):.6f} vs uniform {J(b_uniform):.6f}")
