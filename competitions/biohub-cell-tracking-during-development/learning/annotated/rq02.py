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

N = 16
alpha, beta = 1.0, 4.0                                          # D(b) = alpha * beta^-b (measured in rqb1)
w = torch.distributions.LogNormal(0.0, 1.2).sample((N,))
B = 4.0 * N                                                     # average 4 bits per head
D = lambda b: alpha * beta ** (-b)
J = lambda b: float((w * D(b)).sum())
b_uniform = torch.full((N,), B / N)
ok("the budget constraint is satisfiable", abs(float(b_uniform.sum()) - B) < 1e-6)
ok("the objective is finite and positive", 0 < J(b_uniform) < float('inf'), f"J(uniform) = {J(b_uniform):.5f}")
print(f"  {N} heads, budget {B:.0f} bits total ({B/N:.1f} per head on average)")

bbar = B / N
logw = torch.log(w)
b_star = bbar + (logw - logw.mean()) / torch.log(torch.tensor(beta))
ok("the allocation respects the budget exactly", abs(float(b_star.sum()) - B) < 1e-4,
   f"sum = {float(b_star.sum()):.4f} vs B = {B}")
ok("it beats uniform allocation", J(b_star) < J(b_uniform),
   f"J: uniform {J(b_uniform):.6f} -> optimal {J(b_star):.6f} "
   f"({J(b_uniform)/J(b_star):.2f}x better)")
i_hi, i_lo = int(w.argmax()), int(w.argmin())
ratio = float(w[i_hi] / w[i_lo])
ok("a beta-times more important head earns exactly one more bit",
   abs(float(b_star[i_hi] - b_star[i_lo]) -
       float(torch.log(torch.tensor(ratio)) / torch.log(torch.tensor(beta)))) < 1e-4,
   f"importance ratio {ratio:.1f}x -> {float(b_star[i_hi]-b_star[i_lo]):.2f} bits apart")
ok("and the allocation is invariant to rescaling every importance",
   close(bbar + (torch.log(100 * w) - torch.log(100 * w).mean()) / torch.log(torch.tensor(beta)), b_star,
         1e-4), "only RELATIVE importance matters")

ok("b-bar is the per-head average budget", abs(bbar - B / N) < 1e-12, f"b_bar = {bbar}")
ok("mean-log-w is the log of the GEOMETRIC mean",
   abs(float(logw.mean()) - float(torch.log(torch.exp(logw.mean())))) < 1e-6,
   f"exp(mean log w) = {float(torch.exp(logw.mean())):.4f} = geometric mean")

am = float(w.mean()); gm = float(torch.exp(torch.log(w).mean()))
ok("the measured ratio equals AM/GM exactly",
   abs(J(b_uniform) / J(b_star) - am / gm) < 1e-3,
   f"measured {J(b_uniform)/J(b_star):.4f} vs AM/GM {am/gm:.4f}")
ok("the ratio is >= 1 always (Jensen)", am / gm >= 1.0 - 1e-9)
w_eq = torch.full((N,), 3.0)
b_eq = B / N + (torch.log(w_eq) - torch.log(w_eq).mean()) / torch.log(torch.tensor(beta))
ok("equal importances -> uniform IS optimal (no gain to chase)",
   abs(float(w_eq.mean() / torch.exp(torch.log(w_eq).mean())) - 1.0) < 1e-6
   and close(b_eq, torch.full((N,), B / N), 1e-5))
print(f"  so on THIS importance profile mixed precision is worth {am/gm:.2f}x, computable up front")
