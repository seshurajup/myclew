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
Dgen = [lambda b, a=float(ai), bb=float(bi): a * torch.exp(-bb * b)
        for ai, bi in zip(torch.rand(N) + 0.5, torch.rand(N) * 0.8 + 1.0)]
Jgen = lambda b: float(sum(w[i] * Dgen[i](b[i]) for i in range(N)))
ok("a general per-head distortion still gives a finite objective", 0 < Jgen(b_uniform) < float('inf'),
   f"J_gen(uniform) = {Jgen(b_uniform):.5f}")

bg = b_uniform.clone().requires_grad_(True)
Jt = sum(w[i] * Dgen[i](bg[i]) for i in range(N))
Jt.backward()
ok("marginal gains are NOT equal at a uniform allocation (so uniform is not optimal)",
   float(bg.grad.std() / bg.grad.abs().mean()) > 0.05,
   f"relative spread {float(bg.grad.std()/bg.grad.abs().mean()):.3f}")
print("equalising those marginals is exactly what the multiplier does")

ai = torch.rand(N) + 0.5
bi = torch.rand(N) * 0.8 + 1.0
lam2 = 0.05
b_opt = (torch.log(w * ai * bi / lam2)) / bi                     # eq. 13
lhs = ai * bi * torch.exp(-bi * b_opt)
ok("the stationarity condition holds per head", close(lhs, lam2 / w, 1e-4),
   f"max relative error {float(((lhs - lam2/w).abs()/(lam2/w)).max()):.2e}")

def alloc(lam):
    return (torch.log(w * ai * bi / lam)) / bi
lo, hi = 1e-8, 1e3
for _ in range(80):                                              # bisect lambda for the budget
    mid = (lo * hi) ** 0.5
    if float(alloc(mid).clamp(b_min, b_max).sum()) > B: lo = mid
    else: hi = mid
b_gen = alloc((lo * hi) ** 0.5).clamp(b_min, b_max)
ok("the budget is met with per-head curve steepness", abs(float(b_gen.sum()) - B) < 0.2,
   f"sum {float(b_gen.sum()):.2f} vs {B}")
# a raw correlation confounds importance with curve shape, so CONTROL for importance: two heads with
# identical w and different beta_i, allocated by eq. 13 at the same multiplier
w_eq = torch.tensor([1.0, 1.0]); a_eq = torch.tensor([1.0, 1.0])
b_steep, b_flat = torch.tensor(2.0), torch.tensor(0.8)           # flat = distortion falls slowly
bits_pair = torch.stack([torch.log(w_eq[0] * a_eq[0] * b_steep / 0.05) / b_steep,
                         torch.log(w_eq[1] * a_eq[1] * b_flat / 0.05) / b_flat])
ok("at EQUAL importance, the flatter distortion curve earns more bits",
   float(bits_pair[1]) > float(bits_pair[0]),
   f"steep beta={float(b_steep)} -> {float(bits_pair[0]):.2f} bits; "
   f"flat beta={float(b_flat)} -> {float(bits_pair[1]):.2f} bits")
ok("and importance still dominates the overall ranking",
   float(torch.corrcoef(torch.stack([torch.log(w), b_gen]))[0, 1]) > 0.4,
   f"corr(log w, bits) = {float(torch.corrcoef(torch.stack([torch.log(w), b_gen]))[0,1]):+.3f}")

X = torch.log(w)
lhs = float(torch.exp(X).mean())                                 # E[f(X)], f = exp (convex)
rhs = float(torch.exp(X.mean()))                                 # f(E[X])
ok("Jensen holds for exp on this sample", lhs >= rhs - 1e-9, f"E[e^X] {lhs:.4f} >= e^E[X] {rhs:.4f}")
ok("equality only when the importances are identical",
   abs(float(torch.exp(torch.zeros(N)).mean()) - float(torch.exp(torch.zeros(N).mean()))) < 1e-9)

ratio = float(torch.exp(torch.log(w.mean()) - torch.log(w).mean()))
ok("the exp-of-log-gap form matches AM/GM", abs(ratio - am / gm) < 1e-5, f"{ratio:.5f}")
import math
approx = math.exp(0.5 * float(torch.log(w).var(unbiased=False)))
ok("to leading order it is exp(var of log w / 2)", abs(approx - ratio) / ratio < 0.35,
   f"exp(varlog/2) = {approx:.3f} vs exact {ratio:.3f}")
print(f"  diagnostic: log-variance {float(torch.log(w).var(unbiased=False)):.3f} "
      f"-> expect about {ratio:.2f}x from mixed precision")
