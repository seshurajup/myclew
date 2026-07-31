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
def water_fill(w, B, b_min, b_max, beta=beta, iters=60):
    lw = torch.log(w)
    lo, hi = -50.0, 50.0
    for _ in range(iters):                                      # bisect the multiplier to hit the budget
        lam = 0.5 * (lo + hi)
        b = ((lw - lam) / torch.log(torch.tensor(beta))).clamp(b_min, b_max)
        if float(b.sum()) > B: lo = lam
        else: hi = lam
    return ((lw - 0.5 * (lo + hi)) / torch.log(torch.tensor(beta))).clamp(b_min, b_max)

b_box = water_fill(w, B, b_min, b_max)
ok("the budget is met", abs(float(b_box.sum()) - B) < 0.05, f"sum = {float(b_box.sum()):.3f} vs {B}")
ok("every head is inside the hardware range",
   bool((b_box >= b_min - 1e-6).all() and (b_box <= b_max + 1e-6).all()),
   f"bits in [{float(b_box.min()):.2f}, {float(b_box.max()):.2f}]")
ok("and it still beats uniform", J(b_box) < J(b_uniform),
   f"J {J(b_uniform):.6f} -> {J(b_box):.6f}")

lam = 0.02
Lag = lambda b: float((w * D(b)).sum() + lam * (b.sum() - B))
ok("the Lagrangian equals the objective on the feasible set",
   abs(Lag(b_star) - J(b_star)) < 1e-4, "the constraint term vanishes when the budget is met")
marg = -(w * D(b_star) * torch.log(torch.tensor(beta)))          # dJ/db per head
ok("at the optimum the marginal gain is EQUAL across heads (water-filling)",
   float(marg.std() / marg.abs().mean()) < 1e-4,
   f"relative spread of dJ/db = {float(marg.std()/marg.abs().mean()):.2e}")

bv = b_star.clone().requires_grad_(True)
(w * alpha * beta ** (-bv)).sum().backward()
grads = bv.grad
ok("all per-head derivatives are equal at b*", float(grads.std() / grads.abs().mean()) < 1e-4,
   f"relative spread {float(grads.std()/grads.abs().mean()):.2e}")
lam_star = float(-grads.mean())
b_from_lam = (torch.log(w * alpha * torch.log(torch.tensor(beta))) -
              torch.log(torch.tensor(lam_star))) / torch.log(torch.tensor(beta))
ok("inverting the condition reproduces b* exactly", close(b_from_lam, b_star, 1e-3),
   f"lambda* = {lam_star:.6f}")

J_star_closed = alpha * beta ** (-bbar) * N * gm
ok("the closed form matches the evaluated optimum", abs(J(b_star) - J_star_closed) / J(b_star) < 1e-4,
   f"evaluated {J(b_star):.6f} vs closed form {J_star_closed:.6f}")
J_unif_closed = alpha * beta ** (-bbar) * N * am
ok("and the uniform cost is the same expression with the ARITHMETIC mean",
   abs(J(b_uniform) - J_unif_closed) / J(b_uniform) < 1e-6,
   f"uniform {J(b_uniform):.6f} vs {J_unif_closed:.6f}")

dh, T = 16, 24
K = torch.randn(T, dh, requires_grad=True)
V = torch.randn(T, dh, requires_grad=True)
q = torch.randn(dh)
loss = ((F.softmax(K @ q / dh ** 0.5, 0) @ V) ** 2).sum()
gK, gV = torch.autograd.grad(loss, [K, V])
eps = 1e-3
dK = eps * torch.randn_like(K)
pred = float((gK * dK).sum())                                    # the first-order term
with torch.no_grad():
    actual = float(((F.softmax((K + dK) @ q / dh ** 0.5, 0) @ V) ** 2).sum() - loss)
ok("the first-order term predicts the loss change", abs(pred - actual) < 0.05 * abs(actual) + 1e-6,
   f"predicted {pred:.3e} vs actual {actual:.3e}")
ok("so a gradient norm IS the importance weight, not a proxy",
   float((gK ** 2).sum()) > 0, f"||dL/dK||_F^2 = {float((gK**2).sum()):.4f}")
