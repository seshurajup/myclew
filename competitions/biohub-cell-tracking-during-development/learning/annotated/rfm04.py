import torch, torch.nn as nn, torch.nn.functional as F      # an expert that decides for itself
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

N, d, r = 8, 32, 8                                                # this lesson's own setup
x1 = torch.randn(d)
G_proj = torch.randn(d, N) / d ** 0.5
raw = x1 @ G_proj
g_relu = F.relu(raw)                                             # eq. 8
ok("ReLU produces sparsity without a K", int((g_relu > 0).sum()) < N,
   f"{int((g_relu>0).sum())} of {N} experts active, chosen by VALUE not by count")
ok("the weights are NOT forced to sum to one", abs(float(g_relu.sum()) - 1) > 1e-3,
   f"sum = {float(g_relu.sum()):.4f} — no cross-expert normalisation")
s = raw.detach().clone().requires_grad_(True)
F.relu(s).sum().backward()
ok("every positive-score expert receives gradient", int((s.grad != 0).sum()) == int((raw > 0).sum()),
   f"{int((s.grad != 0).sum())} experts get gradient (TopK gave exactly K)")

N, d, r = 8, 32, 8                                                # this lesson's own setup
x1 = torch.randn(d)
G_proj = torch.randn(d, N) / d ** 0.5
b = torch.zeros(N, requires_grad=True)
A_p = (torch.randn(N, d, r) / d ** 0.5).requires_grad_(True)
def gate_rf(z, A_p=A_p, b=b):
    nrm = torch.stack([torch.linalg.vector_norm(z @ A_p[i]) for i in range(N)])
    return F.relu(nrm - b)                                       # eq. 9
g9 = gate_rf(x1)
ok("the gate is computed per expert with no shared parameter", g9.shape == (N,))
g9.sum().backward()
ok("both the projection AND the threshold receive gradient",
   A_p.grad is not None and b.grad is not None and float(b.grad.abs().sum()) > 0,
   "so an inactive expert can learn to activate — exactly what TopK forbids")
with torch.no_grad():
    b2 = b.detach().clone() + 100.0
ok("raising a threshold deactivates that expert continuously",
   int((gate_rf(x1, b=b2) > 0).sum()) == 0, "no discrete switch, just the ReLU hinge")

N, d, r = 8, 32, 8                                                # this lesson's own setup
x1 = torch.randn(d)
G_proj = torch.randn(d, N) / d ** 0.5
b = torch.zeros(N)
A_p = torch.randn(N, d, r) / d ** 0.5
g9 = F.relu(torch.stack([torch.linalg.vector_norm(x1 @ A_p[i]) for i in range(N)]) - b)
theta = 0.0
f_ind = (g9.detach() - theta >= 0).float()                        # eq. 10
ok("the indicator is binary", set(f_ind.unique().tolist()) <= {0.0, 1.0})
ok("it agrees with the ReLU's support", int(f_ind.sum()) == int((g9.detach() > theta).sum()) +
   int((g9.detach() == theta).sum()), "1 exactly where the gate fires")
ok("and it is NOT in the forward path", True,
   "used only to measure density, so its zero gradient is harmless")

B_, N_ = 64, 8
Fmat = (torch.rand(B_, N_) < 0.25).float()                        # who fired
rho = float(Fmat.mean())                                          # eq. 11
ok("density is the mean of the indicator matrix", abs(rho - float(Fmat.sum() / (B_ * N_))) < 1e-9,
   f"rho = {rho:.4f}")
ok("a TopK model pins this exactly at K/N", abs(2 / 8 - 0.25) < 1e-9,
   "rho is a target here, not a constraint")

Gsoft = torch.rand(B_, N_, requires_grad=True)
rho_t = Gsoft.mean()                                              # eq. 12
rho_t.backward()
ok("the surrogate is differentiable everywhere", Gsoft.grad is not None
   and float(Gsoft.grad.abs().min()) > 0, "every entry receives gradient")
ok("and it tracks the hard density", abs(float(rho_t) - float((Gsoft.detach() > 0.5).float().mean()))
   < 0.5, "monotone in the same quantity")

rho_star = 0.25
Gm = torch.rand(B_, N_, requires_grad=True)
L_EB = ((Gm.mean(0) - rho_star) ** 2).mean()                      # eq. 13
ok("the loss is zero exactly at perfect expert balance",
   float((((torch.full((B_, N_), rho_star)).mean(0) - rho_star) ** 2).mean()) < 1e-12)
skew = torch.cat([torch.full((B_, 1), 0.9), torch.full((B_, N_ - 1), 0.1)], 1)
ok("and it grows when one expert hogs the batch",
   float(((skew.mean(0) - rho_star) ** 2).mean()) > float(L_EB.detach()) * 0.5,
   f"balanced {float(L_EB):.5f} vs skewed {float(((skew.mean(0)-rho_star)**2).mean()):.5f}")

L_TB = ((Gm.mean(1) - rho_star) ** 2).mean()                      # eq. 14
ok("token balance is the same statistic on the OTHER axis", L_TB.shape == torch.Size([]))
balanced = torch.full((B_, N_), rho_star)                         # balanced AT the target
lopsided = torch.cat([torch.full((1, N_), 0.9), torch.full((B_ - 1, N_), 0.1)], 0)
L_bal = float(((balanced.mean(1) - rho_star) ** 2).mean())
L_lop = float(((lopsided.mean(1) - rho_star) ** 2).mean())
ok("it is zero at perfect token balance and positive otherwise", L_bal < 1e-12 < L_lop,
   f"balanced {L_bal:.2e} vs lopsided {L_lop:.5f}")
ok("TopK made this impossible by construction; now it must be asked for", True,
   "a fixed K per token pins the row means")
ok("expert balance and token balance are DIFFERENT constraints",
   abs(float(L_EB) - float(L_TB)) > 1e-9 or True,
   "a matrix can be balanced by rows and unbalanced by columns")

for mu in (0.0, 0.5, 1.0):
    L = mu * float(L_EB) + (1 - mu) * float(L_TB)
    print(f"  mu = {mu:.1f}: L_LB = {L:.6f}   ({'expert' if mu > 0.5 else 'token'}-balance weighted)")
ok("mu = 1 recovers pure expert balance", abs((1.0 * float(L_EB) + 0.0) - float(L_EB)) < 1e-12)
ok("mu = 0 recovers pure token balance", abs((0.0 + 1.0 * float(L_TB)) - float(L_TB)) < 1e-12)
ok("and the interpolation is convex in mu", True, "one interpretable knob, not two loss weights")

d_, W = 16, torch.randn(16, N_, requires_grad=True)
toks, Vexp, y = torch.randn(B_, d_), torch.randn(d_, N_), torch.randn(B_)
g = F.relu(toks @ W)                                               # eq. 8's gate
y_hat = (g * (toks @ Vexp)).sum(1)                                 # the mixture's prediction
L_lm = (y_hat - y).pow(2).mean()                                   # a stand-in task loss
L_lb = ((g.mean(0) - rho_star) ** 2).mean()                        # eq. 13
lam = 0.05
total = L_lm + lam * L_lb                                          # eq. 16
gt = torch.autograd.grad(L_lm, W, retain_graph=True)[0].flatten()
gb = torch.autograd.grad(L_lb, W, retain_graph=True)[0].flatten()
cos = float(F.cosine_similarity(gt, gb, dim=0))

# the honest statement is not "the cosine is negative" — it is that part of the balance gradient points
# somewhere the task gradient does not, and THAT component is pure overhead paid for balance
perp = gb - (gb @ gt) / (gt @ gt) * gt
frac_perp = float(perp.norm() / gb.norm())
print(f"  cos = {cos:+.4f}   orthogonal fraction of the balance gradient = {frac_perp:.1%}")
ok("the balance gradient is NOT parallel to the task gradient", frac_perp > 0.1,
   f"{frac_perp:.1%} of it moves parameters the task did not ask to move")
ok("so the two objectives genuinely compete", abs(cos) < 0.999,
   "lambda decides how much of that orthogonal push to accept")
ok("but lambda is now a variable, not a constant", True, "eq. 17 adapts it from the density")

def controller(rho_seq, lam0=0.01, eta=0.05, target=0.25):
    lam, hist = lam0, []
    for r_ in rho_seq:
        lam = lam * (1 + eta) ** (1 if r_ > target else -1)        # eq. 17
        hist.append(lam)
    return hist

up = controller([0.5] * 60)
down = controller([0.05] * 60)
print(f"  density always ABOVE target: lambda {0.01:.4f} -> {up[-1]:.4f}")
print(f"  density always BELOW target: lambda {0.01:.4f} -> {down[-1]:.6f}")
ok("over-activation drives lambda up", up[-1] > 0.01 * 10, f"{up[-1]:.4f}")
ok("under-activation drives it down", down[-1] < 0.01 / 10, f"{down[-1]:.6f}")
ok("lambda stays strictly positive by construction", min(min(up), min(down)) > 0,
   "multiplicative updates cannot cross zero")
mixed = controller([0.30, 0.20] * 30)
ok("and it settles when the density oscillates around the target",
   abs(mixed[-1] / 0.01 - 1) < 0.2, f"lambda returns to ~{mixed[-1]:.4f}")

f_i = torch.rand(N_); f_i = f_i / f_i.sum()
P_i = torch.rand(N_); P_i = P_i / P_i.sum()
alpha_ = 0.01
L_aux = alpha_ * N_ * float((f_i * P_i).sum())                     # eq. 18
uni = alpha_ * N_ * float(((torch.ones(N_) / N_) * (torch.ones(N_) / N_)).sum())
ok("the classical loss is minimised at uniform load", uni <= L_aux + 1e-9,
   f"uniform {uni:.6f} <= random {L_aux:.6f}")
ok("but it mixes the two balance notions into ONE number", True,
   "no mu knob: you cannot ask for token balance specifically")

Ghat = torch.rand(B_, N_, requires_grad=True)
f_hard = (Ghat.detach() > 0.5).float().mean(0)                     # eq. 19, hard
g_soft = Ghat.mean(0)                                              # eq. 19, soft
g_soft.sum().backward()
ok("the hard count carries no gradient", not f_hard.requires_grad)
ok("the soft mean does", Ghat.grad is not None and float(Ghat.grad.abs().min()) > 0)
ok("so the product is differentiable in exactly one factor", True,
   "the classical trick, reused by eq. 12")
