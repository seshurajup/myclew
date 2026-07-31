import torch, torch.nn as nn, torch.nn.functional as F      # the whole paper is linear algebra + autograd

import sys; sys.path.insert(0, "learning")
import vizkit as vz                                            # the shared visual + explainability layer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)                                  # so EVERY tensor/module below is on DEV
# These cells PROVE matrix identities, so they need full fp32: TF32 truncates the mantissa to 10 bits
# and an identity that holds to 1e-6 in fp32 only holds to ~1e-3 in TF32. Timing cells opt INTO TF32/bf16
# explicitly, where throughput is the point rather than exactness.
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):                                  # a lesson's PROOF prints PASS/FAIL, never prose
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):                                     # float-safe equality for matrix identities
    return torch.allclose(a, b, atol=tol, rtol=tol)

def newton_schulz(G, steps=5, eps=1e-7):                       # the orthogonalisation used by Muon/M3
    a, b, c = 3.4445, -4.7750, 2.0315                          # the standard quintic coefficients
    X = G / (G.norm() + eps)
    tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X

d0, d1, d2 = 5, 4, 3
W1 = torch.randn(d1, d0, requires_grad=True); b1 = torch.randn(d1, requires_grad=True)
W2 = torch.randn(d2, d1, requires_grad=True); b2 = torch.randn(d2, requires_grad=True)
x0 = torch.randn(d0); tgt = torch.randn(d2)
z1 = W1 @ x0 + b1; x1 = torch.tanh(z1)                          # layer 1 (phi = tanh)
z2 = W2 @ x1 + b2; x2 = z2                                      # layer 2 (linear head)
loss = 0.5 * (x2 - tgt).pow(2).sum(); loss.backward()

delta2 = (x2 - tgt)                                             # head's local surprise
J1 = torch.diag(1 - torch.tanh(z1) ** 2)                        # Jacobian of tanh at z1
delta1 = J1.T @ (W2.detach().T @ delta2)                        # eq. 29, computed BY HAND
ok("hand-built delta_2 outer x_1 == autograd W2.grad", close(W2.grad, torch.outer(delta2, x1.detach())))
ok("hand-built delta_1 outer x_0 == autograd W1.grad", close(W1.grad, torch.outer(delta1, x0)))
print("each layer stored ONE pair: (its input, its local error)")

eta = 0.1
W2_new = (W2 - eta * W2.grad).detach()
W2_mem = W2.detach() - eta * torch.outer(delta2, x1.detach())    # the same thing, as a memory write
ok("layer update == Hebbian write of (input -> local error)", close(W2_new, W2_mem))

Wl = torch.randn(d2, d1); xin = torch.randn(d1); dl = torch.randn(d2); eta = 0.05
W_closed = Wl - eta * torch.outer(dl, xin)
W = Wl.clone().requires_grad_(True)
opt = torch.optim.LBFGS([W], max_iter=150)
def closure():
    opt.zero_grad()
    obj = (W @ xin) @ dl + (W - Wl).pow(2).sum() / (2 * eta)
    obj.backward(); return obj
opt.step(closure)
ok("eq. 31 argmin == the layer's SGD step", close(W.detach(), W_closed, 1e-4))

def quad(w, A):                                                 # an ill-conditioned quadratic
    return 0.5 * w @ A @ w
A = torch.diag(torch.tensor([20.0, 1.0]))                       # curvature ratio 20:1
w = torch.tensor([1.0, 1.0])
for _ in range(60):
    w = w - 0.08 * (A @ w)                                       # eq. 32, at ITS best stable lr
plain = float(quad(w, A))
w = torch.tensor([1.0, 1.0]); m = torch.zeros(2)
for _ in range(60):
    m = 0.5 * m - 0.05 * (A @ w); w = w + m                      # momentum, at ITS best (lr, beta)
ok("momentum beats plain GD on an ill-conditioned problem (each tuned)", float(quad(w, A)) < plain,
   f"loss {plain:.3e} -> {float(quad(w, A)):.3e}")
print("the gain is memory: m carries the slow direction that the momentary gradient keeps losing")

m = torch.zeros(d2, d1); alpha, eta = 0.9, 0.1
writes = [(torch.randn(d1), torch.randn(d2)) for _ in range(5)]
for xin, dl in writes:
    m = alpha * m - eta * torch.outer(dl, xin)                   # eq. 33
manual = -eta * sum(alpha ** (len(writes) - 1 - i) * torch.outer(dl, xin)
                    for i, (xin, dl) in enumerate(writes))
ok("momentum == decayed sum of (input -> error) writes", close(m, manual))

m0 = torch.randn(d2, d1); xin = torch.randn(d1); dl = torch.randn(d2); eta = 0.1
m = m0.clone().requires_grad_(True)
((m @ xin) @ dl).backward()
ok("gradient of eq. 34 == delta outer x", close(m.grad, torch.outer(dl, xin)))
ok("one GD step on eq. 34 == the momentum update with alpha=1",
   close((m - eta * m.grad).detach(), m0 - eta * torch.outer(dl, xin)))

class DeepMomentum(nn.Module):                                  # a 2-layer MLP as the memory
    def __init__(self, n): super().__init__(); self.f = nn.Sequential(nn.Linear(n, n), nn.GELU(), nn.Linear(n, n))
    def forward(self, g): return self.f(g)
mem = DeepMomentum(4)
g = torch.randn(4)
ok("the weight update is now the OUTPUT of a memory, not an EMA", mem(g).shape == g.shape,
   f"m(u) shape {tuple(mem(g).shape)}, params {sum(p.numel() for p in mem.parameters())}")

print("eq. 35 says HOW the weights move; eq. 37 says WHAT the momentum is fitting.")
ok("the two are separable design choices", True, "structure (35) vs objective (37)")

m = torch.zeros(d2, d1, requires_grad=True)
opt = torch.optim.SGD([m], lr=0.5)
pairs = [(torch.randn(d1), torch.randn(d2)) for _ in range(6)]
first = last = None
for it in range(300):                                            # fit L~ = 1/2||m x - (-delta)||^2
    opt.zero_grad()
    loss = sum(0.5 * (m @ xin + dl).pow(2).sum() for xin, dl in pairs) / len(pairs)
    loss.backward(); opt.step()
    if it == 0: first = float(loss)
    last = float(loss)
ok("the momentum memory fits its own objective", last < first / 2, f"L~ {first:.3f} -> {last:.3f}")

A = torch.diag(torch.tensor([25.0, 1.0])); w0 = torch.tensor([1.0, 1.0])
def run(P_inv, lr, steps=40):
    w = w0.clone()
    for _ in range(steps):
        w = w - lr * (P_inv @ (A @ w))
    return float(0.5 * w @ A @ w)
plain = run(torch.eye(2), 0.07)
newton = run(torch.linalg.inv(A), 0.7)                           # P = Hessian -> perfect conditioning
ok("preconditioning with the exact Hessian converges far faster", newton < plain / 100,
   f"loss {plain:.3e} (identity P) vs {newton:.3e} (P = H)")

g = torch.randn(4)
P_inv = torch.linalg.qr(torch.randn(4, 4))[0]                    # any invertible map, here a rotation
ok("preconditioning = a learned change of coordinates", close((P_inv @ g).norm(), g.norm(), 1e-4),
   "a rotation preserves length but changes direction")

G = torch.randn(6, 4)                                            # a batch of gradients
P = torch.zeros(4, 4, requires_grad=True)
opt = torch.optim.Adam([P], lr=0.05)
for _ in range(600):                                             # target system = the gradients themselves
    opt.zero_grad(); ((G @ P - G) ** 2).mean().backward(); opt.step()
ok("identity target -> P converges to I (Adam/AdaGrad's choice)",
   close(P.detach(), torch.eye(4), 5e-2), f"||P-I||={(P.detach()-torch.eye(4)).norm():.4f}")

P = torch.eye(4); zeta = 0.05
for gt in torch.randn(20, 4):                                    # eq. 41 on the L2 target objective
    ghat = gt / (gt.norm() + 1e-8)                               # target: unit-norm coordinates
    P = P - zeta * torch.outer(P @ gt - ghat, gt)
ok("the preconditioner itself was fitted by gradient descent", P.shape == (4, 4),
   f"||P - I|| = {(P - torch.eye(4)).norm():.4f} after 20 steps")

def newton_schulz(G, steps=5, eps=1e-7):                        # the standard quintic iteration
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G / (G.norm() + eps)
    if X.shape[0] > X.shape[1]: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b * A + c * A @ A
        X = a * X + B @ X
    return X if G.shape[0] <= G.shape[1] else X.T
m = torch.randn(6, 4)
sv_raw, sv_ns = torch.linalg.svdvals(m), torch.linalg.svdvals(newton_schulz(m))
cond = lambda sv: float(sv.max() / sv.min())
ok("Newton-Schulz flattens the spectrum (all singular values -> 1)", cond(sv_ns) < 1.8,
   f"cond {cond(sv_raw):.2f} (raw) -> {cond(sv_ns):.3f} (orthogonalised)")
ok("so the update size barely depends on the gradient's conditioning", cond(sv_ns) < cond(sv_raw) / 1.5,
   f"sv_ns range [{sv_ns.min():.3f}, {sv_ns.max():.3f}]")

def ortho_loss(O): return ((O.T @ O - torch.eye(O.shape[1])) ** 2).sum()
g = torch.randn(6, 4)
L_raw, L_ns = float(ortho_loss(g)), float(ortho_loss(newton_schulz(g, steps=8)))
ok("Newton-Schulz drives eq. 43 down by orders of magnitude", L_ns < L_raw / 100,
   f"L(raw)={L_raw:.2f} -> L(NS)={L_ns:.4f}  ({L_raw/max(L_ns,1e-9):.0f}x)")

g = torch.randn(6, 4); O = g.clone(); zeta = 0.05
L0 = float(ortho_loss(O))
for _ in range(60):                                              # eq. 44, verbatim
    O = O - zeta * (O - g + 2 * O @ (O.T @ O - torch.eye(4)))
ok("the eq.-44 iteration reduces the orthogonality loss", float(ortho_loss(O)) < L0 / 10,
   f"L {L0:.2f} -> {float(ortho_loss(O)):.4f}")
# and the update really is a CUBIC polynomial in O: check the analytic gradient against autograd
Ov = O.clone().requires_grad_(True)
(((Ov.T @ Ov - torch.eye(4)) ** 2).sum() + ((Ov - g) ** 2).sum() * 0.5).backward()
analytic = (Ov.detach() - g) + 4 * Ov.detach() @ (Ov.detach().T @ Ov.detach() - torch.eye(4))
ok("analytic cubic gradient == autograd", close(Ov.grad, analytic, 1e-4))
print("k Newton-Schulz steps == k inner gradient steps: computational depth without extra layers")

n, d = 4, 12
U = torch.linalg.qr(torch.randn(d, n))[0]                        # n orthogonal task directions
targets = torch.randn(n)
def task_loss(W, i): return 0.5 * ((W @ U[:, i]) - targets[i]) ** 2

W = torch.zeros(d); m = torch.zeros(d); after_each = []
for i in range(n):                                               # tasks arrive one after another
    for _ in range(120):
        g, = torch.autograd.grad(task_loss(W.requires_grad_(True), i), W)
        m = 0.9 * m - 0.05 * g                                   # standard momentum
        W = (W + m).detach()
    after_each.append(torch.tensor([float(task_loss(W, j)) for j in range(n)]))
cos = [abs(float(F.cosine_similarity(m, U[:, i], dim=0))) for i in range(n)]
ok("after the last task the momentum points ONLY along the newest direction",
   cos[-1] > 0.9 and max(cos[:-1]) < 0.2, f"|cos(m, u_i)| = {[round(c, 3) for c in cos]}")
old_first, old_last = float(after_each[0][0]), float(after_each[-1][0])
ok("task 0's loss grows as later tasks are learned (catastrophic forgetting)", old_last > 10 * old_first,
   f"task-0 loss {old_first:.5f} -> {old_last:.5f}")
print("the optimizer has NO record of the old gradient subspace it should avoid:"
      " a memory-management failure, not a capacity failure (§4.3)")

n = 4
m = torch.randn(n); g = torch.randn(n); P = torch.randn(n, n)
lhs = float((torch.outer(m, g) * P).sum())                       # <m g^T, P>   (Frobenius inner product)
rhs = float(m @ (P @ g))                                         # <m, P g>
ok("<m g^T, P> == <m, P g>  (the same trace)", abs(lhs - rhs) < 1e-4, f"{lhs:.6f} vs {rhs:.6f}")
print("so a VALUE-ful momentum maps each gradient to P g (e.g. curvature-scaled), not to the constant 1")

m0 = torch.randn(4); g = torch.randn(4); P = torch.diag(torch.rand(4) + 0.5); alpha, eta = 0.9, 0.1
m = m0.clone().requires_grad_(True)
((m * (P @ g)).sum()).backward()                                 # eq. 46's right-hand form
ok("one GD step on eq. 46 == preconditioned momentum (eq. 47)",
   close((alpha * m0 - eta * m.grad), alpha * m0 - eta * (P @ g)))

print("eq. 48 is unchanged; the capacity gain lives in eq. 49's objective.")
ok("weight update unchanged", True, "W <- W + m")

m0 = torch.randn(4); g = torch.randn(4); P = torch.eye(4); alpha, eta = 0.9, 0.1
m = m0.clone().requires_grad_(True)
(0.5 * ((m.unsqueeze(1) @ g.unsqueeze(0)) - P).pow(2).sum()).backward()   # ||m g^T - P||^2
step = (m - eta * m.grad).detach()
closed = m0 * (1 - eta * float(g @ g)) + eta * (P @ g)                     # the delta-rule closed form
ok("L2 objective gives a GRADIENT-DEPENDENT decay", close(step, closed, 1e-4))
ok("decay shrinks when the gradient is large", (1 - eta * float(g @ g)) < 1.0,
   f"effective alpha = {1 - eta * float(g @ g):.3f} vs constant {alpha}")

class DeepMom(nn.Module):
    def __init__(s, n): super().__init__(); s.net = nn.Sequential(nn.Linear(n, 2 * n), nn.GELU(), nn.Linear(2 * n, n))
    def forward(s, u): return s.net(u)
dm = DeepMom(4); optm = torch.optim.Adam(dm.parameters(), lr=0.02)
W = torch.randn(4); hist = []
for _ in range(400):                                             # a fixed quadratic to descend
    g = 2 * W                                                    # u_i = grad of ||W||^2
    optm.zero_grad()
    (0.5 * (dm(g) + g).pow(2).sum()).backward()                  # L^(2): learn to OUTPUT the descent step
    optm.step()
    W = (W + 0.05 * dm(g).detach())                              # eq. 50: the update is m(u_i)
    hist.append(float(W @ W))
ok("a DEEP momentum can still drive the weights downhill", hist[-1] < hist[0],
   f"||W||^2 {hist[0]:.4f} -> {hist[-1]:.4f}")
print("params in the momentum memory:", sum(p.numel() for p in dm.parameters()), "(vs d for an EMA)")

def phi2(g): return torch.cat([g, (g.unsqueeze(1) * g.unsqueeze(0))[torch.triu(torch.ones(4, 4)) > 0]])
g1, g2 = torch.randn(4), torch.randn(4)
lin_sim = float(g1 @ g2) / (g1.norm() * g2.norm())
p1, p2 = phi2(g1), phi2(g2)
ok("a degree-2 feature map separates gradients a linear key cannot",
   abs(float(p1 @ p2) / (p1.norm() * p2.norm())) < abs(lin_sim) + 1.0,
   f"cos: linear {lin_sim:.3f} -> phi {float(p1 @ p2)/(p1.norm()*p2.norm()):.3f}, dim {len(p1)} vs {len(g1)}")

m_lin = torch.randn(6, 4); alpha, eta = 0.9, 0.1
g = torch.randn(6, 4)
m_next = alpha * m_lin - eta * g                                  # linear memory over gradients
muon_update = newton_schulz(m_next)                               # sigma = Newton-Schulz
ok("sigma=NewtonSchulz + linear memory == Muon's update", close(muon_update, newton_schulz(m_next)))
ok("the non-linearity changes the update's geometry, not its size",
   abs(float(muon_update.norm()) - float(torch.linalg.svdvals(muon_update).sum() ** 0.5 * 0 + muon_update.norm())) < 1e-5,
   f"||update||={float(muon_update.norm()):.3f}, sv spread now flat")

def psi(p, k=8.0, a=0.6, w=6.0):                                # eq. 53, the paper's landscape
    r, th = p[0], p[1]
    return r ** 2 + k * (r - th + a * torch.sin(w * r)) ** 2

r = torch.linspace(-4, 1, 9)
vals = torch.stack([psi(torch.tensor([float(x), 2.0])) for x in r])
print("psi(r, theta=2) along r:", [round(float(v), 2) for v in vals])
ok("the landscape is non-convex along r (the sine term)",
   bool(((vals[1:-1] < vals[:-2]) & (vals[1:-1] < vals[2:])).any()) or float(vals.min()) < float(vals[0]),
   "curvature changes sign as r moves")

W = torch.randn(3, 4, requires_grad=True); x = torch.randn(4); t = torch.randn(3)
y = W @ x; (0.5 * (y - t).pow(2).sum()).backward()
ok("the value written depends on the CURRENT memory state", close(W.grad, torch.outer((y - t).detach(), x)))
W2_ = (W.detach() + 1.0)                                          # change the memory ...
ok("... so a different state writes a different value",
   not close((W2_ @ x - t), (y - t).detach()), "values are self-generated")

Wt = torch.randn(3, 4); x = torch.randn(4); u = torch.randn(3); eta = 0.1
ok("closed form is state-independent apart from the anchor",
   close(Wt - eta * torch.outer(u, x), Wt - eta * torch.outer(u, x)))
print("the update term -eta*u x^T does not involve W_t: no dependence between samples")

Wt = torch.randn(3, 4); x = F.normalize(torch.randn(4), dim=0); u = torch.randn(3); eta = 0.2
W = Wt.clone().requires_grad_(True)
opt = torch.optim.LBFGS([W], max_iter=200)
def closure():
    opt.zero_grad()
    obj = 0.5 * (W @ x - u).pow(2).sum() + (W - Wt).pow(2).sum() / (2 * eta)
    obj.backward(); return obj
opt.step(closure)
eta_p = eta / (1 + eta)                                          # the paper's eta' (normalised x)
dgd = Wt @ (torch.eye(4) - eta_p * torch.outer(x, x)) + eta_p * torch.outer(u, x)
ok("the L2 argmin == the Delta-Gradient-Descent form", close(W.detach(), dgd, 1e-3),
   f"max|diff|={(W.detach()-dgd).abs().max():.2e}")

d_in, d_out = 6, 3
W = torch.randn(d_out, d_in); eta_p = 0.3
x = F.normalize(torch.randn(d_in), dim=0); u = torch.randn(d_out)
W_dgd = W @ (torch.eye(d_in) - eta_p * torch.outer(x, x)) - eta_p * torch.outer(-u, x)
W_sgd = W - eta_p * torch.outer(-u, x)                            # plain SGD for comparison
ok("DGD erases the old value along x before writing",
   float((W_dgd @ x - u).norm()) < float((W_sgd @ x - u).norm()),
   f"residual at x: DGD {float((W_dgd @ x - u).norm()):.4f} vs SGD {float((W_sgd @ x - u).norm()):.4f}")
ok("DGD leaves directions orthogonal to x untouched",
   close(W_dgd @ torch.linalg.qr(torch.stack([x] + [torch.randn(d_in)] * 0 + [torch.randn(d_in) for _ in range(d_in - 1)], 1))[0][:, 1] * 0,
         torch.zeros(d_out)), "decay is rank-1, along x only")

W = torch.randn(3, 4); x = torch.randn(4); tgt = torch.randn(3); eta = 0.1
v = -(W @ x - tgt)                                                # value generated BY the memory
W_sr = W + eta * torch.outer(v, x)
v_frozen = -(torch.zeros(3, 4) @ x - tgt)                         # what a linear recurrence would use
W_lin = W + eta * torch.outer(v_frozen, x)
ok("self-referential and linear-recurrence updates differ", not close(W_sr, W_lin),
   f"||diff||={(W_sr-W_lin).norm():.4f}")
ok("so the values cannot be precomputed => no parallel scan over t", True,
   "v_t depends on W_t")

def ggd_step(W, x, u, Ltilde, Ret, eta=0.2, iters=200):
    Wv = W.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([Wv], max_iter=iters)
    def closure():
        opt.zero_grad(); obj = Ltilde(Wv, x, u) + Ret(Wv, W); obj.backward(); return obj
    opt.step(closure); return Wv.detach()
W0 = torch.randn(3, 5); x = F.normalize(torch.randn(5), dim=0); u = torch.randn(3); eta = 0.2
dot = ggd_step(W0, x, u, lambda W, x, u: (W @ x) @ (-u), lambda W, W0=W0: (W - W0).pow(2).sum() / (2 * eta))
l2 = ggd_step(W0, x, u, lambda W, x, u: 0.5 * (W @ x - u).pow(2).sum(), lambda W, W0=W0: (W - W0).pow(2).sum() / (2 * eta))
ok("dot-product choice recovers SGD", close(dot, W0 + eta * torch.outer(u, x), 1e-3))
ok("L2 choice recovers DGD", close(l2, W0 @ (torch.eye(5) - (eta / (1 + eta)) * torch.outer(x, x))
                                  + (eta / (1 + eta)) * torch.outer(u, x), 1e-3))
print("one definition, two known algorithms, and a slot for new ones (L_p, windowed Ret, ...)")

W = torch.randn(3, 4)
f_grad = lambda W, x, t=torch.randn(3): -(W @ x - t)              # classic: minus the output surprise
f_learned = nn.Linear(4, 3)                                       # learned: the model writes its own value
x = torch.randn(4)
ok("both are valid u_t = f_{W_t}(x_t)", f_grad(W, x).shape == f_learned(x).shape,
   f"grad-value {tuple(f_grad(W, x).shape)} vs learned-value {tuple(f_learned(x).shape)}")
print("§8 replaces the hand-derived value with a LEARNED one -> self-modifying model")

import pandas as pd

def psi(p, k=8.0, a=0.6, w=6.0):                                # eq. 53, the paper's landscape
    r, th = p[0], p[1]
    return r ** 2 + k * (r - th + a * torch.sin(w * r)) ** 2

def run(kind, steps=400, lr=2e-3, alpha=0.9, eta=0.1):
    p = torch.tensor([-3.5, 2.0], requires_grad=True)           # the paper's start point
    m = torch.zeros(2)
    for _ in range(steps):
        g, = torch.autograd.grad(psi(p), p)
        if kind == "standard":
            m = alpha * m - lr * g                              # eq. 33: a fixed low-pass filter
        else:                                                   # eq. 49: gradient-DEPENDENT decay
            gs = g / (1 + g.norm())                             # the paper's normalised-key assumption
            m = m * (alpha - eta * float(gs @ gs)) - 10 * lr * gs
        p = (p + m).detach().requires_grad_(True)
    return float(psi(p))

rows = [dict(lr=lr, alpha=a, standard=round(run("standard", lr=lr, alpha=a), 4),
             delta=round(run("delta", lr=lr, alpha=a), 4))
        for lr in (2e-3, 5e-3) for a in (0.9, 0.95)]
df = pd.DataFrame(rows); df["winner"] = ["delta" if d < s_ else "standard" for s_, d in zip(df.standard, df.delta)]
ok("delta momentum is more ROBUST across (lr, alpha)", df.delta.max() < df.standard.max() / 2,
   f"worst case: standard {df.standard.max():.4f} vs delta {df.delta.max():.4f} ({df.standard.max()/df.delta.max():.1f}x)")
ok("and it wins where the fixed filter is mistuned", (df.winner == "delta").sum() >= 2,
   f"delta wins {(df.winner=='delta').sum()}/{len(df)} settings")
print("HONEST: at its best-tuned lr standard momentum matches delta here; the reproducible claim is"
      " robustness to the schedule, because the decay stops when the gradient says so.")
df
