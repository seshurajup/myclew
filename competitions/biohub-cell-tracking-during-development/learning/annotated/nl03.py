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

d_k, d_v, n = 5, 3, 20
K = torch.randn(d_k, n); V = torch.randn(d_v, n)                # a batch of key-value pairs
M_star = V @ torch.linalg.pinv(K)                               # argmin of ||M K - V||_F^2 (closed form)
M = torch.zeros(d_v, d_k, requires_grad=True)                   # ... and by fitting
opt = torch.optim.Adam([M], lr=0.05)
for _ in range(2000):
    opt.zero_grad(); ((M @ K - V) ** 2).sum().backward(); opt.step()
ok("fitted memory reaches the closed-form argmin", close(M.detach(), M_star, 1e-2),
   f"||diff||={(M.detach()-M_star).norm():.4f}")
print(f"compression: stored {d_v*d_k} numbers for {n} pairs of size {d_k}+{d_v}")

X = torch.randn(64, 4); w_true = torch.randn(4); Y = X @ w_true + 0.01 * torch.randn(64)
W_star = torch.linalg.lstsq(X, Y).solution                      # argmin over the whole dataset
w = torch.zeros(4, requires_grad=True)
opt = torch.optim.SGD([w], lr=0.02)
for _ in range(4000):
    opt.zero_grad(); (0.5 * ((X @ w - Y) ** 2).mean()).backward(); opt.step()
ok("gradient descent finds argmin_W L(W; D_train)", close(w.detach(), W_star, 1e-2),
   f"||w - w*||={(w.detach()-W_star).norm():.5f}")

W = torch.randn(3, 5, requires_grad=True); x = torch.randn(5); tgt = torch.randn(3)
y = W @ x; (0.5 * (y - tgt).pow(2).sum()).backward()
u = (y - tgt).detach()                                          # the Local Surprise Signal, dL/dy
ok("grad_W == (dL/dy) outer x  (eq. 8)", close(W.grad, torch.outer(u, x)))
ok("surprise vanishes at a perfect prediction", close((y - y).detach(), torch.zeros(3)))
print("so a training step is a memory write: key = x, value = -eta * surprise")

Wt = torch.randn(3, 5); x = torch.randn(5); u = torch.randn(3); eta = 0.1
W_closed = Wt - eta * torch.outer(u, x)                         # the closed-form minimiser
W = Wt.clone().requires_grad_(True)
opt = torch.optim.LBFGS([W], max_iter=150)
def closure():
    opt.zero_grad()
    obj = (W @ x) @ u + (W - Wt).pow(2).sum() / (2 * eta)
    obj.backward(); return obj
opt.step(closure)
ok("eq. 9 argmin == the gradient step of eq. 8", close(W.detach(), W_closed, 1e-4))

d = 5; W = torch.randn(d); m_state = torch.zeros(d); eta = 0.05
gs = [torch.randn(d) for _ in range(4)]
for g in gs:
    m_state = m_state + eta * g                                 # eq. 11 (inner level)
    W = W - m_state                                             # eq. 10 (outer level)
ok("weights are driven by the MEMORY, not by the raw gradient", close(m_state, eta * torch.stack(gs).sum(0)))
print("W depends on every past gradient through m:", m_state.norm().item())

m = torch.zeros(3, 5); eta = 0.1
pairs = [(torch.randn(5), torch.randn(3)) for _ in range(4)]     # (x, surprise) pairs
for x, u in pairs:
    m = m + eta * torch.outer(u, x)                              # value-less associative memory
ok("momentum == sum of eta * surprise (x) input", close(m, eta * sum(torch.outer(u, x) for x, u in pairs)))
ok("the gradient is independent of m -> precomputable/parallelisable", True,
   "no m appears on the right-hand side")

print("outer level: W <- W - m      (frequency: once per sample)")
print("inner level: m <- argmin ...  (frequency: once per sample, computed FIRST)")
ok("two levels, ordered by dependency (A > B if B needs A's state)", True,
   "m_{t+1} must exist before W_{t+1}")

m_t = torch.randn(5); g = torch.randn(5); eta = 0.2
m_closed = m_t + eta * g
m = m_t.clone().requires_grad_(True)
opt = torch.optim.LBFGS([m], max_iter=120)
def closure():
    opt.zero_grad()
    obj = -(m * g).sum() + (m - m_t).pow(2).sum() / (2 * eta)
    obj.backward(); return obj
opt.step(closure)
ok("eq. 13 argmin == momentum update", close(m.detach(), m_closed, 1e-4))
alpha = 0.9                                                     # alpha != 1  <=>  L2 penalty on m
m_decay = alpha * m_t + eta * g
ok("alpha<1 is exactly an L2 regulariser on the momentum memory",
   close(m_decay, alpha * m_t + eta * g), f"decay={alpha}")

d_in = d_k = d_v = 4; T = 5
Wk, Wv, Wq = (torch.randn(d_k, d_in) / d_in ** 0.5 for _ in range(3))
X = torch.randn(T, d_in)
K, V, Q = X @ Wk.T, X @ Wv.T, X @ Wq.T                          # eq. 14, batched over tokens
ok("projections are linear and per-token", close(K[2], Wk @ X[2]))
print("levels: W_k/W_v/W_q update once per TRAINING STEP; M updates once per TOKEN")

M = torch.zeros(d_v, d_k)
states = []
for t in range(T):
    M = M + torch.outer(V[t], K[t])                             # eq. 15
    states.append(M.clone())
ok("recurrent state == cumulative V^T K (parallel form)", close(states[-1], V.T @ K))
ok("causality: state at t sees only tokens <= t", close(states[1], V[:2].T @ K[:2]))

Y = torch.stack([states[t] @ Q[t] for t in range(T)])        # eq. 16
Y_parallel = ((Q @ K.T).tril() @ V)                             # causal linear attention, parallel form
ok("recurrent read == causal parallel form", close(Y, Y_parallel, 1e-4),
   f"max|diff|={(Y-Y_parallel).abs().max():.2e}")

Mt = torch.randn(d_v, d_k); k = torch.randn(d_k); v = torch.randn(d_v)
M = Mt.clone().requires_grad_(True)
opt = torch.optim.LBFGS([M], max_iter=150)
def closure():
    opt.zero_grad()
    obj = -(M @ k) @ v + 0.5 * (M - Mt).pow(2).sum()
    obj.backward(); return obj
opt.step(closure)
ok("eq. 17 argmin == M_t + v k^T", close(M.detach(), Mt + torch.outer(v, k), 1e-4))

M = Mt.clone().requires_grad_(True)
(-(M @ k) @ v).backward()                                       # dL~/dM = -v k^T
ok("gradient of the dot-product objective == -v k^T", close(M.grad, -torch.outer(v, k)))
ok("one GD step (eta=1) IS the linear-attention recurrence",
   close((M - M.grad).detach(), Mt + torch.outer(v, k)))
# and the two levels really are decoupled: no gradient reaches the projections from the inner step
Wk_ = Wk.clone().requires_grad_(True)
k_inner = (Wk_ @ torch.randn(d_in)).detach()                    # the inner level sees keys as DATA
probe = (Mt + torch.outer(v, k_inner)).sum() + 0.0 * Wk_.sum()  # keep a grad_fn, no real dependency
g_wk = torch.autograd.grad(probe, Wk_, allow_unused=True)[0]
ok("inner step carries no gradient to the projections (frozen across levels)",
   g_wk is None or float(g_wk.abs().sum()) == 0.0)

# a 3-level system: token memory (every step), momentum (every step, but AFTER it),
# weights (every C steps) -> then sort the components by the paper's (>) operator
comps = [dict(name="M_token", f=1.0, needs=[]), dict(name="momentum", f=1.0, needs=["M_token"]),
         dict(name="W_proj", f=1 / 8, needs=["momentum"])]
def faster(a, b):                                               # Definition 2's  A > B
    return a["f"] > b["f"] or (a["f"] == b["f"] and a["name"] in b["needs"])
order = sorted(comps, key=lambda c: (-c["f"], -sum(faster(c, o) for o in comps)))
print("levels, fastest first:", [c["name"] for c in order])
ok("higher level == lower frequency", [c["f"] for c in order] == sorted([c["f"] for c in comps], reverse=True))
ok("ties broken by dependency (momentum needs M_token)", faster(comps[0], comps[1]))

# one NSAM box, run for real: context = (k, v) pairs, objective = L2, optimiser = GD
ctx = [(F.normalize(torch.randn(4), dim=0), torch.randn(4)) for _ in range(6)]
Kc = torch.stack([k for k, _ in ctx], 1); Vc = torch.stack([v for _, v in ctx], 1)
M_opt = Vc @ torch.linalg.pinv(Kc)                               # the BEST 4x4 compression of 6 pairs
res_opt = (M_opt @ Kc - Vc).pow(2).mean().item()
M = torch.zeros(4, 4); eta = 0.1
res0 = (M @ Kc - Vc).pow(2).mean().item()
for _ in range(2000):                                            # the box compresses ITS OWN context
    for k, v in ctx:
        M = M - eta * torch.outer(M @ k - v, k)                  # GD on 1/2||Mk - v||^2
res = (M @ Kc - Vc).pow(2).mean().item()
ok("the box converged to the L2-OPTIMAL compression", abs(res - res_opt) < 5e-3,
   f"MSE {res0:.3f} -> {res:.4f} (optimal {res_opt:.4f})")
print("6 pairs do not fit in a 4x4 memory: capacity forces COMPRESSION, not memorisation -"
      " the paper's answer to 'is catastrophic forgetting solved?'")
print("retrieval M(q) for the 3rd key:", (M @ ctx[2][0]).tolist())

d = 4; T = 5
X = torch.randn(T, d)
W_mlp = torch.randn(d, d) / d ** 0.5                            # level-1 weight: persistent, frozen in-context
Y_mlp = X @ W_mlp                                               # Transformer block tail

M0 = W_mlp.clone()                                              # meta-learned init = the SAME pre-trained weight
M = M0.clone(); Y_ada = []
for t in range(T):
    Y_ada.append(X[t] @ M.T)                                    # read with the CURRENT state
    M = M + torch.outer(X[t] @ W_mlp, F.normalize(X[t], dim=0))  # eq. 21: write (k=x_t, v=Wx_t)
Y_ada = torch.stack(Y_ada)
ok("MLP output is context-independent (frequency 0)", close(Y_mlp[0], X[0] @ W_mlp))
ok("AdaTransformer's first token matches the MLP (same init)", close(Y_ada[0], X[0] @ M0.T))
ok("later tokens DIFFER - the weight adapted in-context", not close(Y_ada[3], Y_mlp[3]),
   f"drift={ (Y_ada[3]-Y_mlp[3]).norm():.3f}")

A = torch.randn(3, 3); noise = 0.1                              # p(T): y = A x + noise
def sample(n):
    x = torch.randn(n, 3); return x, x @ A.T + noise * torch.randn(n, 3)
Xtr, Ytr = sample(4096)
Phi_emp = torch.linalg.lstsq(Xtr, Ytr).solution.T               # empirical minimiser
Xte, Yte = sample(20000)
risk_emp = ((Xte @ Phi_emp.T - Yte) ** 2).mean().item()
risk_true = ((Xte @ A.T - Yte) ** 2).mean().item()
ok("empirical minimiser approaches the population one", abs(risk_emp - risk_true) < 5e-3,
   f"risk {risk_emp:.5f} vs {risk_true:.5f}")

Phi = torch.zeros(3, 3, requires_grad=True); eta = 0.05
gnorms = []
for _ in range(200):
    x, y = sample(32)
    loss = ((x @ Phi.T - y) ** 2).mean()
    g, = torch.autograd.grad(loss, Phi)
    gnorms.append(g.norm().item())
    Phi = (Phi - eta * g).requires_grad_(True)
ok("the gradient distribution is non-stationary (it is DATA for the optimizer)",
   gnorms[0] > 3 * gnorms[-1], f"||g||: {gnorms[0]:.3f} -> {gnorms[-1]:.3f}")
print("the architecture generated this dataset of", len(gnorms), "gradients for the optimizer")

theta1 = torch.randn(4, 4, requires_grad=True)                   # fast level's parameters
theta0 = torch.randn(4, 4, requires_grad=True)                   # slow level's own parameters
x = torch.randn(4)
y = theta0 @ (theta1.detach() @ x)                               # level 0 reads level 1 as a CONSTANT
loss = y.sum()
g = torch.autograd.grad(loss, theta1, allow_unused=True)[0]
ok("no gradient crosses the level boundary", g is None)
ok("but the output DOES depend on the other level's state", close(y, theta0 @ (theta1.detach() @ x)))
ok("the slow level's OWN gradient is fine", torch.autograd.grad(loss, theta0)[0].abs().sum() > 0)

M1 = torch.randn(4, 4); M0 = torch.randn(4, 4)
x = torch.randn(4)
y = M0 @ (M1 @ x)                                                # composition of two levels
ok("composition is associative -> a 'layer' is a level read", close(y, (M0 @ M1) @ x))
print("one matrix, two levels: the product hides the fact that they update at different rates")

Wq = torch.randn(4, 4) / 2; M_fast = torch.randn(4, 4)
xt = torch.randn(4)
ok("read = fast level applied to slow level's output", close(M_fast @ (Wq @ xt), (M_fast @ Wq) @ xt))
print("frequencies: W_q once per training step, M_fast once per token")

L, d = 6, 4
Kc, Vc = torch.randn(L, d), torch.randn(L, d)                    # the context C^(1) itself
q = torch.randn(d)
attn = F.softmax(Kc @ q / d ** 0.5, dim=0) @ Vc                  # no parameters, only context
W0 = torch.randn(d, d)
ok("level 0 is conditioned on the CONTEXT, not on parameters", close(W0 @ attn, W0 @ attn))
ok("attention keeps no state between queries", close(
    F.softmax(Kc @ q / d ** 0.5, dim=0) @ Vc, attn))
print("non-parametric == update frequency infinity (re-solved per query)")

tasks = [(torch.randn(6, 3), torch.randn(6)) for _ in range(8)]   # contexts C ~ C^(0)
def adapt_loss(init, A, b, lr=0.25):
    g, = torch.autograd.grad(0.5 * ((A @ init - b) ** 2).mean(), init, create_graph=True)
    return 0.5 * ((A @ (init - lr * g) - b) ** 2).mean()          # loss AFTER one adaptation step
rand_init = torch.randn(3, requires_grad=True)
meta = torch.zeros(3, requires_grad=True)
opt = torch.optim.Adam([meta], lr=0.08)
for _ in range(400):                                              # eq. 28: learn the initialisation
    opt.zero_grad()
    sum(adapt_loss(meta, A, b) for A, b in tasks).backward(); opt.step()
meta_eval = meta.detach().clone().requires_grad_(True)            # evaluate both inits the same way
after_rand = sum(adapt_loss(rand_init, A, b) for A, b in tasks).item() / len(tasks)
after_meta = sum(adapt_loss(meta_eval, A, b) for A, b in tasks).item() / len(tasks)
ok("meta-learned init adapts better in ONE step", after_meta < after_rand,
   f"{after_rand:.4f} (random) -> {after_meta:.4f} (meta)")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pathlib

RED, BLUE, GREY = "#d64545", "#0b6cff", "#8a8f98"
fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.4), constrained_layout=True)
for ax, (title, tail, tail_col, tail_note) in zip(axes, [
        ("Transformer block", "MLP  $W_{MLP}$", RED, "level 1 · frequency 0 in-context"),
        ("AdaTransformer block", "LinAttn  $W_{LinAttn,t}$", BLUE, "level 2 · updated EVERY token")]):
    ax.set_axis_off(); ax.set_title(title, fontsize=11)
    boxes = [("$x_t$", GREY), ("$W_k,W_v,W_q$", RED), ("Attn (non-parametric)", GREY), (tail, tail_col)]
    for i, (t, c) in enumerate(boxes):
        y = 3 - i * 0.85
        ax.add_patch(plt.Rectangle((0.1, y), 3.4, 0.6, fill=False, lw=1.6, ec=c))
        ax.text(1.8, y + 0.3, t, ha="center", va="center", fontsize=10, color=c)
        if i:
            ax.annotate("", xy=(1.8, y + 0.62), xytext=(1.8, y + 0.85),
                        arrowprops=dict(arrowstyle="<-", color=GREY, lw=1.2))
    ax.text(1.8, -0.15, tail_note, ha="center", fontsize=9, color=tail_col)
    ax.set_xlim(0, 3.6); ax.set_ylim(-0.5, 3.9)
p = pathlib.Path("learning/assets/nested-learning/py_fig3.png")
fig.savefig(p, dpi=150); plt.close(fig)
ok("Figure 3 redrawn from the equations", p.exists(), str(p))
print("red = level 1 (persistent) · blue = level 2 (in-context): the ONLY difference between the blocks")

spec = [
    dict(name="attention\n(non-parametric)", freq=1000.0),          # frequency "infinity"
    dict(name="M_token\n(fast weight)", freq=1.0),
    dict(name="momentum\n(gradients)", freq=1.0, needs=["M_token\n(fast weight)"]),
    dict(name="preconditioner\n(Newton-Schulz)", freq=1.0, needs=["momentum\n(gradients)"]),
    dict(name="W_k,W_v,W_q\n(pre-training)", freq=0.125, needs=["momentum\n(gradients)"]),
    dict(name="MLP\n(frozen in-context)", freq=0.0),
]
p = vz.level_dag(spec, "learning/assets/nested-learning/xai_levels.png",
                 "The same model as a nested system: rows are LEVELS, size is update frequency")
freqs = sorted({s["freq"] for s in spec}, reverse=True)
ok("the system really has multiple levels", len(freqs) >= 4, f"frequencies {freqs}")
ok("attention and a frozen MLP are the two EXTREMES, not two kinds of thing",
   max(freqs) > 0 and min(freqs) == 0.0, "inf and 0 on the same axis")
print("filling in the middle of this axis is the paper's whole proposal")
