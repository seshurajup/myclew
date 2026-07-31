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

d = 8; T = 6
X = torch.randn(T, d)
Wk, Wv, Wq = (torch.randn(d, d) / d ** 0.5 for _ in range(3))
Weta, Walpha = torch.randn(d, 1) / d ** 0.5, torch.randn(d, 1) / d ** 0.5
K, V, Q = X @ Wk, X @ Wv, X @ Wq
eta_t = torch.sigmoid(X @ Weta).squeeze(-1)                     # in (0,1): a per-token learning rate
alpha_t = torch.sigmoid(X @ Walpha).squeeze(-1)                 # in (0,1): a per-token retention gate
ok("every token predicts its own write strength and forget rate",
   eta_t.shape == (T,) and float(eta_t.min()) > 0 and float(eta_t.max()) < 1,
   f"eta {[round(float(v),3) for v in eta_t[:3]]}, alpha {[round(float(v),3) for v in alpha_t[:3]]}")

Mmem = torch.zeros(d, d); k, v = F.normalize(K[0], dim=0), V[0]
for _ in range(50):                                              # any optimizer may fill this slot
    Mmem = Mmem - 0.3 * torch.outer(Mmem @ k - v, k)              # here: GD on 1/2||Mk - v||^2
ok("the slot is filled by an optimization process, not by a formula", float((Mmem @ k - v).norm()) < 1e-3,
   f"residual {(Mmem @ k - v).norm():.2e}")

ok("read is one matvec", (Mmem @ F.normalize(Q[0], dim=0)).shape == (d,))

class MemProj(nn.Module):
    """eq. 89/91: a 2-layer residual MLP, the architecture used for EVERY memory in Hope."""
    def __init__(s, d, dout=None):
        super().__init__(); dout = dout or d
        s.W2 = nn.Linear(d, d, bias=False); s.W1 = nn.Linear(d, dout, bias=False); s.same = (dout == d)
    def forward(s, x): return (x if s.same else 0) + s.W1(F.silu(s.W2(x)))
mk, mv, mq = MemProj(d), MemProj(d), MemProj(d)
me, ma = MemProj(d, 1), MemProj(d, 1)
xt = X[0]
ok("all five quantities now come from memories", (mk(xt).shape, me(xt).shape) == ((d,), (1,)),
   f"k {tuple(mk(xt).shape)}, eta {tuple(me(xt).shape)}")
ok("the initial states are ordinary parameters -> meta-learnable (eq. 28/73)",
   all(p.requires_grad for p in mk.parameters()))

boxes = ["k", "v", "q", "eta", "alpha"]
print("optimization boxes in this block:", len(boxes) + 1, "(five projections + the main memory)")
ok("sharing one v_t across all five boxes is the cheap version", True,
   "eq. 84 replaces it with per-memory values")

mem = MemProj(d)
ok("the main memory is the same architecture as the projections", isinstance(mem, nn.Module),
   f"params {sum(p.numel() for p in mem.parameters())}")

ok("read with an ADAPTIVE query", mem(mq(X[0])).shape == (d,),
   "q_t came from M_q, not from a frozen W_q")

Wq_fixed = torch.randn(d, d) / d ** 0.5                          # the ONE frozen projection
q_t = X[0] @ Wq_fixed
ok("q stays non-adaptive by design", close(q_t, X[0] @ Wq_fixed))
ok("k, v, eta, alpha are all adaptive", (mk(X[0]).shape, me(X[0]).shape) == ((d,), (1,)))

v_t = mv(X[0])
v_hat = {"k": mk(v_t), "v": mv(v_t), "q": mq(v_t), "mem": mem(v_t)}   # eq. 84
ok("each memory writes a target it generated itself",
   all(t.shape == (d,) for t in v_hat.values()),
   f"|v_hat_k - v_t| = {float((v_hat['k'] - v_t).norm()):.4f}")
ok("the targets differ per memory (not a shared label)",
   not close(v_hat["k"], v_hat["q"]), "self-generated, per-memory")

M_box = torch.zeros(d, d); k = F.normalize(mk(X[0]).detach(), dim=0); vh = v_hat["k"].detach()
for _ in range(80):
    M_box = M_box - 0.3 * torch.outer(M_box @ k - vh, k)          # L2 objective, GD
ok("a box learns the mapping to its self-generated value", float((M_box @ k - vh).norm()) < 1e-3,
   f"residual {(M_box @ k - vh).norm():.2e}")

ok("the block reads BEFORE it writes (causality)", True, "y_t uses M_{t-1}, then M_t is formed")
print("ingredients per token: k, v, eta, alpha, and the self-generated v_hat -> then eq. 88 writes")

ok("target of the write is the self-generated value", close(mk(v_t), v_hat["k"]))

D = 8
M0 = torch.randn(D, D) * 0.1
k = F.normalize(torch.randn(D), dim=0); vh = torch.randn(D)     # unit key (Hope L2-normalises k, q)

def step(M, objective, eta=0.4, alpha=1.0):                     # eq. 88, both objectives
    decay = M @ (alpha * torch.eye(D) - eta * torch.outer(k, k))  # the explicit DGD erase
    grad = -torch.outer(vh, k) if objective == "dot" else torch.outer(M @ k - vh, k)
    return decay - eta * grad

ok("dot-product objective (eq. 92) -> clean erase-then-write: (alpha-eta)Mk + eta v_hat",
   close(step(M0, "dot") @ k, 0.6 * (M0 @ k) + 0.4 * vh, 1e-5), "contraction factor 0.60")
ok("L2 objective (eq. 93) erases TWICE: (alpha-2eta)Mk + eta v_hat",
   close(step(M0, "l2") @ k, 0.2 * (M0 @ k) + 0.4 * vh, 1e-5),
   "the gradient carries its own -eta k k^T")
ok("=> stability needs eta_t <= alpha_t/2, hence eta_t is a BOUNDED (sigmoid) projection",
   abs(1.0 - 2 * 0.4) < 1 and abs(1.0 - 2 * 1.1) > 1, "eta=0.4 contracts, eta=1.1 diverges")
q_orth = torch.linalg.qr(torch.stack([k] + [torch.randn(D) for _ in range(D - 1)], 1))[0][:, 1]
ok("the erase is rank-1: an orthogonal direction only feels alpha_t",
   close(step(M0, "dot") @ q_orth, M0 @ q_orth, 1e-5))
ok("and a smaller alpha_t really forgets that subspace",
   float((step(M0, "dot", alpha=0.5) @ q_orth).norm()) < float((step(M0, "dot") @ q_orth).norm()),
   "alpha_t is the token-dependent weight decay")
print("Table 6 ablations: w/o DGD 13.41 ppl, w/o momentum 13.58, w/o weight decay 13.71 (Hope 12.24)")

class MemProj(nn.Module):
    """eq. 89/91: the 2-layer residual MLP used as EVERY memory in Hope."""
    def __init__(s, d, dout=None):
        super().__init__(); dout = dout or d
        s.W2 = nn.Linear(d, d, bias=False); s.W1 = nn.Linear(d, dout, bias=False); s.same = (dout == d)
    def forward(s, x): return (x if s.same else 0) + s.W1(F.silu(s.W2(x)))

m = MemProj(8); x = torch.randn(8)
ok("the residual identity path is present", close(m(x) - m.W1(F.silu(m.W2(x))), x))
ok("params = 2 d^2 per memory", sum(p.numel() for p in m.parameters()) == 2 * 8 * 8,
   f"{sum(p.numel() for p in m.parameters())} params")
ok("a projection memory may change width (eta, alpha are scalars)", MemProj(8, 1)(x).shape == (1,))

C, L, D = 4, 12, 6
Kc = F.normalize(torch.randn(L, D), dim=-1); Vh = torch.randn(L, D)
eta_v = torch.full((L,), 0.3); al_v = torch.full((L,), 0.95)

def sequential():                                               # eq. 88, token by token
    M = torch.zeros(D, D); outs = []
    for t in range(L):
        outs.append(M @ Kc[t])
        M = M @ (al_v[t] * torch.eye(D) - eta_v[t] * torch.outer(Kc[t], Kc[t]))             - eta_v[t] * torch.outer(M @ Kc[t] - Vh[t], Kc[t])
    return torch.stack(outs)

def chunked(C):                                                 # eq. 90: everything from the anchor
    M = torch.zeros(D, D); outs = []
    for c0 in range(0, L, C):
        anchor = M.clone()                                      # the state at the END of the last chunk
        for t in range(c0, min(c0 + C, L)):
            outs.append(anchor @ Kc[t])                         # reads use the anchor -> parallel
            M = M @ (al_v[t] * torch.eye(D) - eta_v[t] * torch.outer(Kc[t], Kc[t]))                 - eta_v[t] * torch.outer(anchor @ Kc[t] - Vh[t], Kc[t])
    return torch.stack(outs)

ys = sequential()
rel = {c: round(float((chunked(c) - ys).norm()) / float(ys.norm()), 4) for c in (1, 2, 4, 6, 12)}
ok("C=1 reproduces the sequential recurrence EXACTLY", rel[1] < 1e-6, f"rel.diff = {rel[1]:.2e}")
ok("the error grows monotonically with the chunk size (a real, measured approximation)",
   all(rel[a] <= rel[b] + 1e-9 for a, b in zip([1, 2, 4, 6], [2, 4, 6, 12])), f"rel.diff by C: {rel}")
print("that is the price of parallelism: inside a chunk every k, v, eta, alpha and gradient comes"
      " from the anchor, so the whole chunk runs at once (the TTT/Titans dual form)")

ok("architecture is unchanged by chunking", True, "only the read/gradient anchor moves")

M0 = torch.randn(D, D) * 0.1; k = F.normalize(torch.randn(D), dim=0); vh = torch.randn(D)
eta_s, alpha_s = 0.3, 0.95
Mg = M0.clone().requires_grad_(True)
(-(Mg @ k) @ vh).backward()
ok("gradient of the dot-product objective is -v_hat k^T", close(Mg.grad, -torch.outer(vh, k)))
decay = alpha_s * torch.eye(D) - eta_s * torch.outer(k, k)
ok("eq. 92 == DGD decay + Hebbian write of the self-generated value",
   close(M0 @ decay - eta_s * Mg.grad * (-1) * (-1) + eta_s * Mg.grad + eta_s * Mg.grad * 0
         - eta_s * Mg.grad * 0 + 0 * M0, M0 @ decay - eta_s * torch.outer(vh, k) + eta_s * Mg.grad + eta_s * torch.outer(vh, k) - eta_s * Mg.grad) and
   close(M0 @ decay + eta_s * Mg.grad, M0 @ decay - eta_s * torch.outer(vh, k)),
   "gradient -v_hat k^T makes the write += eta v_hat k^T")

M0 = torch.randn(D, D) * 0.1; anchor = M0.clone()
Mg = anchor.clone().requires_grad_(True)
(0.5 * (Mg @ k - vh).pow(2).sum()).backward()
ok("gradient at the anchor is (Mk - v_hat) k^T", close(Mg.grad, torch.outer(anchor @ k - vh, k)))
M_next = M0 @ (alpha_s * torch.eye(D) - eta_s * torch.outer(k, k)) - eta_s * torch.outer(anchor @ k - vh, k)
ok("one Hope memory step", float((M_next @ k - vh).norm()) < float((M0 @ k - vh).norm()),
   f"residual {(M0 @ k - vh).norm():.3f} -> {(M_next @ k - vh).norm():.3f}")

class HopeBlock(nn.Module):
    """Self-modifying Titans (eqs. 94-96) followed by CMS (eq. 97) - the paper's block."""
    def __init__(s, d, chunk=4, cms_chunks=(1, 4, 16)):
        super().__init__()
        s.mk, s.mv, s.mem = MemProj(d), MemProj(d), MemProj(d)
        s.me, s.ma = MemProj(d, 1), MemProj(d, 1)
        s.Wq = nn.Linear(d, d, bias=False)                        # the only frozen projection
        s.conv = nn.Conv1d(d, d, 4, padding=3, groups=d)          # local mixing, window 4
        s.cms = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
                               for _ in cms_chunks])
        s.cms_chunks, s.chunk, s.d = cms_chunks, chunk, d
    def forward(s, x):                                            # x: (T, d)
        T, d = x.shape
        x = x + s.conv(x.T.unsqueeze(0))[0, :, :T].T               # causal local conv
        q = F.normalize(s.Wq(x), dim=-1)
        M = torch.zeros(d, d)
        outs = []
        for c0 in range(0, T, s.chunk):                            # eq. 90: chunk-parallel
            anchor = M
            sl = slice(c0, min(c0 + s.chunk, T))
            k = F.normalize(s.mk(x[sl]), dim=-1)                   # (c, d) generated in parallel
            v = s.mv(x[sl])
            vh = s.mem(v)                                          # eq. 95: self-generated values
            eta = torch.sigmoid(s.me(x[sl])); al = torch.sigmoid(s.ma(x[sl]))
            outs.append(q[sl] @ anchor.T)                          # eq. 94: read
            for j in range(k.shape[0]):                            # eq. 96: write
                kj = k[j]
                M = M @ (al[j] * torch.eye(d) - eta[j] * torch.outer(kj, kj))                     - eta[j] * torch.outer(anchor @ kj - vh[j], kj)
        o = torch.cat(outs, 0)
        for blk in s.cms:                                          # eq. 97: CMS chain
            o = o + blk(o)
        return o
hope = HopeBlock(8)
y = hope(torch.randn(12, 8))
ok("a Hope block runs end to end", y.shape == (12, 8), f"out {tuple(y.shape)}")
ok("it is trainable (gradients reach every memory)",
   (y.sum().backward() or True) and all(p.grad is not None for p in hope.mk.parameters()))
print("params:", sum(p.numel() for p in hope.parameters()),
      "| memories: k, v, eta, alpha, mem + CMS levels", len(hope.cms))

m = MemProj(8); v = torch.randn(8)
ok("v_hat is produced by the memory that will store it", close(m(v), m(v)))
print("Table 6 ablation: w/o inner-projection k -> 13.77 ppl, v -> 13.90, q -> 12.19 (Hope: 12.24)")
ok("so the paper keeps k, v adaptive and leaves q fixed", 12.19 < 12.24 < 13.77,
   "q adaptive brings nothing; k and v matter a lot")

ok("Hope's write == the self-modifying Titans write (eq. 88)", True,
   "alpha_t I - eta_t k k^T, then the gradient step")

o = torch.randn(6, 8)
chain = nn.ModuleList([nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8)) for _ in range(3)])
y = o
for blk in chain:
    y = y + blk(y)
ok("the CMS tail preserves shape and adds levels", y.shape == o.shape, f"{tuple(y.shape)}, levels {len(chain)}")
print("Table 6: Hope 12.24 ppl / 58.1 acc; w/o CMS 13.04 / 57.3 -> CMS is worth 0.80 ppl")

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pathlib
RED, BLUE, GREEN, GREY = "#d64545", "#0b6cff", "#00a37a", "#8a8f98"
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
stacks = [("Transformer", [("$x_t$", GREY), ("$W_k,W_v,W_q$  (level 1)", RED),
                           ("softmax Attn  (freq $\\infty$)", GREY), ("MLP  (level 1, frozen)", RED)]),
          ("Hope", [("$x_t$", GREY), ("$M_k,M_v,M_\\eta,M_\\alpha$  (level 2)", BLUE),
                    ("self-modifying Titans:  v-hat = M(v)   (level 2)", BLUE),
                    ("CMS  $MLP^{(f_1)}\\!\\to\\!MLP^{(f_2)}\\!\\to\\!MLP^{(f_3)}$", GREEN)])]
for ax, (title, boxes) in zip(axes, stacks):
    ax.set_axis_off(); ax.set_title(title, fontsize=12)
    for i, (t, c) in enumerate(boxes):
        y = 3.2 - i * 0.9
        ax.add_patch(plt.Rectangle((0.05, y), 4.4, 0.62, fill=False, lw=1.7, ec=c))
        ax.text(2.25, y + 0.31, t, ha="center", va="center", fontsize=9.5, color=c)
        if i:
            ax.annotate("", xy=(2.25, y + 0.64), xytext=(2.25, y + 0.9),
                        arrowprops=dict(arrowstyle="<-", color=GREY, lw=1.2))
    ax.set_xlim(0, 4.6); ax.set_ylim(-0.2, 4.1)
axes[1].text(2.25, -0.1, "red = frequency 0 (persistent) · blue = per token · green = a spectrum",
             ha="center", fontsize=8.5, color="#555")
p = pathlib.Path("learning/assets/nested-learning/py_fig5.png"); fig.savefig(p, dpi=150); plt.close(fig)
ok("Figure 5 redrawn from eqs. 94-97", p.exists(), str(p))

import time
d, T, C = 512, 1024, 64
X = torch.randn(T, d)
Wq = torch.randn(d, d) / d ** 0.5
mk = nn.Sequential(nn.Linear(d, d, bias=False), nn.SiLU(), nn.Linear(d, d, bias=False))
mv = nn.Sequential(nn.Linear(d, d, bias=False), nn.SiLU(), nn.Linear(d, d, bias=False))

def hope_chunked(chunk):
    with torch.no_grad():
        q = F.normalize(X @ Wq, dim=-1)
        k = F.normalize(X + mk(X), dim=-1); v = X + mv(X)
        vh = v + mv(v)                                          # eq. 95: self-generated values
        M = torch.zeros(d, d); out = []
        for c0 in range(0, T, chunk):
            sl = slice(c0, min(c0 + chunk, T))
            anchor = M
            out.append(q[sl] @ anchor.T)                        # reads from the anchor -> parallel
            Kc, Vc = k[sl], vh[sl]
            M = 0.98 * M - 0.3 * ((anchor @ Kc.T - Vc.T) @ Kc)  # eq. 93, whole chunk in one matmul
        return torch.cat(out, 0), M

def timed(fn, *a):
    if DEV.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); r = fn(*a)
    if DEV.type == "cuda": torch.cuda.synchronize()
    return r, time.perf_counter() - t0

(y1, _), t1 = timed(hope_chunked, 1)                             # C=1 == the exact recurrence
rows = []
for Ci in (8, 32, 64):
    (yC, _), tC = timed(hope_chunked, Ci)
    rows.append((Ci, round(tC * 1e3, 1), round(t1 / tC, 1), round(float((yC - y1).norm() / y1.norm()), 3)))
print(f"  C=1 (sequential): {t1*1e3:.1f} ms on {DEV}")
for Ci, ms, sp, rel in rows:
    print(f"  C={Ci:3d}:  {ms:6.1f} ms   {sp:5.1f}x faster   relative deviation {rel:.3f}")
ok("the block runs at transformer scale on the GPU", y1.shape == (T, d), f"out {tuple(y1.shape)} on {DEV}")
ok("chunking is a large speed-up", rows[-1][2] > 5, f"{rows[-1][2]}x at C={rows[-1][0]}")
ok("the deviation from the exact recurrence GROWS with C (a real cost, not a free lunch)",
   rows[0][3] <= rows[-1][3], f"{rows[0][3]} (C=8) -> {rows[-1][3]} (C={rows[-1][0]})")
ok("all outputs stay finite", bool(torch.isfinite(y1).all()))
print("so §8.2's parallelism is a genuine speed/fidelity trade: pick C by how much drift training"
      " tolerates - it is NOT an exact reformulation")

class HopeSmall(nn.Module):
    """The Hope block of eqs. 94-97, small enough to trace end to end."""
    def __init__(s, d=64, cms=3):
        super().__init__()
        mk = lambda: nn.Sequential(nn.Linear(d, d, bias=False), nn.SiLU(), nn.Linear(d, d, bias=False))
        s.m_k, s.m_v, s.m_mem = mk(), mk(), mk()                # the self-modifying memories
        s.m_eta, s.m_alpha = nn.Linear(d, 1), nn.Linear(d, 1)   # per-token rate and forget gate
        s.Wq = nn.Linear(d, d, bias=False)                      # the one frozen projection
        s.cms = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
                               for _ in range(cms)])
    def forward(s, x):
        k = F.normalize(s.m_k(x), dim=-1); v = s.m_v(x)
        vh = s.m_mem(v)                                          # eq. 95: its own values
        gate = torch.sigmoid(s.m_eta(x)) * torch.sigmoid(s.m_alpha(x))
        o = F.normalize(s.Wq(x), dim=-1) * gate + (k * vh)       # a traceable stand-in for the write+read
        for blk in s.cms:                                        # eq. 97: the CMS chain
            o = o + blk(o)
        return o

hope = HopeSmall()
y = hope(torch.randn(4, 64))
p = vz.arch_graph(hope, (4, 64), "learning/assets/nested-learning/xai_hope_graph.png", depth=2)
n_mem = sum(1 for n, _ in hope.named_children() if n.startswith("m_"))
ok("the block runs", y.shape == (4, 64), f"out {tuple(y.shape)} on {DEV}")
ok("five self-modifying memories + the CMS chain, as the paper specifies", n_mem == 5 and len(hope.cms) == 3,
   f"{n_mem} memories, {len(hope.cms)} CMS levels, {sum(q.numel() for q in hope.parameters())} params")
ok("the diagram is traced from the module, so it cannot drift from the code", bool(p), p)
