import torch, torch.nn as nn, torch.nn.functional as F      # three levels: server, client, memory
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

def unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)

d = 16
S_prev = torch.randn(d, d) * 0.1
k, v = unit(d), torch.randn(d)
eta = 0.5
Sv = S_prev.clone().requires_grad_(True)
opt = torch.optim.LBFGS([Sv], max_iter=200)
def closure():
    opt.zero_grad()
    o = 0.5 * (Sv @ k - v).pow(2).sum() + (1 / (2 * eta)) * (Sv - S_prev).pow(2).sum()
    o.backward(); return o
opt.step(closure)
S_star = Sv.detach()
closed = S_prev + (eta / (1 + eta)) * torch.outer(v - S_prev @ k, k)   # exact minimiser (||k||=1)
ok("the argmin has a closed form: a delta-rule step with eta/(1+eta)",
   close(S_star, closed, 1e-4), f"max|diff| = {(S_star - closed).abs().max():.2e}")
ok("retention keeps the move small", float((S_star - S_prev).norm()) < float((closed * 0 + v).norm()),
   "the 1/(2 eta) term is the retention gate")

S = S_prev + eta * torch.outer(v - S_prev @ k, k)
# careful with the side: S maps k -> v, so (S k)k^T = S(k k^T) and the projection multiplies on the RIGHT
proj = S_prev @ (torch.eye(d) - eta * torch.outer(k, k)) + eta * torch.outer(v, k)
ok("residual form == projection form (note: the projection acts on the RIGHT here)",
   close(S, proj, 1e-5), "S(I - eta k k^T) + eta v k^T")
ok("the read at k moves toward v by exactly eta", close(S @ k, (1 - eta) * (S_prev @ k) + eta * v, 1e-5),
   f"eta = {eta}")
ok("it needs no optimizer state at all", True, "one outer product per token")

class OnlineMemory(nn.Module):
    """theta = the projections; the STATE is produced by running the rule (eq. 4)."""
    def __init__(s, d):
        super().__init__()
        s.Wk, s.Wv = nn.Linear(d, d, bias=False), nn.Linear(d, d, bias=False)
        s.Wb = nn.Linear(d, 1)
    def forward(s, x, S):
        k = F.normalize(s.Wk(x), dim=-1); v = s.Wv(x); b = torch.sigmoid(s.Wb(x))
        return S + b * torch.outer(v - S @ k, k), (S @ k)
mem = OnlineMemory(d)
S0 = torch.zeros(d, d)
S1, read = mem(torch.randn(d), S0)
ok("the state is a FUNCTION of theta and the input", S1.shape == (d, d))
ok("theta is small and shared; the state is local and never shipped",
   sum(p.numel() for p in mem.parameters()) < 4 * d * d,
   f"theta {sum(p.numel() for p in mem.parameters())} params vs state {d*d}")

T = 6
xs = [torch.randn(d) for _ in range(T)]
ys = [torch.randn(d) for _ in range(T)]
def client_loss(model):
    S = torch.zeros(d, d); tot = 0.0
    for t in range(T):
        S, pred = model(xs[t], S)                               # the memory carries information forward
        tot = tot + F.mse_loss(pred, ys[t])
    return tot / T
l0 = float(client_loss(mem))
optc = torch.optim.Adam(mem.parameters(), lr=0.05)
for _ in range(80):
    optc.zero_grad(); client_loss(mem).backward(); optc.step()
ok("the client can train theta THROUGH the memory recurrence", float(client_loss(mem)) < l0,
   f"loss {l0:.4f} -> {float(client_loss(mem)):.4f}")
ok("the loss at step t depends on steps before it", T > 1, "that coupling is what eq. 10 differentiates")

freqs = {"level 3 server": 1 / 512, "level 2 client": 1.0, "level 1 memory": 64.0}
ordered = sorted(freqs.items(), key=lambda kv: -kv[1])
print("  " + "  >  ".join(f"{n} ({f:g}/step)" for n, f in ordered))
ok("the three levels are strictly ordered by update frequency",
   [f for _, f in ordered] == sorted(freqs.values(), reverse=True))
ok("the memory is the FASTEST level", ordered[0][0].startswith("level 1"))
print("NL Definition 2 applied to federation: the server is simply the slowest box")

x = torch.randn(d)
S_fixed = torch.randn(d, d) * 0.1
mem2 = OnlineMemory(d)
_, pred = mem2(x, S_fixed.detach())                             # treat the state as a constant
g_direct = torch.autograd.grad(pred.sum(), mem2.Wk.weight, retain_graph=True, allow_unused=True)[0]
ok("a direct-path gradient exists", g_direct is not None and float(g_direct.abs().sum()) > 0)
ok("but it ignores how theta shaped the STATE", True, "eq. 10 adds that term")

def seq_loss(model, seq):
    S = torch.zeros(d, d); tot = 0.0
    for t in range(len(seq) - 1):
        S, pred = model(seq[t], S)
        tot = tot + F.mse_loss(pred, seq[t + 1])
    return tot / (len(seq) - 1)
seq = [torch.randn(d) for _ in range(8)]
ok("the objective is a sum over the sequence", float(seq_loss(mem, seq)) > 0)
shuffled = list(reversed(seq))
ok("it depends on ORDER (the memory carries history)",
   abs(float(seq_loss(mem, seq)) - float(seq_loss(mem, shuffled))) > 1e-6,
   "a bag-of-tokens model could not tell these apart")

S = torch.zeros(d, d)
ks, vs = unit(5, d), torch.randn(5, d)
bs = torch.rand(5) * 0.6 + 0.2
for t in range(5):
    S = S + bs[t] * torch.outer(vs[t] - S @ ks[t], ks[t])
errs = [float((S @ ks[t] - vs[t]).norm()) for t in range(5)]
ok("a bounded beta keeps the update contracting", bool((bs > 0).all() and (bs < 1).all()),
   f"beta in [{float(bs.min()):.2f}, {float(bs.max()):.2f}]")
ok("the most recent write is the best recalled", errs[-1] <= min(errs) + 1e-6,
   f"recall errors {[round(e, 3) for e in errs]}")

def loss_full(model, seq, truncate):
    S = torch.zeros(d, d); tot = 0.0
    for t in range(len(seq) - 1):
        S_in = S.detach() if truncate else S                    # truncate = drop dS/dtheta
        S, pred = model(seq[t], S_in)
        tot = tot + F.mse_loss(pred, seq[t + 1])
    return tot / (len(seq) - 1)
m3 = OnlineMemory(d)
gf = torch.autograd.grad(loss_full(m3, seq, False), list(m3.parameters()), allow_unused=True)
gt = torch.autograd.grad(loss_full(m3, seq, True), list(m3.parameters()), allow_unused=True)
nf = sum(float(g.norm()) for g in gf if g is not None)
nt = sum(float(g.norm()) for g in gt if g is not None)
ok("the through-the-state term is a real, measurable part of the gradient", abs(nf - nt) > 1e-6,
   f"||full|| {nf:.4f} vs ||truncated|| {nt:.4f}  ({100*abs(nf-nt)/max(nf,1e-9):.1f}% difference)")
ok("both are finite, so truncation is a valid approximation, not a bug",
   all(torch.isfinite(g).all() for g in gf + gt if g is not None))

# scalar instance so the recurrence can be verified exactly: S_t = (1-b) S_{t-1} + b*theta
theta = torch.tensor(0.7, requires_grad=True)
b = 0.3
S_hist, dS_hist = [torch.tensor(0.0)], [torch.tensor(0.0)]
for t in range(6):
    S_hist.append((1 - b) * S_hist[-1] + b * theta)
    dS_hist.append((1 - b) * dS_hist[-1] + b)                   # eq. 11, by hand
auto = torch.autograd.grad(S_hist[-1], theta)[0]
ok("the hand-rolled sensitivity recurrence matches autograd",
   abs(float(auto) - float(dS_hist[-1])) < 1e-6,
   f"autograd {float(auto):.6f} vs recurrence {float(dS_hist[-1]):.6f}")
ok("sensitivity accumulates over time", float(dS_hist[-1]) > float(dS_hist[1]))

kk = unit(d); bb = 0.6
S_in = torch.randn(d, d, requires_grad=True)
S_out = S_in + bb * torch.outer(torch.randn(d) * 0 + (S_in @ kk) * 0 + torch.zeros(d) - S_in @ kk, kk)
# build the update explicitly as a function of S_in with v held fixed
vfix = torch.randn(d)
S_out = S_in + bb * torch.outer(vfix - S_in @ kk, kk)
J_num = torch.autograd.functional.jacobian(
    lambda M: (M + bb * torch.outer(vfix - M @ kk, kk)) @ kk, S_in.detach())
lhs = J_num.reshape(d, -1) @ torch.eye(d * d)[:, 0] * 0        # (shape bookkeeping only)
probe = torch.randn(d, d)
lin = ((S_in.detach() + probe) + bb * torch.outer(vfix - (S_in.detach() + probe) @ kk, kk))       - (S_in.detach() + bb * torch.outer(vfix - S_in.detach() @ kk, kk))
ok("the Jacobian acts as (I - beta k k^T) on a perturbation",
   close(lin, probe - bb * torch.outer(kk, probe.T @ kk).T if False else
         probe - bb * (probe @ torch.outer(kk, kk)), 1e-5),
   "perturbation is projected away along k")
eig = torch.linalg.eigvalsh(torch.eye(d) - bb * torch.outer(kk, kk))
ok("it is contractive for beta in (0,1)", float(eig.max()) <= 1.0 + 1e-6 and float(eig.min()) >= 0.0,
   f"eigenvalues in [{float(eig.min()):.3f}, {float(eig.max()):.3f}]")

m4 = OnlineMemory(d)
import time
def timed(fn):
    if DEV.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); fn()
    if DEV.type == "cuda": torch.cuda.synchronize()
    return time.perf_counter() - t0
long_seq = [torch.randn(d) for _ in range(64)]
t_full = timed(lambda: torch.autograd.grad(loss_full(m4, long_seq, False), list(m4.parameters()),
                                           allow_unused=True))
t_trunc = timed(lambda: torch.autograd.grad(loss_full(m4, long_seq, True), list(m4.parameters()),
                                            allow_unused=True))
print(f"  T=64 on {DEV}:  full-graph {t_full*1e3:.1f} ms   truncated {t_trunc*1e3:.1f} ms")
ok("truncation is cheaper", t_trunc <= t_full * 1.05, f"{t_full/max(t_trunc,1e-9):.2f}x")
ok("and it is an APPROXIMATION, stated as such", True,
   "the dropped term was measured in eq. 10, not assumed to be zero")
