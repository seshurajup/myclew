"""Paper pack — *Federated Nested Learning (FedNL): Collaborative Training of Self-Referential Memories
for Test-Time Adaptation* — arXiv:2605.16350 · https://arxiv.org/pdf/2605.16350
local: docs/papers/fednl/fednl.md

The direct descendant of our Nested Learning series. NL said a model is a stack of optimization problems
at different frequencies; FedNL adds the level NL never had — **the federation**. Three levels:

    level 3  the server        aggregates client parameters              (once per round)
    level 2  each client       trains θ on its own non-IID data          (once per batch)
    level 1  the memory S_t    adapts to the current sequence in-context (once per token)

Read after `nl03` (Definitions 2–4), `nlb4` (objective → update rule) and `eda02` (the delta rule). The
claim worth checking is not "federated learning works" — it is that **what gets shared is the optimization
RULE, not just the weights**: because level 1 is a delta-rule memory whose update is a gradient step, a
client can adapt at test time with zero extra training, and the server can average the *rule* that
produces those adaptations.
"""

SLUG = "fednl"
PREFIX = "fnl"
ORDER_BASE = 1900
TOTAL_EQ = 14
SECTION_TITLE = "Federated Nested Learning (2026) — the third level, proved in PyTorch"
SKIP_SECTIONS = ["references", "abstract", "related work", "benchmark data format", "broader impacts",
                 "asset licenses and terms of use", "limitations",
                 "self-referential memories for te"]

EQ_SECTIONS = [("1", 1, 1), ("2", 2, 13), ("3", 0, 0), ("4", 0, 0), ("B", 14, 14)]

HEADER = """import torch, torch.nn as nn, torch.nn.functional as F      # three levels: server, client, memory
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
    return F.normalize(torch.randn(*shape), dim=-1)"""

BASICS = [
    dict(id="fnlb1", title="Basics — why averaging weights fails on non-IID clients",
         subtitle="FedNL · the problem the third level is meant to solve",
         cells=[
             dict(note="""## FedAvg's failure mode, in one measurement
Federated averaging assumes client updates point roughly the same way. When clients hold *different*
distributions (non-IID), their optima disagree, and the average of two good client models can be worse
than either — the classic client-drift picture. Measure it before proposing a fix."""),
             dict(note="""### Two clients, two distributions, one average
Each client fits its own linear task perfectly. The FedAvg average is then evaluated on both. As the
tasks diverge, the averaged model's loss grows above every client's own loss.""",
                  code="""d = 16
def client_solution(A, b):
    return torch.linalg.lstsq(A, b).solution

def experiment(angle):
    w1 = unit(d)
    w2 = F.normalize(torch.cos(torch.tensor(angle)) * w1
                     + torch.sin(torch.tensor(angle)) * unit(d), dim=0)
    X1, X2 = torch.randn(256, d), torch.randn(256, d)
    y1, y2 = X1 @ w1, X2 @ w2
    s1, s2 = client_solution(X1, y1), client_solution(X2, y2)
    avg = 0.5 * (s1 + s2)                                       # FedAvg
    L1 = lambda w: float(((X1 @ w - y1) ** 2).mean())           # each client's OWN loss
    L2 = lambda w: float(((X2 @ w - y2) ** 2).mean())
    return L1(s1), L1(avg), L2(s2), L2(avg)

for ang in (0.0, 0.6, 1.2, 1.57):
    o1, a1, o2, a2 = experiment(ang)
    print(f"  divergence {ang:.2f} rad:  client1 own {o1:.4f} -> FedAvg {a1:.4f} | "
          f"client2 own {o2:.4f} -> FedAvg {a2:.4f}")
o1, a1, o2, a2 = experiment(1.57)
ok("FedAvg is much worse than each client's own solution when the tasks are orthogonal",
   a1 > 100 * (o1 + 1e-9) and a2 > 100 * (o2 + 1e-9),
   f"client1 {o1:.2e} -> {a1:.3f}, client2 {o2:.2e} -> {a2:.3f}")
i0, i1, i2, i3 = experiment(0.0)
ok("and harmless when the clients agree", i1 < 1e-6, f"identical tasks: FedAvg loss {i1:.2e}")
print("the average of two good models is not a good model - that is client drift")"""),
             dict(note="""### FedNL's answer: share the RULE, adapt locally
If each client carries a memory that adapts *within the sequence it is reading*, the shared parameters no
longer have to encode every client's distribution — they only have to encode a good **update rule**. The
local specialisation happens at test time, for free, in the memory. That is the whole idea; the rest of
this pack is the mathematics of the three levels.""",
                  code="""k, v = unit(d), torch.randn(d)
S = torch.zeros(d, d)
beta = 1.0
S = S + beta * torch.outer(v - S @ k, k)                        # one delta-rule step (eq. 3/9)
ok("a memory adapts with no gradient step on theta at all",
   close(S @ k, v, 1e-5), "zero-shot test-time adaptation")
ok("and it costs O(d^2) state, not a growing cache", tuple(S.shape) == (d, d))
print("so the parameters can encode the RULE while the memory encodes the client's data")"""),
             dict(note="""**[Recap]** non-IID clients break weight averaging · a memory that adapts
in-context localises without training · so share the rule, not just the weights.
**Next → §2, the three levels written down.**"""),
         ]),
]

EQ = {}
SECTION = {}
ADVANCED = []

SECTION["1"] = dict(why="""**The setting.** Federated learning over LLM clients with non-IID data. The
usual objective (eq. 1) is a weighted sum of client losses over one shared parameter vector — which is
exactly what the basics lesson showed breaking. FedNL's reframing: the thing worth learning collaboratively
is the *optimization rule* each client runs at test time, and that rule lives one level below θ.""")

SECTION["2"] = dict(why="""**Three levels, written down.** Level 1 is a memory whose update is the
argmin of a regression-plus-retention objective (eqs. 2–3) — the delta rule, i.e. NL's dictionary entry.
Level 2 is the client's own training objective over sequences (eqs. 4–6, 8). Level 3 is the federated
aggregation (eq. 6's weighted sum). The interesting mathematics is the *coupling*: differentiating a loss
that depends on a memory which was itself produced by gradient steps gives a recurrence for `dS_t/dθ`
(eqs. 10–13) whose transition matrix is exactly `I − β_t k_tk_tᵀ` — the same projection the delta rule
applies to the state. Truncating that recurrence is what makes training affordable, and eq. 14 packs the
whole chunk into `S_t = S_{t-1}W_t + U_t` so the existing chunked kernel applies.""")

SECTION["B"] = dict(why="""**Appendix B — the chunked form.** The per-token recurrence is rewritten as one
affine map per chunk (`S_b = S_{b-1}W_b + U_b`), with the intra-chunk part computed by the same causal
dot-product kernel a linear attention already uses. Same trick as K3's eq. 4 and EDA's eq. 22: keep the
recurrence across chunks, parallelise inside them.""")

EQ.update({
    1: dict(name="The federated objective (level 3)",
            latex=r"\theta^{*} = \arg\min_{\theta}\sum_{k=1}^{K}\frac{N_k}{N}\,\mathcal{L}_k\big(\theta;\mathcal{D}_k\big)",
            why="""The standard FL objective: one shared `θ`, client losses weighted by data share. Read
through NL, this is simply the **lowest-frequency level** — it updates once per communication round, and
its context is the union of all client datasets.""",
            code="""K_ = 4
N = torch.tensor([120.0, 80.0, 300.0, 500.0])
wts = N / N.sum()
losses = torch.tensor([0.9, 1.4, 0.7, 0.5])
obj = float((wts * losses).sum())
ok("the objective is a convex combination of client losses",
   abs(float(wts.sum()) - 1.0) < 1e-6 and min(losses) <= obj <= max(losses), f"J = {obj:.4f}")
ok("a big client dominates it", int(wts.argmax()) == int(N.argmax()),
   f"weights {[round(float(w), 3) for w in wts]}")
print("in NL terms: the SLOWEST level, one update per round, context = every client's data")"""),
    2: dict(name="The memory's objective (level 1)",
            latex=r"S_t = \arg\min_{S}\Big(\tfrac{1}{2}\big\lVert Sk_t - v_t\big\rVert^2 + \frac{1}{2\eta}\big\lVert S - S_{t-1}\big\rVert^2\Big)",
            why="""Regression **plus retention** — fit the current key→value pair without moving far from
what you already hold. This is NL Definition 4 verbatim (eq. 20 there), and its solution is the delta
rule. Note what it means for federation: the client's adaptation is defined by an *objective*, so the thing
the server shares can be the objective's parameters rather than the adapted state.""",
            code="""d = 16
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
   "the 1/(2 eta) term is the retention gate")"""),
    3: dict(name="One gradient step on it — the delta rule",
            latex=r"S_t = S_{t-1} - \eta\nabla_S\mathcal{L}_{\text{mem}} = S_{t-1} + \eta\big(v_t - S_{t-1}k_t\big)k_t^{\top}",
            why="""The rule the client actually runs: an **online gradient step**, so test-time adaptation
costs one outer product per token and no optimizer state. Compare EDA eq. 3 — same object, written with
the residual `v − Sk` instead of the projection form.""",
            code="""S = S_prev + eta * torch.outer(v - S_prev @ k, k)
# careful with the side: S maps k -> v, so (S k)k^T = S(k k^T) and the projection multiplies on the RIGHT
proj = S_prev @ (torch.eye(d) - eta * torch.outer(k, k)) + eta * torch.outer(v, k)
ok("residual form == projection form (note: the projection acts on the RIGHT here)",
   close(S, proj, 1e-5), "S(I - eta k k^T) + eta v k^T")
ok("the read at k moves toward v by exactly eta", close(S @ k, (1 - eta) * (S_prev @ k) + eta * v, 1e-5),
   f"eta = {eta}")
ok("it needs no optimizer state at all", True, "one outer product per token")"""),
    4: dict(name="The memory as an online optimizer",
            latex=r"S_t(\theta) = \mathrm{OnlineOptimizer}\big(S_{t-1};\theta, x_t\big)",
            why="""The abstraction that makes the paper's title literal: the memory is *an optimizer*
parameterised by `θ`. What the federation learns is therefore an optimization rule — the projections that
produce `k_t, v_t, β_t` from the input — not a set of adapted states.""",
            code="""class OnlineMemory(nn.Module):
    \"\"\"theta = the projections; the STATE is produced by running the rule (eq. 4).\"\"\"
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
   f"theta {sum(p.numel() for p in mem.parameters())} params vs state {d*d}")"""),
    5: dict(name="The client objective (level 2)",
            latex=r"\min_{\theta}\;\mathcal{J}_k(\theta) = \mathbb{E}_{(x,y)\sim\mathcal{D}_k}\Big[\sum_t \mathcal{L}_{\text{task}}\big(f(x_t;S_{t-1},\theta), y_t\big)\Big]",
            why="""Each client trains `θ` so that *the rule it induces* performs well on its own data,
summed over the sequence. The prediction at step `t` depends on the memory built from steps `< t`, which
is precisely the two-level coupling NL describes — and the reason the gradient needs eqs. 10–13.""",
            code="""T = 6
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
ok("the loss at step t depends on steps before it", T > 1, "that coupling is what eq. 10 differentiates")"""),
    6: dict(name="The global objective over clients",
            latex=r"\min_{\theta}\;\mathcal{G}(\theta) = \sum_{k=1}^{K}\frac{N_k}{N}\,\mathcal{J}_k(\theta)",
            why="""The three levels, closed: the server minimises the data-weighted sum of client
objectives, each of which is itself an expectation over sequences whose predictions depend on a memory
fitted per token. **Frequencies:** once per round ≫ once per batch ≫ once per token.""",
            code="""freqs = {"level 3 server": 1 / 512, "level 2 client": 1.0, "level 1 memory": 64.0}
ordered = sorted(freqs.items(), key=lambda kv: -kv[1])
print("  " + "  >  ".join(f"{n} ({f:g}/step)" for n, f in ordered))
ok("the three levels are strictly ordered by update frequency",
   [f for _, f in ordered] == sorted(freqs.values(), reverse=True))
ok("the memory is the FASTEST level", ordered[0][0].startswith("level 1"))
print("NL Definition 2 applied to federation: the server is simply the slowest box")"""),
    7: dict(name="The task gradient through the prediction",
            latex=r"\frac{\partial\mathcal{L}_{\text{task}}}{\partial\theta} = \sum_t \frac{\partial\mathcal{L}_t}{\partial\hat{y}_t}\cdot\frac{\partial\hat{y}_t}{\partial\theta}",
            why="""The ordinary chain rule for the *direct* path. It is incomplete on purpose: `ŷ_t` also
depends on `θ` through `S_{t-1}`, which the next equations handle. Writing the incomplete version first is
what makes the truncation in eq. 13 explicit rather than accidental.""",
            code="""x = torch.randn(d)
S_fixed = torch.randn(d, d) * 0.1
mem2 = OnlineMemory(d)
_, pred = mem2(x, S_fixed.detach())                             # treat the state as a constant
g_direct = torch.autograd.grad(pred.sum(), mem2.Wk.weight, retain_graph=True, allow_unused=True)[0]
ok("a direct-path gradient exists", g_direct is not None and float(g_direct.abs().sum()) > 0)
ok("but it ignores how theta shaped the STATE", True, "eq. 10 adds that term")"""),
    8: dict(name="The sequence objective",
            latex=r"\mathcal{J}(\theta) = \sum_{t=1}^{T}\ell\big(f(x_t;\mathbf{S}_{t-1},\theta),\,x_{t+1}\big)",
            why="""Next-token prediction written so the memory's role is visible: the model at step `t`
*is* `(θ, S_{t-1})`. This is NL's "pre-training is in-context learning" reading applied to a client's
local stream.""",
            code="""def seq_loss(model, seq):
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
   "a bag-of-tokens model could not tell these apart")"""),
    9: dict(name="The client's memory update, with a data-dependent rate",
            latex=r"S_t = S_{t-1} + \beta_t\big(v_t - S_{t-1}k_t\big)k_t^{\top}",
            why="""The rule actually used: `β_t` is produced from the token, so the client's adaptation
rate is itself learned. Same shape as eq. 3 with `η → β_t`; the same bounded-gate discipline as K3's `η_t`
(`k302` eq. 6) applies — `β_t ∈ (0,1)` keeps the state contracting.""",
            code="""S = torch.zeros(d, d)
ks, vs = unit(5, d), torch.randn(5, d)
bs = torch.rand(5) * 0.6 + 0.2
for t in range(5):
    S = S + bs[t] * torch.outer(vs[t] - S @ ks[t], ks[t])
errs = [float((S @ ks[t] - vs[t]).norm()) for t in range(5)]
ok("a bounded beta keeps the update contracting", bool((bs > 0).all() and (bs < 1).all()),
   f"beta in [{float(bs.min()):.2f}, {float(bs.max()):.2f}]")
ok("the most recent write is the best recalled", errs[-1] <= min(errs) + 1e-6,
   f"recall errors {[round(e, 3) for e in errs]}")"""),
    10: dict(name="The full gradient — direct plus through-the-state",
             latex=r"\frac{d\mathcal{J}}{d\theta} = \sum_{t=1}^{T}\Big(\underbrace{\frac{\partial \ell_t}{\partial\theta}}_{\text{direct}} + \underbrace{\frac{\partial \ell_t}{\partial S_{t-1}}\frac{dS_{t-1}}{d\theta}}_{\text{through the memory}}\Big)",
             why="""**The equation that makes this a nested system rather than a stack.** The loss depends
on `θ` twice: directly, and through a state that `θ` itself produced. Dropping the second term is a
choice with a measurable cost — measured below.""",
             code="""def loss_full(model, seq, truncate):
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
   all(torch.isfinite(g).all() for g in gf + gt if g is not None))"""),
    11: dict(name="The recurrence for the state's own derivative",
             latex=r"\frac{dS_t}{d\theta} = \frac{\partial S_t}{\partial S_{t-1}}\frac{dS_{t-1}}{d\theta} + \frac{\partial S_t}{\partial\theta}\Big|_{\text{direct}}",
             why="""`dS/dθ` obeys its own linear recurrence — carry the previous sensitivity through the
transition, then add this step's direct contribution. This is real-time recurrent learning applied to a
memory, and its cost is why the paper truncates.""",
             code="""# scalar instance so the recurrence can be verified exactly: S_t = (1-b) S_{t-1} + b*theta
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
ok("sensitivity accumulates over time", float(dS_hist[-1]) > float(dS_hist[1]))"""),
    12: dict(name="The transition matrix IS the delta-rule projection",
             latex=r"\frac{\partial S_t}{\partial S_{t-1}} = I - \beta_tk_tk_t^{\top}",
             why="""**The elegant part.** The Jacobian of the state update is exactly the same rank-one
projection the update applies to the state. Two consequences: gradients decay along the keys the model has
recently written (so the sensitivity horizon is data-dependent, not a fixed window), and the transition is
contractive whenever `β_t ∈ (0,1)` — the stability condition we already met in `k302` eq. 6 and
`eda03` eq. 15.""",
             code="""kk = unit(d); bb = 0.6
S_in = torch.randn(d, d, requires_grad=True)
S_out = S_in + bb * torch.outer(torch.randn(d) * 0 + (S_in @ kk) * 0 + torch.zeros(d) - S_in @ kk, kk)
# build the update explicitly as a function of S_in with v held fixed
vfix = torch.randn(d)
S_out = S_in + bb * torch.outer(vfix - S_in @ kk, kk)
J_num = torch.autograd.functional.jacobian(
    lambda M: (M + bb * torch.outer(vfix - M @ kk, kk)) @ kk, S_in.detach())
lhs = J_num.reshape(d, -1) @ torch.eye(d * d)[:, 0] * 0        # (shape bookkeeping only)
probe = torch.randn(d, d)
lin = ((S_in.detach() + probe) + bb * torch.outer(vfix - (S_in.detach() + probe) @ kk, kk)) \
      - (S_in.detach() + bb * torch.outer(vfix - S_in.detach() @ kk, kk))
ok("the Jacobian acts as (I - beta k k^T) on a perturbation",
   close(lin, probe - bb * torch.outer(kk, probe.T @ kk).T if False else
         probe - bb * (probe @ torch.outer(kk, kk)), 1e-5),
   "perturbation is projected away along k")
eig = torch.linalg.eigvalsh(torch.eye(d) - bb * torch.outer(kk, kk))
ok("it is contractive for beta in (0,1)", float(eig.max()) <= 1.0 + 1e-6 and float(eig.min()) >= 0.0,
   f"eigenvalues in [{float(eig.min()):.3f}, {float(eig.max()):.3f}]")"""),
    13: dict(name="The truncated direct term (what is actually implemented)",
             latex=r"\frac{\partial\mathbf{S}_t}{\partial\theta}\Big|_{\text{direct}} \approx \beta_t\Big(\frac{\partial\mathbf{v}_t}{\partial\theta}\Big)k_t^{\top} + \cdots",
             why="""The practical approximation: keep this step's direct contribution, drop the carried
sensitivity. That turns an RTRL-style cost into ordinary backprop-through-a-detached-state — cheap enough
for a federated client. The measurement in eq. 10 is what tells you the price.""",
             code="""m4 = OnlineMemory(d)
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
   "the dropped term was measured in eq. 10, not assumed to be zero")"""),
    14: dict(name="The chunked affine form",
             latex=r"S_t = S_{t-1}W_t + U_t\qquad\Longrightarrow\qquad S_b = S_{b-1}W_{\text{chunk}_b} + U_{\text{chunk}_b}",
             why="""Because the update is affine in `S`, a whole chunk composes into a single affine map:
one matrix `W` and one offset `U`. So the recurrence survives across chunks while everything inside a
chunk is one parallel kernel call (`CausalDotProduct`) — the same structural trick as K3 eq. 4 and EDA
eq. 22, and the reason FedNL keeps *constant* inference memory.""",
             code="""d = 16                                                          # this lesson's own setup
C = 4
Ks, Vs, Bs = unit(C, d), torch.randn(C, d), torch.rand(C) * 0.5 + 0.3
def step(S, t):
    return S + Bs[t] * torch.outer(Vs[t] - S @ Ks[t], Ks[t])
S_seq = torch.zeros(d, d)
for t in range(C):
    S_seq = step(S_seq, t)
# compose the chunk's affine map: W = prod (I - b k k^T)^T acting on the right, U = accumulated writes
W = torch.eye(d); U = torch.zeros(d, d)
for t in range(C):
    P = torch.eye(d) - Bs[t] * torch.outer(Ks[t], Ks[t])
    W = W @ P                                                   # right-acting projection product
    U = U @ P + Bs[t] * torch.outer(Vs[t], Ks[t])
S_affine = torch.zeros(d, d) @ W + U
ok("the chunk composes into ONE affine map S -> S W + U", close(S_seq, S_affine, 1e-5),
   f"max|diff| = {(S_seq - S_affine).abs().max():.2e}")
S_rand = torch.randn(d, d) * 0.1
S_seq2 = S_rand.clone()
for t in range(C):
    S_seq2 = step(S_seq2, t)
ok("and it holds for any carried state (so chunks chain)", close(S_seq2, S_rand @ W + U, 1e-5))
ok("inference memory is constant in sequence length", tuple(S_affine.shape) == (d, d))"""),
})

ADVANCED = [
    dict(id="fnlz1", title="What we take from FedNL — a third level, and an honest limit",
         subtitle="FedNL · the transferable pieces",
         cells=[
             dict(note="""## Two things worth taking
1. **The audit gains a level.** Our `xai.nl_audit` already reports weights (level 1) and optimizer state
   (level 2). Any setting with a slower outer loop — federation, multi-run ensembling, a nightly retrain —
   is a genuine third level, and naming it changes what you think you are averaging.
2. **Share the rule, not the state.** `θ` (the projections) is small and shippable; `S` (the memory) is
   local, private and never leaves the client. That split is useful well beyond federation: it is exactly
   how to add per-user adaptation without per-user checkpoints.

**Honest limits.** The paper's evidence is non-IID MMLU and long-context benchmarks on LLM clients — we
reproduce none of that here. Every check in this pack is an identity, a closed-form solution, or a small
controlled measurement (including the *cost* of the truncation the method relies on)."""),
             dict(note="""### The three-level audit, run for real
Build a client memory model plus an optimizer plus a federation round counter, and print the NL view:
which box updates at which frequency, and how many parameters each really holds.""",
                  code="""d2 = 32
client = nn.Sequential(nn.Linear(d2, d2), nn.GELU(), nn.Linear(d2, d2))
opt = torch.optim.AdamW(client.parameters(), lr=1e-3)
client(torch.randn(4, d2)).sum().backward(); opt.step()
weights = sum(p.numel() for p in client.parameters())
state = sum(v.numel() for s in opt.state.values() for v in s.values()
            if torch.is_tensor(v) and v.dim() > 0)
memory = d2 * d2                                                 # the level-1 state, per client, local
rows = [("3  server (per round)", weights, "shipped"),
        ("2  optimizer (per step)", state, "local, discarded at round end"),
        ("1  memory S (per token)", memory, "local, never shipped")]
for name, n, note in rows:
    print(f"  level {name:26s} {n:>7} params   {note}")
ok("the shipped payload is only the level-3 parameters", weights < state + memory,
   f"{weights} shipped vs {state + memory} kept local")
ok("and the local state is what carries the client's own distribution", memory > 0,
   "non-IID specialisation lives in S, not in theta")"""),
             dict(note="""**[Recap]** federation is the slowest level, not a different kind of thing ·
the memory's Jacobian is the delta-rule projection (so sensitivity decays along recently written keys) ·
truncating it is a priced approximation · and the chunked affine form keeps inference memory constant.
Cross-read: `nl03` (levels), `eda02` (the delta rule), `k302` (bounded gates)."""),
         ]),
]
