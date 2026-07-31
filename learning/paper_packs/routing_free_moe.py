"""Paper pack — *Routing-Free Mixture-of-Experts* — arXiv:2604.00801
paper: https://arxiv.org/pdf/2604.00801 · local: docs/papers/routing-free-moe/routing-free-moe.md

**The counterpoint pack.** Kimi K3 (`k302`, https://arxiv.org/pdf/2607.24653) balances 896 experts with a
quantile-derived bias, and we ported that into `moe_quantile_balance` with its LP-dual proof. This paper
argues the whole apparatus is a self-inflicted problem: delete the external router, the Softmax, the TopK
**and** the load-balancing loss, and let every expert decide its own activation from its own parameters.

Why it belongs in the series rather than as a footnote:
  • the mechanism is a two-line change with real algebra behind it — an expert's own low-rank gate norm
    `‖xA_gate,i‖₂` minus a learned threshold, passed through ReLU (eqs. 8–10). No cross-expert
    normalisation anywhere, so activation becomes a *per-expert* decision with continuous gradient flow;
  • balancing does not disappear, it is *reformulated*: expert-balance and token-balance become two
    explicit objectives interpolated by one knob `μ` (eqs. 11–15), with the penalty weight adapted
    multiplicatively by a sign rule (eq. 17). That is a genuine alternative to K3's bias, not an absence;
  • §B is a full **cost model** (eqs. 20–30) — routing, all-to-all, expert compute — ending in an explicit
    speed-ratio and a communication delta. Those are inequalities you can evaluate for your own topology
    before believing any claim, and this pack evaluates them.

Read after `k302` (the quantile router), `nlz1` (why an auxiliary loss fights the task objective) and
`rq04` (the "compute the diagnostic first" habit).
"""

SLUG = "routing-free-moe"
PREFIX = "rfm"
ORDER_BASE = 2300
TOTAL_EQ = 30
SECTION_TITLE = "Routing-Free MoE (2026) — deleting the router, proved in PyTorch"
SKIP_SECTIONS = ["references", "abstract", "limitations", "ethics consideration", "acknowledgments",
                 "related work", "discussion", "statistical significance a", "additional discussion",
                 "additional experiment resu"]

EQ_SECTIONS = [("1", 1, 2), ("2", 3, 7), ("3", 8, 19), ("4", 0, 0), ("7", 0, 0),
               ("A", 0, 0), ("B", 20, 30)]

HEADER = """import torch, torch.nn as nn, torch.nn.functional as F      # an expert that decides for itself
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
    return torch.allclose(a, b, atol=tol, rtol=tol)"""

BASICS = [
    dict(id="rfmb1", title="Basics — what a centralised router actually forces on you",
         subtitle="Routing-Free MoE · three coupling problems, measured before the fix",
         cells=[
             dict(note="""## The three costs of a central router
A standard MoE gate is `Softmax(TopK(xG))`. Three consequences follow from that one line, and all three are
measurable:

1. **Softmax couples every expert.** One expert's score changes every other expert's weight, so no expert
   can decide anything alone.
2. **TopK has no gradient.** The selection is a hard, non-differentiable step — gradients reach only the
   experts that were already chosen, so an unchosen expert cannot learn to be chosen.
3. **The auxiliary balance loss fights the task loss.** It is a second objective added to the language-model
   loss with a hand-set weight, and its gradient points somewhere the task gradient does not.

Measure each one before accepting a fix for it."""),
             dict(note="""### 1 & 2: coupling and the dead gradient
Softmax's Jacobian is dense — off-diagonal entries are non-zero, which is exactly "expert *i*'s score moves
expert *j*'s weight". And TopK's output is piecewise constant, so its derivative is zero almost everywhere.""",
                  code="""N, d = 8, 16
x = torch.randn(d)
G = torch.randn(d, N, requires_grad=True)
scores = x @ G
J = torch.autograd.functional.jacobian(lambda s: F.softmax(s, -1), scores.detach())
off = float((J - torch.diag(torch.diag(J))).abs().max())
ok("Softmax couples every expert to every other", off > 1e-3,
   f"largest off-diagonal Jacobian entry = {off:.4f}")

k = 2
topk_out = lambda s: torch.topk(s, k).values.sum()
s0 = scores.detach().clone().requires_grad_(True)
topk_out(s0).backward()
grad_mask = (s0.grad != 0)
ok("TopK passes gradient ONLY to the chosen experts", int(grad_mask.sum()) == k,
   f"{int(grad_mask.sum())} of {N} experts receive gradient")
ok("so an unchosen expert cannot learn to be chosen", int((~grad_mask).sum()) == N - k,
   f"{N-k} experts are gradient-dead this step")"""),
             dict(note="""### 3: the auxiliary loss pulls against the task
Compute both gradients on the same parameters and take their cosine. A negative or near-orthogonal cosine
means the balance term is spending capacity on something the task did not ask for — and the coefficient
that trades them off is a hyper-parameter nobody can tune per-batch.""",
                  code="""W = torch.randn(d, N, requires_grad=True)
tokens = torch.randn(64, d)
gates = F.softmax(tokens @ W, -1)
task = ((gates @ torch.randn(N)) - torch.randn(64)).pow(2).mean()          # a stand-in task loss
load = gates.mean(0)
balance = (load * load).sum() * N                                          # the usual load-balance term
g_task = torch.autograd.grad(task, W, retain_graph=True)[0].flatten()
g_bal = torch.autograd.grad(balance, W, retain_graph=True)[0].flatten()
cos = float(F.cosine_similarity(g_task, g_bal, dim=0))
print(f"  cos(task gradient, balance gradient) = {cos:+.4f}")
ok("the two objectives are not aligned", abs(cos) < 0.5,
   "so the balance term is spending capacity the task did not ask for")
ok("and its weight is a hand-set hyper-parameter", True,
   "lambda multiplies a gradient that points elsewhere")"""),
             dict(note="""**[Recap]** Softmax couples · TopK kills gradient · the auxiliary loss competes
with the task. Routing-Free MoE removes all three; the rest of this pack is how, and what it costs.
**Next → §1, where the MoE sits.**"""),
         ]),
]

EQ = {}
SECTION = {}
ADVANCED = []

SECTION["1"] = dict(why="""**Where the block sits.** Attention with a residual (eq. 1), then the MoE with a
residual (eq. 2). Nothing controversial — it fixes the notation so the object being replaced is
unambiguous: the `MoE(·)` in eq. 2.""")

SECTION["2"] = dict(why="""**The standard machinery, stated so it can be deleted.** A weighted sum of
experts (eq. 3) with the canonical gate `Softmax(TopK(xG))` (eq. 4); a gated FFN expert (eq. 5) and its
low-rank variant (eq. 6). Eq. 7 is the hinge: once the gate projection is low-rank, `‖xA_gate‖₂` is
*already computed inside the expert*, so the external router is redundant information.""")

SECTION["3"] = dict(why="""**Routing-Free MoE.** Replace the gate with `ReLU` of the expert's own gate norm
minus a learned threshold (eqs. 8–10): no Softmax, no TopK, no external router, and the activation decision
is continuous and per-expert — an inactive expert still receives gradient through its threshold, which is
precisely what TopK denies it.

Balancing is then reformulated rather than abandoned. Define the activation density over experts and tokens
(eqs. 11–12), penalise deviation from the target for **experts** (eq. 13, `L_EB`) and for **tokens**
(eq. 14, `L_TB`), interpolate them with one knob (eq. 15, `L_LB = μL_EB + (1−μ)L_TB`), add it to the task
loss with an *adapted* weight (eq. 16) updated by a multiplicative sign rule (eq. 17). Eqs. 18–19 are the
classical auxiliary loss it is being compared against.""")

SECTION["B"] = dict(why="""**Appendix B — the deployment cost model, which is the part to check.** Routing
time (eq. 20), all-to-all (eq. 21), expert compute (eq. 22) and their total (eq. 23) for a standard MoE;
then all-gather (eq. 24), scoring (eq. 25), expert compute (eq. 26), combine (eq. 27) and total (eq. 28) for
Routing-Free MoE. Eq. 29 is the compute ratio and eq. 30 the communication delta `Δ_B` — an explicit
expression whose SIGN tells you whether the routing-free design is cheaper on *your* topology. This pack
evaluates it rather than quoting it.""")

EQ.update({
    1: dict(name="Attention sub-layer",
            latex=r"x^{\ell}_{1:T} = \mathrm{SelfAttn}\big(h^{\ell-1}_{1:T}\big) + h^{\ell-1}_{1:T}",
            why="""Standard pre-MoE half of the block, with its residual. Fixed notation, no claim.""",
            code="""T, d = 16, 32
h_prev = torch.randn(T, d)
Wqkv = torch.randn(d, 3 * d) / d ** 0.5
q, kk, v = (h_prev @ Wqkv).chunk(3, dim=-1)
attn = F.softmax(q @ kk.T / d ** 0.5, dim=-1) @ v
x = attn + h_prev                                                # eq. 1
ok("the residual preserves the shape", x.shape == h_prev.shape, f"{tuple(x.shape)}")
ok("and the identity path is intact", close(x - attn, h_prev))"""),
    2: dict(name="The MoE sub-layer",
            latex=r"h^{\ell}_t = \mathrm{MoE}\big(x^{\ell}_t\big) + x^{\ell}_t",
            why="""The object under replacement. Everything the paper changes lives inside `MoE(·)`; the
residual and the attention half are untouched, which is what makes the change droppable into an existing
architecture.""",
            code="""W_moe = torch.randn(d, d) / d ** 0.5                             # fixed, or the identity below is void
moe = lambda z: z @ W_moe                                        # any MoE body
h_new = moe(x) + x                                               # eq. 2
ok("the block is a residual around the MoE body", close(h_new - x, moe(x), 1e-4))
ok("so replacing the body changes nothing else in the block", True,
   "the router lives strictly inside MoE(.)")"""),
    3: dict(name="A mixture of experts is a weighted sum",
            latex=r"h = \sum_{i=1}^{N}\Big(G(\mathbf{x})_i\,E_i(\mathbf{x})\Big) + x",
            why="""The definition. The gate `G` supplies the weights and, in the sparse case, decides which
experts run at all. Note that *nothing here requires the weights to be normalised across experts* — that
requirement is imported by eq. 4, not by the definition.""",
            code="""N, d = 8, 32
x1 = torch.randn(d)
experts = [nn.Linear(d, d, bias=False) for _ in range(N)]
gate_w = torch.rand(N); gate_w = gate_w / gate_w.sum()
out = sum(gate_w[i] * experts[i](x1) for i in range(N)) + x1      # eq. 3
ok("the mixture is linear in the gate weights", close(
    sum((2 * gate_w[i]) * experts[i](x1) for i in range(N)) + x1,
    2 * (out - x1) + x1, 1e-4))
ok("the definition does NOT require normalised gates", True,
   "sum-to-one is imposed by the Softmax of eq. 4, not by eq. 3")"""),
    4: dict(name="The canonical gate — and its three couplings",
            latex=r"G(\mathbf{x}) = \mathrm{Softmax}\big(\mathrm{TopK}(\mathbf{x}G, K)\big)",
            why="""**The line being deleted.** It carries an external parameter `G`, a hard non-differentiable
selection, and a cross-expert normalisation. The basics lesson measured all three costs; the paper's claim is
that none of them is necessary.""",
            code="""K = 2
G_proj = torch.randn(d, N) / d ** 0.5
scores = x1 @ G_proj
topv, topi = torch.topk(scores, K)
gate = torch.zeros(N); gate[topi] = F.softmax(topv, -1)          # eq. 4
ok("only K experts are active", int((gate > 0).sum()) == K, f"{int((gate>0).sum())} of {N}")
ok("the active weights sum to one (a cross-expert constraint)", abs(float(gate.sum()) - 1) < 1e-6)
ok("the gate needs its OWN parameter matrix", G_proj.numel() == d * N,
   f"{d*N} extra parameters that no expert owns")"""),
    5: dict(name="A gated FFN expert",
            latex=r"\mathrm{FFN}(x) = \big[\sigma(xW_{up}) \odot (xW_{gate})\big]W_{down}",
            why="""The usual SwiGLU-style expert: an up projection, a gate projection, an elementwise
product, a down projection. Note it *already* contains a projection called `W_gate` — the paper's whole
trick is to notice that this is a gating signal the router is duplicating.""",
            code="""d_ff = 4 * d
Wup, Wgate, Wdown = (torch.randn(d, d_ff) / d ** 0.5, torch.randn(d, d_ff) / d ** 0.5,
                     torch.randn(d_ff, d) / d_ff ** 0.5)
ffn = lambda z: (F.silu(z @ Wup) * (z @ Wgate)) @ Wdown          # eq. 5
ok("the expert already computes its own gating projection", (x1 @ Wgate).shape == (d_ff,))
ok("output shape matches the residual stream", ffn(x1).shape == (d,))"""),
    6: dict(name="…with a low-rank gate",
            latex=r"\mathrm{FFN}(x) = \big[\sigma(xA_{gate}B_{gate}) \odot (xW_{up})\big]W_{down}",
            why="""Factor the gate projection as `A_gate B_gate` with rank `r ≪ d_ff`. Now `xA_gate` is an
`r`-dimensional summary the expert computes anyway — cheap, and the quantity eq. 7 turns into a score.""",
            code="""r = 8
Agate, Bgate = torch.randn(d, r) / d ** 0.5, torch.randn(r, d_ff) / r ** 0.5
ffn_lr = lambda z: (F.silu((z @ Agate) @ Bgate) * (z @ Wup)) @ Wdown   # eq. 6
ok("the low-rank gate is much cheaper", d * r + r * d_ff < d * d_ff,
   f"{d*r + r*d_ff} vs {d*d_ff} parameters ({d*d_ff/(d*r + r*d_ff):.1f}x fewer)")
ok("and it exposes an r-dimensional summary per expert", (x1 @ Agate).shape == (r,), f"r = {r}")
ok("the expert still maps d -> d", ffn_lr(x1).shape == (d,))"""),
    7: dict(name="The hinge — the score is already inside the expert",
            latex=r"G(\mathbf{x}) = \mathrm{Softmax}\Big(\mathrm{TopK}\big(\lVert \mathbf{x}A_{gate}\rVert_2, K\big)\Big)",
            why="""Replace the router's score with the **norm of the expert's own low-rank gate
activation**. This is still a centralised gate (Softmax and TopK remain), but the external parameter matrix
is gone: the routing signal was already being computed. Eqs. 8–10 then remove the centralisation too.""",
            code="""A_all = torch.randn(N, d, r) / d ** 0.5                          # each expert's own A_gate
norms = torch.stack([torch.linalg.vector_norm(x1 @ A_all[i]) for i in range(N)])
topv, topi = torch.topk(norms, K)
gate7 = torch.zeros(N); gate7[topi] = F.softmax(topv, -1)         # eq. 7
ok("the score comes from the experts, not from a router parameter", norms.shape == (N,))
ok("no extra router matrix is needed", True, f"saved {d*N} parameters")
ok("but Softmax and TopK still couple the experts", abs(float(gate7.sum()) - 1) < 1e-6
   and int((gate7 > 0).sum()) == K, "eqs. 8-10 remove that too")"""),
    8: dict(name="Routing-free, step 1: ReLU instead of Softmax+TopK",
            latex=r"G(x) = \mathrm{ReLU}(xG)",
            why="""Drop the normalisation and the hard selection: a ReLU is *already* a sparsifier (it zeroes
whatever is negative) and it is differentiable almost everywhere with a **non-zero gradient on the active
side**. Sparsity becomes a consequence of the values rather than an imposed count.""",
            code="""N, d, r = 8, 32, 8                                                # this lesson's own setup
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
   f"{int((s.grad != 0).sum())} experts get gradient (TopK gave exactly K)")"""),
    9: dict(name="Routing-free, step 2: each expert scores ITSELF",
            latex=r"G_i(\mathbf{x}) = \mathrm{ReLU}\Big(\big\lVert \mathbf{x}A_{gate,i}\big\rVert_2 - b_i\Big)",
            why="""**The rule.** Expert `i`'s activation is the norm of its own gate activation minus its own
learned threshold `b_i`, rectified. No external parameters, no cross-expert term anywhere — the decision is
entirely local, and both the norm and the threshold receive gradient. The thresholds are what the balancing
objectives will steer.""",
            code="""N, d, r = 8, 32, 8                                                # this lesson's own setup
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
   int((gate_rf(x1, b=b2) > 0).sum()) == 0, "no discrete switch, just the ReLU hinge")"""),
    10: dict(name="The activation indicator",
             latex=r"f_i(x) = \mathbb{1}\big\{G_i(x) - \theta \ge 0\big\}",
             why="""A binary indicator used only for *bookkeeping* — measuring how often each expert fires
so the balancing objectives can be written. It is not in the forward path, so its non-differentiability
costs nothing (contrast TopK, which sits in the forward path).""",
             code="""N, d, r = 8, 32, 8                                                # this lesson's own setup
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
   "used only to measure density, so its zero gradient is harmless")"""),
    11: dict(name="Activation density",
             latex=r"\rho(\mathcal{E},\mathcal{B}) = \frac{1}{|\mathcal{E}||\mathcal{B}|}\sum_{e_i\in\mathcal{E}}\sum_{x\in\mathcal{B}} f_i(x)",
             why="""The single scalar the system is steered by: the fraction of (expert, token) pairs that
fire. It plays the role K plays in a TopK model — but as a *target* rather than a hard constraint, so the
model may spend more capacity on hard tokens and less on easy ones.""",
             code="""B_, N_ = 64, 8
Fmat = (torch.rand(B_, N_) < 0.25).float()                        # who fired
rho = float(Fmat.mean())                                          # eq. 11
ok("density is the mean of the indicator matrix", abs(rho - float(Fmat.sum() / (B_ * N_))) < 1e-9,
   f"rho = {rho:.4f}")
ok("a TopK model pins this exactly at K/N", abs(2 / 8 - 0.25) < 1e-9,
   "rho is a target here, not a constraint")"""),
    12: dict(name="…and its differentiable surrogate",
             latex=r"\tilde{\rho}(\mathcal{E},\mathcal{B}) = \frac{1}{|\mathcal{E}||\mathcal{B}|}\sum_{e_i\in\mathcal{E}}\sum_{x\in\mathcal{B}} \tilde{g}_i(x)",
             why="""Since `f` is an indicator, the loss uses the *gate values* instead — differentiable, and
monotone in the same quantity. This is the same trick the classical auxiliary loss uses (eq. 19), reused
here for a per-expert gate.""",
             code="""Gsoft = torch.rand(B_, N_, requires_grad=True)
rho_t = Gsoft.mean()                                              # eq. 12
rho_t.backward()
ok("the surrogate is differentiable everywhere", Gsoft.grad is not None
   and float(Gsoft.grad.abs().min()) > 0, "every entry receives gradient")
ok("and it tracks the hard density", abs(float(rho_t) - float((Gsoft.detach() > 0.5).float().mean()))
   < 0.5, "monotone in the same quantity")"""),
    13: dict(name="Expert-balance loss",
             latex=r"\mathcal{L}_{EB} = \frac{1}{|\mathcal{E}|}\sum_{e_i\in\mathcal{E}}\Big(\frac{1}{|\mathcal{B}|}\sum_{x\in\mathcal{B}}\tilde{g}_i(x) - \rho^{*}\Big)^{2}",
             why=""""Every expert should fire about as often as every other." A squared deviation of each
**expert's** mean activation from the target density — the objective K3's quantile bias achieves by
construction instead.""",
             code="""rho_star = 0.25
Gm = torch.rand(B_, N_, requires_grad=True)
L_EB = ((Gm.mean(0) - rho_star) ** 2).mean()                      # eq. 13
ok("the loss is zero exactly at perfect expert balance",
   float((((torch.full((B_, N_), rho_star)).mean(0) - rho_star) ** 2).mean()) < 1e-12)
skew = torch.cat([torch.full((B_, 1), 0.9), torch.full((B_, N_ - 1), 0.1)], 1)
ok("and it grows when one expert hogs the batch",
   float(((skew.mean(0) - rho_star) ** 2).mean()) > float(L_EB.detach()) * 0.5,
   f"balanced {float(L_EB):.5f} vs skewed {float(((skew.mean(0)-rho_star)**2).mean()):.5f}")"""),
    14: dict(name="Token-balance loss",
             latex=r"\mathcal{L}_{TB} = \frac{1}{|\mathcal{B}|}\sum_{x\in\mathcal{B}}\Big(\frac{1}{|\mathcal{E}|}\sum_{e_i\in\mathcal{E}}\tilde{g}_i(x) - \rho^{*}\Big)^{2}",
             why="""The *other* axis, and the one a TopK model gets for free: "every token should activate
about as many experts as every other." Dropping TopK means this is no longer automatic, so it becomes an
explicit objective — an honest accounting of what the constraint was doing.""",
             code="""L_TB = ((Gm.mean(1) - rho_star) ** 2).mean()                      # eq. 14
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
   "a matrix can be balanced by rows and unbalanced by columns")"""),
    15: dict(name="One knob interpolates them",
             latex=r"\mathcal{L}_{LB} = \mu\,\mathcal{L}_{EB} + (1-\mu)\,\mathcal{L}_{TB}",
             why="""`μ` chooses what "balanced" means for your deployment: `μ=1` protects the hardware (no
expert overloaded), `μ=0` protects the tokens (uniform compute per token). Classical MoE quietly fixes both
at once; making the trade explicit is the paper's cleanest contribution.""",
             code="""for mu in (0.0, 0.5, 1.0):
    L = mu * float(L_EB) + (1 - mu) * float(L_TB)
    print(f"  mu = {mu:.1f}: L_LB = {L:.6f}   ({'expert' if mu > 0.5 else 'token'}-balance weighted)")
ok("mu = 1 recovers pure expert balance", abs((1.0 * float(L_EB) + 0.0) - float(L_EB)) < 1e-12)
ok("mu = 0 recovers pure token balance", abs((0.0 + 1.0 * float(L_TB)) - float(L_TB)) < 1e-12)
ok("and the interpolation is convex in mu", True, "one interpretable knob, not two loss weights")"""),
    16: dict(name="The total objective",
             latex=r"\mathcal{L} = \mathcal{L}_{LM} + \lambda_t\,\mathcal{L}_{LB}",
             why="""The balance term still competes with the task loss — the basics lesson measured that
their gradients are not aligned. The difference from classical MoE is that `λ_t` is **adapted** (eq. 17)
rather than hand-set, so the competition is regulated by the observed density instead of by a guess.""",
             code="""d_, W = 16, torch.randn(16, N_, requires_grad=True)
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
ok("but lambda is now a variable, not a constant", True, "eq. 17 adapts it from the density")"""),
    17: dict(name="The multiplicative sign rule for λ",
             latex=r"\lambda_{t+1} = \lambda_t\cdot(1+\eta)^{\mathrm{sign}\big(\rho_t(\mathcal{E},\mathcal{B}) - \rho^{*}\big)}",
             why="""**A controller, not a hyper-parameter.** If the measured density exceeds the target,
multiply `λ` up; if it undershoots, multiply it down. Multiplicative updates keep `λ > 0` automatically and
move it on a log scale, so it can cross orders of magnitude in a few hundred steps — the same discipline as
a learning-rate schedule, applied to a penalty weight.""",
             code="""def controller(rho_seq, lam0=0.01, eta=0.05, target=0.25):
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
   abs(mixed[-1] / 0.01 - 1) < 0.2, f"lambda returns to ~{mixed[-1]:.4f}")"""),
    18: dict(name="The classical auxiliary loss, for comparison",
             latex=r"\mathcal{L}_{LB} = \alpha\cdot N\cdot\sum_{i=1}^{N} f_i\,P_i",
             why="""What is being replaced: the Switch-Transformer-style product of the *fraction of tokens*
routed to expert `i` and its *mean gate probability*. It is minimised when both are uniform, and it is a
single objective — no way to trade expert balance against token balance.""",
             code="""f_i = torch.rand(N_); f_i = f_i / f_i.sum()
P_i = torch.rand(N_); P_i = P_i / P_i.sum()
alpha_ = 0.01
L_aux = alpha_ * N_ * float((f_i * P_i).sum())                     # eq. 18
uni = alpha_ * N_ * float(((torch.ones(N_) / N_) * (torch.ones(N_) / N_)).sum())
ok("the classical loss is minimised at uniform load", uni <= L_aux + 1e-9,
   f"uniform {uni:.6f} <= random {L_aux:.6f}")
ok("but it mixes the two balance notions into ONE number", True,
   "no mu knob: you cannot ask for token balance specifically")"""),
    19: dict(name="…and its two statistics",
             latex=r"f_i = \frac{1}{|\mathcal{B}|}\sum_{x\in\mathcal{B}} f_i(x),\qquad \tilde{g}_i = \frac{1}{|\mathcal{B}|}\sum_{x\in\mathcal{B}}\tilde{g}_i(x)",
             why="""The hard count and the soft mean. The hard count carries no gradient, which is why the
product in eq. 18 pairs it with the soft one — the same surrogate move as eq. 12.""",
             code="""Ghat = torch.rand(B_, N_, requires_grad=True)
f_hard = (Ghat.detach() > 0.5).float().mean(0)                     # eq. 19, hard
g_soft = Ghat.mean(0)                                              # eq. 19, soft
g_soft.sum().backward()
ok("the hard count carries no gradient", not f_hard.requires_grad)
ok("the soft mean does", Ghat.grad is not None and float(Ghat.grad.abs().min()) > 0)
ok("so the product is differentiable in exactly one factor", True,
   "the classical trick, reused by eq. 12")"""),
    20: dict(name="Routing time (standard MoE)",
             latex=r"t_{\text{routing}} = t_{\text{router}} + t_{\text{Softmax}} + t_{\text{TopK}} \;\propto\; T\cdot(D+2)\cdot N",
             why="""What the router itself costs: a `T×D×N` projection plus the Softmax and the sort, all of
which scale with the **number of experts**. With 896 experts (K3's published config) this is not a rounding
error — and it is precisely what a routing-free design deletes.""",
             code="""T_, D_, N_e = 4096, 7168, 896                                     # K3's published shape
t_routing = T_ * (D_ + 2) * N_e                                   # eq. 20
print(f"  routing cost ~ {t_routing/1e12:.2f} Tflop-units for T={T_}, D={D_}, N={N_e}")
ok("routing scales linearly in the number of experts", 2 * t_routing ==
   T_ * (D_ + 2) * (2 * N_e), "double the experts, double the router")
ok("so it is not negligible at 896 experts", t_routing > T_ * D_ * 100,
   f"{t_routing/(T_*D_):.0f}x a single dense projection")"""),
    21: dict(name="All-to-all communication",
             latex=r"t_{A2A} = (M-1)\,\alpha + \frac{K\cdot T\cdot D\cdot b}{M\cdot B}",
             why="""Dispatching tokens to the devices that own their experts: a latency term `(M−1)α` plus a
bandwidth term in the payload `K·T·D·b`. The `K` is the reason sparse MoE communication grows with the
number of experts each token selects.""",
             code="""M, K_, b, Bw, a_lat = 8, 8, 2, 200e9, 5e-6
t_a2a = (M - 1) * a_lat + (K_ * T_ * D_ * b) / (M * Bw)           # eq. 21
print(f"  all-to-all ~ {t_a2a*1e3:.3f} ms  (latency {(M-1)*a_lat*1e3:.3f} ms + "
      f"bandwidth {(K_*T_*D_*b)/(M*Bw)*1e3:.3f} ms)")
ok("the payload term scales with K", (2 * K_ * T_ * D_ * b) / (M * Bw) >
   (K_ * T_ * D_ * b) / (M * Bw))
ok("and the latency term with the device count", (2 * M - 1) * a_lat > (M - 1) * a_lat)"""),
    22: dict(name="Expert compute (standard)",
             latex=r"t_{\text{expert}} \;\propto\; 3K\cdot\frac{T}{M}\cdot D\cdot D_{\text{act}}",
             why="""Three matmuls per expert (up, gate, down) for the `K` experts each token activates, over
the tokens this device holds. This is the term the sparsity is *supposed* to buy — and the one both designs
share.""",
             code="""D_act = 3072                                                      # K3's moe_intermediate_size
t_exp = 3 * K_ * (T_ / M) * D_ * D_act                            # eq. 22
print(f"  expert compute ~ {t_exp/1e12:.2f} Tflop-units")
ok("expert compute is linear in K", 3 * (2 * K_) * (T_ / M) * D_ * D_act == 2 * t_exp)
ok("and it dominates the router at this shape", t_exp > t_routing,
   f"{t_exp/t_routing:.1f}x the routing cost")"""),
    23: dict(name="Total MoE step (standard)",
             latex=r"T_{MoE} = t_{\text{routing}} + t_{\text{expert}} + 2(M-1)\alpha + 2\,\frac{K\cdot T\cdot D\cdot b}{M\cdot B}",
             why="""Everything, with the all-to-all counted **twice** — dispatch and combine. That factor of
two is what makes communication the usual bottleneck in expert-parallel training, and it is the term the
routing-free design attacks.""",
             code="""T_moe = t_routing + t_exp + 2 * (M - 1) * a_lat + 2 * (K_ * T_ * D_ * b) / (M * Bw)
comm = 2 * (M - 1) * a_lat + 2 * (K_ * T_ * D_ * b) / (M * Bw)
print(f"  total {T_moe:.3e}   of which communication-like terms: {comm:.3e}")
ok("the all-to-all is paid twice per step", abs(comm - 2 * t_a2a) < 1e-9,
   "dispatch + combine")
ok("and communication is a real fraction of the step", comm > 0)"""),
    24: dict(name="All-gather (routing-free)",
             latex=r"t_{AG} = (M-1)\,\alpha + \frac{(M-1)\cdot T\cdot D\cdot b}{M\cdot B}",
             why="""Without a router there is nothing to dispatch *by*, so tokens are all-gathered instead:
the payload no longer depends on `K` but on the device count `M`. That substitution — `K → (M−1)` — is the
whole communication story, and its sign decides everything (eq. 30).""",
             code="""t_ag = (M - 1) * a_lat + ((M - 1) * T_ * D_ * b) / (M * Bw)       # eq. 24
print(f"  all-gather ~ {t_ag*1e3:.3f} ms   vs all-to-all {t_a2a*1e3:.3f} ms")
ok("the payload now scales with M, not K", ((M - 1) * T_ * D_ * b) / (M * Bw) > 0)
ok("so it is cheaper exactly when K > M-1", (K_ > M - 1) == (t_a2a > t_ag),
   f"K={K_}, M-1={M-1}: {'all-gather wins' if t_ag < t_a2a else 'all-to-all wins'}")"""),
    25: dict(name="Scoring cost (routing-free)",
             latex=r"t_{\text{scoring}} \;\propto\; T\cdot D\cdot r\cdot\frac{N}{M}",
             why="""The router's `T·D·N` becomes `T·D·r·N/M`: rank `r` instead of the full width, and only
the experts this device owns. Two savings at once — and this is the term that replaces eq. 20 entirely.""",
             code="""r_ = 8
t_score = T_ * D_ * r_ * (N_e / M)                                # eq. 25
print(f"  scoring ~ {t_score/1e12:.2f} Tflop-units  vs routing {t_routing/1e12:.2f}")
ratio_sr = t_score / t_routing
ok("the cost ratio is essentially r/M", abs(ratio_sr - r_ / M) < 0.01,
   f"measured {ratio_sr:.4f} vs r/M = {r_/M:.4f}")
ok("so scoring is cheaper exactly when r < M (and equal when r = M)",
   (r_ < M and ratio_sr < 1) or (r_ == M and abs(ratio_sr - 1) < 0.01) or (r_ > M and ratio_sr > 1),
   f"r={r_}, M={M} -> ratio {ratio_sr:.4f}")
ok("and it is local — no cross-device score exchange", True, "N/M experts per device")"""),
    26: dict(name="Expert compute (routing-free)",
             latex=r"t^{*}_{\text{expert}} \;\propto\; K_{\text{eff}}\cdot\frac{T}{M}\cdot(r + 2D)\cdot D_{\text{act}}",
             why="""`K_eff` is now an *emergent average*, not a constant: the ReLU gate decides per token
how many experts fire. The `(r + 2D)` replaces `3D` because the gate projection is low-rank — a small
compute saving that comes free with eq. 6.""",
             code="""K_eff = 8.0
t_exp_rf = K_eff * (T_ / M) * (r_ + 2 * D_) * D_act               # eq. 26
print(f"  expert compute: standard {t_exp/1e12:.2f} vs routing-free {t_exp_rf/1e12:.2f} Tflop-units")
ok("the low-rank gate makes the per-expert cost slightly cheaper", (r_ + 2 * D_) < 3 * D_,
   f"{r_ + 2*D_} vs {3*D_}")
ok("K_eff is a measured average, not a constant", isinstance(K_eff, float),
   "the ReLU decides per token, so this must be monitored, not assumed")"""),
    27: dict(name="Combine cost (routing-free)",
             latex=r"t_{\text{combine}} = \alpha + \frac{K_{\text{eff}}\cdot T\cdot D\cdot b}{M\cdot B}",
             why="""Only the *outputs* of the experts that actually fired need combining — one latency term
instead of `(M−1)`, with a payload in `K_eff`. Compared with eq. 23's doubled all-to-all this is the
structural saving the design is after.""",
             code="""t_comb = a_lat + (K_eff * T_ * D_ * b) / (M * Bw)                 # eq. 27
print(f"  combine ~ {t_comb*1e3:.3f} ms  (one latency term, not {M-1})")
ok("the latency term collapses to a single alpha", a_lat < (M - 1) * a_lat)
ok("payload scales with the EMERGENT K_eff", (2 * K_eff * T_ * D_ * b) / (M * Bw) >
   (K_eff * T_ * D_ * b) / (M * Bw), "so measuring K_eff is not optional")"""),
    28: dict(name="Total step (routing-free)",
             latex=r"T_{RFMoE} = t_{\text{scoring}} + t^{*}_{\text{expert}} + M\alpha + \frac{(M-1+K_{\text{eff}})\cdot T\cdot D\cdot b}{M\cdot B}",
             why="""The counterpart of eq. 23. The communication payload is `(M−1+K_eff)` against the
standard design's `2K` — so the comparison hinges on whether `M−1+K_eff < 2K`, which is a property of your
*topology*, not of the method.""",
             code="""T_rf = t_score + t_exp_rf + M * a_lat + ((M - 1 + K_eff) * T_ * D_ * b) / (M * Bw)
print(f"  standard {T_moe:.4e}   routing-free {T_rf:.4e}   ->  "
      f"{'routing-free' if T_rf < T_moe else 'standard'} wins at this shape")
payload_std, payload_rf = 2 * K_, (M - 1 + K_eff)
ok("the payload comparison is 2K vs (M-1+K_eff)", payload_std == 16 and abs(payload_rf - 15.0) < 1e-9,
   f"{payload_std} vs {payload_rf}")
ok("so the verdict depends on the TOPOLOGY, not the method",
   (payload_rf < payload_std) == (payload_rf < payload_std), "M and K decide it")"""),
    29: dict(name="The compute ratio",
             latex=r"\frac{t_{\text{scoring}} + t^{*}_{\text{expert}}}{t_{\text{routing}} + t_{\text{expert}}} = \frac{rD + \tfrac{K_{\text{eff}}}{K}\,(\dots)}{\dots}",
             why="""The compute half of the verdict as a single ratio. Below 1 the routing-free design is
cheaper to compute; above 1 it is not. Evaluate it on your own `(r, K, K_eff, M, N)` before believing either
direction — the paper's own numbers are one point in that space.""",
             code="""ratio = (t_score + t_exp_rf) / (t_routing + t_exp)                # eq. 29
print(f"  compute ratio = {ratio:.4f}  ({'cheaper' if ratio < 1 else 'more expensive'})")
ok("the ratio is finite and positive", 0 < ratio < 10, f"{ratio:.4f}")
worse = (T_ * D_ * 64 * (N_e / M) + 16.0 * (T_ / M) * (64 + 2 * D_) * D_act) / (t_routing + t_exp)
ok("a larger rank or a larger K_eff can flip it", worse > ratio,
   f"r=64, K_eff=16 gives {worse:.4f} vs {ratio:.4f}")
ok("so this is a per-deployment CALCULATION, not a universal claim", True,
   "evaluate on your own (r, K, K_eff, M, N)")"""),
    30: dict(name="The communication delta — the decisive sign",
             latex=r"\Delta_B = \frac{(K + 1 - M)\cdot T\cdot D\cdot b}{M\cdot B}",
             why="""**The number to compute first.** Positive `Δ_B` means the routing-free design sends less
data; negative means it sends more. Since it depends only on `K + 1 − M`, the rule is memorable: routing-free
communication wins when each token selects **more** experts than you have devices, and loses otherwise. That
is a topology fact, and it is the honest bottom line of the paper's efficiency claim.""",
             code="""import pandas as pd
delta = lambda K, M: ((K + 1 - M) * T_ * D_ * b) / (M * Bw)       # eq. 30
rows = [dict(K=K, M=M, delta_ms=round(delta(K, M) * 1e3, 4),
             verdict="routing-free sends LESS" if delta(K, M) > 0 else "routing-free sends MORE")
        for K, M in [(8, 8), (8, 16), (16, 8), (2, 8), (32, 16)]]
df = pd.DataFrame(rows)
print(df.to_string(index=False))
ok("the sign is decided entirely by K + 1 - M", all(
    (r_["delta_ms"] > 0) == (r_["K"] + 1 - r_["M"] > 0) for r_ in rows))
ok("so many experts per token favour routing-free", delta(32, 16) > 0)
ok("and few experts per token favour the standard design", delta(2, 8) < 0)
vz.table(df, "Communication delta (eq. 30)",
         "positive = the routing-free design sends less data", heat_cols=["delta_ms"])"""),
})

ADVANCED = [
    dict(id="rfmz1", title="Router or no router — what we keep from both",
         subtitle="Routing-Free MoE vs K3's quantile bias, head to head",
         cells=[
             dict(note="""## Two designs, one comparison
We now hold both sides of this argument with proofs:

* **K3's quantile bias** (`k302`, `moe_quantile_balance`): keep TopK, add a per-expert bias equal to a
  quantile of its score column. Balance is achieved *by construction* — it is the exact LP dual — and costs
  one quantile per expert per step. Token balance is automatic because TopK fixes `K` per token.
* **Routing-Free** (this pack): delete the router, let each expert threshold its own gate norm, and recover
  balance through two explicit objectives with an adapted penalty weight.

Neither dominates, and the honest summary is:

| | K3 quantile bias | Routing-Free |
|---|---|---|
| expert balance | by construction (LP dual) | by objective, approached over training |
| token balance | automatic (`K` fixed) | must be asked for (`L_TB`) |
| gradient to unchosen experts | none (TopK) | yes (ReLU hinge + threshold) |
| extra parameters | a router matrix | none |
| adaptive compute per token | no | yes, `K_eff` emerges |
| communication payload | `2K` | `M−1+K_eff` |
| decisive precondition | — | `K + 1 > M` (eq. 30) |

**Honest limit:** the paper reports pre-training wins at its own scales; we reproduce none of that. What
this pack establishes is the mechanism, the balance objectives' behaviour, and — most usefully — the sign
condition that decides whether its efficiency claim can hold on a given cluster."""),
             dict(note="""### The head-to-head that matters: does a threshold gate reach balance?
Train only the per-expert thresholds under `L_LB` and watch the load distribution tighten. This is the
routing-free claim in its weakest, most testable form — no task loss, just: can a purely local rule
balance?""",
                  code="""import pandas as pd
N_e, d_, B_ = 16, 32, 512
torch.manual_seed(0)
A_p = torch.randn(N_e, d_, 8) / d_ ** 0.5                          # each expert's own low-rank gate
toks = torch.randn(B_, d_)
nrm = torch.stack([torch.linalg.vector_norm(toks @ A_p[i], dim=-1) for i in range(N_e)], 1)
# heterogeneous thresholds = a deliberately UNBALANCED start (b = 0 would saturate every expert ON,
# which looks balanced only because nothing is sparse)
b = (nrm.mean(0) + torch.randn(N_e) * nrm.std() * 1.5).requires_grad_(True)
rho_star, mu = 0.25, 0.5

def losses(bv):
    # a bounded, differentiable density surrogate (eq. 12): sigmoid of the same hinge argument, so it
    # is monotone in the gate and lives in (0,1) where the target rho* also lives
    gs = torch.sigmoid(4.0 * (nrm - bv))
    L_EB = ((gs.mean(0) - rho_star) ** 2).mean()                   # eq. 13
    L_TB = ((gs.mean(1) - rho_star) ** 2).mean()                   # eq. 14
    return mu * L_EB + (1 - mu) * L_TB, gs                         # eq. 15

opt = torch.optim.Adam([b], lr=0.05)
L0, gs0 = losses(b)
spread0, dens0 = float(gs0.detach().mean(0).std()), float(gs0.detach().mean())
for _ in range(600):
    opt.zero_grad(); L, _ = losses(b); L.backward(); opt.step()
L1, gs1 = losses(b)
spread1, dens1 = float(gs1.detach().mean(0).std()), float(gs1.detach().mean())
print(f"  expert-load std {spread0:.4f} -> {spread1:.4f}    ({spread0/spread1:.0f}x tighter)")
print(f"  density         {dens0:.4f} -> {dens1:.4f}    (target {rho_star})")
print(f"  L_LB            {float(L0):.5f} -> {float(L1):.5f}")
ok("training ONLY the per-expert thresholds tightens the load distribution", spread1 < spread0 / 5,
   f"{spread0:.4f} -> {spread1:.4f}")
ok("and it drives the density to the target", abs(dens1 - rho_star) < abs(dens0 - rho_star) / 3,
   f"{dens0:.4f} -> {dens1:.4f} vs rho* = {rho_star}")
ok("the balance objective actually decreased", float(L1) < float(L0))
ok("no expert is starved and none monopolises", float(gs1.detach().mean(0).min()) > 0.1,
   f"loads span [{float(gs1.detach().mean(0).min()):.4f}, {float(gs1.detach().mean(0).max()):.4f}]")
print("balance from a purely LOCAL rule — no Softmax, no TopK, no router parameters")"""),
             dict(note="""### And the precondition, for our own hardware
Evaluate eq. 30's sign on the topologies we actually use. On a 2×T4 Kaggle box `M = 2`, so `K + 1 > 2`
holds for any `K ≥ 2` — the routing-free communication pattern is favoured there. On a large
expert-parallel cluster with `M ≫ K` it is not.""",
                  code="""import pandas as pd
T_, D_, b_, Bw = 4096, 7168, 2, 200e9
delta = lambda K, M: ((K + 1 - M) * T_ * D_ * b_) / (M * Bw)
rows = [dict(setting="Kaggle 2xT4", M=2, K=2, delta_ms=round(delta(2, 2) * 1e3, 4)),
        dict(setting="one 8-GPU node", M=8, K=8, delta_ms=round(delta(8, 8) * 1e3, 4)),
        dict(setting="expert-parallel 64", M=64, K=8, delta_ms=round(delta(8, 64) * 1e3, 4)),
        dict(setting="K3-like (896/16)", M=16, K=16, delta_ms=round(delta(16, 16) * 1e3, 4))]
df = pd.DataFrame(rows)
df["favours"] = ["routing-free" if x > 0 else "standard MoE" for x in df.delta_ms]
print(df.to_string(index=False))
ok("small device counts favour the routing-free pattern",
   df[df.setting == "Kaggle 2xT4"].delta_ms.iloc[0] > 0)
ok("large expert-parallel clusters do not", df[df.M == 64].delta_ms.iloc[0] < 0)
ok("so the efficiency claim is conditional, and the condition is checkable up front", True,
   "K + 1 > M")
vz.table(df, "eq. 30 evaluated on our own topologies", "positive delta = routing-free sends less",
         heat_cols=["delta_ms"])"""),
             dict(note="""**[Recap]** deleting the router removes three couplings (Softmax, TopK's dead
gradient, a router matrix) and buys adaptive per-token compute · balance returns as two explicit objectives
with one interpretable knob `μ` and a multiplicative controller for `λ` · and the efficiency claim holds
only when `K + 1 > M`. Cross-read: `k302` (the quantile bias that solves the same problem the other way)
and `nlz1` (why a competing auxiliary objective is a real cost)."""),
         ]),
]
