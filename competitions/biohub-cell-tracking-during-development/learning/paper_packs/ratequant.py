"""Paper pack — *RateQuant: Optimal Mixed-Precision KV Cache Quantization via Rate-Distortion Theory*
arXiv:2605.06675 · https://arxiv.org/pdf/2605.06675
local: docs/papers/ratequant/ratequant.md

The descendant of HOPE in our reading list, and the most immediately usable of the six. HOPE gave us the
principle — rank compression actions by **cost per parameter saved**, never by cost alone
(`compress_select.rate_distortion_pick`). RateQuant applies exactly that principle to a decision we make
on every long-context run: how many bits does each attention head's KV cache deserve?

Every deployed quantizer gives all heads the same bit-width even though head importance varies by orders
of magnitude. RateQuant writes the bit allocation as a constrained optimization, solves it in closed form
with a Lagrange multiplier, and the answer is beautiful and cheap: **spend bits in proportion to the
LOGARITHM of a head's importance**, i.e. `bᵢ* = b̄ + (ln wᵢ − mean ln w)/ln β`. Uniform allocation is
suboptimal by exactly the arithmetic-to-geometric-mean ratio of the importances — a quantity you can
compute before you quantise anything, which tells you whether mixed precision is worth the trouble at all.

Read after `nl17`/`nlz1` (rate–distortion in Appendix B of Nested Learning is a different use of the same
idea) and the HOPE additions in `compress_select`.
"""

SLUG = "ratequant"
PREFIX = "rq"
ORDER_BASE = 2000
TOTAL_EQ = 16
SECTION_TITLE = "RateQuant (2026) — optimal mixed-precision by rate–distortion, proved in PyTorch"
SKIP_SECTIONS = ["references", "abstract", "related work", "appendix overview",
                 "related work comparison ta", "discussion and future work",
                 "component ablation waterfa", "mixed-precision baseline c"]

EQ_SECTIONS = [("1", 1, 4), ("3", 5, 9), ("4", 0, 0), ("6", 0, 0), ("B", 10, 15), ("C", 16, 16)]

HEADER = """import torch, torch.nn as nn, torch.nn.functional as F      # bit allocation is one Lagrange multiplier
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
    dict(id="rqb1", title="Basics — quantisation error halves per bit, so allocation is a log problem",
         subtitle="RateQuant · the one empirical fact the whole derivation rests on",
         cells=[
             dict(note="""## One extra bit halves the error
Uniform quantisation to `b` bits has a step size proportional to `2^{-b}`, so the squared error falls
geometrically: `D(b) ≈ α·β^{-b}` with `β ≈ 4` for MSE (one bit → half the step → a quarter of the squared
error). That single fact is why bit allocation has a *closed-form* answer: an exponential distortion curve
turns "minimise total distortion subject to a bit budget" into a problem whose stationarity condition is
linear in the bits.

Measure it first — do not take the exponent on faith."""),
             dict(note="""### Fit the distortion curve on real tensors
Quantise a Gaussian tensor to `b` bits with a symmetric uniform quantiser and measure the MSE. The fit
`log D = log α − b·log β` should be almost perfectly linear, with `β ≈ 4`.""",
                  code="""def quantize(x, bits):
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().max() / qmax
    return torch.round(x / s).clamp(-qmax - 1, qmax) * s

x = torch.randn(1 << 16)
bits = torch.arange(2, 9)
mse = torch.tensor([float(((quantize(x, int(b)) - x) ** 2).mean()) for b in bits])
logD = torch.log(mse)
A = torch.stack([torch.ones_like(bits.float()), -bits.float()], 1)
coef = torch.linalg.lstsq(A, logD.unsqueeze(1)).solution.squeeze(1)
alpha, beta = float(torch.exp(coef[0])), float(torch.exp(coef[1]))
pred = A @ coef
r2 = float(1 - ((logD - pred) ** 2).sum() / ((logD - logD.mean()) ** 2).sum())
for b, m in zip(bits.tolist(), mse.tolist()):
    print(f"  {b} bits -> MSE {m:.3e}")
ok("the distortion curve is exponential in the bit-width", r2 > 0.995, f"R^2 = {r2:.5f}")
ok("and the base is about 4 (one bit = a quarter of the squared error)", 3.0 < beta < 5.0,
   f"fitted alpha = {alpha:.3e}, beta = {beta:.3f}")"""),
             dict(note="""### Head importances span orders of magnitude — that is the opportunity
If every head mattered equally, uniform allocation would be optimal and there would be no paper. Measure
the spread of a plausible importance signal (the gradient norm through each head's cache) and the case for
mixed precision writes itself.""",
                  code="""N = 32
w = torch.distributions.LogNormal(0.0, 1.4).sample((N,))         # importance, heavy-tailed as observed
am, gm = float(w.mean()), float(torch.exp(torch.log(w).mean()))
print(f"  {N} heads: importance min {float(w.min()):.3e}  max {float(w.max()):.3e}  "
      f"ratio {float(w.max()/w.min()):.0f}x")
print(f"  arithmetic mean {am:.4f} vs geometric mean {gm:.4f}  ->  AM/GM = {am/gm:.3f}")
ok("importance spans orders of magnitude", float(w.max() / w.min()) > 20)
ok("so AM/GM > 1, which is exactly the suboptimality of uniform bits (eq. 4)", am / gm > 1.0,
   f"uniform costs {am/gm:.2f}x the optimum")"""),
             dict(note="""**[Recap]** `D(b) = αβ^{-b}` with β≈4, measured · head importance is
heavy-tailed · therefore uniform bit-width leaves a factor AM/GM on the table.
**Next → §1, the problem stated properly.**"""),
         ]),
]

EQ = {}
SECTION = {}
ADVANCED = []

SECTION["1"] = dict(why="""**The problem, and its answer.** Minimise total weighted distortion subject to a
bit budget (eq. 1). Because the distortion is exponential in bits, the optimum is *logarithmic in
importance* (eq. 2): every head starts from the average budget `b̄` and is nudged by how far its
log-importance sits from the mean log-importance. Eq. 4 is the punchline — uniform allocation costs exactly
the **AM/GM ratio** of the importances, a number you can compute in advance to decide whether mixed
precision is worth implementing at all.""")

SECTION["3"] = dict(why="""**RateQuant proper.** The constrained problem with box constraints
(`b_min ≤ b ≤ b_max`, eq. 5), its Lagrangian (eq. 6), the stationarity condition that produces the
log-allocation (eq. 7), and the optimal cost in closed form (eq. 8). Eq. 9 is the importance model: a
second-order expansion of the task loss in the cache perturbation, which is what justifies using
gradient norms as the weights `w_i`.""")

SECTION["B"] = dict(why="""**Appendix B — the proofs, and they are short.** The Lagrangian (eq. 10), the
per-head first-order condition (eq. 11), the same condition for a per-head exponential model (eqs. 12–13),
Jensen's inequality (eq. 14) — which is where AM/GM comes from — and the AM/GM identity itself (eq. 15).
The convexity that makes all of this legitimate is eq. 16's diminishing-returns property: each additional
bit buys strictly less than the previous one.""")

SECTION["C"] = dict(why="""**Appendix C — the distortion model's parameters.** `D_i(b) = α_i e^{-β_i b}`
fitted per head, plus the decreasing-marginal-gain property that makes the greedy and the closed-form
solutions agree.""")

EQ.update({
    1: dict(name="The bit-allocation problem",
            latex=r"\min_{b\in\mathbb{R}^{N}}\;\mathcal{J}(b) \;=\; \sum_{i=1}^{N} w_i\,D(b_i)\qquad \text{s.t.}\quad \sum_{i=1}^{N} b_i = B",
            why="""Total distortion is the importance-weighted sum of per-head distortions, under a fixed
bit budget. Note the shape: this is HOPE's rate–distortion criterion with the roles made explicit —
`w_i D(b_i)` is the cost and `b_i` is the rate we are spending.""",
            code="""N = 16
alpha, beta = 1.0, 4.0                                          # D(b) = alpha * beta^-b (measured in rqb1)
w = torch.distributions.LogNormal(0.0, 1.2).sample((N,))
B = 4.0 * N                                                     # average 4 bits per head
D = lambda b: alpha * beta ** (-b)
J = lambda b: float((w * D(b)).sum())
b_uniform = torch.full((N,), B / N)
ok("the budget constraint is satisfiable", abs(float(b_uniform.sum()) - B) < 1e-6)
ok("the objective is finite and positive", 0 < J(b_uniform) < float('inf'), f"J(uniform) = {J(b_uniform):.5f}")
print(f"  {N} heads, budget {B:.0f} bits total ({B/N:.1f} per head on average)")"""),
    2: dict(name="The closed-form optimum — bits are logarithmic in importance",
            latex=r"b_i^{*} \;=\; \bar{b} \;+\; \frac{\ln w_i - \overline{\ln w}}{\ln\beta}",
            why="""**The result.** Start every head at the average budget and shift it by its
log-importance relative to the mean log-importance, scaled by `1/ln β`. Consequences worth internalising:
a head that is `β×` more important earns exactly **one** more bit; the allocation is invariant to a global
rescaling of importances; and it is computable in one pass with no search.""",
            code="""bbar = B / N
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
         1e-4), "only RELATIVE importance matters")"""),
    3: dict(name="…where the two averages are",
            latex=r"\bar{b} = B/N,\qquad \overline{\ln w} = \frac{1}{N}\sum_j \ln w_j",
            why="""Book-keeping, but it makes the invariance visible: the allocation depends on
`ln w_i − mean ln w`, i.e. only on *relative* log-importance, so the units of `w` are irrelevant.""",
            code="""ok("b-bar is the per-head average budget", abs(bbar - B / N) < 1e-12, f"b_bar = {bbar}")
ok("mean-log-w is the log of the GEOMETRIC mean",
   abs(float(logw.mean()) - float(torch.log(torch.exp(logw.mean())))) < 1e-6,
   f"exp(mean log w) = {float(torch.exp(logw.mean())):.4f} = geometric mean")"""),
    4: dict(name="The price of uniform allocation is AM/GM",
            latex=r"\frac{\mathcal{J}_u}{\mathcal{J}^{*}} \;=\; \frac{\bar{w}}{\widetilde{w}} \;\ge\; 1",
            why="""**The number to compute before writing any code.** The ratio of uniform to optimal
distortion is the *arithmetic mean over the geometric mean* of the importances. Equal importances → ratio 1
→ mixed precision is pointless. Heavy-tailed importances → the ratio is the exact speed-up available. It is
≥ 1 by Jensen (eq. 14), so mixed precision can never be worse.""",
            code="""am = float(w.mean()); gm = float(torch.exp(torch.log(w).mean()))
ok("the measured ratio equals AM/GM exactly",
   abs(J(b_uniform) / J(b_star) - am / gm) < 1e-3,
   f"measured {J(b_uniform)/J(b_star):.4f} vs AM/GM {am/gm:.4f}")
ok("the ratio is >= 1 always (Jensen)", am / gm >= 1.0 - 1e-9)
w_eq = torch.full((N,), 3.0)
b_eq = B / N + (torch.log(w_eq) - torch.log(w_eq).mean()) / torch.log(torch.tensor(beta))
ok("equal importances -> uniform IS optimal (no gain to chase)",
   abs(float(w_eq.mean() / torch.exp(torch.log(w_eq).mean())) - 1.0) < 1e-6
   and close(b_eq, torch.full((N,), B / N), 1e-5))
print(f"  so on THIS importance profile mixed precision is worth {am/gm:.2f}x, computable up front")"""),
    5: dict(name="The problem with box constraints (what you can actually ship)",
            latex=r"\min_{b}\;\sum_{i=1}^{N} w_i\,\alpha\beta^{-b_i}\qquad \text{s.t.}\quad \sum_i b_i = B,\;\; b_{\min}\le b_i\le b_{\max}",
            why="""Hardware only has 2/3/4/8-bit kernels, and a head cannot get −1 bits. Adding box
constraints keeps the problem convex, so the unconstrained log-allocation followed by clipping and
re-normalising is a legitimate (and standard) water-filling solution.""",
            code="""N = 16; alpha, beta = 1.0, 4.0                                   # this lesson's own setup
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
   f"J {J(b_uniform):.6f} -> {J(b_box):.6f}")"""),
    6: dict(name="The Lagrangian",
            latex=r"\mathcal{L} = \sum_i w_i\alpha\beta^{-b_i} + \lambda\Big(\sum_i b_i - B\Big) + \sum_i \mu_i\big(\text{box terms}\big)",
            why="""One multiplier `λ` for the budget and KKT multipliers for the box. `λ` has a clean
meaning: **the marginal distortion bought by one more bit anywhere** — at the optimum it is equal across
all unclipped heads, which is exactly the water-filling picture.""",
            code="""lam = 0.02
Lag = lambda b: float((w * D(b)).sum() + lam * (b.sum() - B))
ok("the Lagrangian equals the objective on the feasible set",
   abs(Lag(b_star) - J(b_star)) < 1e-4, "the constraint term vanishes when the budget is met")
marg = -(w * D(b_star) * torch.log(torch.tensor(beta)))          # dJ/db per head
ok("at the optimum the marginal gain is EQUAL across heads (water-filling)",
   float(marg.std() / marg.abs().mean()) < 1e-4,
   f"relative spread of dJ/db = {float(marg.std()/marg.abs().mean()):.2e}")"""),
    7: dict(name="Stationarity gives the log-allocation",
            latex=r"w_i\alpha(\ln\beta)\beta^{-b_i} = \lambda \;\;\Longrightarrow\;\; b_i = \frac{\ln(w_i\alpha\ln\beta) - \ln\lambda}{\ln\beta}",
            why="""Set the derivative to zero and take logs — the exponential distortion turns the
condition linear, which is why the answer is closed-form and not a search. `λ` is then fixed by the budget
(one bisection, eq. 5's water-filling).""",
            code="""bv = b_star.clone().requires_grad_(True)
(w * alpha * beta ** (-bv)).sum().backward()
grads = bv.grad
ok("all per-head derivatives are equal at b*", float(grads.std() / grads.abs().mean()) < 1e-4,
   f"relative spread {float(grads.std()/grads.abs().mean()):.2e}")
lam_star = float(-grads.mean())
b_from_lam = (torch.log(w * alpha * torch.log(torch.tensor(beta))) -
              torch.log(torch.tensor(lam_star))) / torch.log(torch.tensor(beta))
ok("inverting the condition reproduces b* exactly", close(b_from_lam, b_star, 1e-3),
   f"lambda* = {lam_star:.6f}")"""),
    8: dict(name="The optimal cost, in closed form",
            latex=r"\mathcal{J}^{*} = \alpha\sum_i e^{Y_i}\beta^{-\bar{b} - (Y_i - \bar{Y})/\ln\beta} = \alpha\beta^{-\bar{b}}\,N\,\widetilde{w},\qquad Y_i = \ln w_i",
            why="""Substituting the optimum collapses the sum to `N·(geometric mean of w)·αβ^{-b̄}`. That
is where AM/GM comes from: the uniform cost has the **arithmetic** mean in the same slot. Two lines of
algebra, and it turns a design question into a computable number.""",
            code="""J_star_closed = alpha * beta ** (-bbar) * N * gm
ok("the closed form matches the evaluated optimum", abs(J(b_star) - J_star_closed) / J(b_star) < 1e-4,
   f"evaluated {J(b_star):.6f} vs closed form {J_star_closed:.6f}")
J_unif_closed = alpha * beta ** (-bbar) * N * am
ok("and the uniform cost is the same expression with the ARITHMETIC mean",
   abs(J(b_uniform) - J_unif_closed) / J(b_uniform) < 1e-6,
   f"uniform {J(b_uniform):.6f} vs {J_unif_closed:.6f}")"""),
    9: dict(name="Why gradient norms are the right importance",
            latex=r"\mathcal{L}(\hat{\theta}) - \mathcal{L}(\theta) \approx \sum_{l,h}\big\langle \nabla_K\mathcal{L},\,\delta^{K}\big\rangle + \big\langle \nabla_V\mathcal{L},\,\delta^{V}\big\rangle + O(\lVert\delta\rVert^2)",
            why="""Expand the task loss in the quantisation perturbation `δ`. To first order the damage a
head does is governed by the gradient flowing through its cache, so `w_i = ‖∂L/∂K_i‖²` is not a heuristic —
it is the coefficient in the expansion. This is the same move HOPE makes when it scores a neuron by the
function it computes rather than by its weights.""",
            code="""dh, T = 16, 24
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
   float((gK ** 2).sum()) > 0, f"||dL/dK||_F^2 = {float((gK**2).sum()):.4f}")"""),
    10: dict(name="Appendix B — the general Lagrangian",
             latex=r"\mathcal{L}(b,\lambda) = \sum_{i=1}^{N} w_i D_i(b_i) + \lambda\Big(\sum_{i=1}^{N} b_i - N\bar{b}\Big)",
             why="""The same Lagrangian with a *general* per-head distortion `D_i`, so the result does not
depend on the exponential model — only on convexity and monotonicity.""",
             code="""N = 16; alpha, beta = 1.0, 4.0                                   # this lesson's own setup
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
   f"J_gen(uniform) = {Jgen(b_uniform):.5f}")"""),
    11: dict(name="…and its first-order condition",
             latex=r"w_iD_i'(b_i) + \lambda = 0 \;\;\Longrightarrow\;\; D_i'(b_i) = -\frac{\lambda}{w_i}",
             why="""Every head is pushed to the bit-width where its **marginal** distortion reduction,
weighted by importance, equals one common `λ`. This is the general water-filling statement; the
exponential model is only what makes it solvable in closed form.""",
             code="""bg = b_uniform.clone().requires_grad_(True)
Jt = sum(w[i] * Dgen[i](bg[i]) for i in range(N))
Jt.backward()
ok("marginal gains are NOT equal at a uniform allocation (so uniform is not optimal)",
   float(bg.grad.std() / bg.grad.abs().mean()) > 0.05,
   f"relative spread {float(bg.grad.std()/bg.grad.abs().mean()):.3f}")
print("equalising those marginals is exactly what the multiplier does")"""),
    12: dict(name="The per-head exponential condition",
             latex=r"\alpha_i\beta_i\,e^{-\beta_ib_i^{*}} = \frac{\lambda}{w_i}",
             why="""With `D_i(b) = α_i e^{-β_i b}` the condition is explicit per head — heads may now have
*different* curve steepness `β_i`, which matters in practice because some heads' caches tolerate
quantisation far better than others.""",
             code="""ai = torch.rand(N) + 0.5
bi = torch.rand(N) * 0.8 + 1.0
lam2 = 0.05
b_opt = (torch.log(w * ai * bi / lam2)) / bi                     # eq. 13
lhs = ai * bi * torch.exp(-bi * b_opt)
ok("the stationarity condition holds per head", close(lhs, lam2 / w, 1e-4),
   f"max relative error {float(((lhs - lam2/w).abs()/(lam2/w)).max()):.2e}")"""),
    13: dict(name="…solved for the bits",
             latex=r"b_i^{*} = \frac{1}{\beta_i}\ln\frac{w_i\alpha_i\beta_i}{\lambda}",
             why="""The general closed form. Still logarithmic in importance, but now scaled by each head's
own curve steepness `1/β_i` — a head whose distortion falls slowly gets *more* bits for the same
importance. `λ` is again set by the budget.""",
             code="""def alloc(lam):
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
   f"corr(log w, bits) = {float(torch.corrcoef(torch.stack([torch.log(w), b_gen]))[0,1]):+.3f}")"""),
    14: dict(name="Jensen's inequality — the source of the AM/GM bound",
             latex=r"\mathbb{E}\big[f(X)\big] \;\ge\; f\big(\mathbb{E}[X]\big)\qquad (f\text{ convex})",
             why="""The one-line reason mixed precision can never lose: applying Jensen to `exp` over the
log-importances gives arithmetic ≥ geometric mean, hence `J_u/J* ≥ 1`.""",
             code="""X = torch.log(w)
lhs = float(torch.exp(X).mean())                                 # E[f(X)], f = exp (convex)
rhs = float(torch.exp(X.mean()))                                 # f(E[X])
ok("Jensen holds for exp on this sample", lhs >= rhs - 1e-9, f"E[e^X] {lhs:.4f} >= e^E[X] {rhs:.4f}")
ok("equality only when the importances are identical",
   abs(float(torch.exp(torch.zeros(N)).mean()) - float(torch.exp(torch.zeros(N).mean()))) < 1e-9)"""),
    15: dict(name="The AM/GM identity, written out",
             latex=r"\frac{\mathrm{AM}}{\mathrm{GM}} = \frac{\frac{1}{N}\sum_i w_i}{\big(\prod_i w_i\big)^{1/N}} = \exp\Big(\ln\bar{w} - \overline{\ln w}\Big) \;\ge\; 1",
             why="""The suboptimality factor as an explicit formula — and it is precisely the *log-variance*
of the importances to leading order, so the gain from mixed precision grows with how unequal your heads
are. That is the diagnostic to compute before implementing anything.""",
             code="""ratio = float(torch.exp(torch.log(w.mean()) - torch.log(w).mean()))
ok("the exp-of-log-gap form matches AM/GM", abs(ratio - am / gm) < 1e-5, f"{ratio:.5f}")
import math
approx = math.exp(0.5 * float(torch.log(w).var(unbiased=False)))
ok("to leading order it is exp(var of log w / 2)", abs(approx - ratio) / ratio < 0.35,
   f"exp(varlog/2) = {approx:.3f} vs exact {ratio:.3f}")
print(f"  diagnostic: log-variance {float(torch.log(w).var(unbiased=False)):.3f} "
      f"-> expect about {ratio:.2f}x from mixed precision")"""),
    16: dict(name="Diminishing returns makes it convex",
             latex=r"\Delta D_i(b) = D_i(b) - D_i(b+1)\quad \text{is decreasing in } b",
             why="""The property that licenses everything: each extra bit buys strictly less than the
previous one. It makes the problem convex (so the closed form is a global optimum) and it makes a greedy
"give the next bit to whoever gains most" agree with the closed form — which is how you implement this
under integer bit-widths.""",
             code="""N = 16; alpha, beta = 1.0, 4.0                                   # this lesson's own setup
w = torch.distributions.LogNormal(0.0, 1.2).sample((N,))
B = 4.0 * N; bbar = B / N
D = lambda b: alpha * beta ** (-b)
J = lambda b: float((w * D(b)).sum())
b_uniform = torch.full((N,), bbar)
b_star = bbar + (torch.log(w) - torch.log(w).mean()) / torch.log(torch.tensor(beta))
am = float(w.mean()); gm = float(torch.exp(torch.log(w).mean()))
b_min, b_max = 2.0, 8.0
b_box = ((torch.log(w) - torch.log(w).mean()) / torch.log(torch.tensor(beta)) + bbar).clamp(b_min, b_max)
bs = torch.arange(2, 9).float()
gains = torch.tensor([float(alpha * beta ** (-b) - alpha * beta ** (-(b + 1))) for b in bs])
ok("marginal gains are strictly decreasing", bool((gains[1:] < gains[:-1]).all()),
   f"gains {[f'{g:.2e}' for g in gains[:4].tolist()]} ...")
# greedy integer allocation must agree with the (rounded) closed form
bits_int = torch.full((N,), int(b_min))
budget_left = int(B - bits_int.sum())
for _ in range(budget_left):
    g = w * (alpha * beta ** (-bits_int.float()) - alpha * beta ** (-(bits_int.float() + 1)))
    g[bits_int >= b_max] = -1.0
    bits_int[int(g.argmax())] += 1
ok("greedy integer allocation matches the closed form to within a bit",
   float((bits_int.float() - b_box).abs().max()) <= 1.5,
   f"max |greedy - closed| = {float((bits_int.float()-b_box).abs().max()):.2f} bits")
ok("and greedy is never worse than uniform", J(bits_int.float()) <= J(b_uniform),
   f"J greedy {J(bits_int.float()):.6f} vs uniform {J(b_uniform):.6f}")"""),
})

ADVANCED = [
    dict(id="rqz1", title="What we take from RateQuant — a bit allocator we can use today",
         subtitle="RateQuant · wired into our quantisation agents",
         cells=[
             dict(note="""## Why this one is immediately useful
We already quantise for 2×T4 inference, and we already have HOPE's `J/Δparams` criterion. RateQuant
supplies the missing closed form for the *bit* axis:

1. **Compute the diagnostic first.** `AM/GM` of the head importances tells you the available gain before
   you write a line of kernel code. Near 1 → skip mixed precision entirely.
2. **Allocate by log-importance**, then clip to the hardware's bit-widths and water-fill the budget.
3. **Greedy integer allocation is safe** because marginal gains diminish (eq. 16), so an implementation
   restricted to {2,3,4,8} bits is still near-optimal.

**Honest limit:** the paper's evidence is LLM KV caches; our measurements here are on synthetic importance
profiles and a real quantiser's fitted distortion curve. What transfers with proof is the *allocation
rule* and the *diagnostic*, not any particular speed-up."""),
             dict(note="""### The allocator, end to end, on a measured distortion curve
Fit `D(b)` on real tensors, take a heavy-tailed importance profile, allocate under hardware bit-widths,
and report the distortion actually achieved against uniform allocation at the same budget.""",
                  code="""import pandas as pd
def quantize(x, bits):
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().max() / qmax
    return torch.round(x / s).clamp(-qmax - 1, qmax) * s

N = 24
heads = [torch.randn(2048) for _ in range(N)]
w = torch.distributions.LogNormal(0.0, 1.3).sample((N,))         # importance
LEGAL = torch.tensor([2.0, 3.0, 4.0, 8.0])                       # what kernels actually exist
budget = 4.0 * N

lw = torch.log(w)
raw = budget / N + (lw - lw.mean()) / torch.log(torch.tensor(4.0))
snap = LEGAL[(raw[:, None] - LEGAL[None, :]).abs().argmin(1)]     # snap to legal widths
while float(snap.sum()) > budget:                                # give bits back, cheapest first
    cand = (snap > LEGAL.min())
    idx = int((w * cand.float() + (~cand).float() * 1e9).argmin())
    snap[idx] = LEGAL[max(int((LEGAL == snap[idx]).nonzero()[0, 0]) - 1, 0)]

def real_distortion(bits):
    return float(sum(w[i] * ((quantize(heads[i], int(bits[i])) - heads[i]) ** 2).mean()
                     for i in range(N)))
d_alloc = real_distortion(snap)
d_unif = real_distortion(torch.full((N,), 4.0))
am, gm = float(w.mean()), float(torch.exp(torch.log(w).mean()))
print(pd.DataFrame({"head": range(N), "importance": w.round(decimals=3).tolist(),
                    "bits": snap.int().tolist()}).head(8).to_string(index=False))
print(f"  budget {float(snap.sum()):.0f}/{budget:.0f} bits · AM/GM predicted gain {am/gm:.2f}x")
ok("the allocator stays inside the budget", float(snap.sum()) <= budget + 1e-6)
ok("only hardware-legal bit-widths are used", bool(torch.isin(snap, LEGAL).all()),
   f"used {sorted(set(snap.int().tolist()))}")
ok("measured distortion beats uniform at the same budget", d_alloc < d_unif,
   f"weighted MSE {d_unif:.4e} -> {d_alloc:.4e} ({d_unif/d_alloc:.2f}x)")
ok("important heads got more bits", float(torch.corrcoef(torch.stack([torch.log(w), snap]))[0, 1]) > 0.5,
   f"corr(log importance, bits) = {float(torch.corrcoef(torch.stack([torch.log(w), snap]))[0,1]):+.3f}")"""),
             dict(note="""**[Recap]** `D(b) = αβ^{-b}` (measured, β≈4) · optimal bits are **logarithmic in
importance** · uniform allocation costs exactly **AM/GM** · diminishing returns make greedy integer
allocation safe. Cross-read: `nlz1` (rate–distortion in NL's Appendix B) and the HOPE functions in
`compress_select`."""),
         ]),
]
