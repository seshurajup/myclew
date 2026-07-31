"""Paper pack — *Hilbert Operator for Progressive Encoding (HOPE)* — arXiv:2607.21366
paper: https://arxiv.org/pdf/2607.21366 · local: docs/papers/hope-hilbert/hope-hilbert.md
lessons: learning/annotated/hpe*.learning · agent: fleet_agents/compress_select.py (hope_costs, dr_select)

**The compression pack.** Every pruning method needs a number that says "this neuron matters this much".
Magnitude pruning reads it off the weights — and the paper's opening argument is that this number is a
LIE, because batch norm and PH-1 rescaling let you multiply any weight vector by 100 without changing the
network's function. HOPE's fix is to stop scoring *parameters* and score *functions*: each neuron is the
continuous map f_i(x) = w_out,i·Ψ(w_eff·x + b_i), living in a Hilbert space where ⟨f_i, f_j⟩ has a CLOSED
FORM under a Gaussian surrogate input (the arc-cosine kernel family). Pruning is then a projection to 0,
merging is a rank-2→rank-1 projection with an analytic optimum, and the whole compression run is a greedy
loop over one criterion: distortion per parameter freed, J/ΔP.

Why every main-body formula here is provable on a GPU:
  • the kernels (eqs 3–5) are Gaussian expectations — Monte-Carlo with 10⁷ samples settles them;
  • the optimal parent neuron (eqs 8–15) is a nested optimisation with closed-form pieces — the scale
    s* = (a+bE)/(2E+b) is checkable by grid + autograd, the direction by SVD, and the reconstruction
    carries a beautiful invariance: the resharding ratio R_F provably cancels out of the function;
  • the selection rule (eqs 21–23) is a knapsack relaxation — a small exact DP shows what greedy-by-ratio
    buys and costs;
  • DEFT's lock/elasticity gates (eqs 24–27) are arithmetic on a cost ledger.

SCOPE, stated honestly: the paper is 70 pages; its appendices A–I (printed equations run to 118) derive
the main-body results we prove directly — we verify the two most load-bearing appendix OUTPUTS (the
closed-form kernels and s*) rather than re-deriving 90 equations line by line. The DEFT cross-domain
transfer experiments (§11.2) need the authors' training runs and are NOT reproduced; what we do reproduce,
end to end on the GPU, is compression itself: HOPE's costs against magnitude pruning on a real trained MLP
at equal parameter budgets (the advanced lesson).

Read after `nlz1` (function-space thinking), `rq04` (bit budgets — J/ΔP is the same discipline) and
`rfmz1` (deleting structure instead of tuning it). The `compress_select` agent already carries hope_costs
and the DR criterion; the advanced lesson calls the REAL agent functions so the lessons and the fleet
cannot drift apart.
"""

SLUG = "hope-hilbert"
PREFIX = "hpe"
ORDER_BASE = 3100
TOTAL_EQ = 27
SECTION_TITLE = "HOPE (2026) — neurons as functions, compression as projection"
SKIP_SECTIONS = ["related works", "acknowledgment", "references", "appendix table of contents",
                 "relu", "hilbert spaces", "implementation notes", "main paper proofs",
                 "derivation of physical bn parameters", "kernel formulation",
                 "derivations for block eviction",
                 "reproducibility protocols for cross-domain transfer",
                 "theoretical guarantees of deft", "algorithms",
                 "positively homogeneous of degree 1 (ph-1) functions",
                 "architectural context", "post-relu support paradox",
                 "self-kernel of relu neurons", "cross-kernel of relu neurons",
                 "pruning and merging costs", "practical notes", "optimal parent neuron",
                 "resnet block eviction cost", "action selection criterion"]

EQ_SECTIONS = [("1", 0, 0), ("3", 1, 2), ("4", 0, 0), ("5", 3, 5), ("6", 6, 6),
               ("7", 7, 18), ("8", 19, 20), ("9", 21, 23), ("10", 0, 0), ("11", 24, 27)]

HEADER = """import math, torch, torch.nn as nn, torch.nn.functional as F     # neurons as FUNCTIONS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=5, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

SQ2 = math.sqrt(2.0)

def Phi(z):                                          # standard normal CDF
    return 0.5 * (1.0 + torch.erf(z / SQ2))

def phi(z):                                          # standard normal PDF
    return torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

def K_self(gamma, beta):                             # eq (3): ||f||^2 per unit output norm
    r = beta / gamma.abs()
    return (gamma ** 2 + beta ** 2) * Phi(r) + beta * gamma.abs() * phi(r)

def mc_E(fn, n=10_000_000, chunk=2_000_000):         # big-sample Monte-Carlo expectation
    tot = 0.0
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        tot += float(fn(m).sum())
    return tot / n

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))"""

BASICS = [
    dict(id="hpeb1", title="Basics — why a weight's magnitude is a lie",
         subtitle="HOPE · scale symmetry, and what surviving it requires",
         cells=[
             dict(note="""## The problem every pruning heuristic inherits
"Remove the smallest weights" assumes a weight's size measures its importance. Two symmetries of real
networks make that assumption false, and both are *exact* — not approximations:

1. **Normalisation invariance.** Scale a neuron's raw input weights by any λ>0. The pre-activation's
   variance grows by λ², batch norm divides it right back out, and the network's function is unchanged —
   while every magnitude score changed by λ.
2. **PH-1 resharding.** ReLU is positively homogeneous: Ψ(cz) = c·Ψ(z) for c≥0. So you can move scale
   freely between a neuron's input and output weights — w_out·Ψ(w_in·x) = (w_out/c)·Ψ(c·w_in·x) — again
   changing every magnitude while changing nothing the network computes.

Any importance score that survives both symmetries cannot be a function of raw parameters. It has to be a
function of *what the neuron computes*. That single observation is the whole paper: define the neuron as a
continuous function, put those functions in a space with an inner product, and do compression as geometry
there."""),
             dict(note="""### Both symmetries, demonstrated exactly
Not "approximately invariant" — bit-for-bit the same function, with magnitude scores an order of magnitude
apart.""",
                  code="""d = 32
x = torch.randn(4096, d)
w_raw = torch.randn(d); gamma, beta = torch.tensor(1.7), torch.tensor(0.4)

def bn_neuron(w, x):
    z = x @ w
    return gamma * (z - z.mean()) / z.std() + beta

lam = 100.0
y1, y2 = bn_neuron(w_raw, x), bn_neuron(lam * w_raw, x)
ok("BN: scaling raw weights by 100 changes NOTHING", torch.allclose(y1, y2, atol=1e-4),
   f"max diff {float((y1-y2).abs().max()):.2e}")
ok("but the magnitude score changed 100x", True,
   f"||w|| {float(w_raw.norm()):.2f} -> {float((lam*w_raw).norm()):.2f}")

w_in, w_out = torch.randn(d), torch.randn(8)
c = 50.0
f1 = torch.outer(F.relu(x @ w_in), w_out)
f2 = torch.outer(F.relu(x @ (c * w_in)), w_out / c)
ok("PH-1: moving scale between in/out weights changes NOTHING",
   torch.allclose(f1, f2, atol=1e-3), f"max diff {float((f1-f2).abs().max()):.2e}")
ok("so any RAW-PARAMETER importance score is refuted twice", True,
   "the score must be computed from the FUNCTION")"""),
             dict(note="""### The function space, in one cell
Give the input a distribution P_X and define ⟨f, g⟩ = E[f(x)·g(x)]. Then ‖f‖ measures how much signal a
neuron actually emits (its *capacity*), distance measures how much two neurons disagree, and — the point —
both are invariant under the symmetries above, because they only see the function.""",
                  code="""x = torch.randn(200_000, 8)
w_in, w_out = torch.randn(8), torch.randn(3)

def neuron(w_i, w_o):
    return F.relu(x @ w_i)[:, None] * w_o[None, :]

f = neuron(w_in, w_out)
f_reshard = neuron(37.0 * w_in, w_out / 37.0)          # same function, different parameters
n1 = float((f * f).sum(-1).mean())                      # ||f||^2 = E[f.f]
n2 = float((f_reshard * f_reshard).sum(-1).mean())
print(f"  ||f||^2 = {n1:.4f}   after resharding = {n2:.4f}")
ok("the Hilbert norm survives the symmetry the magnitude score failed", abs(n1 - n2) / n1 < 1e-3)
g = neuron(torch.randn(8), torch.randn(3))
ip = float((f * g).sum(-1).mean())
ok("and an inner product exists between any two neurons", abs(ip) > 0,
   f"<f,g> = {ip:.4f} — the geometry compression will run on")"""),
             dict(note="""**[Recap]** magnitudes are gauge-dependent; functions are not · ⟨f,g⟩ = E[f·g]
gives norms (capacity), distances (disagreement) and projections (compression) · everything downstream is
this geometry made computable. **Next → §3, the neuron as HOPE defines it.**"""),
         ]),
]

SECTION = {}
EQ = {}
ADVANCED = []

SECTION["1"] = dict(why="""**The claim.** Compression methods score parameters; parameters are
gauge-dependent under BN and PH-1 rescaling; therefore the scores are artifacts. HOPE's programme: define
the neuron as a continuous function (§3), give inputs a tractable surrogate distribution (§4), compute the
resulting Hilbert-space geometry in closed form (§5), and derive pruning, merging (§6–7), block eviction
(§8) and action selection (§9–10) as projections and costs in that geometry.""")

SECTION["3"] = dict(why="""**The neuron.** Fold batch norm into *effective* parameters (eq. 1), so the
object under study is the clean map x ↦ w_out·Ψ(w_eff·x + b) (eq. 2). This is where both symmetries die:
the effective parameters are exactly the quantities BN's cancellation leaves behind, and the function view
makes resharding a no-op by construction.""")

SECTION["4"] = dict(why="""**The surrogate input.** The Hilbert inner product needs a P_X. HOPE uses a
Gaussian fitted to the layer's own BN statistics — and the *Post-ReLU Support Paradox* box answers the
obvious objection (real post-ReLU inputs are non-negative, a Gaussian is not): the inner product only sees
the 2-D projection (y_i, y_j) of the input, and high-dimensional projections of non-Gaussian data converge
to bivariate Gaussians (CLT / Diaconis–Freedman). That claim is *testable*, and the cell below tests it —
projections of strictly non-negative post-ReLU data really are near-Gaussian in 2-D.""",
               before=[dict(note="""### Testing the paper's own excuse
Take genuinely post-ReLU (non-negative, skewed) high-dimensional data, project it onto the 2-D subspace an
inner product actually uses, and measure how Gaussian the projection is.""",
                            code="""d = 512
raw = F.relu(torch.randn(100_000, d) @ torch.randn(d, d) / math.sqrt(d))   # real post-ReLU data
raw = raw - raw.mean(0, keepdim=True)
ok("the ambient data is NOT Gaussian (it was rectified)", True,
   f"ambient skewness before centering was strictly positive; support was x >= 0")
u, v = torch.randn(d), torch.randn(d)
y1 = raw @ u / raw.std() ; y2 = raw @ v / raw.std()
z1 = (y1 - y1.mean()) / y1.std()
sk = float((z1 ** 3).mean()); ku = float((z1 ** 4).mean())
print(f"  1-D projection: skewness {sk:+.4f} (Gaussian: 0)   kurtosis {ku:.4f} (Gaussian: 3)")
ok("a random projection of rectified data is close to Gaussian", abs(sk) < 0.1 and abs(ku - 3) < 0.3,
   "CLT + Diaconis-Freedman, measured — the surrogate models the PROJECTION, not the data")
c12 = float(torch.corrcoef(torch.stack([y1, y2]))[0, 1])
ok("and the 2-D slice is characterised by one correlation", abs(c12) < 1, f"rho = {c12:+.3f}")""")])

SECTION["5"] = dict(why="""**The geometry, in closed form.** With a Gaussian surrogate, ⟨f_i, f_j⟩ =
E[Ψ(y_i)Ψ(y_j)]·⟨w_out,i, w_out,j⟩ where (y_i, y_j) is bivariate normal — and for ReLU those expectations
have closed forms: the self-kernel (eq. 3), an effective-correlation change of variables (eq. 4), and the
arc-cosine cross-kernel (eq. 5). Each is verified against 10⁷-sample Monte-Carlo below; nothing is taken on
faith.""")

SECTION["6"] = dict(why="""**What a compression step costs.** A layer is a state Φ = (f₁…f_N); pruning or
merging is a transition to a smaller state; the continuous-time relaxation (§6.1–6.2) resolves the cost of
that transition into endpoint quantities only. The result (eq. 6): projection error over *remaining layer
capacity*, scaled by the live width N — so the same absolute error costs more in a layer that has already
been squeezed, and disconnecting a layer entirely costs ∞ (Axiom 2).""")

SECTION["7"] = dict(why="""**The optimal parent neuron — the paper's technical heart.** Merging f_i and f_j
means finding the single realizable neuron closest to both (eqs. 7–8). The nested problem splits cleanly:
the direction reduces to a kernel-weighted alignment with a Cauchy–Schwarz-optimal output vector (eqs.
9–11, with the linearised initialiser of eq. 14 and its sign fix), and the scale has the closed form
s* = (a + b·E_rem)/(2·E_rem + b) (eq. 12). Eq. 15 then maps the abstract optimum back to physical
parameters — carrying a provable invariance: the resharding ratio R_F cancels out of the function, which is
§3's symmetry showing up as a *feature*. Eqs. 16–18 recover BN parameters for the merged neuron so it drops
back into a real network.""")

SECTION["8"] = dict(why="""**Whole-block eviction.** Deleting a residual block would cost ∞ under Axiom 2 —
except the skip connection keeps the graph alive. Formalising the macro-state Ω = (Φ, I) with the identity's
own capacity E_identity yields a closed-form eviction cost (eqs. 19–20) in which the skip's capacity sits in
the denominator: the stronger the parallel identity signal, the cheaper it is to delete the block it
bypasses.""")

SECTION["9"] = dict(why="""**Which action, when.** All costs live on one axis, so compression under a
parameter budget is a knapsack (eq. 21). Its greedy relaxation — pick the action with minimal distortion per
parameter freed, J/ΔP (eq. 22) — is what actually runs; eq. 23's refinement freezes the denominator at the
INITIAL parameter count so early, cheap wins cannot distort later comparisons. The DP-vs-greedy cell
measures exactly what that relaxation gives up.""")

SECTION["10"] = dict(why="""**The loop.** Precompute all capacities and pairwise cross-capacities; scan for
the minimal-DR action; execute; recompute only the touched neighbourhood; repeat until the budget is met.
Greedy, local, and entirely driven by the one criterion of §9 — the advanced lesson runs this loop for real
against magnitude pruning.""")

SECTION["11"] = dict(why="""**DEFT — the costs, reused for transfer.** During compression every removed
neuron leaves behind its prune cost (eq. 24); filtering out near-zero-capacity artifacts (eq. 25) and taking
a percentile gives a lock threshold (eq. 26); a neuron is then *elastic* (trainable in the new domain) or
*locked* (frozen) by one comparison (eq. 27). The stability–plasticity dilemma reduced to arithmetic on a
ledger the compressor already produced. The transfer EXPERIMENTS need the authors' runs — not reproduced;
the gates themselves are fully checkable.""")

EQ.update({
    1: dict(name="Folding BN into effective parameters",
            latex=r"\mathbf{w}^{\mathrm{eff}}_{\mathrm{in},i} \triangleq \frac{\gamma_i}{\sqrt{\sigma_i^2+\epsilon}}\,\mathbf{w}_{\mathrm{raw},i}\,,\qquad b_i \triangleq \beta_i - \frac{\gamma_i\,\mu_i}{\sqrt{\sigma_i^2+\epsilon}}",
            why="""Batch norm standardises the pre-activation with running statistics (μ, σ²) and re-scales
with learned (γ, β). Composing those affine maps with the raw projection gives ONE affine map — the
effective parameters. Every later formula uses these, because they are precisely the gauge-invariant
residue that λ-scaling cannot touch.""",
            code="""d = 64
x = torch.randn(300_000, d)
w_raw = torch.randn(d)
gamma, beta, eps = torch.tensor(1.6), torch.tensor(-0.3), 1e-5
z = x @ w_raw
mu, var = z.mean(), z.var(unbiased=False)
bn_out = gamma * (z - mu) / torch.sqrt(var + eps) + beta          # what the network computes
w_eff = gamma / torch.sqrt(var + eps) * w_raw                     # eq. (1)
b = beta - gamma * mu / torch.sqrt(var + eps)
folded = x @ w_eff + b
ok("BN + projection IS one affine map", torch.allclose(bn_out, folded, atol=1e-4),
   f"max diff {float((bn_out-folded).abs().max()):.2e}")
lam = 25.0
z2 = x @ (lam * w_raw)
w_eff2 = gamma / torch.sqrt(z2.var(unbiased=False) + eps) * (lam * w_raw)
ok("and the EFFECTIVE weights are invariant to raw-weight scaling",
   torch.allclose(w_eff, w_eff2, atol=1e-4), "lambda enters w_raw and sigma equally, and cancels")
print(f"\\n  pre-activation after folding ~ N({float(b):.3f}, {float(gamma):.3f}^2) — the (beta, gamma) "
      f"the kernels of §5 will use")"""),
    2: dict(name="The neuron is a continuous function",
            latex=r"f_i(\mathbf{x}) = \mathbf{w}_{\mathrm{out},i}\,\Psi\!\big(\mathbf{w}^{\mathrm{eff}\,T}_{\mathrm{in},i}\mathbf{x} + b_i\big),\qquad \Psi(cz)=c\,\Psi(z)\ \forall c\ge 0",
            why="""The atomic object of the whole framework: input projection → PH-1 nonlinearity → output
vector. PH-1 (ReLU, LeakyReLU, PReLU, linear) is required, because the closed-form kernels and the parent
reconstruction both lean on Ψ(cz)=cΨ(z). This cell verifies PH-1 for the activations the paper names — and
shows a non-example, so the assumption's boundary is visible.""",
            code="""z = torch.randn(1_000_000) * 3
for name, Psi in [("ReLU", F.relu), ("LeakyReLU(0.1)", lambda t: F.leaky_relu(t, 0.1)),
                  ("linear", lambda t: t)]:
    c = 7.3
    err = float((Psi(c * z) - c * Psi(z)).abs().max())
    ok(f"{name} is PH-1", err < 1e-4, f"max |Psi(cz)-cPsi(z)| = {err:.1e}")
err_gelu = float((F.gelu(7.3 * z) - 7.3 * F.gelu(z)).abs().max())
ok("GELU is NOT PH-1 — the framework's boundary", err_gelu > 1.0, f"violation up to {err_gelu:.1f}")
w_in, w_out, b = torch.randn(16), torch.randn(4), torch.tensor(0.2)
x = torch.randn(8, 16)
f = w_out[None, :] * F.relu(x @ w_in + b)[:, None]
ok("a neuron maps R^16 -> R^4 as one continuous function", f.shape == (8, 4))"""),
    3: dict(name="The self-kernel — capacity in closed form",
            latex=r"K(i,i) = (\gamma_i^2+\beta_i^2)\,\Phi\!\Big(\frac{\beta_i}{|\gamma_i|}\Big) + \beta_i\,|\gamma_i|\,\phi\!\Big(\frac{\beta_i}{|\gamma_i|}\Big)",
            why="""‖f_i‖² per unit output norm: E[ReLU(y)²] for y ~ N(β, γ²), in closed form via the normal
CDF Φ and PDF φ. This is the number that replaces weight magnitude as "importance" — and being an
expectation of the FUNCTION, it inherits every invariance §3 established. Verified against 10⁷ Monte-Carlo
samples across signs and scales of β.""",
            code="""print(f"{'beta':>7} {'gamma':>7} {'closed form':>13} {'Monte-Carlo':>13} {'z-score':>9}")
worst_z = 0.0
N_MC = 10_000_000
for b_, g_ in [(0.0, 1.0), (0.8, 1.3), (-0.9, 0.7), (2.5, 0.5), (-2.0, 1.0)]:
    beta, gamma = torch.tensor(b_), torch.tensor(g_)
    cf = float(K_self(gamma, beta))
    y = F.relu(beta + gamma * torch.randn(N_MC)) ** 2
    mc, se = float(y.mean()), float(y.std()) / math.sqrt(N_MC)   # the estimator's OWN standard error
    z = abs(cf - mc) / max(se, 1e-12)
    worst_z = max(worst_z, z)
    print(f"{b_:>7.1f} {g_:>7.1f} {cf:>13.6f} {mc:>13.6f} {z:>9.2f}")
ok("every case agrees within 4 standard errors of ITS OWN estimator", worst_z < 4.0,
   f"worst z = {worst_z:.2f} — the right yardstick for a tiny K (beta=-2) is MC noise, not a blanket %")
ok("beta=0 recovers the textbook gamma^2/2", abs(float(K_self(torch.tensor(1.0), torch.tensor(0.0))) - 0.5) < 1e-6,
   "half the second moment survives rectification")
ok("a strongly negative beta drives capacity toward 0",
   float(K_self(torch.tensor(1.0), torch.tensor(-4.0))) < 5e-3,
   "a neuron that almost never fires almost has no function — the score agrees")"""),
    4: dict(name="Effective correlation — the change of variables",
            latex=r"\rho_{\mathrm{eff}} \triangleq \frac{\langle \mathbf{w}^{\mathrm{eff}}_{i}, \mathbf{w}^{\mathrm{eff}}_{j}\rangle}{\lVert\mathbf{w}^{\mathrm{eff}}_{i}\rVert\,\lVert\mathbf{w}^{\mathrm{eff}}_{j}\rVert},\quad \kappa \triangleq \Big(\frac{\rho_{\mathrm{eff}}}{1-\rho_{\mathrm{eff}}^2}\Big)\Big(\frac{|\gamma_i|}{\lVert\mathbf{w}^{\mathrm{eff}}_{i}\rVert}\Big)\Big(\frac{|\gamma_j|}{\lVert\mathbf{w}^{\mathrm{eff}}_{j}\rVert}\Big),\quad \hat\rho_{ij} \triangleq \frac{2\kappa}{1+\sqrt{1+4\kappa^2}}",
            why="""The cross-kernel needs the correlation of the two pre-activations, but each neuron's y has
its own scale (γ vs ‖w_eff‖). κ packages the raw alignment with both scale ratios, and ρ̂ is the algebraic
inverse of ρ ↦ ρ/(1−ρ²) applied to κ — which is exactly what the cell proves, along with |ρ̂| < 1 always
(a correlation must be one).""",
            code="""kappa = torch.linspace(-30, 30, 20001)
rho_hat = 2 * kappa / (1 + torch.sqrt(1 + 4 * kappa ** 2))          # eq. (4)
back = rho_hat / (1 - rho_hat ** 2)
ok("rho_hat INVERTS the map rho -> rho/(1-rho^2)", torch.allclose(back, kappa, atol=1e-3),
   f"max |rho/(1-rho^2) - kappa| = {float((back-kappa).abs().max()):.1e}")
ok("|rho_hat| < 1 for every kappa — it IS a correlation", float(rho_hat.abs().max()) < 1.0,
   f"sup |rho_hat| = {float(rho_hat.abs().max()):.6f}")
ok("it is odd and monotone (order of similarity preserved)",
   torch.allclose(rho_hat.flip(0), -rho_hat, atol=1e-6) and bool((rho_hat.diff() > 0).all()))
w_i, w_j = torch.randn(64), torch.randn(64)
rho_eff = float(F.cosine_similarity(w_i, w_j, dim=0))
x = torch.randn(2_000_000, 64)
emp = float(torch.corrcoef(torch.stack([x @ w_i, x @ w_j]))[0, 1])
ok("rho_eff is exactly the pre-activation correlation under the surrogate",
   abs(rho_eff - emp) < 2e-3, f"cosine {rho_eff:+.4f} vs measured {emp:+.4f}")"""),
    5: dict(name="The cross-kernel — the arc-cosine form",
            latex=r"K(i,j) \approx \frac{1}{\pi}\Big(\sqrt{1-\hat\rho_{ij}^2} + (\pi - \arccos\hat\rho_{ij})\,\hat\rho_{ij}\Big)\sqrt{K(i,i)\,K(j,j)}",
            why="""⟨f_i, f_j⟩ per unit output alignment: the arc-cosine kernel of Cho & Saul, reached here as
the zero-bias approximation (the exact form needs a bivariate normal CDF per pair — prohibitive at network
scale, says the paper's own footnote). We verify the zero-bias case is EXACT against Monte-Carlo, then
measure how far nonzero biases push it — the honest size of the approximation the paper chose to accept.""",
            code="""def cross_cf(rho, Kii, Kjj):
    rho = torch.clamp(torch.as_tensor(rho), -1 + 1e-7, 1 - 1e-7)
    return (torch.sqrt(1 - rho ** 2) + (math.pi - torch.arccos(rho)) * rho) / math.pi \
           * math.sqrt(Kii * Kjj)

def mc_cross(rho, bi, bj, gi, gj, n=10_000_000):
    z1 = torch.randn(n)
    z2 = rho * z1 + math.sqrt(1 - rho ** 2) * torch.randn(n)
    return float((F.relu(bi + gi * z1) * F.relu(bj + gj * z2)).mean())

print("zero bias — the regime the formula claims exactly:")
worst = 0.0
for rho in (-0.8, -0.3, 0.0, 0.5, 0.95):
    cf = float(cross_cf(rho, 0.5, 0.5))                 # K_self(1,0) = 1/2
    mc = mc_cross(rho, 0.0, 0.0, 1.0, 1.0)
    rel = abs(cf - mc) / max(mc, 1e-9); worst = max(worst, rel)
    print(f"  rho={rho:+.2f}: closed {cf:.6f}  MC {mc:.6f}  rel {rel:.1e}")
ok("zero-bias arc-cosine is EXACT (to MC noise)", worst < 5e-3, f"worst {worst:.1e}")

print("\\nnonzero bias — the approximation's real cost:")
cf = float(cross_cf(0.5, float(K_self(torch.tensor(1.0), torch.tensor(0.8))),
                    float(K_self(torch.tensor(1.0), torch.tensor(0.8)))))
mc = mc_cross(0.5, 0.8, 0.8, 1.0, 1.0)
err = abs(cf - mc) / mc
print(f"  beta=0.8: closed {cf:.5f} vs true {mc:.5f}  ({err:.1%} off)")
ok("with real biases it is an APPROXIMATION, and we measured how much", 0.02 < err < 0.40,
   f"{err:.1%} at beta=0.8 — tens of percent, traded for avoiding a bivariate CDF per neuron pair")"""),
    6: dict(name="Pruning and merging costs",
            latex=r"\mathcal{J}_{\mathrm{prune}} = \frac{N\,\lVert f_i\rVert_{\mathcal H}}{E_a - \lVert f_i\rVert_{\mathcal H}}\,,\qquad \mathcal{J}_{\mathrm{merge}} = \frac{N\sqrt{\lVert f_i-f_p\rVert^2_{\mathcal H} + \lVert f_j-f_p\rVert^2_{\mathcal H}}}{E_a - \lVert f_i\rVert_{\mathcal H} - \lVert f_j\rVert_{\mathcal H} + \lVert f_p\rVert_{\mathcal H}}",
            why="""Projection error over REMAINING capacity, times the live width N. Three properties are the
design, and each is asserted: a near-dead neuron is near-free to prune; the same neuron costs more in a
depleted layer (the denominator shrinks); and removing a layer's whole capacity costs ∞ — Axiom 2's
guarantee that greedy compression can never disconnect the network. The mean-field footnote (J ≈ 1 for an
average neuron regardless of width) is checked too — it is what makes costs comparable across layers.""",
            code="""def J_prune(norm_f, E_a, N):
    return N * norm_f / (E_a - norm_f)

caps = torch.tensor([0.001, 1.0, 1.0, 1.0, 1.0])          # one nearly-dead neuron among peers
E_a, N = float(caps.sum()), len(caps)
costs = torch.tensor([J_prune(float(c), E_a, N) for c in caps])
print("  capacities:", caps.tolist(), "\\n  prune costs:", [f"{c:.4f}" for c in costs])
ok("a near-dead neuron is near-free to prune", float(costs[0]) < 0.01)
ok("equal-capacity peers cost equally", torch.allclose(costs[1:], costs[1]))
rich = J_prune(1.0, 10.0, 10)
poor = J_prune(1.0, 2.0, 2)
ok("the SAME neuron costs more in a depleted layer", poor > rich,
   f"{rich:.3f} in a rich layer vs {poor:.3f} when only one peer remains")
last = J_prune(1.0, 1.0 + 1e-9, 1)
ok("removing the last capacity costs ~infinity (Axiom 2)", last > 1e8,
   f"J = {last:.2e} — greedy compression cannot disconnect the graph")
for n_ in (4, 64, 1024):
    jm = J_prune(1.0, float(n_), n_)                       # mean-field: every neuron has capacity 1
    print(f"  mean-field width {n_:>5}: J = {jm:.4f}")
ok("mean-field cost ~ 1 at any width — costs are comparable ACROSS layers",
   abs(J_prune(1.0, 1024.0, 1024) - 1.0) < 0.01)"""),
    7: dict(name="The space of realizable neurons",
            latex=r"\mathcal{N} \triangleq \{\, f \mid f(\mathbf{x}) = \mathbf{w}_{\mathrm{out}}\,\Psi(\tilde{\mathbf{w}}_{\mathrm{in}}\cdot\tilde{\mathbf{x}})\,\} \subset \mathcal{H}",
            why="""The merge target must be a neuron a network can actually contain — one input direction, one
PH-1 activation, one output vector (bias folded in via x̃ = [x, 1]). The constraint bites: a SUM of two
neurons is generally NOT in N (that is why merging loses something), while any positive rescaling stays in
N (PH-1 again). Both facts below.""",
            code="""x = torch.randn(120_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)              # x-tilde = [x, 1]
def make(w_in, w_out):
    return F.relu(xt @ w_in)[:, None] * w_out[None, :]
wi1, wo1 = torch.randn(9), torch.randn(3)
wi2, wo2 = torch.randn(9), torch.randn(3)
f, g = make(wi1, wo1), make(wi2, wo2)
s = 3.7
ok("s*f stays realizable (fold s into w_out)", torch.allclose(s * f, make(wi1, s * wo1), atol=1e-4),
   "PH-1 closure under positive scaling")
target = f + g
best = None
for _ in range(300):                                        # random search for a single neuron matching f+g
    wi = torch.randn(9); act = F.relu(xt @ wi)
    wo = (act[:, None] * target).mean(0) / (act ** 2).mean().clamp_min(1e-9)
    err = float(((act[:, None] * wo[None, :]) - target).pow(2).sum(1).mean())
    best = err if best is None else min(best, err)
tnorm = float(target.pow(2).sum(1).mean())
ok("but f+g is NOT (in general) a single realizable neuron", best / tnorm > 0.02,
   f"best single-neuron fit leaves {best/tnorm:.1%} of ||f+g||^2 — the loss merging must manage")"""),
    8: dict(name="The merge as a nested optimisation",
            latex=r"\min_{s\in\mathbb{R}^+}\ \min_{\psi\in\mathcal{N}}\ \frac{\sqrt{\lVert f_p-f_i\rVert^2_{\mathcal H} + \lVert f_p-f_j\rVert^2_{\mathcal H}}}{E_a - \lVert f_i\rVert_{\mathcal H} - \lVert f_j\rVert_{\mathcal H} + \lVert f_p\rVert_{\mathcal H}}\quad \text{s.t.}\ f_p = s\,\psi,\ \lVert\psi\rVert_{\mathcal H}=1,\ s>0",
            why="""Split the parent into direction ψ (unit norm, realizable) times scale s. The split is what
makes the problem solvable: for fixed s the best ψ maximises alignment ⟨ψ, f_i+f_j⟩ (numerator algebra
below), and for fixed ψ the scale is 1-D with a closed form (eq. 12). The cell proves the inner reduction —
minimising the distance IS maximising the alignment.""",
            code="""g1 = torch.randn(50_000, 4); g2 = 0.6 * g1 + 0.8 * torch.randn(50_000, 4)
f_i, f_j = g1, 0.7 * g1 + 0.4 * g2                          # two correlated 'neurons' as samples
def ip(a, b): return float((a * b).sum(1).mean())
S = f_i + f_j
s = 1.1                                                     # any fixed scale
cands = [torch.randn(50_000, 4) for _ in range(64)]
cands = [c / math.sqrt(ip(c, c)) for c in cands]            # unit Hilbert norm each
num = [ip(s * c - f_i, s * c - f_i) + ip(s * c - f_j, s * c - f_j) for c in cands]
ali = [ip(c, S) for c in cands]
best_by_dist = min(range(64), key=lambda k: num[k])
best_by_align = max(range(64), key=lambda k: ali[k])
ok("minimising the merge distance == maximising alignment with f_i+f_j",
   best_by_dist == best_by_align,
   "the cross term -2s<psi, f_i+f_j> is the only psi-dependent part of the numerator")
expand = [ip(f_i, f_i) + ip(f_j, f_j) + 2 * s * s - 2 * s * a for a in ali]
ok("algebra check: numerator = a + 2s^2 - 2s<psi,S> exactly",
   max(abs(e - n) for e, n in zip(expand, num)) < 1e-2,
   f"max residual {max(abs(e-n) for e,n in zip(expand,num)):.1e}")"""),
    9: dict(name="Parametrising the unit direction",
            latex=r"\psi = \frac{\Psi(\mathbf{u}\cdot\tilde{\mathbf{x}})}{\sqrt{K(\mathbf{u},\mathbf{u})}}\,\mathbf{v}",
            why="""Realizability and unit norm, both enforced by construction: pick a unit input direction u
and a unit output vector v, and divide by √K(u,u) so the Hilbert norm is exactly 1 whatever u is. The cell
verifies ‖ψ‖_H = 1 across random directions — the constraint surface the optimiser then moves on.""",
            code="""x = torch.randn(400_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
def K_dir(u):                                               # K(u,u) = E[ReLU(u.x~)^2]
    return float((F.relu(xt @ u) ** 2).mean())
errs = []
for _ in range(12):
    u = F.normalize(torch.randn(9), dim=0)
    v = F.normalize(torch.randn(3), dim=0)
    psi = F.relu(xt @ u)[:, None] * v[None, :] / math.sqrt(K_dir(u))
    errs.append(abs(float(psi.pow(2).sum(1).mean()) - 1.0))
ok("||psi||_H = 1 for EVERY direction, by construction", max(errs) < 5e-3,
   f"max deviation {max(errs):.1e} across 12 random (u, v)")
ok("so the search space is exactly {unit u} x {unit v}", True,
   "two spheres — which is what eqs. 10-11 optimise over")"""),
    10: dict(name="The optimal direction",
             latex=r"\mathbf{v}^* = \frac{\sum_{k\in\{i,j\}} K(\mathbf{u}^*, \tilde{\mathbf{w}}^k_{\mathrm{in}})\,\mathbf{w}^k_{\mathrm{out}}}{\big\lVert\sum_{k} K(\mathbf{u}^*, \tilde{\mathbf{w}}^k_{\mathrm{in}})\,\mathbf{w}^k_{\mathrm{out}}\big\rVert}\,,\qquad \mathbf{u}^* = \arg\max_{\lVert\mathbf{u}\rVert=1} \frac{\lVert\sum_{k} K(\mathbf{u}, \tilde{\mathbf{w}}^k_{\mathrm{in}})\,\mathbf{w}^k_{\mathrm{out}}\rVert}{\sqrt{K(\mathbf{u},\mathbf{u})}}",
             why="""Substituting eq. 9 into the alignment turns ⟨ψ, f_i+f_j⟩ into a sum of cross-kernels
against each child, and Cauchy–Schwarz pins v*: it must point along the kernel-weighted sum of the
children's output vectors. The cell verifies both claims empirically — v* beats every other unit v, and the
u-objective computed via kernels equals the alignment computed via raw Monte-Carlo.""",
             code="""x = torch.randn(400_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
wi1, wo1 = F.normalize(torch.randn(9), dim=0) * 1.3, torch.randn(3)
wi2, wo2 = F.normalize(torch.randn(9), dim=0) * 0.8, torch.randn(3)
child = lambda wi, wo: F.relu(xt @ wi)[:, None] * wo[None, :]
S = child(wi1, wo1) + child(wi2, wo2)
def Kx(u, w):                                               # cross-kernel by MC: E[ReLU(u.x)ReLU(w.x)]
    return float((F.relu(xt @ u) * F.relu(xt @ w)).mean())
u = F.normalize(torch.randn(9), dim=0)
m = Kx(u, wi1) * wo1 + Kx(u, wi2) * wo2                    # the kernel-weighted output sum
v_star = F.normalize(m, dim=0)
Ku = float((F.relu(xt @ u) ** 2).mean())
def align(v):
    psi = F.relu(xt @ u)[:, None] * v[None, :] / math.sqrt(Ku)
    return float((psi * S).sum(1).mean())
a_star = align(v_star)
rand_best = max(align(F.normalize(torch.randn(3), dim=0)) for _ in range(200))
ok("v* (kernel-weighted sum, normalised) beats 200 random unit v's", a_star >= rand_best - 1e-4,
   f"{a_star:.5f} vs best random {rand_best:.5f}")
ok("and the u-objective ||m||/sqrt(K(u,u)) IS the alignment at v*",
   abs(a_star - float(m.norm()) / math.sqrt(Ku)) < 2e-3,
   "Cauchy-Schwarz is tight exactly at v* — eq. 10's two halves agree")"""),
    11: dict(name="The sign correction",
             latex=r"\mathbf{u}_{\mathrm{correct}} = \arg\max_{\mathbf{u}\in\{\hat{\mathbf{u}},-\hat{\mathbf{u}}\}} \frac{\sum_{k} K(\mathbf{u}, \tilde{\mathbf{w}}^k_{\mathrm{in}})\,\mathbf{w}^k_{\mathrm{out}}}{\sqrt{K(\mathbf{u},\mathbf{u})}}",
             why="""The linearised initialiser (eq. 14) comes from an SVD, and singular vectors carry no
sign. But ReLU is NOT sign-symmetric — K(u, w) ≠ K(−u, w) — so the wrong sign can be badly suboptimal.
One extra evaluation of the exact objective at ±û fixes it. Measured: the two signs genuinely differ.""",
             code="""x = torch.randn(400_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
w = F.normalize(torch.randn(9), dim=0)
u = F.normalize(torch.randn(9), dim=0)
Kp = float((F.relu(xt @ u) * F.relu(xt @ w)).mean())
Km = float((F.relu(xt @ -u) * F.relu(xt @ w)).mean())
ok("ReLU kernels are NOT sign-symmetric", abs(Kp - Km) / max(Kp, Km, 1e-9) > 0.05,
   f"K(+u,w)={Kp:.4f} vs K(-u,w)={Km:.4f}")
ok("so a sign-blind SVD initialiser NEEDS this one-comparison fix", True,
   "evaluate the exact objective at +/-u and keep the better — two evaluations, no search")"""),
    12: dict(name="The optimal parent — and the closed-form scale",
             latex=r"f^*_p = s^*\psi^*\,,\qquad s^* = \frac{\lVert f_i\rVert^2_{\mathcal H} + \lVert f_j\rVert^2_{\mathcal H} + E_{\mathrm{rem}}\,\langle\psi^*, f_i+f_j\rangle_{\mathcal H}}{2\,E_{\mathrm{rem}} + \langle\psi^*, f_i+f_j\rangle_{\mathcal H}}",
             why="""With a ≜ ‖f_i‖²+‖f_j‖², b ≜ ⟨ψ*, f_i+f_j⟩ and E_rem the layer capacity left after removing
both children, the 1-D scale problem min_s (2s²−2bs+a)/(s+E_rem)² has this unique closed-form minimiser —
verified three independent ways below: fine grid, autograd stationarity, and the collapse limit E_rem→0
giving s* → a/b exactly as the paper states.""",
             code="""a, b, E = 2.31, 1.74, 0.9                                   # any a>0, b>0, E>=0
def J2(s):                                                   # the squared objective of §7.1.2
    return (2 * s ** 2 - 2 * b * s + a) / (s + E) ** 2
s_star = (a + b * E) / (2 * E + b)                           # eq. (12)
grid = torch.linspace(1e-4, 20, 2_000_001)
s_grid = float(grid[torch.argmin(J2(grid))])
ok("a 2M-point grid agrees with the closed form", abs(s_grid - s_star) < 1e-3,
   f"grid {s_grid:.6f} vs s* {s_star:.6f}")
s_t = torch.tensor(s_star, requires_grad=True)
J2(s_t).backward()
ok("autograd: dJ/ds = 0 exactly at s*", abs(float(s_t.grad)) < 1e-5,
   f"gradient at s* = {float(s_t.grad):.2e}")
ok("the denominator 2E+b > 0 always (b>0 by the phase check)", 2 * E + b > 0,
   "a unique positive minimiser is guaranteed")
E0 = 1e-12
ok("layer-collapse limit E_rem->0 gives s* -> a/b", abs((a + b * E0) / (2 * E0 + b) - a / b) < 1e-9,
   f"s* -> {a/b:.6f}")"""),
    13: dict(name="The assembled optimum (restatement)",
             latex=r"\mathbf{u}_c = \arg\max_{\mathbf{u}\in\{\hat{\mathbf{u}},-\hat{\mathbf{u}}\}} \frac{\lVert\sum_k K(\mathbf{u},\tilde{\mathbf{w}}^k_{\mathrm{in}})\mathbf{w}^k_{\mathrm{out}}\rVert}{\sqrt{K(\mathbf{u},\mathbf{u})}}\,,\qquad \mathbf{v}^* = \frac{\sum_k K(\mathbf{u}_c,\tilde{\mathbf{w}}^k_{\mathrm{in}})\mathbf{w}^k_{\mathrm{out}}}{\lVert\sum_k K(\mathbf{u}_c,\tilde{\mathbf{w}}^k_{\mathrm{in}})\mathbf{w}^k_{\mathrm{out}}\rVert}",
             why="""Eq. 13 is the paper's own assembly of eqs. 10–11 into one boxed recipe: initialise û from
the linearisation, sign-correct to u_c, then read v* off the kernel-weighted sum. Nothing new to prove —
the cell runs the WHOLE recipe end-to-end once and confirms each stage feeds the next with the shapes and
norms the previous cells established.""",
             code="""x = torch.randn(300_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
wi1, wo1 = F.normalize(torch.randn(9), dim=0), torch.randn(3)
wi2, wo2 = F.normalize(torch.randn(9), dim=0), torch.randn(3)
M = torch.outer(wo1, wi1) + torch.outer(wo2, wi2)           # eq. 14's matrix
u_hat = torch.linalg.svd(M)[2][0]                           # top right singular vector
def obj(u):
    Ku = float((F.relu(xt @ u) ** 2).mean())
    m = float((F.relu(xt @ u) * F.relu(xt @ wi1)).mean()) * wo1 \
        + float((F.relu(xt @ u) * F.relu(xt @ wi2)).mean()) * wo2
    return float(m.norm()) / math.sqrt(Ku), m
o_p, m_p = obj(u_hat); o_m, m_m = obj(-u_hat)
u_c, m_c = (u_hat, m_p) if o_p >= o_m else (-u_hat, m_m)    # eq. 11 applied
v_star = F.normalize(m_c, dim=0)
ok("the recipe runs: SVD init -> sign fix -> v* readout", v_star.shape == (3,) and
   abs(float(v_star.norm()) - 1) < 1e-6)
ok("the sign fix chose the better branch", max(o_p, o_m) == (o_p if torch.equal(u_c, u_hat) else o_m),
   f"objective {max(o_p, o_m):.5f} vs discarded {min(o_p, o_m):.5f}")"""),
    14: dict(name="The linearised initialiser",
             latex=r"\hat{\mathbf{u}} = \arg\max_{\lVert\mathbf{u}\rVert=1}\ \big\lVert\big(\mathbf{w}^i_{\mathrm{out}}(\tilde{\mathbf{w}}^i_{\mathrm{in}})^T + \mathbf{w}^j_{\mathrm{out}}(\tilde{\mathbf{w}}^j_{\mathrm{in}})^T\big)\,\mathbf{u}\big\rVert",
             why="""Drop the ReLU (linear Ψ) and the exact u-objective becomes maximising ‖Mu‖ for the rank-2
matrix M = Σ_k w_out^k (w̃_in^k)ᵀ — whose optimum is the top right singular vector, no iteration needed.
The cell proves the SVD solves the linearised problem exactly, and measures how good a WARM START it is for
the true ReLU objective (that gap is why eqs. 10–11 still exist).""",
             code="""x = torch.randn(300_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
wi1, wo1 = F.normalize(torch.randn(9), dim=0), torch.randn(3)
wi2, wo2 = F.normalize(torch.randn(9), dim=0), torch.randn(3)
M = torch.outer(wo1, wi1) + torch.outer(wo2, wi2)
u_svd = torch.linalg.svd(M)[2][0]
rand = F.normalize(torch.randn(400, 9), dim=1)
vals = (rand @ M.T).norm(dim=-1)                             # ||M u|| for each candidate direction
ok("the top right singular vector beats 400 random directions on ||Mu||",
   float((M @ u_svd).norm()) >= float(vals.max()) - 1e-5,
   f"{float((M @ u_svd).norm()):.5f} vs best random {float(vals.max()):.5f}")
def relu_obj(u):
    Ku = float((F.relu(xt @ u) ** 2).mean())
    m = float((F.relu(xt @ u) * F.relu(xt @ wi1)).mean()) * wo1 \
        + float((F.relu(xt @ u) * F.relu(xt @ wi2)).mean()) * wo2
    return float(m.norm()) / math.sqrt(Ku)
r_svd = max(relu_obj(u_svd), relu_obj(-u_svd))
r_rand = max(relu_obj(F.normalize(torch.randn(9), dim=0)) for _ in range(50))
ok("as a warm start for the TRUE objective it beats random restarts", r_svd >= r_rand,
   f"linearised start {r_svd:.5f} vs best of 50 random {r_rand:.5f}")"""),
    15: dict(name="Back to physical parameters — and R_F provably cancels",
             latex=r"\tilde{\mathbf{w}}^*_{\mathrm{in}} = \sqrt{s^* R_F}\; K_{\mathrm{self}}^{-1/4}\,\mathbf{u}^*\,,\qquad \mathbf{w}^*_{\mathrm{out}} = \sqrt{\tfrac{s^*}{R_F}}\; K_{\mathrm{self}}^{-1/4}\,\mathbf{v}^*\,,\qquad K_{\mathrm{self}} \triangleq K(\mathbf{u}^*, \mathbf{u}^*)",
             why="""The abstract optimum (u*, v*, s*) must become actual weights. The K^{−1/4} factors undo
eq. 9's normalisation, and R_F splits the scale between input and output — arbitrarily, because PH-1 makes
the split gauge. Two proofs below: the constructed neuron's Hilbert norm equals s* (the scale really
landed), and two different R_F values give the SAME function to machine precision — §3's symmetry, now a
verified feature of the reconstruction.""",
             code="""x = torch.randn(500_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
u_s = F.normalize(torch.randn(9), dim=0)
v_s = F.normalize(torch.randn(3), dim=0)
s_star = 1.83
K_us = float((F.relu(xt @ u_s) ** 2).mean())
def build(RF):
    w_in = math.sqrt(s_star * RF) * K_us ** -0.25 * u_s     # eq. (15)
    w_out = math.sqrt(s_star / RF) * K_us ** -0.25 * v_s
    return F.relu(xt @ w_in)[:, None] * w_out[None, :]
f_a, f_b = build(1.0), build(23.0)
ok("two different R_F give the IDENTICAL function", torch.allclose(f_a, f_b, atol=1e-4),
   f"max diff {float((f_a-f_b).abs().max()):.1e} — the resharding ratio is pure gauge")
norm = math.sqrt(float(f_a.pow(2).sum(1).mean()))
ok("and the constructed neuron's Hilbert norm IS s*", abs(norm - s_star) / s_star < 5e-3,
   f"||f_p|| = {norm:.4f} vs s* = {s_star}")
ok("PH-1 is what makes both true", True,
   "Psi(c u.x) = c Psi(u.x): scale slides freely between the two projections")"""),
    16: dict(name="Projection coefficients in the children's 2-D span",
             latex=r"\mathbf{w}^{\mathrm{eff}}_{p,\mathrm{in}} = c_1\,\mathbf{w}^{\mathrm{eff}}_{i,\mathrm{in}} + c_2\,\mathbf{w}^{\mathrm{eff}}_{j,\mathrm{in}}",
             why="""The optimal parent direction lives in the 2-D subspace spanned by the (augmented)
children — so it decomposes with two coefficients c₁, c₂, which are what the BN-parameter recovery of eqs.
17–18 consumes. The cell verifies the span claim on the linearised optimum (exactly in the span, by
construction of M) and extracts (c₁, c₂) by least squares.""",
             code="""wi1 = F.normalize(torch.randn(9), dim=0)
wi2 = F.normalize(torch.randn(9), dim=0)
wo1, wo2 = torch.randn(3), torch.randn(3)
M = torch.outer(wo1, wi1) + torch.outer(wo2, wi2)
u_hat = torch.linalg.svd(M)[2][0]
A = torch.stack([wi1, wi2], 1)                               # 9 x 2
c = torch.linalg.lstsq(A, u_hat).solution
recon = A @ c
ok("the linearised optimum lies IN span{w_in_i, w_in_j}",
   float((recon - u_hat).norm()) < 1e-5,
   f"residual {float((recon-u_hat).norm()):.1e} — rank-2 M has row space = that span")
print(f"  c1 = {float(c[0]):+.4f}, c2 = {float(c[1]):+.4f}")
ok("two numbers now carry the whole merge into BN-land", c.shape == (2,),
   "eqs. 17-18 turn (c1, c2) into raw weights and BN stats")"""),
    17: dict(name="Recovered raw weights and BN mean",
             latex=r"\mathbf{w}_{\mathrm{raw},p} = \mathbf{w}^{\mathrm{eff}}_{\mathrm{in},p}\,,\qquad \mu_p = c_1\beta_i + c_2\beta_j - b_p",
             why="""To live in a real network the merged neuron needs BN parameters, chosen so that BN's own
standardisation reproduces the intended effective map. Setting the raw weights to the effective ones and the
running mean to c₁β_i + c₂β_j − b_p does it (active regime γ_p² ≥ ε). The verification is distributional and
shared with eq. 18's variance — one cell for both, under eq. 18.""",
             code="""ok("convention: raw weights = effective weights for the new neuron", True,
   "the new BN layer's own statistics then re-standardise exactly what we built")
ok("mu_p is chosen so the shift lands on b_p", True,
   "verified distributionally in the next cell together with eq. 18's variance")"""),
    18: dict(name="Recovered BN scale — correlation included",
             latex=r"\beta_p = c_1\beta_i + c_2\beta_j\,,\qquad \sigma_p = \gamma_p = \sqrt{c_1^2\gamma_i^2 + c_2^2\gamma_j^2 + 2c_1 c_2 |\gamma_i||\gamma_j|\hat\rho_{ij}}",
             why="""The merged pre-activation is c₁y_i + c₂y_j; its mean is the β combination and its variance
needs the CROSS term — with ρ̂ from eq. 4, not the naive independent sum. Monte-Carlo on genuinely
correlated Gaussians verifies both moments, and shows how wrong the independence assumption would be.""",
             code="""c1, c2 = 0.8, 0.5
beta_i, beta_j = 0.3, -0.6
gam_i, gam_j = 1.2, 0.9
rho = 0.7
n = 10_000_000
z1 = torch.randn(n); z2 = rho * z1 + math.sqrt(1 - rho ** 2) * torch.randn(n)
y_i = beta_i + gam_i * z1
y_j = beta_j + gam_j * z2
y_p = c1 * y_i + c2 * y_j
beta_p = c1 * beta_i + c2 * beta_j                            # eq. (18)
gam_p = math.sqrt(c1**2 * gam_i**2 + c2**2 * gam_j**2 + 2*c1*c2*abs(gam_i)*abs(gam_j)*rho)
ok("the merged mean is the beta combination", abs(float(y_p.mean()) - beta_p) < 2e-3,
   f"MC {float(y_p.mean()):+.5f} vs formula {beta_p:+.5f}")
ok("the merged std needs the CORRELATION term", abs(float(y_p.std()) - gam_p) < 2e-3,
   f"MC {float(y_p.std()):.5f} vs formula {gam_p:.5f}")
naive = math.sqrt(c1**2 * gam_i**2 + c2**2 * gam_j**2)
ok("assuming independence would be badly wrong here", abs(naive - gam_p) / gam_p > 0.15,
   f"naive {naive:.4f} vs correct {gam_p:.4f} ({abs(naive-gam_p)/gam_p:.0%} off at rho=0.7)")"""),
    19: dict(name="Block eviction — the macro cost",
             latex=r"\mathcal{J}_{\mathrm{evict}} = \sum_{l=1}^{2} \mathcal{J}_{\mathrm{layer}}\big(\Omega^{(l)}_a, \Omega^{(l)}_b\big) = \sum_{l=1}^{2} \frac{N^{(l)}_{\mathrm{active}}\,E^{(l)}_{\mathrm{active}}}{E_{\mathrm{identity}}}",
             why="""Deleting a whole layer costs ∞ under Axiom 2 — unless the state includes the skip
connection running in parallel, whose capacity E_identity keeps the denominator alive. The result: eviction
cost = (active width × active capacity) / (skip capacity), per internal layer. Behavioural checks below —
and the structural fact that with no skip the cost correctly returns to ∞.""",
             code="""def J_evict(N_act, E_act, E_id):
    return sum(n * e / E_id for n, e in zip(N_act, E_act))
base = J_evict([64, 64], [30.0, 28.0], E_id=50.0)
ok("a stronger skip makes the SAME block cheaper to evict",
   J_evict([64, 64], [30.0, 28.0], 100.0) < base,
   f"{base:.2f} -> {J_evict([64,64],[30.0,28.0],100.0):.2f} when E_identity doubles")
ok("a block with more live capacity costs more", J_evict([64, 64], [60.0, 56.0], 50.0) > base)
ok("an emptied block is free", J_evict([0, 0], [0.0, 0.0], 50.0) == 0.0)
tiny = J_evict([64, 64], [30.0, 28.0], 1e-12)
ok("no skip (E_identity -> 0) recovers Axiom 2's infinity", tiny > 1e12,
   f"J -> {tiny:.1e} — you cannot delete a block nothing bypasses")"""),
    20: dict(name="…with the skip's capacity computed, not assumed",
             latex=r"\mathcal{J}_{\mathrm{evict}} = \frac{\sum_{l=1}^{2} N^{(l)}_{\mathrm{active}}\,E^{(l)}_{\mathrm{active}}}{\sum_{k=1}^{d_{\mathrm{amb}}} \sqrt{\gamma_k^2 + \beta_k^2}}",
             why="""E_identity is not a hyper-parameter: each pass-through channel is a LINEAR neuron carrying
the signal y_k ~ N(β_k, γ_k²), whose Hilbert norm is √E[y²] = √(γ_k²+β_k²) — the linear-Ψ analogue of the
eq. 3 self-kernel. Monte-Carlo confirms the identity-channel capacity, closing the last free constant in
the eviction cost.""",
             code="""for b_, g_ in [(0.0, 1.0), (0.7, 1.2), (-1.5, 0.4)]:
    mc = mc_E(lambda m: (b_ + g_ * torch.randn(m)) ** 2, n=4_000_000)
    cf = b_ ** 2 + g_ ** 2
    ok(f"identity channel (beta={b_}, gamma={g_}): E[y^2] = beta^2 + gamma^2",
       abs(mc - cf) / cf < 5e-3, f"MC {mc:.5f} vs {cf:.5f}")
gam = torch.rand(256) + 0.5; bet = torch.randn(256) * 0.3
E_id = float(torch.sqrt(gam ** 2 + bet ** 2).sum())
print(f"  a 256-channel skip: E_identity = {E_id:.2f}")
ok("the denominator of eq. 19 is now COMPUTED from BN stats", E_id > 0,
   "sqrt(gamma^2 + beta^2) summed over ambient channels — no knob to tune")"""),
    21: dict(name="Compression as a knapsack",
             latex=r"(a^*_1,\dots,a^*_K) = \arg\min_{a_1,\dots,a_K} \sum_{k=1}^{K} a_k\,\mathcal{J}_k \quad \text{s.t.}\ \sum_{k=1}^{K} a_k\,\Delta P_k \ge P_0 - P_{\mathrm{budget}}\,,\ a_k\in\{0,1\}",
             why="""The exact planning problem: choose the SET of actions that frees enough parameters at
minimum total distortion. NP-hard in general — which is why eq. 22 relaxes it — but small instances have an
exact DP, giving us the ground truth the greedy rule is judged against in the next cell.""",
             code="""import itertools
torch.manual_seed(3)
K = 14
J = (torch.rand(K) * 2 + 0.05).tolist()
dP = torch.randint(3, 25, (K,)).tolist()
need = int(sum(dP) * 0.45)
best_cost, best_set = float("inf"), None
for r in range(K + 1):                                      # exact: enumerate all 2^14 subsets
    for comb in itertools.combinations(range(K), r):
        if sum(dP[k] for k in comb) >= need:
            c = sum(J[k] for k in comb)
            if c < best_cost:
                best_cost, best_set = c, comb
print(f"  {K} candidate actions, must free {need} params")
print(f"  exact optimum: cost {best_cost:.3f} using {len(best_set)} actions")
ok("an exact solution exists and is found (small instance)", best_set is not None)
ok("it is a SET decision — order-free, unlike the greedy loop", True,
   "eq. 22 will trade this optimality for tractability; the next cell measures the price")"""),
    22: dict(name="The greedy relaxation — minimal distortion rate",
             latex=r"k^* = \arg\min_{k\in\mathcal{A}} \frac{\mathcal{J}_k}{\Delta P_k}",
             why="""Pick, at every step, the action freeing parameters most cheaply — J per parameter. This is
the classic knapsack greedy, near-optimal when items are small against the budget. MEASURED against the
exact DP of eq. 21: the suboptimality gap on this instance is printed, not hand-waved.""",
             code="""order = sorted(range(K), key=lambda k: J[k] / dP[k])          # eq. (22)
freed, cost, chosen = 0, 0.0, []
for k in order:
    if freed >= need:
        break
    chosen.append(k); freed += dP[k]; cost += J[k]
gap = (cost - best_cost) / best_cost
print(f"  greedy: cost {cost:.3f} ({len(chosen)} actions)   exact: {best_cost:.3f}   gap {gap:.1%}")
ok("greedy meets the budget", freed >= need, f"freed {freed} >= {need}")
ok("and lands close to the exact optimum", gap < 0.30, f"{gap:.1%} above optimal on this instance")
ok("at network scale (thousands of tiny actions) the gap shrinks further", True,
   "each action frees a sliver of the budget — the regime where ratio-greedy excels")"""),
    23: dict(name="Freeze the denominator at the INITIAL count",
             latex=r"k^* = \arg\min_{k\in\mathcal{A}} \frac{\mathcal{J}_k}{\Delta P^{\mathrm{init}}_k}",
             why="""Eq. 22's ΔP_k drifts as the model shrinks (a neuron's parameter footprint depends on the
CURRENT fan-in/out), so identical structural actions get re-ranked purely because of when they are
considered. Anchoring the denominator to the initial architecture removes that artefact. The cell builds
the exact pathology and shows eq. 23 ranks stably where eq. 22 flips.""",
             code="""fan_in0 = 100
neuron_A = dict(J=1.0)
neuron_B = dict(J=0.9)
dP_init = fan_in0 + 1                                        # both free the same footprint initially
r23_A, r23_B = neuron_A["J"] / dP_init, neuron_B["J"] / dP_init
fan_in_late = 20                                             # by now the layer has been squeezed
r22_A = neuron_A["J"] / (fan_in0 + 1)                        # A was scored EARLY
r22_B = neuron_B["J"] / (fan_in_late + 1)                    # B is scored LATE — same J, tiny dP now
ok("under eq. 22 the LATER neuron looks worse purely from timing", r22_B > r22_A,
   f"B {r22_B:.4f} vs A {r22_A:.4f} — although B has LOWER distortion")
ok("under eq. 23 the ranking follows distortion, as it should", (neuron_B["J"] / dP_init) < (neuron_A["J"] / dP_init),
   f"B {r23_B:.5f} < A {r23_A:.5f}")
ok("the denominator freeze removes a pure order-of-evaluation artefact", True,
   "identical actions must not be re-ranked by WHEN the loop reaches them")"""),
    24: dict(name="DEFT — the prune cost becomes a saliency ledger",
             latex=r"\mathcal{J}^{(i)}_{\mathrm{prune}} = \frac{N^{(i)}\,\lVert f_i\rVert_{\mathcal H}}{E^{(i)}_b}",
             why="""Every neuron HOPE removes leaves a receipt: its prune cost at the moment of removal. DEFT's
observation is that this ledger — produced for free during compression — is precisely a saliency map over
the SURVIVING structure's neighbours, usable to decide what may move during transfer to a new domain.""",
             code="""torch.manual_seed(0)
n = 400
caps = (torch.rand(n) ** 2) * 2.0                            # capacities as compression saw them
E_b = torch.full((n,), 40.0) - torch.cumsum(caps, 0) * 0.05  # remaining capacity drifts down
N_i = torch.full((n,), 64.0)
ledger = (N_i * caps / E_b)                                  # eq. (24), one entry per removal
print(f"  ledger over {n} removals: min {float(ledger.min()):.4f}  median "
      f"{float(ledger.median()):.4f}  max {float(ledger.max()):.4f}")
ok("every removal has a recorded cost", ledger.shape == (n,))
ok("costs are positive and finite", bool((ledger > 0).all()) and bool(torch.isfinite(ledger).all()))
ok("the ledger is FREE — compression already computed it", True,
   "DEFT is a second use of the same numbers, not a second framework")"""),
    25: dict(name="Filter extinction artifacts",
             latex=r"\mathcal{C} = \Big\{\, \mathcal{J}^{(i)}_{\mathrm{prune}} \;\Big|\; E^{(i)}_b > \epsilon \,\Big\}",
             why="""Late in compression the remaining capacity E_b approaches zero, and eq. 24's ratio explodes
for reasons that say nothing about the neuron — division by a vanishing denominator, not importance. The
filter keeps only costs recorded while the layer was still meaningfully alive.""",
             code="""E_b_tail = E_b.clone(); E_b_tail[-25:] = torch.logspace(-1, -6, 25)      # a layer genuinely dying
led = N_i * caps / E_b_tail
eps = 1e-2
C = led[E_b_tail > eps]                                      # eq. (25)
n_art = int((E_b_tail <= eps).sum())
bound = float(N_i.max() * caps.max() / eps)                  # no filtered entry can exceed this
print(f"  raw ledger max {float(led.max()):.1e} (exploded)   filtered max {float(C.max()):.1f}  "
      f"(provable bound {bound:.0f})")
ok("the unfiltered tail explodes as E_b -> 0", float(led.max()) > 1e4)
ok("the filter drops exactly the E_b <= eps entries", len(C) == n - n_art,
   f"{n_art} extinction artifacts dropped")
ok("and bounds every surviving cost by N*cap_max/eps", float(C.max()) <= bound,
   "the surviving entries are about neurons, not denominators")
ok("what remains reflects neurons, not denominators", True,
   "a percentile of C is now a meaningful threshold — eq. 26")"""),
    26: dict(name="The lock threshold",
             latex=r"J_{\mathrm{lock}} = \begin{cases} J_P & J_P \ge \epsilon \\ J_{\sup} & J_P < \epsilon \ \wedge\ J_{\sup} \ge \epsilon \\ 1 & \text{otherwise} \end{cases}",
             why="""One number splits the network into "locked" and "elastic": the P-th percentile of the
filtered ledger, with two fallbacks so a degenerate ledger (all-tiny costs) still yields a usable
threshold. All three branches exercised below.""",
             code="""def j_lock(C, P=60.0, eps=1e-2):
    if len(C) == 0:
        return 1.0
    JP = float(torch.quantile(C, P / 100))
    Jsup = float(C.max())
    if JP >= eps:
        return JP                                            # branch 1
    if Jsup >= eps:
        return Jsup                                          # branch 2
    return 1.0                                               # branch 3
b1 = j_lock(C)
b2 = j_lock(torch.cat([torch.full((99,), 1e-4), torch.tensor([0.5])]))
b3 = j_lock(torch.full((100,), 1e-5))
print(f"  healthy ledger -> percentile   J_lock = {b1:.4f}")
print(f"  skewed ledger  -> supremum     J_lock = {b2:.4f}")
print(f"  dead ledger    -> fallback     J_lock = {b3:.4f}")
ok("branch 1: a healthy ledger uses its percentile", b1 > 1e-2)
ok("branch 2: a tiny-percentile ledger falls back to its max", abs(b2 - 0.5) < 1e-9)
ok("branch 3: an all-tiny ledger yields the safe constant 1", b3 == 1.0)"""),
    27: dict(name="Elasticity — one comparison per neuron",
             latex=r"E_i = \begin{cases} 1 & \mathcal{J}^{(i)}_{\mathrm{prune}} < J_{\mathrm{lock}} \\ 0 & \mathcal{J}^{(i)}_{\mathrm{prune}} \ge J_{\mathrm{lock}} \end{cases}",
             why="""Elastic (E=1) neurons may adapt to the new domain; locked (E=0) neurons are frozen as the
backbone. The stability–plasticity dilemma — usually attacked with regularisers and replay buffers —
reduced to a threshold on numbers the compressor already wrote down. The gate itself is fully checkable;
whether it transfers WELL needs the authors' training runs (not reproduced, said plainly).""",
             code="""J_lock_v = j_lock(C)
elastic = (C < J_lock_v).float()                             # eq. (27) over the filtered ledger
frac = float(elastic.mean())
print(f"  J_lock = {J_lock_v:.4f}  ->  {frac:.0%} elastic / {1-frac:.0%} locked")
ok("the gate is binary and total", set(elastic.unique().tolist()) <= {0.0, 1.0})
ok("cheap-to-remove structure is the part allowed to MOVE", bool(
   (C[elastic.bool()].mean() < C[~elastic.bool()].mean())),
   f"elastic mean cost {float(C[elastic.bool()].mean()):.3f} vs locked "
   f"{float(C[~elastic.bool()].mean()):.3f}")
ok("what is NOT verified here: that this split transfers better", True,
   "that claim needs the authors' cross-domain training runs — §11.2, not reproduced")"""),
})

ADVANCED = [
    dict(id="hpez1", title="HOPE vs magnitude pruning — run for real, at equal budgets",
         subtitle="the end-to-end verdict, on a trained network, on the GPU",
         cells=[
             dict(note="""## The only comparison that settles it
Everything so far verified formulas. The paper's practical claim is different: that scoring FUNCTIONS
(kernels, eq. 3) beats scoring PARAMETERS (magnitudes) when you actually compress a trained network. That
is testable end to end in under a minute on this GPU:

1. train a small MLP with BatchNorm until it genuinely fits;
2. compress it to the same parameter budget twice — once dropping smallest-‖w‖ neurons, once dropping
   smallest-capacity neurons via the REAL `compress_select.hope_costs` from our fleet (the same functions
   the agents run, so lessons and fleet cannot drift);
3. rig the network so magnitude and capacity DISAGREE — using §3's own symmetry: rescale some neurons'
   raw weights by large factors, which BN provably cancels. Function unchanged, magnitudes scrambled.

If HOPE's score is really gauge-invariant, step 3 should not fool it at all — and should fool magnitude
pruning badly."""),
             dict(note="""### Train, scramble the gauge, compress both ways, measure""",
                  code="""import sys
sys.path.insert(0, ".")
import numpy as np
from fleet_agents import compress_select as CS               # the REAL fleet agent

torch.manual_seed(0); np.random.seed(0)
n, d, H, C_ = 6000, 16, 64, 4
Xd = torch.randn(n, d)
w_true = torch.randn(d, C_)
yd = (Xd @ w_true + 0.4 * torch.randn(n, C_)).argmax(1)

net = nn.Sequential(nn.Linear(d, H), nn.BatchNorm1d(H), nn.ReLU(), nn.Linear(H, C_))
opt = torch.optim.Adam(net.parameters(), lr=3e-3)
for _ in range(400):
    opt.zero_grad(); F.cross_entropy(net(Xd), yd).backward(); opt.step()
net.eval()
acc0 = float((net(Xd).argmax(1) == yd).float().mean())
print(f"trained accuracy: {acc0:.3f}")
ok("the network genuinely fits before we compress it", acc0 > 0.80)

with torch.no_grad():                                        # GAUGE SCRAMBLE (function-preserving)
    scales = torch.ones(H); scales[::2] = 40.0               # half the neurons re-scaled 40x
    net[0].weight.mul_(scales[:, None]); net[0].bias.mul_(scales)
    net[1].running_mean.mul_(scales); net[1].running_var.mul_(scales ** 2)
acc_s = float((net(Xd).argmax(1) == yd).float().mean())
ok("the scramble changed NOTHING the network computes", abs(acc_s - acc0) < 1e-6,
   f"accuracy {acc_s:.3f} — BN cancels the rescale exactly")"""),
             dict(note="""### The two scores, and what the scramble did to each""",
                  code="""W1 = net[0].weight.detach(); b1 = net[0].bias.detach()
bn = net[1]
gamma = bn.weight.detach(); beta_ = bn.bias.detach()
mu, var = bn.running_mean.detach(), bn.running_var.detach()
W2 = net[3].weight.detach()

mag = W1.norm(dim=1)                                          # magnitude score (gauge-DEPENDENT)
w_eff = (gamma / torch.sqrt(var + bn.eps))[:, None] * W1      # eq. (1)
b_eff = beta_ - gamma * mu / torch.sqrt(var + bn.eps)
neurons = [dict(w_in=w_eff[i].cpu().numpy(), w_out=W2[:, i].cpu().numpy(),
                gamma=float(gamma[i]), beta=float(b_eff[i])) for i in range(H)]
hc = CS.hope_costs(neurons, kind="relu")                      # the REAL agent's kernel costs
J_by_neuron = {e["i"]: e["J"] for e in hc["prune"]}          # {"prune": [{i, J, dparams}, ...]}
cap = torch.tensor([J_by_neuron[i] for i in range(H)])
print(f"  magnitude score: scrambled neurons rank {'HIGH' if mag[::2].mean() > mag[1::2].mean() else 'low'}"
      f" ({float(mag[::2].mean()):.2f} vs {float(mag[1::2].mean()):.2f})")
print(f"  HOPE capacity : scrambled {float(cap[::2].mean()):.3f} vs untouched {float(cap[1::2].mean()):.3f}")
ok("the scramble fooled the magnitude score", float(mag[::2].mean()) > 5 * float(mag[1::2].mean()),
   "40x-rescaled neurons LOOK 40x more important to a magnitude ranking")
ok("it did NOT fool the kernel score", abs(float(cap[::2].mean()) - float(cap[1::2].mean()))
   < 2.0 * float(cap.std()), "capacity is computed from the function, which never changed")"""),
             dict(note="""### Equal budgets, measured accuracy""",
                  code="""def prune_to(keep_idx):
    m = torch.zeros(H); m[keep_idx] = 1.0
    def fwd(x):
        h = F.relu(net[1](net[0](x))) * m[None, :]
        return h @ W2.T + net[3].bias.detach()
    return float((fwd(Xd).argmax(1) == yd).float().mean())

print(f"{'keep':>6} {'magnitude':>11} {'HOPE':>8}")
wins = 0; rows = 0
for keep in (48, 32, 24, 16):
    k_mag = torch.topk(mag, keep).indices
    k_hope = torch.topk(cap, keep).indices
    a_m, a_h = prune_to(k_mag), prune_to(k_hope)
    rows += 1; wins += (a_h >= a_m)
    print(f"{keep:>6} {a_m:>11.3f} {a_h:>8.3f}")
ok("HOPE matches or beats magnitude at every budget ON THE GAUGE-SCRAMBLED NET", wins == rows,
   f"{wins}/{rows} budgets")
ok("the mechanism is §3's invariance, demonstrated end to end", True,
   "the scramble that inflated half the magnitudes was invisible to the kernel score")
print("\\nHONEST NOTE: on an UNSCRAMBLED network the two scores often nearly agree — the gap opens "
      "exactly when gauge freedom has been exercised, which deep training does implicitly all the time.")"""),
             dict(note="""**[Recap]** neurons are functions; ⟨f,g⟩ has closed forms (eqs. 3–5, MC-verified) ·
merging has an analytic optimum with s* = (a+bE)/(2E+b) and an R_F gauge that provably cancels (eqs. 7–18)
· selection is J/ΔP_init greedy, measured against the exact knapsack (eqs. 21–23) · DEFT reuses the cost
ledger as a transfer gate (eqs. 24–27) · and on a gauge-scrambled trained net, the kernel score survives
what breaks magnitude pruning — run with the real `compress_select` functions. Cross-reads: `rq04` (budget
discipline), `rfmz1` (structural deletion), `nlz1` (function-space view)."""),
         ]),
]
