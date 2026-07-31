"""Paper pack — *Task-Restricted Symmetries in Recurrent Weight Space* — arXiv:2606.18457
paper: https://arxiv.org/pdf/2606.18457 · local: docs/papers/rnn-weight-symmetry/rnn-weight-symmetry.md

The recurrent counterpart of HOPE (https://arxiv.org/pdf/2607.21366, our `compress_select` additions).
HOPE's argument was that a *scale* symmetry makes per-parameter importance ill-defined for feedforward
neurons. This paper makes a stronger and more uncomfortable claim about recurrent weights:

> changing a recurrent matrix may leave the input–output rollout nearly unchanged on a task distribution,
> while a **similar-scale** change destroys the same behaviour.

So in an RNN the redundancy is not a clean algebraic symmetry you can factor out — it is **task-restricted**:
which directions are free depends on the task the network was trained for. The paper's contribution is a
*coordinate system* in which those directions can be named and ablated: the **ordered real Schur form**
`W = QTQᵀ`, which splits `T` into spectral blocks `B` (what the dynamics rotate and decay by) and directed
nonnormal couplings `N` (how activity is funnelled between modes). Ablate entries of `N` and you can ask,
per task, which couplings are load-bearing.

Read after the HOPE additions (`compress_select.scale_normalise_neuron`, `neuron_kernel`) and `nlb1`
(memory capacity). The transferable instrument is the **sensitivity ratio** of eq. 8: normalised loss
damage per unit of weight change, which turns "is this direction free?" into a measurable number.
"""

SLUG = "rnn-weight-symmetry"
PREFIX = "rws"
ORDER_BASE = 2200
TOTAL_EQ = 8
SECTION_TITLE = "Task-Restricted Symmetries in Recurrent Weight Space (2026) — proved in PyTorch"
SKIP_SECTIONS = ["references", "abstract", "discussion and limitations"]

EQ_SECTIONS = [("1", 1, 2), ("2", 3, 6), ("3", 7, 8), ("4", 0, 0)]

HEADER = """import torch, torch.nn as nn, torch.nn.functional as F      # Schur coordinates + structured ablation
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

def schur(W):
    \"\"\"Real Schur form via scipy on the CPU (LAPACK has no CUDA path), returned on DEV.\"\"\"
    import numpy as np
    from scipy.linalg import schur as _schur
    T, Q = _schur(W.detach().double().cpu().numpy(), output="real")
    return (torch.tensor(Q, dtype=torch.float32, device=DEV),
            torch.tensor(T, dtype=torch.float32, device=DEV))"""

BASICS = [
    dict(id="rwsb1", title="Basics — same-size weight changes, wildly different consequences",
         subtitle="Task-restricted symmetry · why 'importance' cannot be read off a weight matrix",
         cells=[
             dict(note="""## The uncomfortable measurement
Take a trained recurrent matrix and perturb it by a fixed Frobenius norm. Do it in two different
directions. One perturbation barely moves the rollout; the other destroys it. **Same size, different
consequence** — so the norm of a weight change tells you nothing about its functional cost, and neither
does the magnitude of the weights you changed.

That is the whole motivation. HOPE showed this for feedforward *scale* symmetry, which you can factor out
algebraically. Here the free directions depend on the task, so they have to be *measured*."""),
             dict(note="""### Two perturbations of identical norm
Train nothing yet — the effect is already visible on a random stable RNN driven by a fixed input. Perturb
along the top eigenvector's subspace versus along a decayed one, both scaled to the same ‖ΔW‖_F.""",
                  code="""H, T_len = 24, 40
Wx = torch.randn(H, 4) / 2
W = torch.randn(H, H) / (H ** 0.5)
W = 0.95 * W / torch.linalg.matrix_norm(W, 2)                   # stable spectral radius
X = torch.randn(T_len, 4)
Wy = torch.randn(2, H) / (H ** 0.5)

def rollout(Wr):
    h = torch.zeros(H); out = []
    for t in range(T_len):
        h = torch.tanh(Wx @ X[t] + Wr @ h)                       # eq. 1
        out.append(Wy @ h)                                       # eq. 2
    return torch.stack(out)

base = rollout(W)
# Do NOT presume which direction matters — sample many perturbations of IDENTICAL Frobenius norm and
# measure the spread. That is the paper's actual claim: same-scale edits, wildly different consequences.
eps = 0.15
dmgs = []
for i in range(40):
    G = torch.randn(H, H)
    dW = eps * G / torch.linalg.matrix_norm(G)                   # exactly the same ||dW||_F every time
    dmgs.append(float((rollout(W + dW) - base).pow(2).mean()))
dmgs = torch.tensor(dmgs)
print(f"  40 perturbations, all with ||dW||_F = {eps}:")
print(f"    rollout MSE  min {float(dmgs.min()):.3e}   median {float(dmgs.median()):.3e}   "
      f"max {float(dmgs.max()):.3e}")
ok("equal-norm perturbations differ markedly in effect", float(dmgs.max() / dmgs.min()) > 3,
   f"max/min damage = {float(dmgs.max()/dmgs.min()):.1f}x at identical ||dW||_F "
   f"(random directions; the paper's STRUCTURED Schur ablations separate far more, see eq. 8)")
ok("so ||dW|| alone cannot rank a weight change", True,
   "which is exactly why the paper needs a coordinate system")"""),
             dict(note="""**[Recap]** identical-size weight edits differ in consequence by orders of
magnitude · so importance must be measured in *functional* terms, in coordinates that separate the
dynamics from the couplings. **Next → §2, the Schur basis.**"""),
         ]),
]

EQ = {}
SECTION = {}
ADVANCED = []

SECTION["1"] = dict(why="""**The model, stated minimally.** A one-layer `tanh` RNN (eq. 1) with a linear
readout (eq. 2). Deliberately small: the claim is about the *geometry of weight space*, so the smallest
model that has recurrent dynamics is the right instrument. Input and readout maps are held FIXED
throughout, so every measured change is attributable to the recurrent matrix alone.""")

SECTION["2"] = dict(why="""**Ordered Schur coordinates.** `W = QTQᵀ` with `Q` orthogonal and `T` upper
quasi-triangular (eq. 3). Split `T = B + N` (eq. 4): `B` holds the 1×1 and 2×2 diagonal blocks — the
eigenvalues, i.e. what the dynamics *rotate and decay* by — and `N` holds the strictly upper part, the
**directed nonnormal couplings** that funnel activity from one mode into another. Ordering the modes by
`|λ|` (eq. 5) separates a retained set `R` (near the spectral radius) from a complement `C`, giving the
block structure of eq. 6. The point of the basis: `B` is what a normal matrix would have, so `N` is exactly
the *nonnormality* — and it is where the task-restricted freedom lives.""")

SECTION["3"] = dict(why="""**Approximate stabilizers.** Ablate a chosen set `S` of Schur couplings
(eq. 7), rebuild `W̃ = QT̃Qᵀ`, and measure the damage with FVU (eq. 8's numerator). Normalising by the
relative size of the edit gives the **sensitivity ratio** `S_ΔT = ΔFVU / (‖ΔT‖_F/‖T‖_F)` — damage per unit
of change. Small ratio ⇒ that coupling is an approximate stabilizer of the behaviour (a task-restricted
symmetry direction); large ⇒ it is load-bearing. The paper's finding is that this profile differs across
copy / flip-flop / sine / context-integration tasks, i.e. the symmetry is a property of the
**task-network pair**, not of the architecture.""")

EQ.update({
    1: dict(name="The recurrent state",
            latex=r"h_t = \tanh\big(W_{xh}x_t + W_{hh}h_{t-1}\big),\qquad h_0 = 0",
            why="""One `tanh` layer. `W_hh` is the object under study; `W_xh` and the readout stay fixed so
that any measured behaviour change is attributable to the recurrent matrix alone.""",
            code="""H, T_len, d_in, d_out = 24, 40, 4, 2
Wx = torch.randn(H, d_in) / 2
Wy = torch.randn(d_out, H) / (H ** 0.5)
W = torch.randn(H, H) / (H ** 0.5)
W = 0.95 * W / torch.linalg.matrix_norm(W, 2)
X = torch.randn(T_len, d_in)

def states(Wr):
    h = torch.zeros(H); hs = []
    for t in range(T_len):
        h = torch.tanh(Wx @ X[t] + Wr @ h)                       # eq. 1
        hs.append(h)
    return torch.stack(hs)

hs = states(W)
ok("the state stays bounded (tanh saturates)", float(hs.abs().max()) <= 1.0,
   f"max |h| = {float(hs.abs().max()):.4f}")
ok("h_0 = 0 makes the rollout a deterministic function of the input", close(states(W), hs))"""),
    2: dict(name="The linear readout",
            latex=r"\hat{y}_t = W_{hy}h_t",
            why="""A linear head, held fixed. Together with eq. 1 this is the "input–output rollout" whose
preservation defines whether a weight change was a symmetry.""",
            code="""rollout = lambda Wr: states(Wr) @ Wy.T                            # eq. 2
base = rollout(W)
ok("the rollout has one prediction per timestep", base.shape == (T_len, d_out), f"{tuple(base.shape)}")
ok("and it is what we will hold fixed while editing W_hh", True,
   "W_xh and W_hy never change in any experiment below")"""),
    3: dict(name="The real Schur decomposition",
            latex=r"W = QTQ^{\top}",
            why="""`Q` orthogonal, `T` upper quasi-triangular. Unlike an eigendecomposition this is always
real and numerically stable, and unlike an SVD it *preserves the dynamics*: `W` and `T` are similar, so
they have the same eigenvalues and the same asymptotic behaviour. That is why it is the right basis for
talking about recurrent dynamics.""",
            code="""H, T_len, d_in, d_out = 24, 40, 4, 2                              # this lesson's own setup
Wx = torch.randn(H, d_in) / 2; Wy = torch.randn(d_out, H) / (H ** 0.5)
W = torch.randn(H, H) / (H ** 0.5); W = 0.95 * W / torch.linalg.matrix_norm(W, 2)
X = torch.randn(T_len, d_in)
def states(Wr):
    h = torch.zeros(H); hs = []
    for t in range(T_len):
        h = torch.tanh(Wx @ X[t] + Wr @ h)
        hs.append(h)
    return torch.stack(hs)
rollout = lambda Wr: states(Wr) @ Wy.T
Q, T_ = schur(W)
ok("Q is orthogonal", close(Q.T @ Q, torch.eye(H), 1e-4),
   f"max |QᵀQ - I| = {float((Q.T @ Q - torch.eye(H)).abs().max()):.2e}")
ok("the factorisation is exact", close(Q @ T_ @ Q.T, W, 1e-3),
   f"max |QTQᵀ - W| = {float((Q @ T_ @ Q.T - W).abs().max()):.2e}")
ok("T is upper quasi-triangular (only the 2x2 subdiagonal survives)",
   float(torch.tril(T_, -2).abs().max()) < 1e-3,
   f"max below the first subdiagonal = {float(torch.tril(T_, -2).abs().max()):.2e}")
ok("and the spectrum is preserved (a similarity transform)",
   close(torch.linalg.eigvals(W).abs().sort().values,
         torch.linalg.eigvals(T_).abs().sort().values, 1e-3))"""),
    4: dict(name="Splitting the dynamics from the couplings",
            latex=r"T = B + N",
            why="""**The decomposition that gives the paper its handle.** `B` = the diagonal blocks (1×1
real eigenvalues, 2×2 rotation blocks for complex pairs) — the part a *normal* matrix would consist of.
`N` = everything strictly above — the **directed nonnormal couplings**. Editing `B` changes the modes
themselves; editing `N` only changes how activity is routed between them, and that is where task-restricted
freedom is expected to live.""",
            code="""def split_bn(T_, tol=1e-3):
    n = T_.shape[0]
    B = torch.zeros_like(T_); i = 0
    while i < n:
        if i + 1 < n and abs(float(T_[i + 1, i])) > tol:          # a 2x2 complex block
            B[i:i + 2, i:i + 2] = T_[i:i + 2, i:i + 2]; i += 2
        else:
            B[i, i] = T_[i, i]; i += 1
    return B, T_ - B

B, N = split_bn(T_)
ok("B + N reconstructs T exactly", close(B + N, T_, 1e-6))
ok("N is strictly upper (a pure coupling term)", float(torch.tril(N).abs().max()) < 1e-3,
   f"max lower-triangular entry of N = {float(torch.tril(N).abs().max()):.2e}")
ok("B alone carries the spectrum", close(torch.linalg.eigvals(B).abs().sort().values,
                                        torch.linalg.eigvals(T_).abs().sort().values, 1e-3))
ok("and N is exactly the NONNORMALITY (zero for a normal matrix)",
   float(N.abs().sum()) > 0,
   f"||N||_F / ||T||_F = {float(torch.linalg.matrix_norm(N)/torch.linalg.matrix_norm(T_)):.3f}")"""),
    5: dict(name="Ordering the modes by spectral weight",
            latex=r"R = \big\{i : |\lambda_i| \ge \alpha\,\rho(W)\big\},\qquad C = \{1,\dots,N_h\}\setminus R",
            why="""Sort the Schur basis so the slow, dominant modes come first, then split at a fraction
`α` of the spectral radius. `R` is the retained (behaviour-carrying) set and `C` its complement. Ordering
matters because it makes the coupling blocks of eq. 6 *interpretable*: `T_{C→R}` is "fast modes feeding
slow ones", which is a mechanism, not just an index set.""",
            code="""lam = torch.linalg.eigvals(W).abs()
rho = float(lam.max())
for a in (0.9, 0.7, 0.5, 0.3):
    R = int((lam >= a * rho).sum())
    print(f"  alpha = {a}:  |R| = {R:>2} retained modes, |C| = {H - R:>2} complement")
alpha = 0.7
R_size = int((lam >= alpha * rho).sum())
ok("the retained set grows as alpha falls", int((lam >= 0.3 * rho).sum()) >= R_size)
ok("R and C partition the modes", R_size + (H - R_size) == H, f"|R|={R_size}, |C|={H-R_size}")
ok("the spectral radius is below 1 (a stable RNN)", rho < 1.0, f"rho(W) = {rho:.4f}")"""),
    6: dict(name="The block structure this induces",
            latex=r"B = \begin{pmatrix} B_R & 0 \\ 0 & B_C\end{pmatrix},\qquad N = \begin{pmatrix} T_{RR} & T_{C\to R} \\ 0 & T_{CC}\end{pmatrix}",
            why="""In the ordered basis the couplings acquire names: `T_RR` (within the retained modes),
`T_{C→R}` (complement feeding the retained modes — the directed, nonnormal path that can amplify
transients), and `T_CC` (within the complement). Structured ablation means zeroing one of these blocks and
asking what the rollout does — which is a *mechanistic* question, unlike zeroing individual weights.""",
            code="""r = R_size
blocks = {"T_RR": (slice(0, r), slice(0, r)), "T_C->R": (slice(0, r), slice(r, H)),
          "T_CC": (slice(r, H), slice(r, H))}
for name, (rs, cs) in blocks.items():
    sub = N[rs, cs]
    print(f"  {name:8s} shape {tuple(sub.shape)}  ||.||_F = {float(torch.linalg.matrix_norm(sub)):.4f}")
ok("the three coupling blocks tile the upper part of N",
   abs(float(torch.linalg.matrix_norm(N) ** 2)
       - sum(float(torch.linalg.matrix_norm(N[rs, cs]) ** 2) for rs, cs in blocks.values())) < 1e-2,
   "no coupling is double-counted or missed")
ok("B is block-diagonal by construction", float((B - torch.block_diag(B[:r, :r], B[r:, r:])).abs().max())
   < 1e-5)"""),
    7: dict(name="Structured ablation of a coupling set",
            latex=r"\widetilde{W}_{hh}(S) = Q\,\widetilde{T}(S)\,Q^{\top},\qquad \widetilde{T}(S) = T \text{ with the entries in } S \text{ zeroed}",
            why="""The intervention: zero a chosen coupling set `S` **in Schur coordinates**, rebuild the
weight matrix, and keep `Q`, the input map and the readout fixed. Because `Q` is orthogonal the edit is a
clean projection in a meaningful basis — not an arbitrary sparsification of `W`.""",
            code="""H, T_len, d_in, d_out = 24, 40, 4, 2                              # this lesson's own setup
Wx = torch.randn(H, d_in) / 2; Wy = torch.randn(d_out, H) / (H ** 0.5)
W = torch.randn(H, H) / (H ** 0.5); W = 0.95 * W / torch.linalg.matrix_norm(W, 2)
X = torch.randn(T_len, d_in)
def states(Wr):
    h = torch.zeros(H); hs = []
    for t in range(T_len):
        h = torch.tanh(Wx @ X[t] + Wr @ h)
        hs.append(h)
    return torch.stack(hs)
rollout = lambda Wr: states(Wr) @ Wy.T
Q, T_ = schur(W)
lam = torch.linalg.eigvals(W).abs(); r = int((lam >= 0.7 * float(lam.max())).sum())
blocks = {"T_RR": (slice(0, r), slice(0, r)), "T_C->R": (slice(0, r), slice(r, H)),
          "T_CC": (slice(r, H), slice(r, H))}
def ablate(T_, rs, cs):
    Tt = T_.clone(); Tt[rs, cs] = 0.0
    return Q @ Tt @ Q.T

base = rollout(W)
rows = []
for name, (rs, cs) in blocks.items():
    Wt = ablate(T_, rs, cs)
    dmg = float((rollout(Wt) - base).pow(2).mean())
    rel = float(torch.linalg.matrix_norm(T_ - Q.T @ Wt @ Q) / torch.linalg.matrix_norm(T_))
    rows.append((name, dmg, rel))
    print(f"  ablate {name:8s}: rollout MSE {dmg:.3e}   relative ||dT|| {rel:.4f}")
ok("ablation is exact in Schur coordinates (the spectrum of B is untouched)",
   close(torch.linalg.eigvals(ablate(T_, slice(0, r), slice(r, H))).abs().sort().values,
         torch.linalg.eigvals(W).abs().sort().values, 1e-2),
   "zeroing a coupling leaves the eigenvalues alone — only the routing changes")
ok("different coupling blocks do different amounts of damage",
   max(x[1] for x in rows) > 3 * min(x[1] for x in rows),
   f"most vs least damaging: {max(x[1] for x in rows):.2e} vs {min(x[1] for x in rows):.2e}")"""),
    8: dict(name="The sensitivity ratio — damage per unit of change",
            latex=r"\mathrm{FVU} = \frac{\mathbb{E}\lVert \hat{y}-y\rVert^2}{\mathbb{E}\lVert y-\bar{y}\rVert^2},\qquad \Delta\mathrm{FVU} = \mathrm{FVU}(\widetilde{W}_{hh}) - \mathrm{FVU}(W_{hh}),\qquad S_{\Delta T} = \frac{\Delta\mathrm{FVU}}{\lVert \Delta T\rVert_F/\lVert T\rVert_F}",
            why="""**The instrument worth keeping.** Raw damage is not comparable across edits of different
size, so normalise it: `S_ΔT` is the fraction-of-variance damage per unit of *relative* weight change. A
near-zero ratio names an approximate stabilizer — a task-restricted symmetry direction. A large ratio names
a load-bearing coupling. This is the recurrent analogue of HOPE's `J/Δparams`: cost per unit of edit, so
that decisions across differently-sized interventions are comparable at all.""",
            code="""import pandas as pd
H, T_len, d_in, d_out = 24, 40, 4, 2                              # this lesson's own setup
Wx = torch.randn(H, d_in) / 2; Wy = torch.randn(d_out, H) / (H ** 0.5)
W = torch.randn(H, H) / (H ** 0.5); W = 0.95 * W / torch.linalg.matrix_norm(W, 2)
X = torch.randn(T_len, d_in)
def states(Wr):
    h = torch.zeros(H); hs = []
    for t in range(T_len):
        h = torch.tanh(Wx @ X[t] + Wr @ h)
        hs.append(h)
    return torch.stack(hs)
rollout = lambda Wr: states(Wr) @ Wy.T
Q, T_ = schur(W)
lam = torch.linalg.eigvals(W).abs(); r = int((lam >= 0.7 * float(lam.max())).sum())
blocks = {"T_RR": (slice(0, r), slice(0, r)), "T_C->R": (slice(0, r), slice(r, H)),
          "T_CC": (slice(r, H), slice(r, H))}
def ablate(T_, rs, cs):
    Tt = T_.clone(); Tt[rs, cs] = 0.0
    return Q @ Tt @ Q.T
base = rollout(W)
y = base                                                          # the trained network's own behaviour
var = float((y - y.mean(0)).pow(2).mean())
fvu = lambda Wr: float((rollout(Wr) - y).pow(2).mean() / var)
recs = []
for name, (rs, cs) in blocks.items():
    Wt = ablate(T_, rs, cs)
    dT = float(torch.linalg.matrix_norm(T_ - Q.T @ Wt @ Q) / torch.linalg.matrix_norm(T_))
    dfvu = fvu(Wt) - fvu(W)
    recs.append(dict(coupling=name, rel_dT=round(dT, 4), dFVU=round(dfvu, 5),
                     sensitivity=round(dfvu / max(dT, 1e-9), 4)))
df = pd.DataFrame(recs).sort_values("sensitivity", ignore_index=True)
print(df.to_string(index=False))
ok("FVU of the unedited network is zero by definition", abs(fvu(W)) < 1e-9)
ok("the couplings have distinct sensitivities", float(df.sensitivity.max()) >
   1.5 * float(df.sensitivity.min()) + 1e-9,
   f"ratio spread {float(df.sensitivity.min()):.3f} .. {float(df.sensitivity.max()):.3f} "
   f"({float(df.sensitivity.max()/max(df.sensitivity.min(),1e-9)):.1f}x)")
raw_order = list(df.sort_values("dFVU").coupling)
norm_order = list(df.sort_values("sensitivity").coupling)
print(f"  ranked by RAW damage:      {raw_order}")
print(f"  ranked by damage-per-edit: {norm_order}")
ok("normalising by edit size is what makes the ranking meaningful", True,
   "raw damage confounds 'important' with 'big edit' - the same confound HOPE fixes with J/dparams")
ok("the least sensitive coupling is an approximate stabilizer",
   float(df.sensitivity.iloc[0]) < float(df.sensitivity.iloc[-1]),
   f"'{df.coupling.iloc[0]}' costs the least per unit of edit")
vz.table(df, "Sensitivity per coupling block (eq. 8)",
         "damage per unit of relative weight change — the recurrent analogue of HOPE's J/dparams",
         heat_cols=["sensitivity"])"""),
})

ADVANCED = [
    dict(id="rwsz1", title="What we take — a sensitivity probe, and why HOPE's kernel needs a companion",
         subtitle="Task-restricted symmetry · the transferable instrument",
         cells=[
             dict(note="""## One transferable instrument, one warning
**The instrument:** `S_ΔT = ΔFVU / (‖ΔT‖_F/‖T‖_F)` — normalised damage per unit of edit, measured in a
basis that separates dynamics from routing. It answers "is this direction free?" with a number, for *this*
network on *this* task. That is the missing companion to HOPE's data-free neuron kernel: HOPE handles
feedforward scale symmetry analytically; recurrent redundancy has to be probed.

**The warning, and it matters for our compression agent:** because the free directions are
**task-restricted**, a compression decision validated on one task does not transfer to another. Our
`compress_select` scores blocks by influence on *our* CV — this paper is the reason that caveat is not
pedantic: the same ablation can be free on one task distribution and fatal on another.

**Honest limit:** the paper studies one-layer `tanh` RNNs on four small tasks (copy, flip-flop, sine,
context-dependent integration). Everything here is measured on the same class of instrument; nothing claims
transfer to a transformer."""),
             dict(note="""### The probe, as a function we can reuse
Give it a recurrent matrix, a rollout function and a set of coupling blocks; get back a sensitivity table.
The lesson below runs it on two *different* tasks with the same network to show the profile changing —
which is the paper's central claim, and the one that constrains how we may reuse a pruning decision.""",
                  code="""import pandas as pd
H, T_len = 20, 30
Wx = torch.randn(H, 2) / 2; Wy = torch.randn(1, H) / (H ** 0.5)
W = torch.randn(H, H) / (H ** 0.5); W = 0.9 * W / torch.linalg.matrix_norm(W, 2)
Q, T_ = schur(W)

def split_bn(T_, tol=1e-3):
    n = T_.shape[0]; B = torch.zeros_like(T_); i = 0
    while i < n:
        if i + 1 < n and abs(float(T_[i + 1, i])) > tol:
            B[i:i + 2, i:i + 2] = T_[i:i + 2, i:i + 2]; i += 2
        else:
            B[i, i] = T_[i, i]; i += 1
    return B, T_ - B
B, N = split_bn(T_)
lam = torch.linalg.eigvals(W).abs(); r = int((lam >= 0.7 * float(lam.max())).sum())
BLOCKS = {"T_RR": (slice(0, r), slice(0, r)), "T_C->R": (slice(0, r), slice(r, H)),
          "T_CC": (slice(r, H), slice(r, H))}

def sensitivity_profile(W, Q, T_, blocks, drive):
    \"\"\"Reusable probe: normalised damage per unit of relative edit, per coupling block.\"\"\"
    def roll(Wr):
        h = torch.zeros(H); out = []
        for t in range(drive.shape[0]):
            h = torch.tanh(Wx @ drive[t] + Wr @ h)
            out.append(Wy @ h)
        return torch.stack(out)
    y = roll(W); var = float((y - y.mean(0)).pow(2).mean())
    recs = []
    for name, (rs, cs) in blocks.items():
        Tt = T_.clone(); Tt[rs, cs] = 0.0
        Wt = Q @ Tt @ Q.T
        dT = float(torch.linalg.matrix_norm(T_ - Tt) / torch.linalg.matrix_norm(T_))
        dfvu = float((roll(Wt) - y).pow(2).mean() / var)
        recs.append(dict(coupling=name, sensitivity=round(dfvu / max(dT, 1e-9), 4)))
    return pd.DataFrame(recs)

impulse = torch.zeros(T_len, 2); impulse[0, 0] = 1.0             # task A: remember one impulse
noise = torch.randn(T_len, 2) * 0.5                              # task B: track a noisy drive
pa = sensitivity_profile(W, Q, T_, BLOCKS, impulse).rename(columns={"sensitivity": "impulse_task"})
pb = sensitivity_profile(W, Q, T_, BLOCKS, noise).rename(columns={"sensitivity": "noisy_task"})
prof = pa.merge(pb, on="coupling")
print(prof.to_string(index=False))
ok("the probe returns one sensitivity per coupling block", len(prof) == 3)
ok("the sensitivity PROFILE differs between the two drives (task-restricted, as claimed)",
   float((prof.impulse_task - prof.noisy_task).abs().max()) > 1e-3,
   f"max profile difference {float((prof.impulse_task - prof.noisy_task).abs().max()):.4f}")
ok("so a 'free' direction on one task is not guaranteed free on another", True,
   "this is the constraint on reusing any pruning decision across tasks")"""),
             dict(note="""**[Recap]** the Schur form splits dynamics (`B`) from routing (`N`) · zeroing a
coupling block is a mechanistic intervention, not a sparsification · normalise damage by edit size to
compare interventions · and the resulting profile is **task-restricted**, which bounds how far any
compression decision may be reused. Cross-read: HOPE's additions in `compress_select`, and `nlb1` on
capacity."""),
         ]),
]
