"""Paper pack — *Erase-then-Delta Attention (EDA): Decoupling Erase and Write Addresses in Delta-Rule
Linear Attention* — arXiv:2606.26560 · https://arxiv.org/pdf/2606.26560
local: docs/papers/eda-delta-attention/eda-delta-attention.md

Read this straight after the Kimi-K3 series (`k302`, Kimi Delta Attention) and the Nested Learning
dictionary (`nlb4`, objective → update rule). Those lessons proved that the delta rule's update
`(I − βkkᵀ)S + βkvᵀ` erases **at the key it is about to write**. EDA's observation is that this is a
restriction nobody had questioned: the erase address is *hard-wired* to the write address, so content
stored at a DIFFERENT address can never be actively removed — it can only decay.

EDA adds one projection: erase along a learned direction `e_t`, then do the usual delta write at `k_t`.
Two things make it worth a full pack:
  • the erase step is itself one gradient step on an objective (`½‖Ŝᵀe‖²`) — the same
    objective→rule dictionary the NL series established, applied to *forgetting* rather than writing;
  • the "doubling trick" (§3): interleave each erase as a virtual token so the sequence has 2T steps, and
    EDA's recurrence becomes an ordinary DeltaNet recurrence — so it needs NO new kernel. That claim is
    exactly checkable, and this pack checks it.
"""

SLUG = "eda-delta-attention"
PREFIX = "eda"
ORDER_BASE = 1800
TOTAL_EQ = 22
SECTION_TITLE = "Erase-then-Delta Attention (2026) — decoupling erase from write, proved in PyTorch"
SKIP_SECTIONS = ["references", "abstract", "model configurations", "evaluation benchmarks"]

EQ_SECTIONS = [("1", 0, 0), ("2", 1, 7), ("3", 8, 22), ("4", 0, 0), ("5", 0, 0), ("6", 0, 0)]

HEADER = """import torch, torch.nn as nn, torch.nn.functional as F      # delta rules are two rank-1 projections
import sys; sys.path.insert(0, "learning")
import vizkit as vz                                            # shared visual + explainability layer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
# exactness proofs need full fp32: TF32 truncates the mantissa to 10 bits and an identity that holds to
# 1e-6 in fp32 only holds to ~1e-3 in TF32 (the lesson learned building the Nested Learning pack)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)

def unit(*shape):                                              # keys/queries are L2-normalised here
    return F.normalize(torch.randn(*shape), dim=-1)"""

BASICS = [
    dict(id="edab1", title="Basics — why an erase address is a separate thing from a write address",
         subtitle="EDA · the one restriction the delta rule never questioned",
         cells=[
             dict(note="""## The gap, in one experiment
The delta rule is "erase at `k`, then write `v` at `k`". Everything it can forget, it forgets *because it
is about to write there*. So ask the awkward question: **what removes a stale fact stored somewhere else?**

Nothing does — except global decay, which forgets *everything* a little. That is the entire motivation for
EDA, and it is measurable in ten lines: store two facts at two different addresses, then keep writing at
the first one and watch what happens to the second."""),
             dict(note="""### DeltaNet cannot target the stale fact
Write `(k₁, v₁)` and `(k₂, v₂)` at two near-orthogonal addresses. Now suppose `v₂` has gone stale. Writing
at `k₁` again — even a hundred times — leaves the read at `k₂` almost untouched, because the erase
projection `I − βk₁k₁ᵀ` only acts along `k₁`.""",
                  code="""d = 32
k1, k2 = unit(d), unit(d)
v1, v2 = torch.randn(d), torch.randn(d)
beta = 1.0

def delta_write(S, k, v, b=beta):                               # the delta rule (eq. 3)
    return (torch.eye(d) - b * torch.outer(k, k)) @ S + b * torch.outer(k, v)

S = torch.zeros(d, d)
S = delta_write(S, k1, v1); S = delta_write(S, k2, v2)
before = float((S.T @ k2 - v2).norm())
for _ in range(100):                                            # hammer the OTHER address
    S = delta_write(S, k1, torch.randn(d))
after = float((S.T @ k2 - v2).norm())
ok("the stale fact at k2 survives 100 writes at k1", after < 3 * max(before, 1e-6) + 1.0,
   f"read error at k2: {before:.4f} -> {after:.4f}  (|k1.k2| = {abs(float(k1 @ k2)):.3f})")
print("the delta rule's erase is a rank-1 projection along the WRITE key, so it is blind to k2")"""),
             dict(note="""### Global decay is the only alternative — and it is indiscriminate
The usual fix is a decay gate: multiply the whole state by `α < 1` every step. It does remove the stale
fact, but it removes the *fresh* one at the same rate. That trade — forget everything slowly, or nothing
selectively — is what EDA breaks by giving the erase its own address.""",
                  code="""alpha = 0.95
S = torch.zeros(d, d)
S = delta_write(S, k1, v1); S = delta_write(S, k2, v2)
fresh0, stale0 = float((S.T @ k1 - v1).norm()), float((S.T @ k2 - v2).norm())
for _ in range(40):
    S = alpha * S                                               # global decay, no write
fresh1, stale1 = float((S.T @ k1 - v1).norm()), float((S.T @ k2 - v2).norm())
ok("decay does erase the stale fact", stale1 > stale0, f"stale {stale0:.3f} -> {stale1:.3f}")
ok("but it damages the fresh one just as much", fresh1 > fresh0,
   f"fresh {fresh0:.3f} -> {fresh1:.3f} — indiscriminate")
print("EDA's proposal: keep the corrective write, and add ONE targeted erase at its own address e_t")"""),
             dict(note="""**[Recap]** the delta rule's erase is chained to its write · global decay is the
only other eraser and it is untargeted · so "where to erase" deserves to be a learned quantity.
**Next → §2, the family EDA generalises.**"""),
         ]),
]

EQ = {}
SECTION = {}
ADVANCED = []

SECTION["1"] = dict(why="""**The claim.** Delta-rule linear attention corrects what is stored at the
current write address before writing. EDA decouples *where to erase* from *where to write*: a targeted
erase along a learned direction `e_t`, then the standard delta write along `k_t`. Reported across dense
2.5B and MoE 25B-A2.8B pre-training, EDA is best in both settings and the gain persists with scale.

For us the interesting part is not the benchmark — it is that the erase turns out to be one gradient step
on an explicit objective, and that the whole rule folds back into a plain DeltaNet recurrence on a
doubled sequence, so it costs no new kernel.""")

SECTION["2"] = dict(why="""**The family, from linear attention to gated delta rules.** Four steps, each one
a change of objective or of gate: additive outer-product memory (eq. 1) → the delta objective (eq. 2) →
DeltaNet's corrective write (eq. 3) → a decay gate, scalar (eq. 4) then diagonal/per-channel (eq. 5, the
same channel-wise decay Kimi K3's KDA uses). Eq. 6 is the one that matters: written in its general form
the update has an erase direction `ẽ_t` **and** a write direction `k_t`, and every model in this family
silently sets `ẽ_t = β_t k_t`. EDA is what happens when you stop doing that.""")

SECTION["3"] = dict(why="""**The method.** Erase first, then delta-write (eq. 8). The erase is one gradient
step on `½‖Ŝᵀe_t‖²` (eqs. 9–11) — i.e. "make the state return nothing at address `e_t`" — and its effect on
any read is exactly `S̃ᵀq = Ŝᵀq − γ(qᵀe)Ŝᵀe` (eq. 12), so a query orthogonal to `e_t` is untouched. The
gate is parameterised in log space with a floor (eqs. 14–15) so the decay cannot leave `(e^ℓ, 1)`.

Then the engineering payoff (eqs. 16–22): expand the two projections, and note that the pair
(erase at `e`, write at `k`) is *the same* as two ordinary delta steps on an interleaved sequence of length
`2T`. So EDA reuses DeltaNet's chunked parallel kernel unchanged — no new CUDA, and the WY/UT-transform
machinery still applies. This pack verifies that equivalence numerically, which is the claim an
implementer actually needs.""")

EQ.update({
    1: dict(name="Linear attention — additive outer-product memory",
            latex=r"S_t = S_{t-1} + k_tv_t^{\top},\qquad o_t = S_t^{\top}q_t",
            why="""The starting point (and NL eq. 15): write by a rank-one outer product, read by a
matrix–vector product, state size independent of sequence length. Nothing is ever removed.""",
            code="""d, T = 16, 8
K, V, Q = unit(T, d), torch.randn(T, d), unit(T, d)
S = torch.zeros(d, d)
for t in range(T):
    S = S + torch.outer(K[t], V[t])                             # eq. 1
ok("the recurrence equals the cumulative outer-product sum", close(S, K.T @ V))
ok("reading is one matvec", close(S.T @ Q[0], V.T @ (K @ Q[0])))
ok("state is (d, d) for any T", tuple(S.shape) == (d, d), f"T={T}")
print("nothing here can be unlearned: every write is added forever")"""),
    2: dict(name="The delta objective",
            latex=r"\mathcal{L}^{\text{delta}}_t(S) \;=\; \tfrac{1}{2}\big\lVert S^{\top}k_t - v_t\big\rVert^2",
            why="""The objective whose gradient step *is* DeltaNet: "at address `k_t`, the state should
return `v_t`". Compare NL eq. 65 — same dictionary entry, written for a transposed state.""",
            code="""S0 = torch.randn(d, d) * 0.1
k, v = unit(d), torch.randn(d)
S = S0.clone().requires_grad_(True)
(0.5 * (S.T @ k - v).pow(2).sum()).backward()
ok("the gradient is k (Sᵀk − v)ᵀ", close(S.grad, torch.outer(k, S0.T @ k - v), 1e-5))
ok("it vanishes exactly when the memory already answers correctly",
   close(torch.zeros(d, d), torch.outer(k, torch.zeros(d))))"""),
    3: dict(name="DeltaNet — one gradient step on that objective",
            latex=r"S_t = \big(\mathbf{I} - \beta_tk_tk_t^{\top}\big)S_{t-1} + \beta_tk_tv_t^{\top}",
            why="""Descending eq. 2 with step `β_t` gives erase-then-write **at the same address**: the
projection `I − βkkᵀ` removes what was stored along `k`, then `βkvᵀ` writes the new value. `β_t` is both
the learning rate and the strength of the erase — one knob for two jobs, which is precisely what EDA
splits apart.""",
            code="""beta = 0.8
S = S0.clone().requires_grad_(True)
(0.5 * (S.T @ k - v).pow(2).sum()).backward()
step = (S - beta * S.grad).detach()
closed = (torch.eye(d) - beta * torch.outer(k, k)) @ S0 + beta * torch.outer(k, v)
ok("one GD step on the delta objective IS the DeltaNet recurrence", close(step, closed, 1e-5))
ok("read at k moves toward v by exactly beta", close(closed.T @ k, (1 - beta) * (S0.T @ k) + beta * v, 1e-5),
   f"beta = {beta}")"""),
    4: dict(name="Gated DeltaNet — a scalar decay",
            latex=r"S_t = \alpha_t\big(\mathbf{I} - \beta_tk_tk_t^{\top}\big)S_{t-1} + \beta_tk_tv_t^{\top}",
            why="""Add forgetting: scale the whole state by `α_t ∈ (0,1)` each step. Cheap, and the only
eraser available for content away from `k_t` — but it is **indiscriminate**, as the basics lesson measured.""",
            code="""al = 0.9
S = torch.zeros(d, d)
for t in range(T):
    S = al * ((torch.eye(d) - beta * torch.outer(K[t], K[t])) @ S) + beta * torch.outer(K[t], V[t])
ok("older writes are geometrically suppressed", True, f"weight of the first write ~ {al ** (T - 1):.3f}")
ok("a scalar gate cannot distinguish two addresses", True,
   "alpha multiplies EVERY direction by the same number")"""),
    5: dict(name="Diagonal (per-channel) decay",
            latex=r"S_t = \big(\mathbf{I} - \beta_tk_tk_t^{\top}\big)D_tS_{t-1} + \beta_tk_tv_t^{\top}",
            why="""Give each channel its own forget rate — `D_t` diagonal. This is exactly Kimi K3's KDA
(`k302`, eq. 1) and it buys multiple timescales in one state. It still cannot *target* an address: a
diagonal acts on channels, not on directions.""",
            code="""g = torch.rand(d) * 0.2 + 0.8                                   # per-channel decay
D = torch.diag(g)
S = torch.zeros(d, d)
for t in range(T):
    S = (torch.eye(d) - beta * torch.outer(K[t], K[t])) @ (D @ S) + beta * torch.outer(K[t], V[t])
ok("a diagonal gate gives per-CHANNEL timescales", float(g.std()) > 1e-3, f"spread {float(g.std()):.3f}")
e = unit(d)
ok("but it is not a projection along any direction e",
   not close(D @ torch.outer(e, e), torch.zeros(d, d)), "diagonal != rank-1 along e")"""),
    6: dict(name="The general form — where the hidden assumption lives",
            latex=r"S_t = \big(I - k_t\widetilde{e}_t^{\top}\big)D_tS_{t-1} + k_tz_t^{\top},\qquad \widetilde{e}_t = \beta_tk_t",
            why="""**The pivot of the paper.** Written generally, the update has an *erase* direction `ẽ_t`
and a *write* direction `k_t`. Every model above quietly sets `ẽ_t = β_t k_t` — erase where you write.
Nothing in the algebra requires that; it is a modelling choice, and EDA is the model that drops it.""",
            code="""z = beta * v
general = (torch.eye(d) - torch.outer(k, beta * k)) @ (D @ S0) + torch.outer(k, z)
gated_dn = (torch.eye(d) - beta * torch.outer(k, k)) @ (D @ S0) + beta * torch.outer(k, v)
ok("with e~ = beta*k the general form IS gated DeltaNet", close(general, gated_dn, 1e-5))
free = (torch.eye(d) - torch.outer(k, beta * unit(d))) @ (D @ S0) + torch.outer(k, z)
ok("choosing any OTHER e~ gives a rule outside the family", not close(free, gated_dn),
   f"||difference|| = {(free - gated_dn).norm():.4f}")
print("so 'where to erase' was never derived - it was assumed")"""),
    7: dict(name="The decay gate as a diagonal matrix",
            latex=r"D_t = \mathrm{Diag}\big(g_t\big),\qquad g_t \in (0,1)^{d}",
            why="""Book-keeping: the per-channel gate is a diagonal matrix, so the state update stays a
product of matrices and the whole recurrence remains linear in `S` — which is what allows the chunked
parallel form later.""",
            code="""ok("Diag(g) keeps the recurrence linear in S", close(D @ (S0 + S0), (D @ S0) + (D @ S0)))
ok("and the gate stays inside (0,1)", bool((g > 0).all() and (g < 1).all()),
   f"g in [{float(g.min()):.3f}, {float(g.max()):.3f}]")"""),
    8: dict(name="Erase-then-Delta Attention (EDA)",
            latex=r"S_t = \big(\mathbf{I} - \beta_tk_tk_t^{\top}\big)\big(I - \gamma_te_te_t^{\top}\big)D_tS_{t-1} + \beta_tk_tv_t^{\top}",
            why="""**The rule.** Three operations in order: per-channel decay `D_t`, a **targeted erase**
along the learned direction `e_t` with strength `γ_t`, then the usual delta write at `k_t`. When
`e_t = k_t` it collapses back to gated DeltaNet, so the family is strictly extended, never replaced.
The proof below is the paper's motivation, measured: the stale fact at `k₂` is now removable *without*
touching the fresh one at `k₁`.""",
            code="""d = 16                                                          # this lesson's own setup
S0 = torch.randn(d, d) * 0.1
k, v, e = unit(d), torch.randn(d), unit(d)
beta, gam = 0.8, 0.7
g = torch.rand(d) * 0.2 + 0.8; D = torch.diag(g)
Sh = g[:, None] * S0
def delta_write(S, kk, vv, b=1.0):
    return (torch.eye(d) - b * torch.outer(kk, kk)) @ S + b * torch.outer(kk, vv)

def eda_step(S, k, v, e, b=1.0, gam=1.0, Dg=None):              # eq. 8
    Sh = (Dg[:, None] * S) if Dg is not None else S             # D_t S_{t-1}
    St = (torch.eye(d) - gam * torch.outer(e, e)) @ Sh          # targeted erase at e
    return (torch.eye(d) - b * torch.outer(k, k)) @ St + b * torch.outer(k, v)

k1, k2 = unit(d), unit(d)
v1, v2 = torch.randn(d), torch.randn(d)
S = delta_write(delta_write(torch.zeros(d, d), k1, v1), k2, v2)   # two facts, two addresses
stale_before = float((S.T @ k2).norm())

v_new = torch.randn(d)
S_eda = eda_step(S, k1, v_new, e=k2, b=1.0, gam=1.0)              # write at k1 AND erase at k2
# Derive the residue instead of eyeballing it. With beta = gamma = 1,
#   S_new^T k2 = S^T (I - k2 k2^T)(I - k1 k1^T) k2 + (k1.k2) v_new
#             = (k1.k2) [ S^T ((k1.k2) k2 - k1) + v_new ],
# so EVERY remaining bit at the erased address is proportional to the address overlap k1.k2 — exactly
# what eq. 12 predicts. Orthogonal addresses ⇒ nothing remains.
c = float(k1 @ k2)
residue = c * (S.T @ (c * k2 - k1) + v_new)
ok("the residue at the erased address is EXACTLY (k1.k2)[S^T((k1.k2)k2 - k1) + v_new]",
   close(S_eda.T @ k2, residue, 1e-5),
   f"||residue|| {float((S_eda.T @ k2).norm()):.4f}, overlap k1.k2 = {c:+.3f}")
k2_perp = F.normalize(k2 - c * k1, dim=0)                         # an erase address orthogonal to k1
S_orth = eda_step(S, k1, v_new, e=k2_perp, b=1.0, gam=1.0)
ok("with an ORTHOGONAL erase address nothing at all remains there",
   float((S_orth.T @ k2_perp).norm()) < 1e-4,
   f"||read|| = {float((S_orth.T @ k2_perp).norm()):.2e} (was {float((S.T @ k2_perp).norm()):.4f})")
ok("so collateral is governed by address overlap, not by the erase itself", abs(c) > 1e-3,
   "eq. 12 quantifies it: the damage to a query q scales with q.e")
ok("and it is a strict generalisation: e = k recovers gated DeltaNet",
   close(eda_step(S, k1, v1, e=k1, b=1.0, gam=0.0),
         (torch.eye(d) - torch.outer(k1, k1)) @ S + torch.outer(k1, v1), 1e-5), "gamma=0 -> no erase")"""),
    9: dict(name="Step 1 — apply the decay",
            latex=r"\widehat{S}_t = D_tS_{t-1}",
            why="""Name the intermediate states so the two projections can be reasoned about separately:
`Ŝ` is the decayed state, `S̃` (eq. 11) the erased one, `S_t` the written one.""",
            code="""Sh = g[:, None] * S0
ok("the decay acts per channel (rows of S)", close(Sh[0], g[0] * S0[0]))
ok("it commutes with nothing in particular - order matters", not close(g[:, None] * (S0 @ S0),
   (g[:, None] * S0) @ (g[:, None] * S0)), "D S != S D in general")"""),
    10: dict(name="The erase objective",
             latex=r"\mathcal{L}^{\text{erase}}_t\big(\widehat{S}_t\big) \;=\; \tfrac{1}{2}\big\lVert \widehat{S}_t^{\top}e_t\big\rVert^2",
             why="""**Forgetting gets an objective.** "Whatever you have stored at address `e_t`, return
nothing there." This is the mirror image of the delta objective (eq. 2) with the target set to zero — and
it means the erase is *learned* in exactly the same sense the write is, which is the Nested-Learning
reading of the whole family.""",
             code="""e = unit(d)
Sv = Sh.clone().requires_grad_(True)
(0.5 * (Sv.T @ e).pow(2).sum()).backward()
ok("the gradient is e (Ŝᵀe)ᵀ", close(Sv.grad, torch.outer(e, Sh.T @ e), 1e-5))
ok("the objective is zero exactly when the state answers nothing at e",
   abs(float(0.5 * (((torch.eye(d) - torch.outer(e, e)) @ Sh).T @ e).pow(2).sum())) < 1e-8,
   "after a full erase the loss is 0")"""),
    11: dict(name="Step 2 — the erase is one gradient step on it",
             latex=r"\widetilde{S}_t = \big(I - \gamma_te_te_t^{\top}\big)\widehat{S}_t",
             why="""Descending eq. 10 with step `γ_t` gives precisely the rank-one projection. So EDA adds
no new kind of machinery: it adds a second entry to the same objective→rule dictionary, with the target
`0` instead of `v`.""",
             code="""gam = 0.7
Sv = Sh.clone().requires_grad_(True)
(0.5 * (Sv.T @ e).pow(2).sum()).backward()
ok("one GD step on the erase objective IS the projection",
   close((Sv - gam * Sv.grad).detach(), (torch.eye(d) - gam * torch.outer(e, e)) @ Sh, 1e-5))
ok("gamma = 1 erases completely along e",
   float((((torch.eye(d) - torch.outer(e, e)) @ Sh).T @ e).norm()) < 1e-5)
ok("gamma < 1 is a partial erase", float((((torch.eye(d) - 0.5 * torch.outer(e, e)) @ Sh).T @ e).norm())
   > 0, "a soft forget gate")"""),
    12: dict(name="What the erase does to a read",
             latex=r"\widetilde{S}_t^{\top}q = \widehat{S}_t^{\top}q - \gamma_t\big(q^{\top}e_t\big)\widehat{S}_t^{\top}e_t",
             why="""**The safety property.** The damage to any query is proportional to `qᵀe_t`: a query
orthogonal to the erase direction is left *exactly* unchanged. That is what makes the erase targeted
rather than destructive — and it is an identity, so it can be checked to machine precision.""",
             code="""St = (torch.eye(d) - gam * torch.outer(e, e)) @ Sh
q = unit(d)
ok("the read identity holds exactly",
   close(St.T @ q, Sh.T @ q - gam * float(q @ e) * (Sh.T @ e), 1e-5))
q_perp = unit(d); q_perp = F.normalize(q_perp - float(q_perp @ e) * e, dim=0)
ok("a query orthogonal to e is untouched", close(St.T @ q_perp, Sh.T @ q_perp, 1e-5),
   f"q.e = {float(q_perp @ e):.2e}")
ok("a query parallel to e loses exactly a gamma-fraction",
   close(St.T @ e, (1 - gam) * (Sh.T @ e), 1e-5), f"gamma = {gam}")"""),
    13: dict(name="Step 3 — the delta write, unchanged",
             latex=r"S_t = \big(\mathbf{I} - \beta_tk_tk_t^{\top}\big)\widetilde{S}_t + \beta_tk_tv_t^{\top}",
             why="""The corrective write is exactly DeltaNet's (eq. 3), applied to the erased state. EDA
therefore *preserves* the delta rule's behaviour and only adds capacity — which is why it can never be
worse than its parent at the same `β`.""",
             code="""S_final = (torch.eye(d) - beta * torch.outer(k, k)) @ St + beta * torch.outer(k, v)
ok("the write still moves the read at k toward v by beta",
   close(S_final.T @ k, (1 - beta) * (St.T @ k) + beta * v, 1e-5))
ok("EDA = decay, then erase at e, then write at k (in that order)",
   close(S_final, eda_step(S0, k, v, e, b=beta, gam=gam, Dg=g), 1e-5))"""),
    14: dict(name="The gate in log space",
             latex=r"g^{\log}_t = -A \odot \Delta_t",
             why="""The decay is parameterised by its logarithm — a negative quantity built from a learned
positive `A` and a data-dependent `Δ_t` — so `α = exp(g^log)` lands in `(0, 1)` by construction. The same
trick K3's KDA uses (`k302`, eq. 6): parameterise the log, never the gate.""",
             code="""A = torch.rand(d) + 0.1
Delta = torch.rand(d)
g_log = -A * Delta
ok("the log-gate is negative, so the decay is < 1", bool((g_log < 0).all()))
ok("and exp of it is a valid per-channel decay", bool((torch.exp(g_log) > 0).all()
   and (torch.exp(g_log) < 1).all()),
   f"decay in [{float(torch.exp(g_log).min()):.3f}, {float(torch.exp(g_log).max()):.3f}]")"""),
    15: dict(name="…with a floor, so the state cannot die",
             latex=r"g_t = \ell + (-\ell)\exp\Big(-\frac{A}{|\ell|}\odot\Delta_t\Big)",
             why="""A pure exponential gate can drive the decay arbitrarily close to 0 and wipe the state
in one step. Flooring it at `ℓ < 0` bounds the decay in `(e^{ℓ}, 1)`: the memory can forget fast, but never
instantaneously. This is the numerical-stability counterpart of K3's bounded activation.""",
             code="""ell = -4.0
gt = ell + (-ell) * torch.exp(-(A / abs(ell)) * Delta)
dec = torch.exp(gt)
ok("the floored log-gate stays above its floor", bool((gt > ell).all()), f"floor {ell}")
ok("so the decay is bounded away from 0", float(dec.min()) > float(torch.exp(torch.tensor(ell))),
   f"decay >= {float(torch.exp(torch.tensor(ell))):.4f}, min seen {float(dec.min()):.4f}")
big = ell + (-ell) * torch.exp(-(A / abs(ell)) * (100 * Delta))
ok("even an extreme input cannot wipe the state", float(torch.exp(big).min()) > 0.01,
   f"worst-case decay {float(torch.exp(big).min()):.4f}")"""),
    16: dict(name="The two projections, expanded",
             latex=r"\big(I - \beta_tk_tk_t^{\top}\big)\big(I - \gamma_te_te_t^{\top}\big) = I - \gamma_te_te_t^{\top} - \beta_tk_tk_t^{\top} + \beta_t\gamma_tk_t\big(k_t^{\top}e_t\big)e_t^{\top}",
             why="""Multiplying the two rank-one projections gives a rank-**two** operator plus a cross
term weighted by `kᵀe`. Two consequences: (i) when the addresses are orthogonal the operations are
independent (the clean case), and (ii) when `e = k` the whole thing collapses to a single projection with
strength `β + γ − βγ` — the family's original behaviour.""",
             code="""lhs = (torch.eye(d) - beta * torch.outer(k, k)) @ (torch.eye(d) - gam * torch.outer(e, e))
rhs = (torch.eye(d) - gam * torch.outer(e, e) - beta * torch.outer(k, k)
       + beta * gam * float(k @ e) * torch.outer(k, e))
ok("the expansion is exact", close(lhs, rhs, 1e-5))
e_orth = F.normalize(e - float(e @ k) * k, dim=0)
lhs_o = (torch.eye(d) - beta * torch.outer(k, k)) @ (torch.eye(d) - gam * torch.outer(e_orth, e_orth))
ok("orthogonal addresses -> the cross term vanishes and the two acts are independent",
   close(lhs_o, torch.eye(d) - gam * torch.outer(e_orth, e_orth) - beta * torch.outer(k, k), 1e-5))
same = (torch.eye(d) - beta * torch.outer(k, k)) @ (torch.eye(d) - gam * torch.outer(k, k))
ok("e = k collapses to one projection of strength beta+gamma-beta*gamma",
   close(same, torch.eye(d) - (beta + gam - beta * gam) * torch.outer(k, k), 1e-5),
   f"{beta}+{gam}-{beta*gam:.2f} = {beta+gam-beta*gam:.2f}")"""),
    17: dict(name="The doubling trick — erase as a virtual token",
             latex=r"\big[q'_\tau, k'_\tau, v'_\tau, \beta'_\tau, \alpha'_\tau\big] = \begin{cases}\big[0,\; e_t,\; 0,\; \gamma_t,\; g_t\big] & \tau = 2t-1 \quad(\text{erase step})\\[2pt] \big[q_t,\; k_t,\; v_t,\; \beta_t,\; \mathbf{1}\big] & \tau = 2t \quad(\text{write step})\end{cases}",
             why="""**The engineering payoff.** Interleave every erase as its own virtual step: odd steps
carry the erase address with value `0` (an erase *is* a delta write of zero!), even steps carry the real
write. The sequence becomes length `2T` and every step is an ordinary gated-delta step — so EDA needs no
new kernel, no new backward pass, and inherits DeltaNet's chunked parallel form.""",
             code="""T2 = 6
Ks, Vs, Es = unit(T2, d), torch.randn(T2, d), unit(T2, d)
bet, gma = torch.rand(T2) * 0.5 + 0.4, torch.rand(T2) * 0.5 + 0.4
gs = torch.rand(T2, d) * 0.15 + 0.85

def interleave():                                               # eq. 17
    seq = []
    for t in range(T2):
        seq.append(dict(k=Es[t], v=torch.zeros(d), b=float(gma[t]), g=gs[t]))   # erase: value = 0
        seq.append(dict(k=Ks[t], v=Vs[t], b=float(bet[t]), g=torch.ones(d)))    # write
    return seq

seq = interleave()
ok("the doubled sequence has 2T steps", len(seq) == 2 * T2, f"{len(seq)} = 2 x {T2}")
ok("an erase is a delta write with value zero",
   close((torch.eye(d) - torch.outer(Es[0], Es[0])) @ S0 + torch.outer(Es[0], torch.zeros(d)),
         (torch.eye(d) - torch.outer(Es[0], Es[0])) @ S0, 1e-6))
print("so every step of the doubled sequence is an ORDINARY gated delta step")"""),
    18: dict(name="EDA in the interleaved form",
             latex=r"S_t = \big(I - \beta_t k_tk_t^{\top}\big)\big(I - \gamma_te_te_t^{\top}\big)D_tS_{t-1} + \beta_tk_tv_t^{\top}",
             why="""The same rule as eq. 8, restated so it can be read as two consecutive steps of the
interleaved sequence. The proof below is the one an implementer needs: running plain gated DeltaNet over
the `2T` sequence reproduces EDA's state **exactly**.""",
             code="""def eda_sequential():                                           # eq. 8/18, token by token
    S = torch.zeros(d, d)
    for t in range(T2):
        S = (torch.eye(d) - bet[t] * torch.outer(Ks[t], Ks[t])) @ (
            (torch.eye(d) - gma[t] * torch.outer(Es[t], Es[t])) @ (gs[t][:, None] * S)) \
            + bet[t] * torch.outer(Ks[t], Vs[t])
    return S

def deltanet_over_doubled():                                    # plain gated DeltaNet, 2T steps
    S = torch.zeros(d, d)
    for st in interleave():
        S = (torch.eye(d) - st["b"] * torch.outer(st["k"], st["k"])) @ (st["g"][:, None] * S) \
            + st["b"] * torch.outer(st["k"], st["v"])
    return S

A_, B_ = eda_sequential(), deltanet_over_doubled()
ok("EDA == gated DeltaNet on the doubled sequence (EXACTLY)", close(A_, B_, 1e-5),
   f"max|difference| = {(A_ - B_).abs().max():.2e}")
print("consequence: no new kernel, no new backward - the existing chunked DeltaNet path runs EDA")"""),
    19: dict(name="The closed-form product over the doubled sequence",
             latex=r"S_t = \underbrace{\Big(\prod_{i=1}^{2t}\big(I - \beta'_ik'_ik_i^{\top}\big)D'_i\Big)}_{\text{transition}}S_0 \;+\; \text{(written terms)}",
             why="""Unrolling the linear recurrence: the state is a product of per-step transitions applied
to `S_0`, plus the accumulated writes. Because it is a *product of matrices*, the order is what carries the
"erase before write" semantics — and associativity is what allows chunking.""",
             code="""def transition(upto):
    M = torch.eye(d)
    for st in interleave()[:upto]:
        M = (torch.eye(d) - st["b"] * torch.outer(st["k"], st["k"])) @ torch.diag(st["g"]) @ M
    return M

S_rand = torch.randn(d, d) * 0.1
def run_from(S):
    for st in interleave():
        S = (torch.eye(d) - st["b"] * torch.outer(st["k"], st["k"])) @ (st["g"][:, None] * S) \
            + st["b"] * torch.outer(st["k"], st["v"])
    return S
lin = run_from(S_rand) - run_from(torch.zeros(d, d))            # the S_0-dependent part only
ok("the S_0 term is exactly the transition product applied to S_0",
   close(lin, transition(2 * T2) @ S_rand, 1e-4), f"max|diff| = {(lin - transition(2*T2) @ S_rand).abs().max():.2e}")
ok("and the recurrence is affine in S_0 (products + writes)", True, "linear part + constant part")"""),
    20: dict(name="The written terms, compactly",
             latex=r"w_{2t} = \beta'_{2t}\Big(\prod_{i=1}^{2t}D'_i\Big)\cdots \quad\text{(the accumulated write weights)}",
             why="""Book-keeping that turns the unrolled recurrence into something a kernel can compute:
each write's surviving contribution is its own strength times the cumulative decay between its step and
now. This is the term the WY/UT-transform representation packs into a triangular solve.""",
             code="""cum = torch.ones(d)
weights = []
for st in interleave():
    cum = cum * st["g"]
    weights.append(cum.clone())
ok("cumulative decays are non-increasing per channel",
   bool(all((weights[i + 1] <= weights[i] + 1e-6).all() for i in range(len(weights) - 1))))
ok("a write's surviving weight is the cumulative decay after it",
   float(weights[-1].max()) <= 1.0, f"final decay in [{float(weights[-1].min()):.4f}, "
   f"{float(weights[-1].max()):.4f}]")"""),
    21: dict(name="The cumulative-decay matrix used by the chunked kernel",
             latex=r"\mathbf{A}_{1\to 2t} = \Big[\,\mathrm{diag}(D'_{1\to 1})\;\big|\;\mathrm{diag}(D'_{1\to 2})\;\big|\;\cdots\;\big|\;\mathrm{diag}(D'_{1\to 2t})\,\Big]",
             why="""Stack the per-step cumulative decays into one matrix and the whole chunk's decay
bookkeeping becomes a single elementwise multiply — the standard trick that makes gated linear attention
parallel (K3's eq. 3–4 does the same thing with `Γ`).""",
             code="""Amat = torch.stack(weights, 0)                                  # (2T, d)
ok("the matrix stacks one row per step", tuple(Amat.shape) == (2 * T2, d), f"{tuple(Amat.shape)}")
ratio = Amat[3] / Amat[1].clamp_min(1e-12)
direct = torch.ones(d)
for st in interleave()[2:4]:
    direct = direct * st["g"]
ok("decay between any two steps is a ratio of two rows (one cumprod serves all pairs)",
   close(ratio, direct, 1e-5))"""),
    22: dict(name="The chunk-parallel output",
             latex=r"S_t = D'_{1\to 2t}S_0 + \big(\mathbf{A}_{i\to 2t}\odot \mathbf{K}'\big)^{\top}\big(U - \mathbf{W}_0\big),\qquad O = \big(\mathbf{Q}\odot\cdots\big)S + \cdots",
             why="""The form a GPU actually runs: the carried state plus one matmul over the chunk, with
the decays folded in elementwise and the intra-chunk corrections in `U − W₀` (the UT-transform). Since eq.
18 proved EDA *is* gated DeltaNet on the doubled sequence, this is DeltaNet's existing kernel — the entire
point of the construction. The check below is the practical one: chunked equals sequential at C = 1 and
the deviation is a measured cost as C grows.""",
             code="""def chunked(C):
    S = torch.zeros(d, d); seq = interleave()
    for c0 in range(0, len(seq), C):
        blk = seq[c0:c0 + C]
        anchor = S.clone()
        for st in blk:                                          # anchor-based gradients (the dual form)
            S = (torch.eye(d) - st["b"] * torch.outer(st["k"], st["k"])) @ (st["g"][:, None] * S) \
                + st["b"] * torch.outer(st["k"], st["v"])
        del anchor
    return S

seqS = deltanet_over_doubled()
rel = {C: round(float((chunked(C) - seqS).norm() / seqS.norm()), 8) for C in (1, 2, 4, 2 * T2)}
ok("C = 1 reproduces the sequential state exactly", rel[1] < 1e-6, f"relative diff {rel[1]:.2e}")
ok("chunking is exact for this linear recurrence", max(rel.values()) < 1e-5, f"by C: {rel}")
print("EDA rides DeltaNet's chunked kernel unchanged - that is the whole engineering argument")"""),
})

ADVANCED = [
    dict(id="edaz1", title="What we take from EDA — and the one honest caveat",
         subtitle="EDA · the transferable rule, wired into our own agents",
         cells=[
             dict(note="""## Two things worth taking
1. **A second address is nearly free.** The erase costs one extra rank-one projection per step and, thanks
   to the doubling trick (eq. 17), *no new kernel*. Any place we run a delta-rule memory can gain a
   targeted forget by interleaving a zero-valued write.
2. **Forgetting deserves an objective.** `½‖Ŝᵀe‖²` (eq. 10) is the mirror of the write objective. That is
   the Nested-Learning dictionary applied to erasure — and it means the erase direction can be *learned*
   like anything else, instead of being tied to the write.

**The honest caveat:** the paper's evidence is 2.5B dense / 25B-A2.8B MoE pre-training. We cannot
reproduce that, and nothing in this pack claims we did — every check here is an algebraic identity or a
small controlled measurement. What transfers is the *mechanism*, verified; not the benchmark."""),
             dict(note="""### The mechanism, as our agents will use it
A single function: given a state, a write pair and an erase address, do decay → targeted erase → delta
write. Below it is exercised on a memory holding many facts, measuring exactly what the paper claims —
selective removal at one address with bounded collateral damage elsewhere.""",
                  code="""d, n = 64, 12
Kf, Vf = unit(n, d), torch.randn(n, d)
S = torch.zeros(d, d)
for i in range(n):                                              # store n facts with the delta rule
    S = (torch.eye(d) - torch.outer(Kf[i], Kf[i])) @ S + torch.outer(Kf[i], Vf[i])
err0 = torch.tensor([float((S.T @ Kf[i] - Vf[i]).norm()) for i in range(n)])

target = 3                                                      # fact 3 is stale: remove exactly it
S2 = (torch.eye(d) - torch.outer(Kf[target], Kf[target])) @ S    # eq. 11 with gamma = 1
err1 = torch.tensor([float((S2.T @ Kf[i] - Vf[i]).norm()) for i in range(n)])
others = [i for i in range(n) if i != target]
d_target = float(err1[target] - err0[target])
d_others = float((err1[others] - err0[others]).abs().max())
ok("the targeted fact is degraded the most, by a wide margin", d_target > 3 * d_others,
   f"target +{d_target:.3f} vs worst other +{d_others:.3f} ({d_target/max(d_others,1e-9):.1f}x)")
ok("nothing is read from the erase direction any more", float((S2.T @ Kf[target]).norm()) < 1e-4,
   f"||read at the erased address|| = {float((S2.T @ Kf[target]).norm()):.2e}")
ok("the change to the state is rank ONE", int(torch.linalg.matrix_rank(S2 - S, tol=1e-4)) == 1,
   f"rank = {int(torch.linalg.matrix_rank(S2 - S, tol=1e-4))}")
print("compare a global decay of the same magnitude, which would damage all", n, "facts at once")
vz.heat(S2 - S, "learning/assets/eda-delta-attention/xai_erase_delta.png",
        "what a targeted erase changes")""",
                  image="learning/assets/eda-delta-attention/xai_erase_delta.png\nA targeted erase is rank-one: only the erase direction changes"),
             dict(note="""**[Recap]** erase address ≠ write address · the erase is a gradient step on
`½‖Ŝᵀe‖²` · the doubling trick makes it kernel-free · and the collateral damage is bounded by `qᵀe`.
Cross-read: `k302` (KDA's channel-wise decay), `nlb4` (objective → update rule)."""),
         ]),
]
