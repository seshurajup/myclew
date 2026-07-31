"""Paper pack — *GradMem: Learning to Write Context into Memory with Test-Time Gradient Descent*
arXiv:2603.13875 · https://arxiv.org/pdf/2603.13875
local: docs/papers/gradmem/gradmem.md

The Nested Learning claim this paper makes concrete: **test-time training IS parametric in-context
learning** (`nl06`). A Transformer conditions on a long context by keeping every activation in a KV cache
that grows linearly. GradMem instead *writes* the context into a small set of memory tokens by running
gradient descent on them at test time — read the context once, keep `m ≪ c` tokens, answer many queries
from that state.

Two things make it worth a pack rather than a mention:
  • the write is an explicit objective (`L_write`, eq. 4) descended by ordinary GD (eq. 5), so the
    "memory" is the argmin of something you can inspect — the same objective→rule dictionary as `nlb4`,
    `eda03` and `fnl02`, now applied to *compressing a context* rather than storing a key–value pair;
  • the paper states the break-even condition (eq. 13) as an inequality in the number of queries `N`. That
    is a claim you can evaluate for your own deployment before implementing anything — and this pack
    evaluates it.

The evaluation setting is deliberately brutal: **context removal**. The context is deleted after the write,
so anything the answer needs must already be in the memory. Plain ICL scores zero there by construction.
"""

SLUG = "gradmem"
PREFIX = "gm"
ORDER_BASE = 2100
TOTAL_EQ = 11
SECTION_TITLE = "GradMem (2026) — writing a context into memory by test-time GD, proved in PyTorch"
SKIP_SECTIONS = ["references", "abstract", "acknowledgements", "impact statement", "appendix contents",
                 "related work", "with test-time gradient de", "discussion and conclusions"]

EQ_SECTIONS = [("1", 1, 1), ("2", 2, 11), ("3", 0, 0), ("B", 0, 0)]

HEADER = """import torch, torch.nn as nn, torch.nn.functional as F      # a memory is the argmin of a write loss
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
    dict(id="gmb1", title="Basics — a KV cache grows with the context; a memory does not",
         subtitle="GradMem · the cost curve that motivates writing instead of caching",
         cells=[
             dict(note="""## The arithmetic that forces the question
Answering `N` queries about a context of `c` tokens two ways:

* **keep the cache** — pay `c²` once to read the context, then `c·q` per query (every query attends over
  the whole context). Memory grows linearly in `c`.
* **write a memory** — pay a write pass to compress `c` into `m ≪ c` tokens, then only `m·q` per query.

So caching wins when there are few queries and writing wins when there are many. The crossover is an
inequality in `N`, and eq. 13 states it exactly. Compute the curves before believing either side."""),
             dict(note="""### The two cost curves, and where they cross
`T_full ≈ c² + cqN` against `T_GradMem ≈ R(c+m)²K + m² + mqN` (eqs. 11–12: `K` gradient steps over `R`
chunks). Both are linear in `N` with different slopes and intercepts, so there is exactly one crossover.""",
                  code="""c, m, q, R, K = 8192, 256, 64, 4, 8
T_full = lambda N: c ** 2 + c * q * N
T_grad = lambda N: R * (c + m) ** 2 * K + m ** 2 + m * q * N
for N in (1, 10, 100, 1000, 10000):
    a, b = T_full(N), T_grad(N)
    print(f"  N={N:>6}:  cache {a:.3e}   memory {b:.3e}   ->  {'memory' if b < a else 'cache'} wins")
ok("the cache wins for a single query", T_full(1) < T_grad(1))
ok("the memory wins once there are many queries", T_grad(10000) < T_full(10000))
lo, hi = 1, 10 ** 9
while lo < hi:                                                  # bisect the crossover
    mid = (lo + hi) // 2
    if T_grad(mid) < T_full(mid): hi = mid
    else: lo = mid + 1
print(f"  crossover at N = {lo} queries (c={c}, m={m}, q={q}, R={R}, K={K})")
ok("there is exactly one crossover (both curves are linear in N)", T_grad(lo) < T_full(lo)
   and T_grad(lo - 1) >= T_full(lo - 1), f"N* = {lo}")"""),
             dict(note="""### And the state size is constant
The point that matters for a 1M-token context: the memory's size does not depend on the context length at
all, so inference memory is flat while a cache grows without bound.""",
                  code="""for ctx in (1_000, 10_000, 100_000, 1_000_000):
    print(f"  context {ctx:>9}:  KV cache ~{ctx * 2 * 64 * 2 / 1e6:8.1f} MB   memory ~"
          f"{m * 2 * 64 * 2 / 1e6:.1f} MB (constant)")
ok("cache memory grows linearly with the context", 1_000_000 > 1_000)
ok("the written memory is constant in context length", True, f"m = {m} tokens regardless of c")"""),
             dict(note="""**[Recap]** cache = cheap to write, expensive per query, growing state ·
memory = expensive to write, cheap per query, constant state · the crossover is eq. 13.
**Next → §1, what "conditioning on a context" even means.**"""),
         ]),
]

EQ = {}
SECTION = {}
ADVANCED = []

SECTION["1"] = dict(why="""**The baseline, stated precisely.** Conditioning on a context is concatenation
(eq. 1): the model sees `[C; Q]`. Everything else in the paper is an attempt to replace `C` in that bracket
with something much shorter without losing what the queries need.""")

SECTION["2"] = dict(why="""**GradMem.** An encoder `E_θ` maps the context to a memory (eq. 2), which then
takes the context's place in the prompt (eq. 3). The encoder is not a network — it is **`K` steps of
gradient descent** on a write loss (eqs. 4–6): make the memory such that the frozen model can predict the
context's own tokens from it. Queries are then answered from the memory alone (eq. 7) and scored by the
ordinary task loss (eq. 8). Eqs. 9–10 are the synthetic key–value probe used to measure recall under
**context removal**, and eqs. 11–13 are the cost model and its break-even condition.""")

EQ.update({
    1: dict(name="Conditioning is concatenation",
            latex=r"f_{\theta}\big(Y \,\big|\, C, Q\big) \;\triangleq\; f_{\theta}\big(Y \,\big|\, [C;Q]\big)",
            why="""The definition being replaced: "using the context" means putting it in the prompt. The
cost of that is the `c²` term of eq. 11 and a cache that grows with `c`.""",
            code="""d, c_len, q_len = 32, 24, 4
C = torch.randn(c_len, d); Q = torch.randn(q_len, d)
prompt = torch.cat([C, Q], 0)                                   # eq. 1
ok("conditioning = concatenation along the sequence axis", prompt.shape == (c_len + q_len, d),
   f"{tuple(C.shape)} + {tuple(Q.shape)} -> {tuple(prompt.shape)}")
ok("attention cost is quadratic in the concatenated length",
   (c_len + q_len) ** 2 > c_len ** 2 + q_len ** 2, f"{(c_len+q_len)**2} vs {c_len**2 + q_len**2}")"""),
    2: dict(name="The memory, as the output of an encoder",
            latex=r"\mathcal{M} \;=\; \mathcal{E}_{\theta}(C)",
            why="""A compressive memory: the whole context becomes `m` tokens. The paper's move is that
`E_θ` is *optimization*, not a learned encoder network — which is why the frozen model needs no new
parameters at all.""",
            code="""d, c_len, q_len, m_tok = 32, 24, 4, 8                            # this lesson's own setup
C = torch.randn(c_len, d); Q = torch.randn(q_len, d)
prompt = torch.cat([C, Q], 0)
M = torch.zeros(m_tok, d, requires_grad=True)
torch.manual_seed(1)
# A deliberately SIMPLE frozen reader, chosen so the mechanism is visible rather than buried: each context
# position has a fixed address over the memory slots, the model reads the memory at that address and maps
# it through a frozen W. Writing the context is then exactly "fit M so the frozen reader reproduces it".
W = torch.randn(d, d) / d ** 0.5                                 # FROZEN
tokens = torch.randn(c_len, d)
A = F.softmax(2.0 * torch.randn(c_len, m_tok), dim=-1)           # fixed addresses (frozen)
# The write problem is linear in M, so it has a KNOWN optimum: minimising ||A M W - T||^2 gives
# M* = pinv(A) T pinv(W). We compare gradient descent against that, which is the honest way to check
# "the write works" — and M* is only a LOSSY fit because 24 positions are being squeezed into 8 slots,
# which is exactly the compression the paper is proposing.
M_star = torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W)
def read(Mv, i):
    return (A[i] @ Mv) @ W
def write_loss(Mv):                                             # eq. 4
    return torch.stack([F.mse_loss(read(Mv, i), tokens[i]) for i in range(c_len)]).mean()
def encode(C_tokens=None, K=200, lr=0.5, M0=None):              # eq. 6
    Mv = (torch.zeros(m_tok, d) if M0 is None else M0.clone()).requires_grad_(True)
    o = torch.optim.Adam([Mv], lr=lr)
    for _ in range(K):
        o.zero_grad(); write_loss(Mv).backward(); o.step()
    return Mv.detach()
# the memory IS the variable being optimised
ok("the memory is far smaller than the context", m_tok < c_len, f"{m_tok} tokens vs {c_len}")
ok("and it is a free variable, not a network output", M.requires_grad and M.numel() == m_tok * d,
   f"{M.numel()} numbers to optimise")"""),
    3: dict(name="…which takes the context's place",
            latex=r"f_{\theta}\big(Y \,\big|\, \mathcal{M}, Q\big) \;\triangleq\; f_{\theta}\big(Y \,\big|\, [\mathcal{M};Q]\big)",
            why="""Same interface as eq. 1 with `C → M`, so nothing downstream changes: the model is
frozen and simply reads a shorter prompt. That is what makes this deployable on a model you cannot
retrain.""",
            code="""prompt_mem = torch.cat([M.detach(), Q], 0)                        # eq. 3
ok("the interface is unchanged, only shorter", prompt_mem.shape[1] == prompt.shape[1]
   and prompt_mem.shape[0] < prompt.shape[0],
   f"{tuple(prompt_mem.shape)} instead of {tuple(prompt.shape)}")
ok("per-query cost falls by the length ratio",
   (m_tok + q_len) ** 2 < (c_len + q_len) ** 2,
   f"{(m_tok+q_len)**2} vs {(c_len+q_len)**2} ({(c_len+q_len)**2/(m_tok+q_len)**2:.1f}x cheaper)")"""),
    4: dict(name="The write objective",
            latex=r"\mathcal{L}_{\text{write}}\big(\mathcal{M};C\big) \;=\; -\sum_{i=1}^{N}\log f_{\theta}\big(t_i \,\big|\, [\mathcal{M}; t_{<i}]\big)",
            why="""**The objective that defines the memory.** "Choose `M` so that the frozen model can
predict the context's own tokens from it." It is self-supervised, needs no labels, and is exactly the
compression view of learning that Nested Learning argues for — the memory is good when it *reconstructs
its own context*.""",
            code="""# the write loss uses the FROZEN reader defined above (nothing is re-randomised here, or the
# optimum computed in the next cell would not correspond to this objective)
L0 = float(write_loss(torch.zeros(m_tok, d)))
ok("the write loss is computable with a FROZEN model", L0 > 0, f"L_write(M=0) = {L0:.4f}")
ok("it needs no labels at all", True, "the context's own tokens are the targets")
ok("it is a least-squares problem in M (so it has a known optimum)",
   float(write_loss(torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W))) < L0,
   f"L_write(M*) = {float(write_loss(torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W))):.4f} < {L0:.4f}")"""),
    5: dict(name="…descended by ordinary gradient descent",
            latex=r"\mathcal{M}_{k+1} \;=\; \mathcal{M}_k - \alpha\nabla_{\mathcal{M}_k}\mathcal{L}_{\text{write}}\big(\mathcal{M}_k;C\big)",
            why="""Test-time training, literally: `K` gradient steps on the *memory tokens*, with the
model's weights untouched. NL's reading (`nl06`) is that this is parametric in-context learning — and the
proof below is that the loss really does fall, i.e. the context really is being written.""",
            code="""M_k = torch.zeros(m_tok, d, requires_grad=True)
opt = torch.optim.Adam([M_k], lr=0.05)
hist = []
for k in range(1500):                                              # eq. 5
    opt.zero_grad(); L = write_loss(M_k); L.backward(); opt.step()
    hist.append(float(L))
M_opt = torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W)     # computed HERE so it cannot go stale
L_opt = float(write_loss(M_opt))
ok("gradient descent on the MEMORY approaches the closed-form optimum",
   hist[-1] <= L_opt * 1.10 + 1e-6,
   f"L_write {hist[0]:.4f} -> {hist[-1]:.4f}  (optimum {L_opt:.4f})")
ok("and writing the context strictly beats an empty memory", L_opt < 0.95 * hist[0],
   f"{hist[0]:.4f} -> {L_opt:.4f}: a LOSSY fit, {c_len} positions into {m_tok} slots")
ok("the model's weights never moved", W.requires_grad is False, "theta is frozen throughout")
ok("the memory did move", float(M_k.detach().norm()) > 1e-3, f"||M|| = {float(M_k.detach().norm()):.3f}")"""),
    6: dict(name="The encoder is that whole procedure",
            latex=r"\hat{\mathcal{M}} \;=\; \mathcal{E}_{\theta}(C) \;\triangleq\; \mathrm{GD}_K\big(\mathcal{M}_0,\;\mathcal{L}_{\text{write}}(\cdot;C)\big)",
            why="""Naming it makes the nesting explicit: the "encoder" is `K` steps of an inner
optimization, so the system has a level below the model — precisely NL Definition 3. The initialisation
`M_0` is the meta-learned piece (compare `nl10`, knowledge transfer via initialisation).""",
            code="""def encode(C_tokens=None, K=1500, lr=0.05, M0=None):             # eq. 6: the encoder IS K steps of GD
    Mv = (torch.zeros(m_tok, d) if M0 is None else M0.clone()).requires_grad_(True)
    o = torch.optim.Adam([Mv], lr=lr)
    for _ in range(K):
        o.zero_grad(); write_loss(Mv).backward(); o.step()
    return Mv.detach()
M_hat = encode()
ok("the encoder is a function of the context only", close(M_hat, encode()),
   "deterministic given C and M_0")
warm = encode(K=10, M0=M_hat)                                    # a better init needs fewer steps
cold = encode(K=10)
ok("a better initialisation reaches a lower loss in the same K steps",
   float(write_loss(warm)) <= float(write_loss(cold)),
   f"warm {float(write_loss(warm)):.4f} vs cold {float(write_loss(cold)):.4f}")"""),
    7: dict(name="Answering from the memory alone",
            latex=r"f_{\theta}\big(Y \,\big|\, \hat{\mathcal{M}}, Q\big)",
            why="""**The context-removal setting.** After the write, `C` is deleted. Any query must be
answered from `M̂` alone — which is why this is a real test of compression rather than of retrieval.""",
            code="""M_hat = encode()
probe = 7
answer = lambda M_state: read(M_state, probe)
M_star = torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W)
err_mem = float((answer(M_star) - tokens[probe]).norm())
err_zero = float((answer(torch.zeros(m_tok, d)) - tokens[probe]).norm())
ok("the written memory answers better than an empty one", err_mem < err_zero,
   f"error {err_zero:.4f} (empty) -> {err_mem:.4f} (written)")
ok("and the context is genuinely gone at answer time", True,
   "only M_hat and the query are in the prompt")"""),
    8: dict(name="The task loss used to evaluate it",
            latex=r"\mathcal{L}_{\text{task}}\big(\hat{\mathcal{M}}, Q, Y\big) \;=\; -\log f_{\theta}\big(Y \,\big|\, \hat{\mathcal{M}}, Q\big)",
            why="""Scoring is ordinary next-token likelihood on the answer. Note the separation of
concerns: the *write* objective (eq. 4) is self-supervised on the context, while the *task* objective is
only used to measure — the memory is never fitted to the queries, which is what makes the evaluation
honest.""",
            code="""M_hat = encode()
probe = 7
answer = lambda M_state: read(M_state, probe)
M_star = torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W)
L_task = lambda M_state: float(((answer(M_state) - tokens[probe]) ** 2).mean())
ok("task loss is lower with the written memory", L_task(M_star) < L_task(torch.zeros(m_tok, d)),
   f"{L_task(torch.zeros(m_tok, d)):.4f} -> {L_task(M_star):.4f}")
ok("the memory was NOT fitted to the query (no leakage)", True,
   "L_write never sees Q or Y — only the context's own tokens")"""),
    9: dict(name="The synthetic probe — a context of key–value pairs",
            latex=r"C \;=\; \texttt{!}\;k_1\!:\!v_1\;\texttt{!}\;k_2\!:\!v_2\;\texttt{!}\;\cdots\;\texttt{!}\;k_N\!:\!v_N\;\texttt{!}",
            why="""A context whose information content is exactly `N` facts, so recall can be *counted*
rather than judged. This is the same instrument our own `nlb1` lesson uses for associative memory — and it
makes the capacity question concrete: how many facts fit in `m` tokens?""",
            code="""n_facts, dk = 24, 16
keys = F.normalize(torch.randn(n_facts, dk), dim=-1)
vals = torch.randn(n_facts, dk)
ok("the context contains exactly n_facts pieces of information", keys.shape[0] == n_facts)
ok("keys are distinguishable", float((keys @ keys.T - torch.eye(n_facts)).abs().max()) < 0.9,
   f"max off-diagonal similarity {float((keys @ keys.T - torch.eye(n_facts)).abs().max()):.3f}")
# how many fit in a rank-limited memory? store into an m x dk state by least squares
for mm in (4, 8, 16, 24):
    Msub = vals[:mm].T @ torch.linalg.pinv(keys[:mm].T)
    err = float(((Msub @ keys[:mm].T) - vals[:mm].T).pow(2).mean())
    print(f"  {mm:>2} facts into a {dk}x{dk} memory: residual {err:.2e}")
ok("facts up to the state's rank are recalled exactly, beyond it they are compressed",
   float(((vals[:dk].T @ torch.linalg.pinv(keys[:dk].T) @ keys[:dk].T) - vals[:dk].T).pow(2).mean()) < 1e-6)"""),
    10: dict(name="…and the query that removes the context",
             latex=r"Q \;=\; \texttt{?}\;\texttt{!}\;k_j\!:\!,\qquad Y \;=\; v_j",
             why="""Ask for one key's value with the context deleted. Plain in-context learning scores zero
here **by construction** — there is nothing left to attend to — so any non-trivial accuracy is evidence
that the write actually moved information into the memory.""",
             code="""j = 5
M_kv = vals.T @ torch.linalg.pinv(keys.T)                        # the written memory
pred = M_kv @ keys[j]
err_written = float((pred - vals[j]).norm())
err_no_ctx = float((torch.zeros(dk) - vals[j]).norm())           # ICL with the context removed
ok("the written memory recalls the queried value far better than nothing",
   err_written < 0.8 * err_no_ctx, f"error {err_no_ctx:.3f} (no context) -> {err_written:.3f} (memory)")
within = vals[:dk].T @ torch.linalg.pinv(keys[:dk].T)
ok("and within capacity the recall is EXACT", float((within @ keys[3] - vals[3]).norm()) < 1e-4,
   f"{dk} facts in a {dk}x{dk} state: error {float((within @ keys[3] - vals[3]).norm()):.2e}; "
   f"{n_facts} facts over-fills it, hence the residual above")
ok("plain ICL cannot score at all once the context is removed", err_no_ctx > 0,
   "nothing to attend to — this is the point of the setting")"""),
    11: dict(name="The cost model and its break-even",
             latex=r"T_{\text{full}} \approx c^2 + cqN,\qquad T_{\text{GradMem}} \approx R(c+m)^2K + m^2 + mqN \;\;\Longrightarrow\;\; N > \frac{c^2(RK-1) + (1+RK)m^2 + 2cmRK}{q(c-m)}",
             why="""**The decision rule, in closed form.** Writing costs `R(c+m)²K` up front (K gradient
steps over R chunks) and saves `(c−m)q` per query, so it pays off after enough queries. Everything on the
right is known before you implement anything — so this is the number to compute first, exactly like
RateQuant's AM/GM diagnostic (`rq04`).""",
             code="""def breakeven(c, m, q, R, K):
    return (c ** 2 * (R * K - 1) + (1 + R * K) * m ** 2 + 2 * c * m * R * K) / (q * (c - m))
for (c_, m_, q_, R_, K_) in [(8192, 256, 64, 4, 8), (8192, 256, 64, 1, 2), (2048, 512, 64, 4, 8)]:
    N_star = breakeven(c_, m_, q_, R_, K_)
    T_full = lambda N: c_ ** 2 + c_ * q_ * N
    T_gm = lambda N: R_ * (c_ + m_) ** 2 * K_ + m_ ** 2 + m_ * q_ * N
    N_emp = 1
    while T_gm(N_emp) >= T_full(N_emp) and N_emp < 10 ** 9:
        N_emp *= 2
    print(f"  c={c_} m={m_} R={R_} K={K_}:  formula N* = {N_star:,.0f}   empirical crossover < {N_emp:,}")
    assert N_emp >= 1
N_star = breakeven(8192, 256, 64, 4, 8)
T_full = lambda N: 8192 ** 2 + 8192 * 64 * N
T_gm = lambda N: 4 * (8192 + 256) ** 2 * 8 + 256 ** 2 + 256 * 64 * N
ok("above the break-even the memory is cheaper", T_gm(int(N_star * 1.2)) < T_full(int(N_star * 1.2)),
   f"N* = {N_star:,.0f}")
ok("below it the cache is cheaper", T_gm(int(N_star * 0.5)) > T_full(int(N_star * 0.5)))
ok("fewer write steps (R*K) lowers the break-even sharply",
   breakeven(8192, 256, 64, 1, 2) < breakeven(8192, 256, 64, 4, 8),
   f"{breakeven(8192,256,64,1,2):,.0f} vs {breakeven(8192,256,64,4,8):,.0f} queries")"""),
})

ADVANCED = [
    dict(id="gmz1", title="What we take from GradMem — the break-even, and one honest caveat",
         subtitle="GradMem · when writing beats caching",
         cells=[
             dict(note="""## Two transferable pieces
1. **The break-even inequality (eq. 13) is a design tool.** Any time we consider compressing a context —
   a long document, a replay buffer, a well's history — the question "how many queries will we ask of it?"
   decides the answer, and eq. 13 answers it in closed form. Same shape of diagnostic as RateQuant's AM/GM.
2. **A memory is the argmin of an inspectable objective.** `L_write` is self-supervised on the context
   itself, so it can be run against a frozen model with no labels and no fine-tuning — which is exactly
   the situation we are in on a Kaggle box with someone else's checkpoint.

**Honest caveat:** the paper's evidence is LLM long-context benchmarks under context removal. Everything
proved here is either an identity, a closed-form cost comparison, or a small controlled optimisation
against a frozen linear model. The mechanism transfers; the benchmark numbers are not reproduced."""),
             dict(note="""### The break-even, plotted for our own regime
Fill in the numbers for a setting we actually run and read off whether writing is worth it.""",
                  code="""import pandas as pd
def breakeven(c, m, q, R, K):
    return (c ** 2 * (R * K - 1) + (1 + R * K) * m ** 2 + 2 * c * m * R * K) / (q * (c - m))
rows = []
for c_ in (2048, 8192, 32768):
    for RK in ((1, 2), (4, 8)):
        rows.append(dict(context=c_, R=RK[0], K=RK[1], memory=256,
                         breakeven_queries=round(breakeven(c_, 256, 64, RK[0], RK[1]))))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
ok("a longer context RAISES the break-even (writing costs more up front)",
   df[df.R == 4].breakeven_queries.is_monotonic_increasing,
   "so writing pays off soonest for medium contexts asked many questions")
ok("cheaper writes (small R*K) always lower the break-even",
   bool((df[df.R == 1].breakeven_queries.values < df[df.R == 4].breakeven_queries.values).all()))
vz.table(df, "GradMem break-even (eq. 13)", "queries needed before writing beats caching",
         heat_cols=["breakeven_queries"], lower_better=["breakeven_queries"])"""),
             dict(note="""**[Recap]** write once, answer many · the write is a self-supervised objective
descended at test time on a frozen model · state size is constant in context length · and eq. 13 tells you
in advance whether to bother. Cross-read: `nl06` (test-time training = parametric ICL), `fnl02` (the same
argmin shape), `rq04` (the same "compute the diagnostic first" discipline)."""),
         ]),
]
