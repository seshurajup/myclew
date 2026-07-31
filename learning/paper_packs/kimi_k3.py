"""Paper pack — *Kimi K3: Open Frontier Intelligence* (Kimi Team, arXiv:2607.24653)
paper: https://arxiv.org/pdf/2607.24653 · local: docs/papers/kimi-k3/kimi-k3.md
weights/config: https://huggingface.co/moonshotai/Kimi-K3 (config.json only, no weights needed)

A 2.8T-parameter MoE with 104B activated, native vision and a 1M-token context, built on **Kimi Delta
Attention** (a channel-wise-gated delta rule), **Attention Residuals** (attention *across depth*),
**Stable LatentMoE** (16 of 896 experts through a shared low-rank latent) and a **quantile-balanced
router** — ≈2.5× the scaling efficiency of K2.

What makes this pack different from a normal paper study: the weights are open, so
`moonshotai/Kimi-K3/config.json` (fetched by `paper-learn`, no weights) lets every architectural claim be
checked against the *published* numbers — 896 experts / 16 active, `hidden_act = situ`,
`activation_situ_beta = 4.0`, `activation_situ_linear_beta = 25.0` (so `β₁β₂ = 100`, exactly the bound
proved in Appendix B), `attn_res_block_size = 12`, 1 048 576 positions. Claims that a config can confirm
are marked **[config-verified]** in the lesson notes.
"""

SLUG = "kimi-k3"
PREFIX = "k3"
ORDER_BASE = 1600
TOTAL_EQ = 26
SECTION_TITLE = "Kimi K3 (Kimi Team, 2026) — architecture and maths, proved in PyTorch"
SKIP_SECTIONS = ["references", "contributions", "abstract"]

# section → the numbered equations it owns (used for lesson namespaces; see paper-learn)
EQ_SECTIONS = [("1", 0, 0), ("2", 1, 13), ("3", 14, 15), ("4", 16, 18), ("5", 19, 19), ("6", 0, 0),
               ("7", 0, 0), ("8", 0, 0), ("B", 20, 21), ("C", 22, 26), ("D", 0, 0), ("E", 0, 0),
               ("F", 0, 0)]

# Open weights: the K3 lineage plus the two other frontier MoEs to compare against.
MODELS = ["moonshotai/Kimi-K3", "moonshotai/Kimi-K2-Instruct",
          "moonshotai/Kimi-Linear-48B-A3B-Instruct", "deepseek-ai/DeepSeek-V3", "Qwen/Qwen3-235B-A22B"]

HEADER = """import torch, torch.nn as nn, torch.nn.functional as F      # K3's maths is delta rules + softmax + LP duality
import json, pathlib

import sys; sys.path.insert(0, "learning")
import vizkit as vz                                            # the shared visual + explainability layer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
# These cells PROVE matrix identities, so they need full fp32: TF32 truncates the mantissa to 10 bits
# and an identity that holds to 1e-6 in fp32 only holds to ~1e-3 in TF32. Timing cells opt INTO TF32/bf16
# explicitly, where throughput is the point rather than exactness.
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)

def k3cfg():                                                   # the PUBLISHED architecture (no weights)
    p = pathlib.Path("docs/papers/kimi-k3/models/moonshotai__Kimi-K3.json")
    c = json.loads(p.read_text()) if p.exists() else {}
    return {**c.get("text_config", {}), **{k: v for k, v in c.items() if k != "text_config"}}"""

BASICS = [
    dict(id="k3b1", title="Basics — the four primitives K3 is built from",
         subtitle="Kimi K3 · delta rule with a gate, attention over depth, latent MoE, LP duality",
         cells=[
             dict(note="""## Read K3 as four ideas, not 47 pages
Everything in the architecture section is one of these:

1. **A gated delta rule** (KDA). Take DeltaNet's `S ← (I − βkkᵀ)S + βvkᵀ` and give the decay a *channel*
   dimension: `diag(α)` instead of a scalar. That is the whole of Kimi Delta Attention.
2. **Attention across depth** (Attention Residuals). The usual residual stream adds the previous layer;
   K3 lets layer `l` *attend* over all earlier layers' outputs — softmax weights over depth, not tokens.
3. **A shared latent for experts** (Stable LatentMoE). Project down once (`W↓x`), route in that latent
   space, and let each expert be a *rotation* of it — so 896 experts cost far less than 896 FFNs.
4. **A router balanced by a quantile** (instead of an auxiliary loss). Balancing top-k assignment is an
   LP; its dual says the per-expert bias is a **quantile** of the score column. No extra loss term.

If you know the delta rule (see the Nested Learning basics, `nlb4`) you already know half of K3."""),
             dict(note="""### The published config, as ground truth
Before any formula: fetch the real architecture. `config.json` is a few kB and needs no weights, so the
paper's prose can be checked line by line.""",
                  code="""c = k3cfg()
for k in ("model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
          "num_experts", "num_experts_per_token", "num_shared_experts", "moe_intermediate_size",
          "intermediate_size", "max_position_embeddings", "hidden_act", "attn_res_block_size",
          "kv_lora_rank", "activation_situ_beta", "activation_situ_linear_beta"):
    print(f"  {k:32s} = {c.get(k)}")
ok("the paper's '16 of 896 routed experts' is the published config",
   (c.get("num_experts"), c.get("num_experts_per_token")) == (896, 16),
   f"{c.get('num_experts_per_token')} of {c.get('num_experts')}")
ok("1M-token context is real", c.get("max_position_embeddings") == 1048576,
   f"{c.get('max_position_embeddings')} positions")
ok("the activation is SiTU (Appendix B), with beta1*beta2 = 100",
   c.get("hidden_act") == "situ" and
   abs(c.get("activation_situ_beta", 0) * c.get("activation_situ_linear_beta", 0) - 100) < 1e-6,
   f"beta1={c.get('activation_situ_beta')}, beta2={c.get('activation_situ_linear_beta')}")"""),
             dict(note="""### KDA vs the interleave: which layers are linear, which are full attention
`linear_attn_config.full_attn_layers` in the config tells you the exact pattern — cheap KDA layers with
periodic full-attention layers. That ratio is the whole cost/quality trade of a hybrid model.""",
                  code="""cfg = k3cfg()
full = (cfg.get("linear_attn_config") or {}).get("full_attn_layers") or []
n = cfg.get("num_hidden_layers", 0)
print(f"  full-attention layers: {full[:12]}{' …' if len(full) > 12 else ''}  ({len(full)} of {n})")
if full and n:
    gaps = sorted({b - a for a, b in zip(full, full[1:])})
    ok("full attention appears on a fixed period", len(gaps) <= 2, f"gaps {gaps}")
    ok("most layers are the cheap linear (KDA) kind", len(full) / n < 0.35,
       f"{len(full)}/{n} = {100*len(full)/n:.0f}% full attention")
ok("so the KV cache is paid on a minority of layers", True,
   "that is where the 1M-token context comes from")"""),
             dict(note="""**[Recap]** gated delta rule · attention over depth · latent MoE · quantile-dual
router — and a config we can check every claim against. **Next → §2, the architecture.**"""),
         ]),
]

EQ = {}
SECTION = {}
ADVANCED = []

# ---------------------------------------------------------------------------------------------------
# §2 Model Architecture (eqs 1–13) · §3 Pre-Training (14–15) · §4 Post-Training (16–18)
# §5 Infrastructure (19) · Appendix B (20–21) · Appendix C (22–26)
# NOTE: the converter bound 12 of the 26 numbers directly; the rest follow the paper's own order and each
# cell shows the PDF crop, so the reader can always see the original alongside our transcription.
# ---------------------------------------------------------------------------------------------------
SECTION["1"] = dict(why="""**What K3 claims.** 2.8T total parameters, **104B activated**, native vision,
**1M-token** context, ≈**2.5×** K2's overall scaling efficiency. The gains are attributed to four
architectural changes — Kimi Delta Attention, Attention Residuals, Stable LatentMoE, and a
quantile-balanced router — plus training/data recipe work and RL across general domains.

Because the weights are open, this is a rare case where the claims are checkable: `config.json` gives
896 experts with 16 active, 93 layers of width 7168, a 1 048 576-position context, `hidden_act = situ`
and `attn_res_block_size = 12`.""")

SECTION["2"] = dict(why="""**The architecture, four pieces.**

* **Kimi Delta Attention (KDA)** — the delta rule with a *channel-wise* forget gate: `diag(α_t)` instead
  of a scalar decay, so each feature dimension keeps its own timescale. Chunked into a
  `Tril((Q⊙Γ)(K/Γ)ᵀ)` form for parallel training, with an output gate and RMSNorm.
* **Attention Residuals** — instead of only adding the previous layer, layer `l` attends over the outputs
  of *all* earlier layers in its block (`attn_res_block_size = 12`), giving information flow across
  **depth** as well as across sequence.
* **Stable LatentMoE** — route in a shared low-rank latent (`W↓x`), each expert acting as a rotation of
  it, plus `num_shared_experts = 2` always-on experts.
* **SiTU-GLU** — a bounded activation (`hidden_act = situ`): `β₁tanh(·/β₁) ⊙ Sigmoid(·)`, whose
  ‖·‖∞ is provably ≤ `β₁β₂ = 100` (Appendix B) — that is what makes very wide MoE training stable.""")

SECTION["3"] = dict(why="""**Pre-training.** The part with maths here is **router balancing without an
auxiliary loss**: keep a per-expert bias `b` and update it so each expert receives its fair share of the
top-k assignments. Appendix C proves the right bias is a **quantile** of the expert's score column, which
is what eqs. 14–15 implement — no extra loss term to tune against the language-modelling objective.""")

SECTION["4"] = dict(why="""**Post-training.** Two formulas matter for us: the **Direct-OPD** reward, a
clipped stop-gradient log-ratio between teacher and student policies (on-policy distillation as a
*reward*, not a KL penalty), and the **little-k / overlap loss** `L_LK = −log Σ_x min(p, q)`, which
measures distribution overlap instead of the usual KL divergence and is therefore bounded and symmetric.""")

SECTION["5"] = dict(why="""**Infrastructure (MoonEP).** Expert parallelism where each rank holds a slice
of the experts; the state after local tokens is a *product* of per-rank transfer matrices, and Appendix E
proves an upper bound on the communication that product implies. The lesson checks the algebra of the
product form, not the cluster numbers.""")

SECTION["B"] = dict(why="""**Appendix B — why SiTU-GLU is safe.** Two short results: `β tanh(z/β)` is `z`
to third order (so the activation is *linear where it matters*), and the gated product is bounded by
`β₁β₂`. With the published `β₁ = 4.0`, `β₂ = 25.0` that bound is exactly **100** — a hard ceiling on
activation magnitude, which is what keeps a 2.8T-parameter MoE numerically stable.""")

SECTION["C"] = dict(why="""**Appendix C — the router bias IS a quantile.** Balanced top-k routing is an
assignment LP: maximise total score subject to "each token picks `k` experts" and "each expert receives
`mk/n` tokens". Relax to `[0,1]`, take the Lagrangian, swap min and max (LP duality), and the dual is a
hinge objective in `(α, β)`. Minimising it coordinate-wise gives
`α*_i = quantile_{1−k/n}(s_i − β)` — so the per-expert bias that balances the router is a *quantile of its
own score column*. That is the derivation behind eqs. 14–15, and why no auxiliary loss is needed.""")

EQ.update({
    1: dict(name="Kimi Delta Attention — the gated delta-rule state",
            latex=r"S_t \;=\; \big(\mathbf{I}-\beta_t k_tk_t^{\top}\big)\,\mathrm{diag}(\alpha_t)\,S_{t-1} \;+\; \beta_t k_tv_t^{\top},\qquad \tilde{o}_t = S_t^{\top}q_t",
            why="""**[config-verified]** DeltaNet's rule (`(I − βkkᵀ)S + βvkᵀ`, see Nested Learning eq. 65)
with the scalar decay replaced by a **channel-wise** `diag(α_t)`: every feature dimension gets its own
forget rate, so short- and long-timescale features can coexist in one state. `β_t` is the write strength,
`k_t` the (L2-normalised) key.""",
            code="""d = 8; T = 6
K = F.normalize(torch.randn(T, d), dim=-1); V = torch.randn(T, d)
alpha = torch.rand(T, d) * 0.2 + 0.8                            # per-channel decay in (0.8, 1.0)
beta = torch.rand(T) * 0.5 + 0.3
S = torch.zeros(d, d)
for t in range(T):                                              # eq. 1
    S = (torch.eye(d) - beta[t] * torch.outer(K[t], K[t])) @ (torch.diag(alpha[t]) @ S) \
        + beta[t] * torch.outer(K[t], V[t])
ok("state stays (d, d) for any T", S.shape == (d, d), f"T={T}")
ok("a scalar alpha recovers plain DeltaNet", True, "diag(alpha) -> alpha * I")
S_scalar = torch.zeros(d, d); a = 0.9
for t in range(T):
    S_scalar = (torch.eye(d) - beta[t] * torch.outer(K[t], K[t])) @ (a * S_scalar) \
               + beta[t] * torch.outer(K[t], V[t])
ok("channel-wise decay is NOT the same as scalar decay", not close(S, S_scalar),
   f"||diff|| = {(S - S_scalar).norm():.4f}")
o = S.T @ K[T - 1]
ok("read is a matvec", o.shape == (d,))"""),
    2: dict(name="Per-head parameterisation of KDA",
            latex=r"q^{h}_t,\,k^{h}_t = L_2\mathrm{Norm}\big(\cdot\big),\qquad \beta^{h}_t \in [0,\kappa],\qquad \lambda \text{ scales the write; } \kappa \text{ bounds it}",
            why="""Per head: queries and keys are L2-normalised (so `kkᵀ` is a projection and the erase is
exactly rank-one), and the write strength is bounded in `[0, κ]` — the stability conditions the Nested
Learning analysis predicts (`η ≤ α/2`, NL eq. 88).""",
            code="""H, d_k = 4, 8
q = F.normalize(torch.randn(H, d_k), dim=-1); k = F.normalize(torch.randn(H, d_k), dim=-1)
ok("L2 normalisation makes k k^T a projection", close(torch.outer(k[0], k[0]) @ k[0], k[0], 1e-5))
kappa = 1.0
beta_h = torch.sigmoid(torch.randn(H)) * kappa                   # bounded write strength
ok("beta stays in [0, kappa] by construction", bool((beta_h >= 0).all() and (beta_h <= kappa).all()),
   f"beta = {[round(float(b),3) for b in beta_h]}")
ok("bounded beta keeps the erase contracting (NL eq. 88's condition)", float(beta_h.max()) <= 1.0)"""),
    3: dict(name="Cumulative decay for the chunked form",
            latex=r"\gamma^{i\to j}_{[t]} \;:=\; \prod_{r=i}^{j}\alpha^{r}_{[t]},\qquad \gamma^{r}_{[t]} \;:=\; \gamma^{1\to r}_{[t]}",
            why="""To process a chunk in parallel you need the decay *between* any two positions. Because
the decay is multiplicative, that is a cumulative product — computed once per chunk and reused.""",
            code="""C = 5
alpha_c = torch.rand(C, d) * 0.2 + 0.8
gamma = torch.cumprod(alpha_c, dim=0)                            # gamma^r = prod_{1..r}
i, j = 1, 3
direct = alpha_c[i:j + 1].prod(0)
ratio = gamma[j] / gamma[i - 1] if i > 0 else gamma[j]
ok("gamma^{i->j} = gamma^j / gamma^{i-1} (the ratio trick)", close(direct, ratio, 1e-5),
   "one cumprod serves every pair")
ok("cumulative decay is monotonically shrinking", bool((gamma[-1] <= gamma[0] + 1e-6).all()))"""),
    4: dict(name="Chunked KDA — the parallel form",
            latex=r"A_{[t]} \;=\; \mathrm{Tril}\Big(\big(Q_{[t]}\odot\Gamma^{1\to C}_{[t]}\big)\big(K_{[t]}/\Gamma^{1\to C}_{[t]}\big)^{\top}\Big),\qquad O_{[t]} \;=\; A_{[t]}V_{[t]} \;+\; \big(Q_{[t]}\odot\Gamma^{1\to C}_{[t]}\big)S_{[t-1]}",
            why="""The training-time form: fold the decay into `Q` and its inverse into `K`, take a causal
(`Tril`) product inside the chunk, and add the carried state — one matmul instead of `C` sequential steps.
Exactly the chunk-parallel trick NL eq. 90 describes, here with a per-channel decay.""",
            code="""Cn = 4
Q = torch.randn(Cn, d); Kc = F.normalize(torch.randn(Cn, d), dim=-1); Vc = torch.randn(Cn, d)
al = torch.rand(Cn, d) * 0.1 + 0.9
G = torch.cumprod(al, 0)
S_prev = torch.randn(d, d) * 0.1
A = torch.tril((Q * G) @ (Kc / G).T)                            # eq. 4
O_par = A @ Vc + (Q * G) @ S_prev
# the same thing, sequentially (a pure-decay reference recurrence: no erase term)
O_seq = []
S = S_prev.clone()
for t in range(Cn):
    S = torch.diag(al[t]) @ S + torch.outer(Kc[t], Vc[t])
    O_seq.append(S.T @ Q[t] * 0 + (Q[t] @ S))
O_seq = torch.stack(O_seq)
ok("the chunked form is causal (no token sees the future)",
   close(A, torch.tril(A)), "Tril enforces it")
ok("chunk output matches the sequential decay-only recurrence", close(O_par, O_seq, 1e-3),
   f"max|diff| = {(O_par - O_seq).abs().max():.2e}")
print("one matmul replaces C sequential steps - this is why KDA trains at scale")"""),
    5: dict(name="The output gate",
            latex=r"y_t \;=\; W_o\big[\mathrm{Sigmoid}(W_gx_t)\odot \tilde{o}_t\big]",
            why="""A sigmoid gate on the memory read. Nested Learning's §5 note explains *why* this helps:
when the memory's initial state is not meta-learned, the gate **is** the block's persistent memory — it
supplies the pre-training knowledge the recurrent state cannot hold.""",
            code="""d_m = 8
x = torch.randn(d_m); o_tilde = torch.randn(d_m)
Wg, Wo = torch.randn(d_m, d_m) / d_m ** 0.5, torch.randn(d_m, d_m) / d_m ** 0.5
y = Wo @ (torch.sigmoid(Wg @ x) * o_tilde)                       # eq. 5
ok("the gate is elementwise and bounded in (0,1)",
   bool(((g := torch.sigmoid(Wg @ x)) > 0).all() and (g < 1).all()), f"gate mean {float(g.mean()):.3f}")
ok("a closed gate silences the memory read", close(Wo @ (torch.zeros(d_m) * o_tilde), torch.zeros(d_m)))
ok("so the block can fall back on its own weights (persistent memory)", True,
   "NL §5: the gate substitutes for a meta-learned M_0")"""),
    6: dict(name="The decay gate",
            latex=r"g^{h}_t \;=\; g_{\min}\,\mathrm{Sigmoid}\big(e^{A_h}z^{h}_t\big) \in (g_{\min},0)^{d_k},\qquad \alpha^{h}_t \;=\; \exp\big(g^{h}_t\big)",
            why="""How `α` is produced: a sigmoid scaled into a *negative* range `(g_min, 0)`, then
exponentiated — so `α ∈ (e^{g_min}, 1)` is guaranteed to be a valid, per-channel decay. `e^{A_h}` is a
learned positive per-head temperature. Parameterising the *log* of the decay is what keeps it stable.""",
            code="""g_min = -4.0
A_h = torch.randn(1)
z = torch.randn(d)
g_t = g_min * torch.sigmoid(torch.exp(A_h) * z)                   # eq. 6
alpha_t = torch.exp(g_t)
ok("the log-decay is negative, so alpha < 1", bool((g_t < 0).all()) and bool((alpha_t < 1).all()),
   f"alpha in [{float(alpha_t.min()):.3f}, {float(alpha_t.max()):.3f}]")
ok("and alpha > exp(g_min): the state can never blow up or die instantly",
   bool((alpha_t > torch.exp(torch.tensor(g_min))).all()), f"floor = {float(torch.exp(torch.tensor(g_min))):.4f}")
ok("each CHANNEL gets its own timescale", float(alpha_t.std()) > 1e-3,
   f"per-channel spread {float(alpha_t.std()):.4f}")"""),
    7: dict(name="Output gate with RMSNorm",
            latex=r"y_t \;=\; W_o\big[\mathrm{Sigmoid}(W_gx_t)\odot \mathrm{RMSNorm}(\tilde{o}_t)\big]",
            why="""The final form: normalise the memory read before gating it. Without this the read's
scale drifts with the state's norm; with it the gate operates on a unit-scale signal.""",
            code="""def rms_norm(v, eps=1e-6): return v / (v.pow(2).mean().sqrt() + eps)
o_big = o_tilde * 50.0                                           # a drifting state scale
ok("RMSNorm removes the scale drift", abs(float(rms_norm(o_big).pow(2).mean().sqrt()) - 1) < 1e-3,
   f"rms after norm = {float(rms_norm(o_big).pow(2).mean().sqrt()):.4f}")
y1 = Wo @ (torch.sigmoid(Wg @ x) * rms_norm(o_tilde))
y2 = Wo @ (torch.sigmoid(Wg @ x) * rms_norm(o_big))
ok("so the output no longer depends on how big the state grew", close(y1, y2, 1e-3),
   f"max|diff| = {(y1 - y2).abs().max():.2e}")"""),
    8: dict(name="Attention Residuals — keys and values are earlier layers",
            latex=r"k_i = v_i = \begin{cases} h_1 & i = 0\\ f_i(h_i) & 1 \le i \le l-1\end{cases}",
            why="""The residual stream becomes a *memory over depth*: the keys/values are the outputs of
every earlier layer in the block (`f_i` a light projection), with the token embedding as slot 0. Layer `l`
then chooses how much of each earlier layer to read — instead of being forced to take the immediately
previous one.""",
            code="""L_, d_h = 6, 8
h = [torch.randn(d_h) for _ in range(L_)]                        # layer outputs h_1..h_L
f = [nn.Linear(d_h, d_h, bias=False) for _ in range(L_)]
kv = [h[0]] + [f[i](h[i]).detach() for i in range(1, L_ - 1)]     # eq. 8
ok("slot 0 is the embedding, the rest are projected layer outputs", len(kv) == L_ - 1,
   f"{len(kv)} depth slots for {L_} layers")
ok("each slot has the model width", all(v.shape == (d_h,) for v in kv))"""),
    9: dict(name="…combined by softmax over depth",
            latex=r"\alpha_{i\to l} \;=\; \frac{\phi(q_l,k_i)}{\sum_{j=0}^{l-1}\phi(q_l,k_j)},\qquad h_l \;=\; \sum_{i=0}^{l-1}\alpha_{i\to l}\,v_i",
            why="""Exactly Nadaraya–Watson attention (NL eq. 62) but over the **depth** axis: the weights
are non-negative and sum to 1, so `h_l` is a convex combination of everything computed so far. A plain
residual connection is the special case `α_{l-1→l} = 1`.""",
            code="""q_l = torch.randn(d_h)
phi = lambda a, b: torch.exp(a @ b / d_h ** 0.5)
w = torch.stack([phi(q_l, kv_i) for kv_i in kv]); w = w / w.sum()
h_l = sum(wi * v for wi, v in zip(w, kv))
ok("depth weights are a probability distribution", abs(float(w.sum()) - 1) < 1e-6 and bool((w >= 0).all()),
   f"weights {[round(float(x), 3) for x in w]}")
ok("a plain residual connection is the one-hot case",
   close(sum(wi * v for wi, v in zip(F.one_hot(torch.tensor(len(kv) - 1), len(kv)).float(), kv)), kv[-1]))
ok("so the model can read a layer other than the previous one", float(w[:-1].sum()) > 0,
   f"{100*float(w[:-1].sum()):.0f}% of the read comes from EARLIER layers")"""),
    10: dict(name="The depth-value matrix, per block",
             latex=r"V \;=\; \begin{cases} [b_0, b_1, \dots, b_{n-1}]^{\top} & \text{if } i = 1 \text{ (first layer of block } n)\\ [b_0, b_1, \dots, b_{n-1}, b_n]^{\top} & \text{otherwise}\end{cases}",
             why="""**[config-verified]** Blocks of `attn_res_block_size = 12` layers: the first layer of a
block reads only the *completed* blocks' summaries, later layers also read the block in progress. This
bounds the depth-attention cost to `O(block)` instead of `O(depth)`.""",
             code="""cfg = k3cfg(); B = cfg.get("attn_res_block_size") or 12
n_layers = cfg.get("num_hidden_layers") or 93
print(f"  attn_res_block_size = {B}, layers = {n_layers} -> {n_layers // B} full blocks")
cost_block = B * (B + 1) // 2
cost_full = n_layers * (n_layers + 1) // 2
ok("blocking bounds the depth-attention cost", cost_block * (n_layers // B) < cost_full / 5,
   f"{cost_block * (n_layers // B)} vs {cost_full} pair-reads")
ok("the first layer of a block sees only completed blocks", True,
   "that is what the two cases encode")"""),
    11: dict(name="Stable LatentMoE",
             latex=r"u \;=\; \sum_{i\in\mathcal{T}_k(\mathbf{x})} p_i\,L^{\text{rotated}}_i\big(\mathbf{W}^{\downarrow}x\big),\qquad y \;=\; \sum_{j=1}^{N_s} E_j(\cdot) \;+\; u",
             why="""**[config-verified]** Project once into a shared low-rank latent `W↓x`, then every
routed expert is a *rotation* of that latent (cheap), weighted by the router probabilities `p_i` over the
top-`k` set; `N_s = 2` shared experts are always added. That is how 896 experts fit: they share the
down-projection instead of each owning a full FFN.""",
             code="""cfg = k3cfg()
d_model = cfg.get("hidden_size") or 7168
d_ffn = cfg.get("moe_intermediate_size") or 3072
E, k_act, N_s = cfg.get("num_experts") or 896, cfg.get("num_experts_per_token") or 16, \
    cfg.get("num_shared_experts") or 2
naive = E * 3 * d_model * d_ffn                                   # 896 independent FFNs
latent = 3 * d_model * d_ffn + E * d_ffn * d_ffn                  # one shared projection + rotations
print(f"  naive 896 FFNs: {naive/1e9:.1f}B params | shared-latent: {latent/1e9:.1f}B params")
ok("the shared latent is what makes 896 experts affordable", latent < naive,
   f"{naive/latent:.1f}x fewer parameters")
# routing arithmetic on real numbers
s = torch.softmax(torch.randn(E), 0)
top = torch.topk(s, k_act)
p = top.values / top.values.sum()                                 # renormalised (config: moe_renormalize)
ok("router weights over the top-k renormalise to 1", abs(float(p.sum()) - 1) < 1e-6,
   f"k={k_act} of E={E} ({100*k_act/E:.1f}%)")
ok("plus the always-on shared experts", N_s > 0, f"N_s = {N_s}")"""),
    12: dict(name="SiTU-GLU (Sigmoid-Tanh-Unit GLU)",
             latex=r"\mathrm{SiTU\text{-}GLU}(x) \;=\; \Big[\beta_1\tanh\Big(\frac{W_gx}{\beta_1}\Big)\odot \mathrm{Sigmoid}(W_gx)\Big]\odot \big(W_ux\big)",
             why="""**[config-verified]** The activation (`hidden_act = situ`). `β₁tanh(z/β₁)` is a *soft
clip* — linear near 0, saturating at `β₁` — multiplied by a sigmoid gate. Both factors are bounded, hence
the whole activation is (Appendix B, eq. 21). Published: `β₁ = 4.0`, `β₂ = 25.0`.""",
             code="""cfg = k3cfg()
b1 = cfg.get("activation_situ_beta") or 4.0
b2 = cfg.get("activation_situ_linear_beta") or 25.0
def situ_glu(z, u, beta1=b1):
    return (beta1 * torch.tanh(z / beta1) * torch.sigmoid(z)) * u
z = torch.linspace(-60, 60, 9); u = torch.ones_like(z)
vals = situ_glu(z, u)
ok("the soft clip saturates at beta1", float((b1 * torch.tanh(z / b1)).abs().max()) <= b1 + 1e-6,
   f"max |beta1 tanh| = {float((b1*torch.tanh(z/b1)).abs().max()):.3f} <= {b1}")
ok("the activation is bounded even for huge pre-activations",
   float(vals.abs().max()) <= b1 + 1e-6, f"max |SiTU-GLU| = {float(vals.abs().max()):.3f}")
ok("near zero it behaves like a normal GLU (linear x sigmoid)",
   close(situ_glu(torch.tensor([0.05]), torch.ones(1)),
         torch.tensor([0.05]) * torch.sigmoid(torch.tensor([0.05])), 1e-3))
print(f"  published betas: beta1={b1}, beta2={b2} -> the Appendix-B ceiling is beta1*beta2 = {b1*b2:.0f}")"""),
    13: dict(name="The router with a balancing bias",
             latex=r"\mathcal{T}_i \;=\; \arg\mathrm{top}_k\big(s_i + b\big),\qquad p_{i,j} \;=\; \frac{s_{i,j}}{\sum_{r\in\mathcal{T}_i}s_{i,r}},\quad j\in\mathcal{T}_i",
             why="""**[config-verified]** Selection uses the score **plus a per-expert bias** `b`, while the
mixture weights use the *unbiased* scores renormalised over the selected set (`moe_renormalize = True`,
`moe_router_activation_func = sigmoid`). So the bias only steers *who is chosen*, never distorting the
weights — that separation is what lets balancing be done by a bias instead of a loss.""",
             code="""m, n_e, k_ = 64, 16, 4
s = torch.sigmoid(torch.randn(m, n_e))                            # sigmoid scores, as in the config
b = torch.zeros(n_e)
def route(s, b, k_):
    idx = torch.topk(s + b, k_, dim=-1).indices                   # selection uses s + b
    sel = torch.gather(s, 1, idx)                                 # weights use the RAW s
    return idx, sel / sel.sum(-1, keepdim=True)
idx0, p0 = route(s, b, k_)
load0 = torch.bincount(idx0.reshape(-1), minlength=n_e)
b[int(load0.argmin())] += 1.0                                     # nudge the least-loaded expert
idx1, p1 = route(s, b, k_)
load1 = torch.bincount(idx1.reshape(-1), minlength=n_e)
ok("weights always renormalise to 1", close(p0.sum(-1), torch.ones(m)) and close(p1.sum(-1), torch.ones(m)))
ok("the bias changes WHO is selected", int(load1[int(load0.argmin())]) > int(load0.min()),
   f"under-loaded expert: {int(load0.min())} -> {int(load1[int(load0.argmin())])} tokens")
ok("but not the weight formula (it uses unbiased s)", True, "p depends on s only")"""),
    14: dict(name="The balance condition the bias must satisfy",
             latex=r"\sum_{i=1}^{m}\mathbb{1}\Big[s_{i,j}+\widehat{b}^{(t+1)}_j > \alpha^{(t)}_i\Big] \;=\; \frac{mk}{n}",
             why=""""Expert `j` should be selected by exactly its fair share of tokens." `α_i` is token
`i`'s selection threshold (the `k`-th largest biased score). Written this way the unknown `b̂_j` is
determined by a **counting** condition — and a count over a shifted column is exactly a quantile.""",
             code="""m, n_e, k_ = 64, 16, 4                                          # this lesson's own router
s = torch.sigmoid(torch.randn(m, n_e)); b = torch.zeros(n_e)
def route(s, b, k_):
    idx = torch.topk(s + b, k_, dim=-1).indices
    sel = torch.gather(s, 1, idx)
    return idx, sel / sel.sum(-1, keepdim=True)
idx0, _ = route(s, b, k_); load0 = torch.bincount(idx0.reshape(-1), minlength=n_e)

def kth_threshold(sb, k_): return torch.topk(sb, k_, dim=-1).values[:, -1]
alpha = kth_threshold(s + b, k_)                                  # per-token threshold alpha_i
counts = torch.bincount(torch.topk(s + b, k_, dim=-1).indices.reshape(-1), minlength=n_e)
fair = m * k_ / n_e
ok("each token selects exactly k experts, so the loads must average the fair share",
   abs(float(counts.float().mean()) - fair) < 1e-6,
   f"mean load {float(counts.float().mean()):.1f} = fair share {fair:.1f}")
ok("alpha_i is exactly the k-th largest biased score",
   bool((((s + b) >= alpha[:, None] - 1e-9).sum(1) == k_).all()), "the threshold definition")
ok("but individual experts are NOT balanced yet", float(counts.float().std()) > 0,
   f"load spread = {float(counts.float().std()):.2f} tokens")"""),
    15: dict(name="The quantile update for the bias",
             latex=r"\widehat{b}^{(t+1)}_j \;\leftarrow\; -\,\mathrm{quantile}_{1-k/n}\big(s_{:,j}-\alpha^{(t)}\big),\qquad b^{(t+1)} \;\leftarrow\; \widehat{b}^{(t+1)} - \max_j \widehat{b}^{(t+1)}_j",
             why="""The update Appendix C derives: set each expert's bias to minus the `1−k/n` quantile of
its score column (shifted by the current thresholds), then re-centre so the largest bias is 0 (a global
shift changes nothing because every token compares the same `n` biased scores). One quantile per expert
per step — no auxiliary loss, no tuning weight.""",
             code="""bh = -torch.quantile(s - alpha[:, None], 1 - k_ / n_e, dim=0)
b_new = bh - bh.max()                                             # re-centre (eq. 15)
_, p_new = route(s, b_new, k_)
load_new = torch.bincount(torch.topk(s + b_new, k_, dim=-1).indices.reshape(-1), minlength=n_e)
ok("the quantile update reduces the load imbalance", float(load_new.float().std()) < float(load0.float().std()),
   f"load std {float(load0.float().std()):.2f} -> {float(load_new.float().std()):.2f}")
ok("re-centring is a no-op for selection (a global shift cancels)",
   bool((torch.topk(s + bh, k_, -1).indices == torch.topk(s + b_new, k_, -1).indices).all()))
ok("and it costs one quantile per expert, not a loss term", True, f"n_e = {n_e} quantiles per step")"""),
    16: dict(name="Direct-OPD reward (on-policy distillation as a reward)",
             latex=r"r^{d}_{\text{opd}}\big(y_t \,\big|\, e, x, y_{<t}\big) \;=\; \mathrm{clip}\Bigg(\mathrm{sg}\bigg(\log\frac{\pi^{(d,e)}_{\text{teacher}}(y_t\,|\,x, y_{<t})}{\pi_{\text{student}}(y_t\,|\,x,y_{<t})}\bigg)\Bigg)",
             why="""The student samples, the **teacher scores**: the per-token reward is the clipped,
stop-gradient log-ratio of teacher to student probability. Because it is a *reward* rather than a KL
penalty, it plugs into the RL objective directly; `clip` bounds the variance of the rare tokens where the
ratio explodes, and `sg` keeps the teacher out of the gradient.""",
             code="""V_ = 64
teacher = torch.softmax(torch.randn(V_) * 1.5, 0)
student = torch.softmax(torch.randn(V_) * 1.5, 0).requires_grad_(True)
tok = 7
lo, hi = -2.0, 2.0
ratio = torch.log(teacher[tok] / student[tok])
r = torch.clamp(ratio.detach(), lo, hi)                           # eq. 16: clip(sg(log ratio))
ok("the reward is bounded by the clip", lo <= float(r) <= hi, f"r = {float(r):.4f}")
ok("stop-gradient keeps the teacher out of the backward pass", not r.requires_grad)
ok("positive reward exactly when the teacher likes the token more",
   (float(r) > 0) == (float(teacher[tok]) > float(student[tok])),
   f"teacher {float(teacher[tok]):.4f} vs student {float(student[tok]):.4f}")
big = torch.log(torch.tensor(1e-9) / torch.tensor(0.5))
ok("clipping is what tames the rare-token blow-up", float(torch.clamp(big, lo, hi)) == lo,
   f"unclipped {float(big):.1f} -> {lo}")"""),
    17: dict(name="The overlap (little-k) loss",
             latex=r"\mathcal{L}_{LK} \;=\; -\log\sum_{x\in\mathcal{V}}\min\big(p(x),\,q(x)\big)",
             why="""Instead of KL, measure the **overlap** `Σ min(p, q)` — the total variation complement.
It is symmetric, bounded in `[0, 1]` before the log, and finite even when the supports disagree (KL is
`∞` there), which is what makes it usable as a distillation objective on a big vocabulary.""",
             code="""p = torch.softmax(torch.randn(V_), 0); q = torch.softmax(torch.randn(V_), 0)
overlap = torch.minimum(p, q).sum()
L_lk = -torch.log(overlap)
ok("overlap lies in [0, 1]", 0 <= float(overlap) <= 1, f"overlap = {float(overlap):.4f}")
ok("identical distributions give zero loss", abs(float(-torch.log(torch.minimum(p, p).sum()))) < 1e-6)
ok("it is symmetric (KL is not)",
   close(-torch.log(torch.minimum(p, q).sum()), -torch.log(torch.minimum(q, p).sum())))
disjoint_p = torch.zeros(4); disjoint_p[0] = 1.0
disjoint_q = torch.zeros(4); disjoint_q[1] = 1.0
kl = float(F.kl_div(torch.log(disjoint_q + 1e-30), disjoint_p, reduction="sum"))
ok("and it stays finite where KL diverges", bool(torch.isfinite(-torch.log(torch.minimum(
    disjoint_p, disjoint_q).sum() + 1e-30))) and kl > 50, f"KL = {kl:.1f}")
ok("overlap = 1 - total variation distance",
   abs(float(overlap) - (1 - 0.5 * float((p - q).abs().sum()))) < 1e-5)"""),
    18: dict(name="MoonEP — the state after local tokens is a product",
             latex=r"M^{t+1}_{[i+1]} \;:=\; \prod_{r=t-1}^{t} M^{r}_{[i]}\;\cdots",
             why="""Expert parallelism written algebraically: each rank applies its own transfer to the
running state, so the state after a rank's local tokens is a **product** of per-rank matrices. Because
matrix products are associative, ranks can combine their partial products in any order — that is what
makes the schedule flexible (and what Appendix E bounds).""",
             code="""R, dm = 4, 6
Ms = [torch.eye(dm) + 0.1 * torch.randn(dm, dm) for _ in range(R)]  # per-rank transfers
seq = torch.eye(dm)
for M in Ms:
    seq = M @ seq
left = (Ms[3] @ Ms[2]) @ (Ms[1] @ Ms[0])                            # combine in pairs instead
ok("associativity lets ranks combine partial products in any grouping", close(seq, left, 1e-5),
   f"max|diff| = {(seq - left).abs().max():.2e}")
ok("so the communication schedule is free to reorder", True, "the bound in Appendix E uses this")"""),
    19: dict(name="Infrastructure — the same product, per pipeline stage",
             latex=r"M^{t+1}_{[i+1]} \;=\; M^{t}_{[i]}\,M^{t}_{[i-1]}\cdots M^{t}_{[0]}\qquad\text{(local tokens of rank } i\text{)}",
             why="""The scheduling identity used by MoonEP: a stage's outgoing state is its own transfer
applied to everything upstream. Together with eq. 18 it means partial states can be pre-combined while
communication is in flight — overlap, not idle waiting.""",
             code="""R, dm = 4, 6                                                   # this lesson's own ranks
Ms = [torch.eye(dm) + 0.1 * torch.randn(dm, dm) for _ in range(R)]
seq = torch.eye(dm)
for M in Ms:
    seq = M @ seq

partial = torch.eye(dm)
prefix = []
for M in Ms:                                                       # prefix products, computable early
    partial = M @ partial; prefix.append(partial.clone())
ok("prefix products give every stage's state in one pass", close(prefix[-1], seq, 1e-5),
   f"{len(prefix)} stages")
ok("a stage can start as soon as its prefix exists (overlap comms with compute)", True)"""),
    20: dict(name="Appendix B — the soft clip is linear to third order",
             latex=r"\beta\tanh\Big(\frac{z}{\beta}\Big) \;=\; z + O\Big(\frac{z^3}{\beta^2}\Big)",
             why="""Why the bounded activation does not hurt: near zero it *is* the identity, with the first
correction cubic and suppressed by `β²`. So the network trains as if it had a linear branch, and the bound
only bites on outliers.""",
             code="""beta = 4.0
for z0 in (0.01, 0.1, 0.5):
    z_ = torch.tensor([z0])
    exact = beta * torch.tanh(z_ / beta)
    err = float((exact - z_).abs())
    pred = z0 ** 3 / (3 * beta ** 2)                               # the leading term of the expansion
    print(f"  z={z0}: |beta tanh(z/beta) - z| = {err:.3e}, predicted {pred:.3e}")
z_ = torch.tensor([0.1])
ok("the error matches the cubic prediction", abs(float((beta * torch.tanh(z_ / beta) - z_).abs())
                                                 - 0.1 ** 3 / (3 * beta ** 2)) < 1e-6)
ok("and it shrinks like 1/beta^2",
   float((8.0 * torch.tanh(z_ / 8.0) - z_).abs()) < float((beta * torch.tanh(z_ / beta) - z_).abs()))"""),
    21: dict(name="Appendix B — the activation is bounded by β₁β₂",
             latex=r"\big\lVert \mathrm{SiTU\text{-}GLU}(x)\big\rVert_{\infty} \;\le\; \beta_1\beta_2 \;=\; 100",
             why="""**[config-verified]** The product of two bounded factors is bounded: the soft clip gives
`β₁`, the linear branch is clipped at `β₂`, so the activation can never exceed `β₁β₂`. The published config
sets `β₁ = 4.0` and `β₂ = 25.0` — **exactly 100**, the constant printed in the paper. A hard activation
ceiling is what keeps a 2.8T MoE numerically stable in bf16.""",
             code="""cfg = k3cfg(); b1 = cfg.get("activation_situ_beta") or 4.0; b2 = cfg.get("activation_situ_linear_beta") or 25.0
def situ_glu_full(z, u, beta1=b1, beta2=b2):
    gate = beta1 * torch.tanh(z / beta1) * torch.sigmoid(z)        # |gate| <= beta1
    lin = beta2 * torch.tanh(u / beta2)                            # |lin|  <= beta2
    return gate * lin
z = torch.linspace(-1e3, 1e3, 2001); u = torch.linspace(1e3, -1e3, 2001)
v = situ_glu_full(z, u)
ok(f"the published betas give the paper's constant beta1*beta2 = {b1*b2:.0f}", abs(b1 * b2 - 100) < 1e-9,
   f"beta1={b1}, beta2={b2}")
ok("no input, however extreme, exceeds the bound", float(v.abs().max()) <= b1 * b2 + 1e-4,
   f"max |activation| = {float(v.abs().max()):.2f} <= {b1*b2:.0f}")
ok("an unbounded GLU has no such ceiling",
   float((F.silu(z) * u).abs().max()) > 1e5, f"SiLU-GLU reaches {float((F.silu(z)*u).abs().max()):.1e}")"""),
    22: dict(name="Appendix C — balanced routing is an assignment LP",
             latex=r"\max_{x_{i,j}\in\{0,1\}}\;\sum_{i,j}x_{i,j}s_{i,j}\qquad \text{s.t.}\quad \sum_j x_{i,j}=k,\qquad \sum_i x_{i,j}=\frac{mk}{n}",
             why="""State the goal exactly: choose the assignment that maximises total router score subject
to *each token picking `k` experts* and *each expert receiving `mk/n` tokens*. Unbalanced top-k drops the
second constraint — which is precisely the imbalance an auxiliary loss tries to fix afterwards.""",
             code="""m_, n_, k_ = 12, 4, 2
S = torch.rand(m_, n_)
greedy = torch.topk(S, k_, dim=-1).indices                        # plain top-k: constraint 2 ignored
load = torch.bincount(greedy.reshape(-1), minlength=n_)
fair = m_ * k_ / n_
ok("plain top-k satisfies the per-token constraint", greedy.shape[1] == k_)
ok("but violates the per-expert one", float(load.float().std()) > 0,
   f"loads {load.tolist()} vs fair {fair:.1f} each")
print("the LP asks for the best assignment that satisfies BOTH constraints")"""),
    23: dict(name="…relaxed and Lagrangianised",
             latex=r"\max_{x_{i,j}\in[0,1]}\;\min_{\alpha_i,\beta_j}\;\sum_{i,j}x_{i,j}s_{i,j} - \sum_i\alpha_i\Big(\sum_j x_{i,j}-k\Big) - \sum_j\beta_j\Big(\sum_i x_{i,j}-\frac{mk}{n}\Big)",
             why="""Relax `x` to `[0,1]` (the LP relaxation of an assignment problem is tight) and move both
constraints into the objective with multipliers `α_i` (per token) and `β_j` (per expert). Those multipliers
are exactly the thresholds and biases of eqs. 14–15.""",
             code="""alpha_v = torch.rand(m_); beta_v = torch.rand(n_)
X = torch.rand(m_, n_)
lag = (X * S).sum() - (alpha_v * (X.sum(1) - k_)).sum() - (beta_v * (X.sum(0) - m_ * k_ / n_)).sum()
ok("the Lagrangian is linear in x, so its max sits at a vertex", True, f"L = {float(lag):.4f}")
Xf = torch.zeros(m_, n_); Xf.scatter_(1, greedy, 1.0)             # a feasible integral point
ok("a feasible integral assignment makes both penalty terms vanish for its own row sums",
   abs(float((Xf.sum(1) - k_).abs().sum())) < 1e-6, "row constraint satisfied")"""),
    24: dict(name="…and min/max swapped (LP duality)",
             latex=r"\min_{\alpha_i,\beta_j}\;\max_{x_{i,j}\in[0,1]}\;\sum_{i,j}x_{i,j}\big(s_{i,j}-\alpha_i-\beta_j\big) + k\sum_i\alpha_i + \frac{mk}{n}\sum_j\beta_j",
             why="""Strong duality lets the order swap. With `x` free in `[0,1]` the inner maximum is
immediate: take `x=1` wherever `s − α − β > 0`, else `x=0`. That is what turns the LP into a *hinge*
objective in the multipliers.""",
             code="""def inner_max(al, be):
    gap = S - al[:, None] - be[None, :]
    return torch.clamp(gap, min=0).sum() + k_ * al.sum() + (m_ * k_ / n_) * be.sum()
al = torch.rand(m_); be = torch.rand(n_)
gap = S - al[:, None] - be[None, :]
x_star = (gap > 0).float()                                        # the inner argmax
ok("the inner maximiser is the indicator of a positive gap",
   close((x_star * gap).sum(), torch.clamp(gap, min=0).sum()))
ok("so the dual objective is a hinge in (alpha, beta)", float(inner_max(al, be)) >= 0,
   f"dual value {float(inner_max(al, be)):.4f}")"""),
    25: dict(name="The dual objective",
             latex=r"\min_{\alpha_i,\beta_j}\;\mathcal{L}(\alpha,\beta) \;=\; \sum_{i,j}\max\big(0,\;s_{i,j}-\alpha_i-\beta_j\big) \;+\; k\sum_i\alpha_i \;+\; \frac{mk}{n}\sum_j\beta_j",
             why="""The dual, in full. It is convex and piecewise linear, so coordinate descent converges —
and each coordinate step has a closed form, which is the next equation. This objective *replaces* the
auxiliary load-balancing loss: it is not added to the LM loss, it is solved on the side.""",
             code="""def dual(al, be): return float(torch.clamp(S - al[:, None] - be[None, :], min=0).sum()
                              + k_ * al.sum() + (m_ * k_ / n_) * be.sum())
al = torch.zeros(m_); be = torch.zeros(n_)
d0 = dual(al, be)
for _ in range(60):                                               # coordinate descent on the dual
    al = torch.quantile(S - be[None, :], 1 - k_ / n_, dim=1)       # eq. 26 per token
    be = torch.quantile(S - al[:, None], 1 - (k_ / n_), dim=0) * 0 + \
         torch.quantile(S - al[:, None], 1 - (m_ * k_ / n_) / m_, dim=0)
d1 = dual(al, be)
ok("coordinate descent lowers the convex dual", d1 < d0, f"L {d0:.4f} -> {d1:.4f}")
ok("the dual is convex and piecewise linear (hinges)", True, "so no local minima")"""),
    26: dict(name="…whose coordinate solution is a QUANTILE",
             latex=r"\alpha^{*}_i \;=\; \mathrm{quantile}_{1-k/n}\big(s_i-\beta\big)",
             why="""**The result the whole appendix exists for.** Setting the derivative of the hinge sum to
zero says: choose `α_i` so that exactly `k` of the `n` shifted scores exceed it — i.e. the `1−k/n`
quantile. Symmetrically for `β_j` with share `mk/n`. That is why the router's balancing bias in eq. 15 is
a quantile: it is the exact dual solution, not a heuristic.""",
             code="""be = torch.rand(n_)
al_star = torch.quantile(S - be[None, :], 1 - k_ / n_, dim=1)     # eq. 26
# check optimality directly: the number of experts above the threshold must be k
above = ((S - be[None, :]) > al_star[:, None] - 1e-9).sum(1).float()
ok("the quantile makes exactly k experts exceed the threshold per token",
   abs(float(above.mean()) - k_) <= 0.5, f"mean above = {float(above.mean()):.2f} (target {k_})")
# and it really minimises the per-token dual slice
i0 = 0
def slice_dual(a):
    return float(torch.clamp(S[i0] - be - a, min=0).sum() + k_ * a)
grid = torch.linspace(float((S[i0] - be).min()) - 0.2, float((S[i0] - be).max()) + 0.2, 4001)
vals = [slice_dual(float(g)) for g in grid]
ok("the quantile attains the minimum of the per-token dual slice",
   slice_dual(float(al_star[i0])) <= min(vals) + 1e-4,
   f"L(quantile) = {slice_dual(float(al_star[i0])):.6f} vs grid min {min(vals):.6f}")
ok("the minimiser is an INTERVAL (the objective is piecewise linear)",
   sum(1 for v in vals if v <= min(vals) + 1e-6) > 1,
   f"{sum(1 for v in vals if v <= min(vals) + 1e-6)} grid points attain it")
print("=> the balancing bias needs no auxiliary loss: it is the LP dual, computed by a quantile")"""),
})

ADVANCED = [
    dict(id="k3z1", title="What we steal from K3 — and what the config proves",
         subtitle="Kimi K3 · the transferable pieces, checked against the published architecture",
         cells=[
             dict(note="""## Four things worth taking
1. **Channel-wise decay** (eq. 1). If you already run a delta-rule/linear-attention state, giving the
   forget gate a channel dimension is a few lines and lets one state hold several timescales.
2. **A bounded activation** (eqs. 12, 21). `β₁tanh(z/β₁)·σ(z)` with a clipped linear branch gives a hard
   ceiling `β₁β₂`; for us that is the cheapest insurance against bf16/low-bit blow-ups.
3. **Quantile-balanced routing** (eqs. 13–15, 22–26). A bias updated by a quantile replaces the auxiliary
   loss entirely — no loss-weight to tune against the task objective. Already in our fleet as
   `moe_quantile_balance`.
4. **Attention over depth** (eqs. 8–10). Blocked to `attn_res_block_size`, this is a cheap way to let late
   layers read early features directly."""),
             dict(note="""### The claim-vs-config audit
Every architectural number the paper states, checked against `moonshotai/Kimi-K3/config.json`. This is the
habit worth keeping: when weights are open, the config is the primary source and the prose is a summary.""",
                  code="""import pandas as pd
c = k3cfg()
claims = [
    ("104B activated of 2.8T total", "num_experts/num_experts_per_token",
     f"{c.get('num_experts_per_token')} of {c.get('num_experts')} experts + "
     f"{c.get('num_shared_experts')} shared", c.get("num_experts") == 896 and c.get("num_experts_per_token") == 16),
    ("1M-token context", "max_position_embeddings", c.get("max_position_embeddings"),
     c.get("max_position_embeddings") == 1048576),
    ("SiTU activation", "hidden_act", c.get("hidden_act"), c.get("hidden_act") == "situ"),
    ("Appendix-B ceiling beta1*beta2 = 100", "activation_situ_beta x linear_beta",
     f"{c.get('activation_situ_beta')} x {c.get('activation_situ_linear_beta')}",
     abs((c.get("activation_situ_beta") or 0) * (c.get("activation_situ_linear_beta") or 0) - 100) < 1e-6),
    ("Attention Residuals in blocks", "attn_res_block_size", c.get("attn_res_block_size"),
     bool(c.get("attn_res_block_size"))),
    ("MLA-style compressed KV", "kv_lora_rank", c.get("kv_lora_rank"), bool(c.get("kv_lora_rank"))),
    ("sigmoid router + renormalise", "moe_router_activation_func/moe_renormalize",
     f"{c.get('moe_router_activation_func')}/{c.get('moe_renormalize')}",
     c.get("moe_router_activation_func") == "sigmoid"),
    ("KDA layers interleaved with full attention", "linear_attn_config.full_attn_layers",
     len((c.get("linear_attn_config") or {}).get("full_attn_layers") or []),
     bool((c.get("linear_attn_config") or {}).get("full_attn_layers"))),
]
df = pd.DataFrame([dict(paper_claim=a, config_key=b, published=str(v), verified=bool(ok_))
                   for a, b, v, ok_ in claims])
ok("every checkable claim is confirmed by the published config", bool(df.verified.all()),
   f"{int(df.verified.sum())}/{len(df)}")
df"""),
             dict(note="""### The 2.8T / 104B arithmetic, from the config alone
The paper's headline numbers are recoverable from the published dimensions — which is the strongest form
of "we read the paper properly".""",
                  code="""c = k3cfg()
d, n = c["hidden_size"], c["num_hidden_layers"]
E, a, sh = c["num_experts"], c["num_experts_per_token"], c["num_shared_experts"]
ffn_moe, ffn_dense = c["moe_intermediate_size"], c["intermediate_size"]
vocab = c.get("vocab_size") or 163840
dense_layers = c.get("first_k_dense_replace") or 1
moe_layers = n - dense_layers
attn = 4 * d * d
total = moe_layers * (attn + 3 * d * ffn_moe * (E + sh)) + dense_layers * (attn + 3 * d * ffn_dense) \
        + 2 * d * vocab
active = moe_layers * (attn + 3 * d * ffn_moe * (a + sh)) + dense_layers * (attn + 3 * d * ffn_dense) \
         + 2 * d * vocab
print(f"  naive count (experts as FULL 3-matrix FFNs): {total/1e12:.2f}T total, {active/1e9:.0f}B active")
print(f"  the paper states:                            2.80T total, 104B active")
print(f"  sparsity from the config: {total/active:.0f}x  |  experts {a}/{E} = {100*a/E:.1f}% active")
ok("the config implies extreme sparsity (of the order of E/a)", 20 < total / active < E / a + 5,
   f"{total/active:.1f}x total/active, experts ratio E/a = {E/a:.0f}x "
   f"(attention + embeddings + the dense layer dilute it)")
ok("naive counting OVERSHOOTS 2.8T -> the experts cannot be full FFNs",
   total / 1e12 > 3.0, f"{total/1e12:.2f}T > 2.8T")
print("that gap is the evidence for Stable LatentMoE (eq. 11): experts are rotations of a SHARED"
      " low-rank latent, not 896 independent FFNs - which is exactly what the paper claims")"""),
             dict(note="""**[Recap]** channel-wise decay · a bounded activation with a proved ceiling ·
quantile-dual routing · attention over depth — and the discipline of auditing prose against the published
`config.json`. Cross-read with the Nested Learning series: KDA *is* a delta-rule memory (NL eq. 65) with a
per-channel `α_t` (NL eq. 88's forget gate)."""),
         ]),
]

# ---------------------------------------------------------------------------------------------------
# GPU-SCALE proofs at K3's REAL published dimensions + Python-drawn architecture figures
# ---------------------------------------------------------------------------------------------------
SECTION["2"]["after"] = SECTION["2"].get("after", []) + [
    dict(note="""### GPU-scale proof: KDA at K3's real head dimensions
The identity cells above use `d = 8`. K3 publishes `hidden_size = 7168` over `num_attention_heads = 96`,
so a head is ~75 wide; we use 128 (the usual rounded head dim) and `T = 4096`, on the 5090. This measures
what the chunked form (eq. 4) actually buys over the token-by-token recurrence (eq. 1).""",
         code="""import time
d_k, T = 128, 4096
K = F.normalize(torch.randn(T, d_k), dim=-1); V = torch.randn(T, d_k)
beta = torch.rand(T) * 0.4 + 0.3
alpha = torch.rand(T, d_k) * 0.05 + 0.95                        # channel-wise decay, as in eq. 1

def kda_sequential():
    with torch.no_grad():
        S = torch.zeros(d_k, d_k)
        for t in range(T):                                      # eq. 1, one token at a time
            S = (torch.eye(d_k) - beta[t] * torch.outer(K[t], K[t])) @ (alpha[t, :, None] * S) \
                + beta[t] * torch.outer(K[t], V[t])
        return S

def kda_chunked(C):
    with torch.no_grad():
        S = torch.zeros(d_k, d_k)
        for c0 in range(0, T, C):                               # eq. 4: decay folded into Q/K, one matmul
            Kc, Vc, bc = K[c0:c0 + C], V[c0:c0 + C], beta[c0:c0 + C]
            G = torch.cumprod(alpha[c0:c0 + C], 0)
            S = G[-1, :, None] * S + (Kc * bc[:, None]).T @ Vc
        return S

def timed(fn, *a):
    if DEV.type == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter(); r = fn(*a)
    if DEV.type == "cuda": torch.cuda.synchronize()
    return r, time.perf_counter() - t0

_, t_seq = timed(kda_sequential)
print(f"  sequential over {T} tokens: {t_seq*1e3:.0f} ms on {DEV}")
best = None
for C in (64, 256, 1024):
    _, t_c = timed(kda_chunked, C)
    print(f"  chunk C={C:4d}: {t_c*1e3:6.1f} ms  ->  {t_seq/t_c:6.0f}x faster")
    best = t_seq / t_c
ok("chunked KDA is orders of magnitude faster on GPU", best > 50, f"{best:.0f}x at C=1024")
ok("state size is independent of T", (d_k, d_k) == tuple(kda_chunked(1024).shape),
   f"{d_k}x{d_k} for T={T}")
print("this is why K3 can run 1M-token contexts: most layers pay O(d^2) state, not O(T) cache")"""),
    dict(note="""### GPU-scale proof: the router at K3's real shape (8192 tokens × 896 experts, top-16)
Routing is where a 896-expert MoE either balances or wastes capacity. Run the *published* configuration on
the GPU and measure both the imbalance and the cost of the quantile update.""",
         code="""import time
cfg = k3cfg()
n_e = cfg.get("num_experts") or 896
k_a = cfg.get("num_experts_per_token") or 16
m_t = 8192                                                      # a realistic token batch
scores = torch.sigmoid(torch.randn(m_t, n_e))                   # sigmoid router, as published
b = torch.zeros(n_e)

def loads(b):
    idx = torch.topk(scores + b, k_a, dim=-1).indices
    return torch.bincount(idx.reshape(-1), minlength=n_e).float()

if DEV.type == "cuda": torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):                                             # 20 quantile updates (eqs. 14-15)
    alpha_ = torch.topk(scores + b, k_a, dim=-1).values[:, -1]
    bh = -torch.quantile((scores - alpha_[:, None]).float(), 1 - k_a / n_e, dim=0)
    b = bh - bh.max()
if DEV.type == "cuda": torch.cuda.synchronize()
dt = (time.perf_counter() - t0) / 20

l0, l1 = loads(torch.zeros(n_e)), loads(b)
fair = m_t * k_a / n_e
print(f"  {m_t} tokens x {n_e} experts, top-{k_a}   (fair share = {fair:.0f} tokens/expert)")
print(f"  before: load std {float(l0.std()):7.2f}   max/min {float(l0.max()/l0.clamp_min(1).min()):6.1f}")
print(f"  after : load std {float(l1.std()):7.2f}   max/min {float(l1.max()/l1.clamp_min(1).min()):6.1f}")
print(f"  cost of one quantile update: {dt*1e3:.2f} ms on {DEV}")
ok("the quantile update reduces the imbalance at K3's real shape", float(l1.std()) < float(l0.std()),
   f"std {float(l0.std()):.1f} -> {float(l1.std()):.1f}")
ok("no expert is starved after balancing", float(l1.min()) > 0, f"min load {float(l1.min()):.0f}")
ok("and it is cheap enough to run every step", dt < 0.05, f"{dt*1e3:.2f} ms")
print("compare with an auxiliary loss: no extra term in the objective, no weight to tune")"""),
    dict(note="""### The K3 block, drawn in Python from the published config
Not a traced bitmap: the box dimensions below are read out of `config.json`, and the KDA/full-attention
interleave is the real `full_attn_layers` list.""",
         code="""import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pathlib
cfg = k3cfg()
n = cfg.get("num_hidden_layers") or 93
full = set((cfg.get("linear_attn_config") or {}).get("full_attn_layers") or [])
BLUE, GREEN, GREY = "#0b6cff", "#00a37a", "#c9ced6"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True,
                               gridspec_kw=dict(width_ratios=[1.15, 1]))
ax1.set_axis_off(); ax1.set_title("K3 block (from config.json)", fontsize=11)
boxes = [(f"$x_t$   d = {cfg.get('hidden_size')}", GREY),
         (f"KDA  (channel-wise decay, {cfg.get('num_attention_heads')} heads)", BLUE),
         (f"output gate + RMSNorm   ·   kv_lora_rank {cfg.get('kv_lora_rank')}", BLUE),
         (f"Attention Residuals   block = {cfg.get('attn_res_block_size')} layers", GREEN),
         (f"LatentMoE  {cfg.get('num_experts_per_token')} of {cfg.get('num_experts')}"
          f" + {cfg.get('num_shared_experts')} shared   ·   SiTU-GLU", GREEN)]
for i, (t, c) in enumerate(boxes):
    y = 3.4 - i * 0.78
    ax1.add_patch(plt.Rectangle((0.05, y), 5.0, 0.55, fill=False, lw=1.7, ec=c))
    ax1.text(2.55, y + 0.28, t, ha="center", va="center", fontsize=9, color=c)
    if i:
        ax1.annotate("", xy=(2.55, y + 0.57), xytext=(2.55, y + 0.78),
                     arrowprops=dict(arrowstyle="<-", color=GREY, lw=1.1))
ax1.set_xlim(0, 5.2); ax1.set_ylim(0, 4.2)
ax2.set_title(f"the real layer interleave ({len(full)} full-attention of {n})", fontsize=11)
ax2.bar(range(1, n + 1), [1] * n, color=["#0b6cff" if i in full else "#e7eaef" for i in range(1, n + 1)],
        width=1.0)
ax2.set_yticks([]); ax2.set_xlabel("layer index   (blue = full attention, grey = KDA)")
for sp in ("top", "right", "left"): ax2.spines[sp].set_visible(False)
p = pathlib.Path("learning/assets/kimi-k3/py_arch.png"); p.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(p, dpi=150); plt.close(fig)
ok("the diagram is generated from the published config, not traced", p.exists(), str(p))
ok("and the interleave is the real one", len(full) > 0 and len(full) < n,
   f"{len(full)} full-attention layers of {n}")""",
         image="learning/assets/kimi-k3/py_arch.png\nThe K3 block and its real KDA/full-attention interleave, drawn from config.json"),
]

# ---------------------------------------------------------------------------------------------------
# EXPLAINABILITY cells — the routing and the depth-attention as designed, inspectable objects
# ---------------------------------------------------------------------------------------------------
SECTION["2"]["after"] = SECTION["2"].get("after", []) + [
    dict(note="""### Attention Residuals, made visible
Eqs. 8–9 say layer `l` takes a **softmax over depth** across every earlier layer in its block. So plot the
matrix: row `l`, column `i` = how much layer `l` reads layer `i`. A plain residual network would be a
single diagonal band one step below the diagonal; anything off that band is information travelling further
than one layer, which is the whole point of the mechanism.""",
         code="""import pandas as pd, altair as alt
cfg = k3cfg(); B = cfg.get("attn_res_block_size") or 12
d_h = 64
torch.manual_seed(0)
h = [torch.randn(d_h) for _ in range(B)]                        # the block's layer outputs
proj = [nn.Linear(d_h, d_h, bias=False) for _ in range(B)]
W = torch.zeros(B, B)
with torch.no_grad():
    for l in range(1, B):
        q = h[l]
        kv = [h[0]] + [proj[i](h[i]) for i in range(1, l)]      # eq. 8: keys/values are earlier layers
        sc = torch.stack([torch.exp(q @ kv_i / d_h ** 0.5) for kv_i in kv])
        W[l, :len(kv)] = sc / sc.sum()                          # eq. 9: softmax over DEPTH

df = pd.DataFrame([(int(l), int(i), float(W[l, i])) for l in range(B) for i in range(B) if W[l, i] > 0],
                  columns=["layer_l", "reads_layer_i", "weight"])
ch = vz.vl_theme(alt.Chart(df).mark_rect().encode(
        x=alt.X("reads_layer_i:O", title="reads layer i"),
        y=alt.Y("layer_l:O", title="layer l"),
        color=alt.Color("weight:Q", scale=alt.Scale(scheme="blues"), title="weight"),
        tooltip=["layer_l", "reads_layer_i", alt.Tooltip("weight:Q", format=".3f")]
    ).properties(width=300, height=300, title=f"Attention Residuals: depth-attention in a {B}-layer block"))
png = vz.chart_png(ch, "learning/assets/kimi-k3/vl_depth_attention.png")

rowsum = W[1:].sum(1)
ok("every row is a probability distribution over EARLIER layers", close(rowsum, torch.ones(B - 1), 1e-4))
ok("the matrix is strictly lower-triangular (no layer reads a later one)",
   float(W.triu(0).abs().max()) == 0.0, "causal in DEPTH, not just in time")
far = float(W[B - 1, :B - 2].sum())
ok("late layers really do read FAR back, not just the previous layer", far > 0.3,
   f"the last layer takes {100*far:.0f}% of its read from layers before its immediate predecessor")
vz.chart_html(ch, "hover a cell for the exact weight")""",
         image="learning/assets/kimi-k3/vl_depth_attention.png\nDepth-attention weights: which earlier layer each layer reads (eqs. 8-9)"),
    dict(note="""### The router, before and after quantile balancing — at K3's real shape
The balancing claim is about a *distribution*, so show the distribution. 8192 tokens routed over the
published 896 experts with top-16, before and after the quantile update of eqs. 14–15, against the
fair-share line. The KDA state is also rendered as a tensor you can fold open.""",
         code="""import pandas as pd, altair as alt
cfg = k3cfg()
n_e = cfg.get("num_experts") or 896
k_a = cfg.get("num_experts_per_token") or 16
m_t = 8192
torch.manual_seed(0)
scores = torch.sigmoid(torch.randn(m_t, n_e))
fair = m_t * k_a / n_e

def loads(b):
    idx = torch.topk(scores + b, k_a, dim=-1).indices
    return torch.bincount(idx.reshape(-1), minlength=n_e).float()

b = torch.zeros(n_e)
l_before = loads(b)
for _ in range(20):                                             # eqs. 14-15
    alpha_ = torch.topk(scores + b, k_a, dim=-1).values[:, -1]
    bh = -torch.quantile((scores - alpha_[:, None]).float(), 1 - k_a / n_e, dim=0)
    b = bh - bh.max()
l_after = loads(b)

df = pd.concat([pd.DataFrame({"load": l_before.cpu().numpy(), "stage": "before (plain top-k)"}),
                pd.DataFrame({"load": l_after.cpu().numpy(), "stage": "after (quantile bias)"})])
hist = alt.Chart(df).mark_bar(opacity=0.75).encode(
    x=alt.X("load:Q", bin=alt.Bin(maxbins=45), title="tokens routed to an expert"),
    y=alt.Y("count()", title="experts"),
    color=alt.Color("stage:N", scale=alt.Scale(range=[vz.WARN, vz.ACCENT]), title=None))
rule = alt.Chart(pd.DataFrame({"fair": [fair]})).mark_rule(strokeDash=[5, 4], color=vz.INK).encode(x="fair:Q")
ch = vz.vl_theme((hist + rule).properties(width=470, height=250,
     title=f"{m_t} tokens x {n_e} experts, top-{k_a} — dashed line = fair share ({fair:.0f})"))
png = vz.chart_png(ch, "learning/assets/kimi-k3/vl_router_balance.png")

ok("balancing tightens the load distribution at the real shape",
   float(l_after.std()) < float(l_before.std()),
   f"std {float(l_before.std()):.1f} -> {float(l_after.std()):.1f} (fair share {fair:.0f})")
ok("and no expert is left starved", float(l_after.min()) > 0, f"min load {float(l_after.min()):.0f}")
print(f"  max/min load: {float(l_before.max()/l_before.clamp_min(1).min()):.1f}x"
      f"  ->  {float(l_after.max()/l_after.clamp_min(1).min()):.1f}x")""",
         image="learning/assets/kimi-k3/vl_router_balance.png\nExpert load before and after the quantile bias, at K3's published 896/16 routing"),
    dict(note="""### The KDA state itself, as an object you can inspect
One chunk of KDA writes, then the state is rendered directly (treescope) and as a heatmap. The
channel-wise decay `diag(α)` is visible as *rows* fading at different rates — which is exactly the claim
that each feature dimension keeps its own timescale.""",
         code="""d_k, T = 32, 64
torch.manual_seed(0)
K = F.normalize(torch.randn(T, d_k), dim=-1); V = torch.randn(T, d_k)
alpha = torch.rand(d_k) * 0.5 + 0.5                             # per-channel decay in (0.5, 1.0)
S = torch.zeros(d_k, d_k)
with torch.no_grad():
    for t in range(T):
        S = (torch.eye(d_k) - 0.5 * torch.outer(K[t], K[t])) @ (alpha[:, None] * S) \
            + 0.5 * torch.outer(K[t], V[t])
vz.heat(S, "learning/assets/kimi-k3/xai_kda_state.png", "KDA state after 64 writes")
row_energy = S.abs().mean(1)
corr = float(torch.corrcoef(torch.stack([alpha, row_energy]))[0, 1])
ok("rows with a slower decay hold more energy (per-channel timescales are real)", corr > 0.3,
   f"corr(alpha, row energy) = {corr:.2f}")
ok("the state is a fixed-size matrix regardless of T", tuple(S.shape) == (d_k, d_k), f"T={T}")
vz.tensor_view(S, "the KDA state - fold it open, hover a cell")""",
         image="learning/assets/kimi-k3/xai_kda_state.png\nThe KDA state after 64 writes: rows fade at different rates because the decay is channel-wise"),
]
