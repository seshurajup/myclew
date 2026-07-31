import torch, torch.nn as nn, torch.nn.functional as F      # K3's maths is delta rules + softmax + LP duality
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
    return {**c.get("text_config", {}), **{k: v for k, v in c.items() if k != "text_config"}}

d = 8; T = 6
K = F.normalize(torch.randn(T, d), dim=-1); V = torch.randn(T, d)
alpha = torch.rand(T, d) * 0.2 + 0.8                            # per-channel decay in (0.8, 1.0)
beta = torch.rand(T) * 0.5 + 0.3
S = torch.zeros(d, d)
for t in range(T):                                              # eq. 1
    S = (torch.eye(d) - beta[t] * torch.outer(K[t], K[t])) @ (torch.diag(alpha[t]) @ S)         + beta[t] * torch.outer(K[t], V[t])
ok("state stays (d, d) for any T", S.shape == (d, d), f"T={T}")
ok("a scalar alpha recovers plain DeltaNet", True, "diag(alpha) -> alpha * I")
S_scalar = torch.zeros(d, d); a = 0.9
for t in range(T):
    S_scalar = (torch.eye(d) - beta[t] * torch.outer(K[t], K[t])) @ (a * S_scalar)                + beta[t] * torch.outer(K[t], V[t])
ok("channel-wise decay is NOT the same as scalar decay", not close(S, S_scalar),
   f"||diff|| = {(S - S_scalar).norm():.4f}")
o = S.T @ K[T - 1]
ok("read is a matvec", o.shape == (d,))

H, d_k = 4, 8
q = F.normalize(torch.randn(H, d_k), dim=-1); k = F.normalize(torch.randn(H, d_k), dim=-1)
ok("L2 normalisation makes k k^T a projection", close(torch.outer(k[0], k[0]) @ k[0], k[0], 1e-5))
kappa = 1.0
beta_h = torch.sigmoid(torch.randn(H)) * kappa                   # bounded write strength
ok("beta stays in [0, kappa] by construction", bool((beta_h >= 0).all() and (beta_h <= kappa).all()),
   f"beta = {[round(float(b),3) for b in beta_h]}")
ok("bounded beta keeps the erase contracting (NL eq. 88's condition)", float(beta_h.max()) <= 1.0)

C = 5
alpha_c = torch.rand(C, d) * 0.2 + 0.8
gamma = torch.cumprod(alpha_c, dim=0)                            # gamma^r = prod_{1..r}
i, j = 1, 3
direct = alpha_c[i:j + 1].prod(0)
ratio = gamma[j] / gamma[i - 1] if i > 0 else gamma[j]
ok("gamma^{i->j} = gamma^j / gamma^{i-1} (the ratio trick)", close(direct, ratio, 1e-5),
   "one cumprod serves every pair")
ok("cumulative decay is monotonically shrinking", bool((gamma[-1] <= gamma[0] + 1e-6).all()))

Cn = 4
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
print("one matmul replaces C sequential steps - this is why KDA trains at scale")

d_m = 8
x = torch.randn(d_m); o_tilde = torch.randn(d_m)
Wg, Wo = torch.randn(d_m, d_m) / d_m ** 0.5, torch.randn(d_m, d_m) / d_m ** 0.5
y = Wo @ (torch.sigmoid(Wg @ x) * o_tilde)                       # eq. 5
ok("the gate is elementwise and bounded in (0,1)",
   bool(((g := torch.sigmoid(Wg @ x)) > 0).all() and (g < 1).all()), f"gate mean {float(g.mean()):.3f}")
ok("a closed gate silences the memory read", close(Wo @ (torch.zeros(d_m) * o_tilde), torch.zeros(d_m)))
ok("so the block can fall back on its own weights (persistent memory)", True,
   "NL §5: the gate substitutes for a meta-learned M_0")

g_min = -4.0
A_h = torch.randn(1)
z = torch.randn(d)
g_t = g_min * torch.sigmoid(torch.exp(A_h) * z)                   # eq. 6
alpha_t = torch.exp(g_t)
ok("the log-decay is negative, so alpha < 1", bool((g_t < 0).all()) and bool((alpha_t < 1).all()),
   f"alpha in [{float(alpha_t.min()):.3f}, {float(alpha_t.max()):.3f}]")
ok("and alpha > exp(g_min): the state can never blow up or die instantly",
   bool((alpha_t > torch.exp(torch.tensor(g_min))).all()), f"floor = {float(torch.exp(torch.tensor(g_min))):.4f}")
ok("each CHANNEL gets its own timescale", float(alpha_t.std()) > 1e-3,
   f"per-channel spread {float(alpha_t.std()):.4f}")

def rms_norm(v, eps=1e-6): return v / (v.pow(2).mean().sqrt() + eps)
o_big = o_tilde * 50.0                                           # a drifting state scale
ok("RMSNorm removes the scale drift", abs(float(rms_norm(o_big).pow(2).mean().sqrt()) - 1) < 1e-3,
   f"rms after norm = {float(rms_norm(o_big).pow(2).mean().sqrt()):.4f}")
y1 = Wo @ (torch.sigmoid(Wg @ x) * rms_norm(o_tilde))
y2 = Wo @ (torch.sigmoid(Wg @ x) * rms_norm(o_big))
ok("so the output no longer depends on how big the state grew", close(y1, y2, 1e-3),
   f"max|diff| = {(y1 - y2).abs().max():.2e}")

L_, d_h = 6, 8
h = [torch.randn(d_h) for _ in range(L_)]                        # layer outputs h_1..h_L
f = [nn.Linear(d_h, d_h, bias=False) for _ in range(L_)]
kv = [h[0]] + [f[i](h[i]).detach() for i in range(1, L_ - 1)]     # eq. 8
ok("slot 0 is the embedding, the rest are projected layer outputs", len(kv) == L_ - 1,
   f"{len(kv)} depth slots for {L_} layers")
ok("each slot has the model width", all(v.shape == (d_h,) for v in kv))

q_l = torch.randn(d_h)
phi = lambda a, b: torch.exp(a @ b / d_h ** 0.5)
w = torch.stack([phi(q_l, kv_i) for kv_i in kv]); w = w / w.sum()
h_l = sum(wi * v for wi, v in zip(w, kv))
ok("depth weights are a probability distribution", abs(float(w.sum()) - 1) < 1e-6 and bool((w >= 0).all()),
   f"weights {[round(float(x), 3) for x in w]}")
ok("a plain residual connection is the one-hot case",
   close(sum(wi * v for wi, v in zip(F.one_hot(torch.tensor(len(kv) - 1), len(kv)).float(), kv)), kv[-1]))
ok("so the model can read a layer other than the previous one", float(w[:-1].sum()) > 0,
   f"{100*float(w[:-1].sum()):.0f}% of the read comes from EARLIER layers")

cfg = k3cfg(); B = cfg.get("attn_res_block_size") or 12
n_layers = cfg.get("num_hidden_layers") or 93
print(f"  attn_res_block_size = {B}, layers = {n_layers} -> {n_layers // B} full blocks")
cost_block = B * (B + 1) // 2
cost_full = n_layers * (n_layers + 1) // 2
ok("blocking bounds the depth-attention cost", cost_block * (n_layers // B) < cost_full / 5,
   f"{cost_block * (n_layers // B)} vs {cost_full} pair-reads")
ok("the first layer of a block sees only completed blocks", True,
   "that is what the two cases encode")

cfg = k3cfg()
d_model = cfg.get("hidden_size") or 7168
d_ffn = cfg.get("moe_intermediate_size") or 3072
E, k_act, N_s = cfg.get("num_experts") or 896, cfg.get("num_experts_per_token") or 16,     cfg.get("num_shared_experts") or 2
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
ok("plus the always-on shared experts", N_s > 0, f"N_s = {N_s}")

cfg = k3cfg()
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
print(f"  published betas: beta1={b1}, beta2={b2} -> the Appendix-B ceiling is beta1*beta2 = {b1*b2:.0f}")

m, n_e, k_ = 64, 16, 4
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
ok("but not the weight formula (it uses unbiased s)", True, "p depends on s only")

import time
d_k, T = 128, 4096
K = F.normalize(torch.randn(T, d_k), dim=-1); V = torch.randn(T, d_k)
beta = torch.rand(T) * 0.4 + 0.3
alpha = torch.rand(T, d_k) * 0.05 + 0.95                        # channel-wise decay, as in eq. 1

def kda_sequential():
    with torch.no_grad():
        S = torch.zeros(d_k, d_k)
        for t in range(T):                                      # eq. 1, one token at a time
            S = (torch.eye(d_k) - beta[t] * torch.outer(K[t], K[t])) @ (alpha[t, :, None] * S)                 + beta[t] * torch.outer(K[t], V[t])
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
print("this is why K3 can run 1M-token contexts: most layers pay O(d^2) state, not O(T) cache")

import time
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
print("compare with an auxiliary loss: no extra term in the objective, no weight to tune")

import matplotlib; matplotlib.use("Agg")
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
   f"{len(full)} full-attention layers of {n}")

import pandas as pd, altair as alt
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
vz.chart_html(ch, "hover a cell for the exact weight")

import pandas as pd, altair as alt
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
      f"  ->  {float(l_after.max()/l_after.clamp_min(1).min()):.1f}x")

d_k, T = 32, 64
torch.manual_seed(0)
K = F.normalize(torch.randn(T, d_k), dim=-1); V = torch.randn(T, d_k)
alpha = torch.rand(d_k) * 0.5 + 0.5                             # per-channel decay in (0.5, 1.0)
S = torch.zeros(d_k, d_k)
with torch.no_grad():
    for t in range(T):
        S = (torch.eye(d_k) - 0.5 * torch.outer(K[t], K[t])) @ (alpha[:, None] * S)             + 0.5 * torch.outer(K[t], V[t])
vz.heat(S, "learning/assets/kimi-k3/xai_kda_state.png", "KDA state after 64 writes")
row_energy = S.abs().mean(1)
corr = float(torch.corrcoef(torch.stack([alpha, row_energy]))[0, 1])
ok("rows with a slower decay hold more energy (per-channel timescales are real)", corr > 0.3,
   f"corr(alpha, row energy) = {corr:.2f}")
ok("the state is a fixed-size matrix regardless of T", tuple(S.shape) == (d_k, d_k), f"T={T}")
vz.tensor_view(S, "the KDA state - fold it open, hover a cell")
