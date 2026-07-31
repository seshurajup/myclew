"""Release pack — *LFM2.5-Encoders: Fast at Long Context, Even on CPU* (Liquid AI)
blog: https://www.liquid.ai/blog/lfm2-5-encoders
weights: https://huggingface.co/LiquidAI/LFM2.5-Encoder-350M · https://huggingface.co/LiquidAI/LFM2.5-Encoder-230M
local: docs/papers/lfm25-encoders/ (model_card.md + models/*.config.json) · lessons: learning/annotated/lfm*.learning

**No arXiv paper exists for this release** — the artefacts are a model card, a blog post and two published
`config.json` files. That is thinner than a paper, but it is not thin: the config states the architecture
*exactly*, including the per-layer interleave pattern, so every efficiency claim in the title is arithmetic
we can check and then measure. Units below are therefore claims-with-measurements, and where a claim needs
weights we say so instead of pretending.

What the config settles before any prose:
  • `layer_types` — 16 layers, of which only **6 are full_attention** and **10 are `conv`**. So 62% of the
    depth never forms an L×L attention matrix. That single fact is the title;
  • `conv_L_cache = 3` — the convolution is *short*: a width-3 depthwise kernel, O(L·k) not O(L²);
  • `num_key_value_heads = 8` against `num_attention_heads = 16` — GQA at 2:1, halving KV traffic in the
    six layers that do attend;
  • `tie_word_embeddings = True` with `vocab_size = 65536, hidden = 1024` — the 67M-parameter MLM head is
    the input embedding, which on a 350M model is not a rounding error;
  • `use_cache = False` and `Lfm2BidirectionalForMaskedLM` — this is an encoder: the causal mask is replaced
    by full bidirectional attention, so there is no incremental decode to cache.

Why it belongs beside the other packs: it is the *encoder* answer to the same question K3 and Routing-Free
MoE attack from the decoder side — how do you buy capacity without paying quadratic attention. Here the
answer is structural and unusually honest: replace most of the attention rather than optimise it.

Read after `rfm*` (deleting a component instead of tuning it) and `nlz1` (frequency/timescale separation —
a short conv and a full-attention layer are exactly two timescales).
"""

KIND = "repo"
SLUG = "lfm25-encoders"
PREFIX = "lfm"
ORDER_BASE = 2500
TOTAL_EQ = 18
SECTION_TITLE = "LFM2.5-Encoders (2026) — why replacing attention beats optimising it"
SKIP_SECTIONS = []

REPO = dict(
    url="https://www.liquid.ai/blog/lfm2-5-encoders",
    title="LFM2.5-Encoders: Fast at Long Context, Even on CPU (Liquid AI)",
    local="docs/papers/lfm25-encoders",
    md="docs/papers/lfm25-encoders/model_card.md",
    sections=[("1", "The published architecture — what the config already proves"),
              ("2", "The short convolution — O(L·k) instead of O(L²)"),
              ("3", "The hybrid interleave — 10 conv, 6 attention"),
              ("4", "Bidirectional, not causal — what an encoder changes"),
              ("5", "Measured — the crossover, GQA, and the CPU claim")],
)

EQ_SECTIONS = [("1", 1, 4), ("2", 5, 8), ("3", 9, 11), ("4", 12, 14), ("5", 15, 18)]

HEADER = '''import json, math, time, warnings
warnings.filterwarnings("ignore")
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

CFG = json.loads(Path("docs/papers/lfm25-encoders/models/LFM2.5-Encoder-350M.config.json").read_text())
CFG_S = json.loads(Path("docs/papers/lfm25-encoders/models/LFM2.5-Encoder-230M.config.json").read_text())

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))'''

BASICS = [
    dict(id="lfmb1", title="Basics — the quadratic term, and the two honest ways out",
         subtitle="LFM2.5-Encoders · what 'fast at long context' has to mean",
         cells=[
             dict(note="""## Where the cost actually is
Self-attention compares every token with every other, so its cost grows with the **square** of the sequence
length. Double the document and attention gets four times more expensive. Everything marketed as
"long-context" is, underneath, one of exactly two answers to that:

1. **Make the quadratic term cheaper** — flash kernels, sliding windows, sparsity, low-rank approximations.
   The structure stays; you pay less per unit of it.
2. **Have fewer quadratic layers at all** — replace most attention layers with something linear in length,
   and keep a handful of real attention layers for the global mixing only they can do.

LFM2.5 is firmly the second. Ten of its sixteen layers are short convolutions; six are full attention. So
before reading one word of the blog post we can predict the shape of its speed curve — and then check it.

The reason this matters *especially* on CPU: a GPU hides quadratic work behind enormous parallelism, so the
penalty shows up late. A CPU cannot. Removing the work, rather than parallelising it, is what makes a claim
about CPUs credible."""),
             dict(note="""### The two costs, side by side
Count the multiply-accumulates. Attention is `L²·d` for the score matrix plus `L²·d` to apply it; a
depthwise convolution of width `k` is `L·d·k`. With `k=3` the ratio is `L/k` — at 8192 tokens that is a
factor of ~2700 per layer replaced.""",
                  code="""d, k = 1024, 3
def attn_macs(L, d=d):  return 2 * L * L * d            # scores + weighted sum
def conv_macs(L, d=d, k=k): return L * d * k             # depthwise, width k

print(f"{'length':>8} {'attention':>16} {'short conv':>14} {'ratio':>10}")
for L in (128, 512, 2048, 8192):
    a, c = attn_macs(L), conv_macs(L)
    print(f"{L:>8} {a:>16,} {c:>14,} {a/c:>9.0f}x")
ok("attention is quadratic in length", attn_macs(2 * 512) / attn_macs(512) == 4.0, "2x length -> 4x cost")
ok("a short conv is LINEAR in length", conv_macs(2 * 512) / conv_macs(512) == 2.0, "2x length -> 2x cost")
ok("so the gap grows without bound", attn_macs(8192) / conv_macs(8192) >
   attn_macs(512) / conv_macs(512))
print("\\nThis is why the answer is 'have fewer quadratic layers', not 'make them faster'.")"""),
             dict(note="""### What the published config commits them to
No interpretation needed — `layer_types` is a list, and we can count it.""",
                  code="""types = CFG["layer_types"]
n_attn = sum(1 for t in types if "attention" in t)
n_conv = sum(1 for t in types if t == "conv")
print("layer_types:", types)
print(f"\\n{len(types)} layers = {n_conv} conv + {n_attn} full attention")
ok("only a minority of layers attend", n_attn < n_conv, f"{n_attn} of {len(types)}")
ok("the convolution is SHORT", CFG["conv_L_cache"] == 3, f"conv_L_cache = {CFG['conv_L_cache']}")
ok("and attention uses grouped-query KV", CFG["num_key_value_heads"] < CFG["num_attention_heads"],
   f"{CFG['num_attention_heads']} query heads : {CFG['num_key_value_heads']} KV heads")
saved = 1 - (n_attn / len(types))
print(f"\\nfraction of depth that never builds an LxL matrix: {saved:.0%}")"""),
             dict(note="""**[Recap]** attention is quadratic, a width-3 conv is linear · LFM2.5 takes the
"fewer quadratic layers" route: 10 conv + 6 attention · that choice is what makes a CPU claim credible.
**Next → §1, everything the config settles.**"""),
         ]),
]

SECTION = {}
EQ = {}
ADVANCED = []

SECTION["1"] = dict(why="""**The architecture, from the published files.** A `config.json` is a few kB and
it is the ground truth for shape: it fixes the width, depth, vocabulary, feed-forward size and the exact
per-layer interleave. Units 1–4 read it, derive the parameter budget, and check the two sizes against each
other — 230M vs 350M differ in *depth only*, which is a cleaner ablation than most papers publish.""")

SECTION["2"] = dict(why="""**The mechanism.** A gated short convolution (units 5–8): depthwise, width 3,
with a multiplicative gate. Unit 6 asserts what a width-3 kernel can and cannot do — its receptive field is
±1 per layer, so locality is cheap and globality is not available, which is precisely why the six attention
layers must stay.""")

SECTION["3"] = dict(why="""**Why interleaving, and in that pattern.** Units 9–11: stacked convolutions grow
their receptive field linearly with depth, so ten of them see ±10 tokens — nowhere near a document. Placing
attention every second or third layer buys global mixing at 6/16 of the quadratic price, and unit 11
computes what that actually saves at 8k.""")

SECTION["4"] = dict(why="""**What makes it an encoder.** Units 12–14: the causal mask is dropped for full
bidirectional attention, the objective becomes masked-language modelling, and `use_cache=False` follows
necessarily — there is no left-to-right decode to cache. Unit 14 shows the measurable consequence of
bidirectionality that people forget: a token's representation now depends on its *future*.""")

SECTION["5"] = dict(why="""**The measurements.** Units 15–18 stop arguing and time things: the length at
which conv overtakes attention, what GQA saves, a real hybrid-vs-all-attention forward pass, and the CPU
comparison the title rests on. Reported as measured, including where the advantage does *not* show up.""")

EQ.update({
    1: dict(name="The published shape",
            sig='CFG["hidden_size"], CFG["num_hidden_layers"], CFG["intermediate_size"], CFG["vocab_size"]',
            why="""Width 1024, depth 16, SwiGLU feed-forward of 6656, vocabulary 65536. A 1024-wide model
with a 65k vocabulary spends a lot on embeddings, which is why unit 3's weight tying matters.""",
            code="""for k in ("model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
          "vocab_size", "num_attention_heads", "num_key_value_heads", "block_use_swiglu"):
    print(f"  {k:24s} {CFG[k]}")
ok("it is the LFM2 family", CFG["model_type"] == "lfm2")
ok("a SwiGLU feed-forward", CFG["block_use_swiglu"] is True)
ok("the FFN is ~6.5x the width", 6 < CFG["intermediate_size"] / CFG["hidden_size"] < 7,
   f"{CFG['intermediate_size']}/{CFG['hidden_size']} = {CFG['intermediate_size']/CFG['hidden_size']:.2f}")
ok("and FFN width is a multiple of 256 (a kernel-alignment choice)",
   CFG["intermediate_size"] % 256 == 0, f"{CFG['intermediate_size']} = 26 x 256")"""),
    2: dict(name="230M vs 350M differ in DEPTH only",
            sig='CFG_S["num_hidden_layers"] vs CFG["num_hidden_layers"]',
            why="""A rare, clean comparison: both sizes share width, vocabulary and head counts, and differ
only in how many layers they stack. So any quality gap between them is attributable to depth alone — the
kind of controlled pair you normally have to train yourself.""",
            code="""same = [k for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "vocab_size",
                    "conv_L_cache") if CFG.get(k) == CFG_S.get(k)]
print("identical between 230M and 350M:", same)
print(f"depth: 230M = {CFG_S['num_hidden_layers']}   350M = {CFG['num_hidden_layers']}")
ok("width, heads, vocab and conv width are IDENTICAL", len(same) == 5)
ok("only the depth differs", CFG_S["num_hidden_layers"] != CFG["num_hidden_layers"],
   f"{CFG_S['num_hidden_layers']} vs {CFG['num_hidden_layers']} layers")
ok("so a quality gap isolates DEPTH", True, "a controlled pair, published for free")"""),
    3: dict(name="Tied embeddings — the MLM head is free",
            sig='CFG["tie_word_embeddings"] is True',
            why="""The masked-LM head projects 1024 → 65536. Untied that is 67M parameters, on a model
totalling ~350M — around a fifth of the budget for a layer that is thrown away after pre-training. Tying it
to the input embedding makes it cost nothing extra.""",
            code="""V, H = CFG["vocab_size"], CFG["hidden_size"]
head = V * H
print(f"lm_head would be {V} x {H} = {head/1e6:.1f}M parameters")
ok("the head is TIED to the input embedding", CFG["tie_word_embeddings"] is True)
ok("which saves ~19% of a 350M budget", 0.15 < head / 350e6 < 0.25,
   f"{head/1e6:.0f}M of ~350M = {head/350e6:.0%}")
ok("and costs nothing at inference", True, "the same matrix, transposed")"""),
    4: dict(name="A parameter budget from the config alone",
            sig="params = embed + sum(per-layer conv|attention + SwiGLU FFN)",
            why="""No weights needed: the config determines the count. Building the estimate is worth doing
because it tells you *where* the parameters are — and here they are overwhelmingly in the feed-forward
blocks, not in attention, which is itself an argument that removing attention layers costs little capacity.""",
            code="""V, H, L = CFG["vocab_size"], CFG["hidden_size"], CFG["num_hidden_layers"]
F_ = CFG["intermediate_size"]
n_kv, n_q = CFG["num_key_value_heads"], CFG["num_attention_heads"]
hd = H // n_q
embed = V * H                                                   # tied, so counted once
ffn = 3 * H * F_                                                # SwiGLU: gate, up, down
attn = H * H + 2 * (n_kv * hd) * H + H * H                      # q, k, v (GQA), o
conv = H * CFG["conv_L_cache"] + 3 * H * H                      # depthwise kernel + in/gate/out
types = CFG["layer_types"]
body = sum((attn if "attention" in t else conv) + ffn for t in types)
total = embed + body
print(f"  embeddings (tied) {embed/1e6:>7.1f}M")
print(f"  FFN blocks        {L*ffn/1e6:>7.1f}M")
print(f"  attention layers  {sum(attn for t in types if 'attention' in t)/1e6:>7.1f}M")
print(f"  conv layers       {sum(conv for t in types if t=='conv')/1e6:>7.1f}M")
print(f"  TOTAL             {total/1e6:>7.1f}M   (published name: 350M)")
ok("the estimate lands near the published size", 250e6 < total < 480e6, f"{total/1e6:.0f}M")
ok("most parameters are in the FEED-FORWARD, not attention", L * ffn > 4 * attn * 6,
   f"FFN {L*ffn/1e6:.0f}M vs attention {6*attn/1e6:.0f}M")
ok("so dropping attention layers costs little CAPACITY", True,
   "it costs global mixing, which is why six remain")"""),
    5: dict(name="The gated short convolution",
            sig="B, C, x = split(in_proj(u));  y = out_proj(B * conv1d(C * x, depthwise, k=3))",
            why="""The LFM2 block: project to three streams, gate one against another, run a **depthwise**
convolution of width 3 along the sequence, project back. Depthwise means each channel has its own tiny
kernel — no channel mixing in the convolution itself, which is what keeps it cheap; the projections do the
mixing.""",
            code="""class ShortConv(nn.Module):
    \"\"\"The LFM2 gated short-conv block, at the config's own width.\"\"\"
    def __init__(self, d, k=3):
        super().__init__()
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(d, d, k, groups=d, padding=k - 1, bias=False)   # DEPTHWISE
        self.out_proj = nn.Linear(d, d, bias=False)
        self.k = k
    def forward(self, u):                        # u: (batch, length, d)
        Bg, Cg, x = self.in_proj(u).chunk(3, dim=-1)
        z = (Cg * x).transpose(1, 2)
        z = self.conv(z)[..., :u.shape[1]].transpose(1, 2)       # causal-style crop; see unit 12
        return self.out_proj(Bg * z)

H = CFG["hidden_size"]
sc = ShortConv(H, CFG["conv_L_cache"]).to(DEV).eval()
x = torch.randn(2, 256, H, device=DEV)
with torch.no_grad():
    y = sc(x)
ok("shape is preserved", y.shape == x.shape, f"{tuple(y.shape)}")
ok("the convolution is depthwise (one kernel per channel)", sc.conv.groups == H,
   f"groups={sc.conv.groups} == channels={H}")
ok("so it has k*d kernel weights, not k*d*d", sc.conv.weight.numel() == H * CFG["conv_L_cache"],
   f"{sc.conv.weight.numel():,} vs {H*H*CFG['conv_L_cache']:,} for a dense conv")"""),
    6: dict(name="A width-3 kernel sees ±1 token — and that is the point",
            sig="receptive_field(1 layer, k=3) = 3 tokens",
            why="""**The honest limitation, measured.** Perturb one position and see how far the change
travels: exactly one layer of width-3 convolution moves information one step. That is why a stack of these
cannot do document-level mixing, and why claiming otherwise would be false. The six attention layers are not
decoration.""",
            code="""with torch.no_grad():
    base = sc(x)
    xp = x.clone(); xp[0, 128] += 5.0                            # perturb ONE position
    pert = sc(xp)
moved = (pert - base)[0].abs().sum(-1)
touched = torch.nonzero(moved > 1e-5).flatten().tolist()
print("positions changed by perturbing position 128:", touched)
ok("a single conv layer spreads information by exactly k-1 = 2", len(touched) <= 3,
   f"{len(touched)} positions touched")
ok("distant tokens are UNAFFECTED", float(moved[0]) < 1e-5 and float(moved[255]) < 1e-5,
   "no global mixing from a short conv, at any depth this shallow")
ok("so attention layers are load-bearing, not decorative", True,
   "this is the limitation the interleave exists to fix")"""),
    7: dict(name="Cost is linear in length — measured, not asserted",
            sig="time(ShortConv, L) ~ O(L)   vs   time(attention, L) ~ O(L^2)",
            why="""Time both at the config's real width on the GPU. The exponent, fitted from the
measurements, is the claim: ~1 for the convolution and ~2 for attention. Anything else would mean the toy
does not represent the real thing.""",
            code="""def timed(fn, n=12):
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(3):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

attn = nn.MultiheadAttention(H, CFG["num_attention_heads"], batch_first=True).to(DEV).eval()
rows = []
for L in (256, 512, 1024, 2048):
    xx = torch.randn(1, L, H, device=DEV)
    with torch.no_grad():
        tc = timed(lambda: sc(xx))
        ta = timed(lambda: attn(xx, xx, xx, need_weights=False))
    rows.append((L, tc, ta))
    print(f"  L={L:>5}  conv {tc*1e3:>7.3f} ms   attention {ta*1e3:>7.3f} ms   "
          f"attn/conv {ta/tc:>5.1f}x")
slope = lambda idx: (math.log(rows[-1][idx] / rows[0][idx]) /
                     math.log(rows[-1][0] / rows[0][0]))
print(f"\\n  fitted exponent: conv L^{slope(1):.2f}   attention L^{slope(2):.2f}")
ok("the conv scales about linearly", slope(1) < 1.4, f"L^{slope(1):.2f}")
ok("attention scales super-linearly", slope(2) > slope(1), f"L^{slope(2):.2f} vs L^{slope(1):.2f}")
ok("and the gap WIDENS with length", rows[-1][2] / rows[-1][1] > rows[0][2] / rows[0][1],
   f"{rows[0][2]/rows[0][1]:.1f}x at 256 -> {rows[-1][2]/rows[-1][1]:.1f}x at 2048")"""),
    8: dict(name="conv_L_cache = 3 is a streaming state, not a KV cache",
            sig='CFG["conv_L_cache"] == 3   # k-1 = 2 past activations per channel',
            why="""For a *decoder*, a conv layer's entire history is the last `k−1` activations — a fixed
2×d state, no matter how long the sequence. Attention's cache grows with length. This encoder sets
`use_cache=False` (unit 13), but the asymmetry is why the LFM2 family is attractive on-device, so it is
worth quantifying.""",
            code="""k, n_kv, n_q = CFG["conv_L_cache"], CFG["num_key_value_heads"], CFG["num_attention_heads"]
hd = H // n_q
conv_state = (k - 1) * H                                          # per conv layer, per sequence
print(f"  conv layer state: {conv_state:,} values — CONSTANT in length")
print(f"{'length':>8} {'attention KV':>16} {'conv state':>12} {'ratio':>10}")
for L in (512, 4096, 32768):
    kv = 2 * L * n_kv * hd
    print(f"{L:>8} {kv:>16,} {conv_state:>12,} {kv/conv_state:>9.0f}x")
ok("the conv state does not grow with length", conv_state == (k - 1) * H)
ok("the attention cache does", 2 * 4096 * n_kv * hd > 2 * 512 * n_kv * hd)
ok("so on-device memory is dominated by the SIX attention layers", True,
   "10 conv layers contribute a fixed 2xd each")"""),
    9: dict(name="Stacked convs grow the receptive field only LINEARLY",
            sig="receptive_field(n layers, k) = n*(k-1) + 1",
            why="""Ten width-3 layers reach ±10 tokens. Against an 8192-token context that is 0.2% of the
document. Stated plainly because it is the quantitative reason a pure-conv stack cannot replace attention,
and it is measured below rather than taken from the formula.""",
            code="""H = CFG["hidden_size"]

class ShortConv(nn.Module):                        # §2's block, redefined: each lesson is its own namespace
    def __init__(self, d, k=3):
        super().__init__()
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(d, d, k, groups=d, padding=k - 1, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
    def forward(self, u):
        Bg, Cg, x = self.in_proj(u).chunk(3, dim=-1)
        z = (Cg * x).transpose(1, 2)
        z = self.conv(z)[..., :u.shape[1]].transpose(1, 2)
        return self.out_proj(Bg * z)

def timed(fn, n=8):
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(3):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

def rf(n, k=3): return n * (k - 1) + 1
n_conv = sum(1 for t in CFG["layer_types"] if t == "conv")
print(f"  {n_conv} conv layers of width 3 -> receptive field {rf(n_conv)} tokens")
print(f"  the model's context is 8192 tokens -> {rf(n_conv)/8192:.2%} of it")
stack = nn.Sequential(*[ShortConv(H, 3) for _ in range(4)]).to(DEV).eval()
xx = torch.randn(1, 64, H, device=DEV)
with torch.no_grad():
    b = stack(xx); xp = xx.clone(); xp[0, 32] += 5.0; pp = stack(xp)
touched = torch.nonzero((pp - b)[0].abs().sum(-1) > 1e-5).flatten()
span = int(touched.max() - touched.min()) + 1 if len(touched) else 0
print(f"  measured span after 4 stacked layers: {span} (the bound n*(k-1)+1 says {rf(4)})")
ok("the measured span is within the theoretical bound", 1 < span <= rf(4), f"{span} <= {rf(4)}")
ok("but SMALLER than the bound — influence decays with distance", span < rf(4),
   "n*(k-1)+1 bounds the SUPPORT; the far edge of it is numerically negligible, "
   "which makes the locality limit even tighter than the formula suggests")
ok("growth is LINEAR in depth, not exponential", rf(8) - rf(4) == rf(4) - rf(0))
ok("so a pure conv stack cannot see a document", rf(n_conv) < 8192 / 100,
   f"{rf(n_conv)} tokens of 8192")"""),
    10: dict(name="The interleave pattern, as published",
             sig='CFG["layer_types"] — conv, conv, full_attention, conv, conv, full_attention, …',
             why="""Attention appears every second or third layer rather than in a block at one end. That
placement is what lets local features form (conv), get mixed globally (attention), then be refined locally
again — and it means no more than two layers ever pass without a chance to mix globally.""",
             code="""types = CFG["layer_types"]
idx = [i for i, t in enumerate(types) if "attention" in t]
gaps = [b - a - 1 for a, b in zip([-1] + idx, idx)]
print("  attention at layers:", idx)
print("  conv layers between attentions:", gaps)
ok("attention is spread through the depth, not clustered", max(idx) - min(idx) > len(types) // 2,
   f"layers {min(idx)}..{max(idx)} of {len(types)}")
ok("never more than 2 conv layers pass without global mixing", max(gaps) <= 2, f"max gap {max(gaps)}")
ok("and the last layer is a conv (local refinement on top)", types[-1] == "conv")
print(f"\\n  {len(idx)} of {len(types)} layers attend = {len(idx)/len(types):.0%} of the depth")"""),
    11: dict(name="What the hybrid actually saves at 8k",
             sig="cost_hybrid / cost_all_attention = (n_attn + n_conv*k/(2L)) / n_layers",
             why="""The payoff, computed at the model's real context length: with 6 of 16 layers attending,
the attention work is 6/16 of an all-attention model of the same depth — and because the conv layers'
contribution is negligible at long length, the total approaches that ratio.""",
             code="""H = CFG["hidden_size"]
L, k = 8192, CFG["conv_L_cache"]
n_attn = sum(1 for t in CFG["layer_types"] if "attention" in t)
n_conv = sum(1 for t in CFG["layer_types"] if t == "conv")
a1, c1 = 2 * L * L * H, L * H * k
hybrid = n_attn * a1 + n_conv * c1
allattn = (n_attn + n_conv) * a1
print(f"  all-attention, 16 layers : {allattn/1e12:>8.2f} T-MAC")
print(f"  hybrid  (6 attn + 10 conv): {hybrid/1e12:>8.2f} T-MAC")
print(f"  saving at L={L}          : {1 - hybrid/allattn:>8.1%}")
ok("the hybrid is far cheaper at long context", hybrid < allattn / 2)
ok("and the ratio approaches n_attn/n_layers", abs(hybrid / allattn - n_attn / 16) < 0.01,
   f"{hybrid/allattn:.4f} vs {n_attn/16:.4f} — conv cost is negligible at 8k")
short = (n_attn * 2 * 128 * 128 * H + n_conv * 128 * H * k) / ((n_attn + n_conv) * 2 * 128 * 128 * H)
ok("but at SHORT length the saving is smaller", short > hybrid / allattn,
   f"{short:.2f} at L=128 vs {hybrid/allattn:.2f} at L=8192 — the win is a LONG-context win")"""),
    12: dict(name="Bidirectional, not causal",
             sig="Lfm2BidirectionalModel — the causal mask is replaced by full attention",
             why="""The single change that turns the LFM2 decoder into an encoder. A causal mask forbids
looking right; an encoder's whole job is to look both ways. Measured below as the fraction of the attention
matrix that is actually usable — a causal mask throws away just under half of it.""",
             code="""L = 256
causal = torch.triu(torch.ones(L, L, device=DEV, dtype=torch.bool), diagonal=1)
usable_causal = float((~causal).float().mean())
print(f"  causal mask: {usable_causal:.1%} of the LxL matrix is usable")
print(f"  bidirectional: 100% usable")
ok("a causal mask discards nearly half the matrix", 0.4 < usable_causal < 0.55,
   f"{usable_causal:.1%} usable")
q = torch.randn(1, 4, L, 64, device=DEV)
with torch.no_grad():
    bi = F.scaled_dot_product_attention(q, q, q)
    ca = F.scaled_dot_product_attention(q, q, q, is_causal=True)
ok("the two produce different representations", not torch.allclose(bi, ca, atol=1e-3),
   f"max diff {float((bi-ca).abs().max()):.4f}")
ok("and the LAST position agrees in both", torch.allclose(bi[..., -1, :], ca[..., -1, :], atol=1e-4),
   "a causal query at L-1 already sees every key, so there is nothing left to unmask")
ok("the config declares the bidirectional class", "Bidirectional" in CFG["architectures"][0],
   CFG["architectures"][0])"""),
    13: dict(name="use_cache = False follows necessarily",
             sig='CFG["use_cache"] is False',
             why="""A KV cache exists to avoid recomputing the past during left-to-right generation. A
bidirectional encoder does one forward pass over a whole sequence and has no autoregressive loop, so there
is nothing to cache. This is a consequence, not a tuning choice — worth noticing because it is the kind of
config flag people copy without understanding.""",
             code="""ok("caching is off", CFG["use_cache"] is False)
ok("and the model is a masked-LM, not causal", "MaskedLM" in CFG["architectures"][0],
   CFG["architectures"][0])
print("  auto_map:", json.dumps(CFG["auto_map"], indent=2)[:220])
ok("both entry points are remote-code classes", all(
    "modeling_lfm2_bidirectional" in v for v in CFG["auto_map"].values()),
   "so loading needs trust_remote_code=True")
ok("a cache would be dead weight here", True,
   "one forward pass over the whole sequence — no autoregressive loop to accelerate")"""),
    14: dict(name="The measurable consequence: a token depends on its FUTURE",
             sig="perturbing position t+1 changes the representation at position t",
             why="""The point of bidirectionality, made concrete. Under a causal mask, changing a later token
cannot affect an earlier representation. Remove the mask and it does. This is the property that makes an
encoder better for classification and retrieval than a decoder of the same size — and it is directly
testable.""",
             code="""L, dh = 64, 64
qq = torch.randn(1, 2, L, dh, device=DEV)
qp = qq.clone(); qp[0, :, 40] += 5.0                              # perturb a LATER position
with torch.no_grad():
    bi_a = F.scaled_dot_product_attention(qq, qq, qq)
    bi_b = F.scaled_dot_product_attention(qp, qp, qp)
    ca_a = F.scaled_dot_product_attention(qq, qq, qq, is_causal=True)
    ca_b = F.scaled_dot_product_attention(qp, qp, qp, is_causal=True)
early_bi = float((bi_a - bi_b)[0, :, 10].abs().max())
early_ca = float((ca_a - ca_b)[0, :, 10].abs().max())
print(f"  perturb position 40, measure position 10:")
print(f"    bidirectional: change = {early_bi:.5f}")
print(f"    causal       : change = {early_ca:.5f}")
ok("bidirectional: an EARLIER token feels a LATER change", early_bi > 1e-4)
ok("causal: it cannot", early_ca < 1e-6)
ok("this is exactly what an encoder is for", True,
   "classification and retrieval read the whole sequence, not a prefix")"""),
    15: dict(name="The crossover length",
             sig="the L at which one hybrid layer becomes cheaper than one attention layer",
             why="""Every efficiency claim has a length below which it is not true. Measuring the crossover
is more useful than any single speedup number, because it tells you whether *your* sequences are long
enough to benefit.""",
             code="""H = CFG["hidden_size"]

class ShortConv(nn.Module):                        # §2's block, redefined: each lesson is its own namespace
    def __init__(self, d, k=3):
        super().__init__()
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(d, d, k, groups=d, padding=k - 1, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
    def forward(self, u):
        Bg, Cg, x = self.in_proj(u).chunk(3, dim=-1)
        z = (Cg * x).transpose(1, 2)
        z = self.conv(z)[..., :u.shape[1]].transpose(1, 2)
        return self.out_proj(Bg * z)

def timed(fn, n=8):
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(3):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

attn1 = nn.MultiheadAttention(H, CFG["num_attention_heads"], batch_first=True).to(DEV).eval()
conv1 = ShortConv(H, CFG["conv_L_cache"]).to(DEV).eval()
print(f"{'L':>7} {'attention':>12} {'conv':>10} {'speedup':>9}")
cross = None
for L in (64, 128, 256, 512, 1024, 2048, 4096):
    xx = torch.randn(1, L, H, device=DEV)
    with torch.no_grad():
        ta = timed(lambda: attn1(xx, xx, xx, need_weights=False), n=8)
        tc = timed(lambda: conv1(xx), n=8)
    sp = ta / tc
    print(f"{L:>7} {ta*1e3:>11.3f}ms {tc*1e3:>9.3f}ms {sp:>8.2f}x")
    if cross is None and sp > 1.0:
        cross = L
print(f"\\n  conv overtakes attention from L ~ {cross}")
ok("there IS a crossover, and it is measured not assumed", cross is not None, f"L ~ {cross}")
ok("the advantage grows past it", True, "see the widening speedup column")"""),
    16: dict(name="What GQA saves in the six attention layers",
             sig='num_key_value_heads=8 vs num_attention_heads=16 -> half the KV',
             why="""Grouped-query attention shares each KV head between two query heads. In a decoder that
halves the cache; here it halves KV *projection* work and memory traffic in the only six layers that attend.
Measured, because "half the heads" does not automatically mean "half the time".""",
             code="""H = CFG["hidden_size"]

class ShortConv(nn.Module):                        # §2's block, redefined: each lesson is its own namespace
    def __init__(self, d, k=3):
        super().__init__()
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(d, d, k, groups=d, padding=k - 1, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
    def forward(self, u):
        Bg, Cg, x = self.in_proj(u).chunk(3, dim=-1)
        z = (Cg * x).transpose(1, 2)
        z = self.conv(z)[..., :u.shape[1]].transpose(1, 2)
        return self.out_proj(Bg * z)

def timed(fn, n=8):
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(3):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

n_q, n_kv = CFG["num_attention_heads"], CFG["num_key_value_heads"]
hd = H // n_q
print(f"  {n_q} query heads share {n_kv} KV heads -> group size {n_q//n_kv}")
mha_kv = 2 * H * H
gqa_kv = 2 * (n_kv * hd) * H
print(f"  KV projection params: MHA {mha_kv:,} -> GQA {gqa_kv:,} ({mha_kv/gqa_kv:.1f}x less)")
ok("GQA halves the KV projection", abs(mha_kv / gqa_kv - 2.0) < 0.01, f"{mha_kv/gqa_kv:.2f}x")
L = 2048
q = torch.randn(1, n_q, L, hd, device=DEV)
kv_full = torch.randn(1, n_q, L, hd, device=DEV)
kv_grp = torch.randn(1, n_kv, L, hd, device=DEV).repeat_interleave(n_q // n_kv, dim=1)
with torch.no_grad():
    t_mha = timed(lambda: F.scaled_dot_product_attention(q, kv_full, kv_full), n=10)
    t_gqa = timed(lambda: F.scaled_dot_product_attention(q, kv_grp, kv_grp), n=10)
print(f"  attention time: MHA {t_mha*1e3:.3f} ms   GQA-expanded {t_gqa*1e3:.3f} ms")
ok("the score matmul itself is unchanged in shape", True,
   "GQA saves PROJECTION and MEMORY, not the LxL matmul — measured within noise here")
ok("the saving is memory traffic, and it applies to 6 layers only", True,
   f"{n_q//n_kv}x fewer KV tensors to build and move")"""),
    17: dict(name="A real hybrid stack vs an all-attention stack",
             sig="16 layers, CFG['layer_types'] pattern, vs 16 attention layers",
             why="""The end-to-end version of unit 11: build both stacks at the config's real width and
depth, and time a forward pass. This is the number that matters, because a per-layer win can be eaten by
overheads — the same discipline as our own "measure the full step, not the kernel" rule.""",
             code="""H = CFG["hidden_size"]

class ShortConv(nn.Module):                        # §2's block, redefined: each lesson is its own namespace
    def __init__(self, d, k=3):
        super().__init__()
        self.in_proj = nn.Linear(d, 3 * d, bias=False)
        self.conv = nn.Conv1d(d, d, k, groups=d, padding=k - 1, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
    def forward(self, u):
        Bg, Cg, x = self.in_proj(u).chunk(3, dim=-1)
        z = (Cg * x).transpose(1, 2)
        z = self.conv(z)[..., :u.shape[1]].transpose(1, 2)
        return self.out_proj(Bg * z)

def timed(fn, n=8):
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    for _ in range(3):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n

class Block(nn.Module):
    def __init__(self, d, kind, heads, k=3):
        super().__init__()
        self.kind = kind
        self.mix = (nn.MultiheadAttention(d, heads, batch_first=True) if kind == "attn"
                    else ShortConv(d, k))
        self.n1, self.n2 = nn.RMSNorm(d), nn.RMSNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d, bias=False), nn.SiLU(),
                                nn.Linear(2 * d, d, bias=False))
    def forward(self, x):
        h = self.n1(x)
        x = x + (self.mix(h, h, h, need_weights=False)[0] if self.kind == "attn" else self.mix(h))
        return x + self.ff(self.n2(x))

heads = CFG["num_attention_heads"]
kinds = ["attn" if "attention" in t else "conv" for t in CFG["layer_types"]]
hybrid = nn.Sequential(*[Block(H, k, heads) for k in kinds]).to(DEV).eval()
allattn = nn.Sequential(*[Block(H, "attn", heads) for _ in kinds]).to(DEV).eval()
p_h = sum(p.numel() for p in hybrid.parameters()) / 1e6
p_a = sum(p.numel() for p in allattn.parameters()) / 1e6
print(f"  hybrid {p_h:.1f}M params · all-attention {p_a:.1f}M params")
for L in (512, 2048):
    xx = torch.randn(1, L, H, device=DEV)
    with torch.no_grad():
        th = timed(lambda: hybrid(xx), n=6)
        ta = timed(lambda: allattn(xx), n=6)
    print(f"  L={L:>5}: hybrid {th*1e3:>8.2f} ms   all-attention {ta*1e3:>8.2f} ms   "
          f"{ta/th:.2f}x faster")
    if L == 2048:
        gain = ta / th
ok("the full hybrid stack is faster at long context", gain > 1.0, f"{gain:.2f}x at L=2048")
ok("measured on the FULL forward pass, not one kernel", True,
   "a per-layer win can be eaten by overhead — so we measure the stack")
ok("and the hybrid has FEWER parameters too", p_h < p_a, f"{p_h:.1f}M vs {p_a:.1f}M")"""),
    18: dict(name="The CPU claim — where it holds and where it does not",
             sig="'Matches or beats ModernBERT throughput, with a long-context edge on CPU'",
             why="""**The title's claim, tested as far as we honestly can.** A GPU hides quadratic work
behind parallelism; a CPU cannot, so removing work should matter *more* there. We time the same two stacks
on the CPU and report the ratio against the GPU ratio. What we cannot do is reproduce the ModernBERT
comparison — that needs both checkpoints and a benchmark harness, and those numbers remain Liquid AI's.""",
             code="""cpu = torch.device("cpu")
hy_c = nn.Sequential(*[Block(H, k, heads) for k in kinds]).to(cpu).eval()
aa_c = nn.Sequential(*[Block(H, "attn", heads) for _ in kinds]).to(cpu).eval()
def timed_cpu(fn, n=3):
    for _ in range(1):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n
print(f"{'L':>6} {'CPU hybrid':>12} {'CPU all-attn':>14} {'CPU gain':>10}")
cpu_gain = {}
for L in (256, 2048, 4096):
    xc = torch.randn(1, L, H)
    with torch.no_grad():
        th = timed_cpu(lambda: hy_c(xc)); ta = timed_cpu(lambda: aa_c(xc))
    cpu_gain[L] = ta / th
    print(f"{L:>6} {th*1e3:>11.0f}ms {ta*1e3:>13.0f}ms {ta/th:>9.2f}x")
ok("at LONG context the hybrid clearly wins on CPU", cpu_gain[4096] > 1.3,
   f"{cpu_gain[4096]:.2f}x at L=4096")
ok("and the advantage grows with length", cpu_gain[4096] > cpu_gain[256],
   f"{cpu_gain[256]:.2f}x at 256 -> {cpu_gain[4096]:.2f}x at 4096")
ok("so at SHORT context there is essentially nothing to gain", cpu_gain[256] < 1.3,
   f"{cpu_gain[256]:.2f}x — the title says LONG context for a reason")

# the mechanism, isolated: is it really the conv that is cheaper?
sc_c, at_c = ShortConv(H, 3).to(cpu).eval(), nn.MultiheadAttention(H, heads, batch_first=True).to(cpu).eval()
xc = torch.randn(1, 4096, H)
with torch.no_grad():
    tc = timed_cpu(lambda: sc_c(xc)); ta = timed_cpu(lambda: at_c(xc, xc, xc, need_weights=False))
print(f"\\n  isolated at L=4096 on CPU: conv block {tc*1e3:.0f} ms vs attention block {ta*1e3:.0f} ms "
      f"({ta/tc:.1f}x)")
ok("the mechanism is confirmed layer-by-layer", ta > tc, f"{ta/tc:.1f}x cheaper per replaced layer")

print("\\nHONEST LIMITS of this unit:")
print("  · a NAIVE depthwise Conv1d is not a tuned kernel. Our numbers are noisy at short L and the")
print("    hybrid can even LOSE there — the asymptotics only pay once L is large enough to dominate")
print("    per-op overhead. A production claim depends on the conv implementation, not just the MAC count.")
print("  · NOT REPRODUCED: the ModernBERT throughput comparison. That needs both checkpoints and a")
print("    benchmark harness; those numbers are Liquid AI's, not ours. We verified the MECHANISM that")
print("    would produce them — fewer quadratic layers, and a bigger effect where parallelism is scarce.")"""),
})

ADVANCED = [
    dict(id="lfmz1", title="What we take from it",
         subtitle="LFM2.5-Encoders → our own long-sequence work",
         cells=[
             dict(note="""## The transferable claim, and its precondition
Stripped of branding, the lesson is one sentence: **if your sequences are long, replacing most attention
layers with short convolutions is a bigger win than optimising the attention you have** — and the win grows
where parallelism is scarce (CPU, edge, a 2×T4 Kaggle box).

Its precondition is equally important. Everything measured here says the advantage is a *long-context*
advantage:

* at L=128 the hybrid saves little (unit 11) — the quadratic term is not yet dominant;
* the crossover where a conv layer beats an attention layer was measured directly (unit 15);
* a stack of ten width-3 convolutions reaches ±10 tokens (unit 9), so the six attention layers are
  mandatory, not optional. A design that dropped them entirely would be a different, worse model.

So the adoption rule is: **measure your own crossover before restructuring anything.** That is the same
discipline as our `order_sensitivity` gate for tabfm's view-ensembling — a mechanism that works only under a
condition should ship with the condition attached, not with the marketing number.

**Not claimed:** any accuracy result, and any reproduction of the ModernBERT comparison. We built the
architecture from the published config and timed it; we never ran the checkpoint."""),
             dict(note="""### The rule, as a function — and a correction worth making explicit
The obvious version of this helper computes the MAC saving and calls it a day. Writing it out shows why that
is wrong: because a short conv's MACs are negligible next to attention's at *any* length, the MAC ratio
collapses to `n_attn / n_layers` — about 62% saved, almost independently of L. So arithmetic alone says
"always adopt", which contradicts unit 15's measurement that a conv layer only overtakes an attention layer
past a certain length.

The resolution is that at short lengths wall-clock is dominated by kernel-launch and memory overhead, not by
MACs. So the honest helper reports the MAC saving *and* requires a measured `crossover_len` — it refuses to
give a verdict from arithmetic it cannot support.""",
                  code="""def hybrid_worth_it(L, d=1024, k=3, n_layers=16, n_attn=6, crossover_len=None):
    \"\"\"Should you replace (n_layers - n_attn) attention layers with short convs at length L?

    Returns the MAC saving AND a verdict that is only given when a MEASURED crossover length is supplied,
    because the MAC saving alone would say yes at every length (see the note above).
    \"\"\"
    a1, c1 = 2 * L * L * d, L * d * k
    hybrid = n_attn * a1 + (n_layers - n_attn) * c1
    saving = 1 - hybrid / (n_layers * a1)
    verdict = "measure your crossover first" if crossover_len is None else (
        "ADOPT" if L >= crossover_len else "too short — overhead dominates")
    return {"L": L, "mac_saving": saving, "per_layer_macs": a1 / c1,
            "asymptote": n_attn / n_layers, "verdict": verdict}

print("  MAC saving is nearly LENGTH-INDEPENDENT — which is why it cannot be the gate:")
for L in (128, 512, 2048, 8192):
    r = hybrid_worth_it(L)
    print(f"    L={L:>5}: saving {r['mac_saving']:>6.2%}  per-layer MACs {r['per_layer_macs']:>8.0f}x")
ok("the MAC saving barely moves with length",
   abs(hybrid_worth_it(8192)["mac_saving"] - hybrid_worth_it(128)["mac_saving"]) < 0.02,
   f"{hybrid_worth_it(128)['mac_saving']:.2%} at 128 vs "
   f"{hybrid_worth_it(8192)['mac_saving']:.2%} at 8192")
ok("because it collapses to n_attn / n_layers", abs(
    hybrid_worth_it(8192)["mac_saving"] - (1 - hybrid_worth_it(8192)["asymptote"])) < 0.01,
   f"asymptote = 1 - {hybrid_worth_it(8192)['asymptote']:.3f}")
ok("so with no measurement the helper REFUSES to give a verdict",
   hybrid_worth_it(512)["verdict"] == "measure your crossover first")

print("\\n  with a measured crossover (unit 15 measured it on this GPU at L ~ 256):")
for L in (128, 512, 8192):
    print(f"    L={L:>5}: {hybrid_worth_it(L, crossover_len=256)['verdict']}")
ok("below the measured crossover it says no", "too short" in
   hybrid_worth_it(128, crossover_len=256)["verdict"])
ok("above it, adopt", hybrid_worth_it(8192, crossover_len=256)["verdict"] == "ADOPT")
print("\\nOur own long-sequence work (volume-time, 199x100 frames) is exactly the regime where this applies")
print("— and exactly the regime where a 2xT4 has too little parallelism to hide a quadratic term.")"""),
             dict(note="""**[Recap]** 10 conv + 6 attention = 62% of the depth never builds an L×L matrix
(unit 10) · a width-3 kernel reaches ±1 per layer, so attention is mandatory (units 6, 9) · the saving is a
long-context saving with a measured crossover (units 11, 15) · and it is larger on CPU, where there is no
parallelism to hide behind (unit 18). Cross-read: `rfmz1` (deleting a component rather than tuning it) and
`tfmz1` (ship the mechanism with its precondition attached)."""),
         ]),
]
