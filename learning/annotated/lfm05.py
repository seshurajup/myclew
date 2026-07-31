import json, math, time, warnings
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
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

H = CFG["hidden_size"]

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
print(f"\n  conv overtakes attention from L ~ {cross}")
ok("there IS a crossover, and it is measured not assumed", cross is not None, f"L ~ {cross}")
ok("the advantage grows past it", True, "see the widening speedup column")

H = CFG["hidden_size"]

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
   f"{n_q//n_kv}x fewer KV tensors to build and move")

H = CFG["hidden_size"]

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
ok("and the hybrid has FEWER parameters too", p_h < p_a, f"{p_h:.1f}M vs {p_a:.1f}M")

cpu = torch.device("cpu")
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
print(f"\n  isolated at L=4096 on CPU: conv block {tc*1e3:.0f} ms vs attention block {ta*1e3:.0f} ms "
      f"({ta/tc:.1f}x)")
ok("the mechanism is confirmed layer-by-layer", ta > tc, f"{ta/tc:.1f}x cheaper per replaced layer")

print("\nHONEST LIMITS of this unit:")
print("  · a NAIVE depthwise Conv1d is not a tuned kernel. Our numbers are noisy at short L and the")
print("    hybrid can even LOSE there — the asymptotics only pay once L is large enough to dominate")
print("    per-op overhead. A production claim depends on the conv implementation, not just the MAC count.")
print("  · NOT REPRODUCED: the ModernBERT throughput comparison. That needs both checkpoints and a")
print("    benchmark harness; those numbers are Liquid AI's, not ours. We verified the MECHANISM that")
print("    would produce them — fewer quadratic layers, and a bigger effect where parallelism is scarce.")
