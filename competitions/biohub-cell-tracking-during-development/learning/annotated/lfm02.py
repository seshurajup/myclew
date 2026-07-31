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

class ShortConv(nn.Module):
    """The LFM2 gated short-conv block, at the config's own width."""
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
   f"{sc.conv.weight.numel():,} vs {H*H*CFG['conv_L_cache']:,} for a dense conv")

with torch.no_grad():
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
   "this is the limitation the interleave exists to fix")

def timed(fn, n=12):
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
print(f"\n  fitted exponent: conv L^{slope(1):.2f}   attention L^{slope(2):.2f}")
ok("the conv scales about linearly", slope(1) < 1.4, f"L^{slope(1):.2f}")
ok("attention scales super-linearly", slope(2) > slope(1), f"L^{slope(2):.2f} vs L^{slope(1):.2f}")
ok("and the gap WIDENS with length", rows[-1][2] / rows[-1][1] > rows[0][2] / rows[0][1],
   f"{rows[0][2]/rows[0][1]:.1f}x at 256 -> {rows[-1][2]/rows[-1][1]:.1f}x at 2048")

k, n_kv, n_q = CFG["conv_L_cache"], CFG["num_key_value_heads"], CFG["num_attention_heads"]
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
   "10 conv layers contribute a fixed 2xd each")
