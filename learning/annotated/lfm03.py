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
   f"{rf(n_conv)} tokens of 8192")

types = CFG["layer_types"]
idx = [i for i, t in enumerate(types) if "attention" in t]
gaps = [b - a - 1 for a, b in zip([-1] + idx, idx)]
print("  attention at layers:", idx)
print("  conv layers between attentions:", gaps)
ok("attention is spread through the depth, not clustered", max(idx) - min(idx) > len(types) // 2,
   f"layers {min(idx)}..{max(idx)} of {len(types)}")
ok("never more than 2 conv layers pass without global mixing", max(gaps) <= 2, f"max gap {max(gaps)}")
ok("and the last layer is a conv (local refinement on top)", types[-1] == "conv")
print(f"\n  {len(idx)} of {len(types)} layers attend = {len(idx)/len(types):.0%} of the depth")

H = CFG["hidden_size"]
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
   f"{short:.2f} at L=128 vs {hybrid/allattn:.2f} at L=8192 — the win is a LONG-context win")
