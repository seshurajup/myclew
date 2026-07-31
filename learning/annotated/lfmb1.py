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

d, k = 1024, 3
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
print("\nThis is why the answer is 'have fewer quadratic layers', not 'make them faster'.")

types = CFG["layer_types"]
n_attn = sum(1 for t in types if "attention" in t)
n_conv = sum(1 for t in types if t == "conv")
print("layer_types:", types)
print(f"\n{len(types)} layers = {n_conv} conv + {n_attn} full attention")
ok("only a minority of layers attend", n_attn < n_conv, f"{n_attn} of {len(types)}")
ok("the convolution is SHORT", CFG["conv_L_cache"] == 3, f"conv_L_cache = {CFG['conv_L_cache']}")
ok("and attention uses grouped-query KV", CFG["num_key_value_heads"] < CFG["num_attention_heads"],
   f"{CFG['num_attention_heads']} query heads : {CFG['num_key_value_heads']} KV heads")
saved = 1 - (n_attn / len(types))
print(f"\nfraction of depth that never builds an LxL matrix: {saved:.0%}")
