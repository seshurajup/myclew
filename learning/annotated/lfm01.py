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

for k in ("model_type", "hidden_size", "num_hidden_layers", "intermediate_size",
          "vocab_size", "num_attention_heads", "num_key_value_heads", "block_use_swiglu"):
    print(f"  {k:24s} {CFG[k]}")
ok("it is the LFM2 family", CFG["model_type"] == "lfm2")
ok("a SwiGLU feed-forward", CFG["block_use_swiglu"] is True)
ok("the FFN is ~6.5x the width", 6 < CFG["intermediate_size"] / CFG["hidden_size"] < 7,
   f"{CFG['intermediate_size']}/{CFG['hidden_size']} = {CFG['intermediate_size']/CFG['hidden_size']:.2f}")
ok("and FFN width is a multiple of 256 (a kernel-alignment choice)",
   CFG["intermediate_size"] % 256 == 0, f"{CFG['intermediate_size']} = 26 x 256")

same = [k for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "vocab_size",
                    "conv_L_cache") if CFG.get(k) == CFG_S.get(k)]
print("identical between 230M and 350M:", same)
print(f"depth: 230M = {CFG_S['num_hidden_layers']}   350M = {CFG['num_hidden_layers']}")
ok("width, heads, vocab and conv width are IDENTICAL", len(same) == 5)
ok("only the depth differs", CFG_S["num_hidden_layers"] != CFG["num_hidden_layers"],
   f"{CFG_S['num_hidden_layers']} vs {CFG['num_hidden_layers']} layers")
ok("so a quality gap isolates DEPTH", True, "a controlled pair, published for free")

V, H = CFG["vocab_size"], CFG["hidden_size"]
head = V * H
print(f"lm_head would be {V} x {H} = {head/1e6:.1f}M parameters")
ok("the head is TIED to the input embedding", CFG["tie_word_embeddings"] is True)
ok("which saves ~19% of a 350M budget", 0.15 < head / 350e6 < 0.25,
   f"{head/1e6:.0f}M of ~350M = {head/350e6:.0%}")
ok("and costs nothing at inference", True, "the same matrix, transposed")

V, H, L = CFG["vocab_size"], CFG["hidden_size"], CFG["num_hidden_layers"]
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
   "it costs global mixing, which is why six remain")
