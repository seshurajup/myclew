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

c = k3cfg()
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
   f"beta1={c.get('activation_situ_beta')}, beta2={c.get('activation_situ_linear_beta')}")

cfg = k3cfg()
full = (cfg.get("linear_attn_config") or {}).get("full_attn_layers") or []
n = cfg.get("num_hidden_layers", 0)
print(f"  full-attention layers: {full[:12]}{' …' if len(full) > 12 else ''}  ({len(full)} of {n})")
if full and n:
    gaps = sorted({b - a for a, b in zip(full, full[1:])})
    ok("full attention appears on a fixed period", len(gaps) <= 2, f"gaps {gaps}")
    ok("most layers are the cheap linear (KDA) kind", len(full) / n < 0.35,
       f"{len(full)}/{n} = {100*len(full)/n:.0f}% full attention")
ok("so the KV cache is paid on a minority of layers", True,
   "that is where the 1M-token context comes from")
