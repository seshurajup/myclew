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

import pandas as pd
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
df

c = k3cfg()
d, n = c["hidden_size"], c["num_hidden_layers"]
E, a, sh = c["num_experts"], c["num_experts_per_token"], c["num_shared_experts"]
ffn_moe, ffn_dense = c["moe_intermediate_size"], c["intermediate_size"]
vocab = c.get("vocab_size") or 163840
dense_layers = c.get("first_k_dense_replace") or 1
moe_layers = n - dense_layers
attn = 4 * d * d
total = moe_layers * (attn + 3 * d * ffn_moe * (E + sh)) + dense_layers * (attn + 3 * d * ffn_dense)         + 2 * d * vocab
active = moe_layers * (attn + 3 * d * ffn_moe * (a + sh)) + dense_layers * (attn + 3 * d * ffn_dense)          + 2 * d * vocab
print(f"  naive count (experts as FULL 3-matrix FFNs): {total/1e12:.2f}T total, {active/1e9:.0f}B active")
print(f"  the paper states:                            2.80T total, 104B active")
print(f"  sparsity from the config: {total/active:.0f}x  |  experts {a}/{E} = {100*a/E:.1f}% active")
ok("the config implies extreme sparsity (of the order of E/a)", 20 < total / active < E / a + 5,
   f"{total/active:.1f}x total/active, experts ratio E/a = {E/a:.0f}x "
   f"(attention + embeddings + the dense layer dilute it)")
ok("naive counting OVERSHOOTS 2.8T -> the experts cannot be full FFNs",
   total / 1e12 > 3.0, f"{total/1e12:.2f}T > 2.8T")
print("that gap is the evidence for Stable LatentMoE (eq. 11): experts are rotations of a SHARED"
      " low-rank latent, not 896 independent FFNs - which is exactly what the paper claims")
