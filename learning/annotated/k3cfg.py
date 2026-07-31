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

import json, pathlib, pandas as pd                              # configs cached at build time
rows = [json.loads(p.read_text()) for p in sorted(pathlib.Path("docs/papers/kimi-k3/models").glob("*.json"))]
def row(c):                                                       # the comparable fields
    src = {**c.get("text_config", {}), **{k: v for k, v in c.items() if k != "text_config"}}
    e = src.get("num_experts") or src.get("n_routed_experts") or src.get("num_local_experts")
    return dict(repo=c.get("repo", "?"), type=src.get("model_type"), d=src.get("hidden_size"),
                layers=src.get("num_hidden_layers"), heads=src.get("num_attention_heads"),
                kv_heads=src.get("num_key_value_heads"), experts=e,
                active=src.get("num_experts_per_tok"), ctx=src.get("max_position_embeddings"))
df = pd.DataFrame([row(c) for c in rows])
if "kv_heads" in df and "heads" in df:                            # GQA ratio: KV-cache saving per token
    df["gqa_ratio"] = (df["heads"] / df["kv_heads"]).round(1)
if "experts" in df and "active" in df:
    df["sparsity"] = (df["experts"] / df["active"]).round(1)      # experts held / experts used
df

# the budget arithmetic a design is actually compared on (no weights involved)
def budget(c):
    s = {**c.get("text_config", {}), **{k: v for k, v in c.items() if k != "text_config"}}
    d, n = s.get("hidden_size", 0), s.get("num_hidden_layers", 0)
    e = s.get("num_experts") or s.get("n_routed_experts") or s.get("num_local_experts") or 1
    a = s.get("num_experts_per_tok") or s.get("num_experts_per_token") or 1
    sh = s.get("n_shared_experts") or s.get("num_shared_experts") or 0
    ffn = s.get("moe_intermediate_size") or s.get("intermediate_size") or 4 * d
    kvh = s.get("num_key_value_heads") or s.get("num_attention_heads") or 1
    hd = d // max(s.get("num_attention_heads", 1), 1)
    tot = n * (4 * d * d + 3 * d * ffn * e) + 2 * d * s.get("vocab_size", 0)
    act = n * (4 * d * d + 3 * d * ffn * (a + sh)) + 2 * d * s.get("vocab_size", 0)
    return dict(repo=c.get("repo", "?"), total_B=round(tot / 1e9, 1), active_B=round(act / 1e9, 2),
                sparsity=round(tot / max(act, 1), 1), kv_kB_per_token=round(2 * n * kvh * hd * 2 / 1024, 1),
                gflops_per_token=round(2 * act / 1e9, 1), ctx_M=round(s.get("max_position_embeddings", 0) / 1e6, 3))

bud = pd.DataFrame([budget(c) for c in rows]).sort_values("total_B", ascending=False)
print("total_B counts every expert (checkpoint size); active_B is what one token touches")
bud

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(9.5, 3.6), constrained_layout=True)
ax.set_axis_off()
for i, c in enumerate(rows):
    s = {**c.get("text_config", {}), **{k: v for k, v in c.items() if k != "text_config"}}
    e = s.get("num_experts") or s.get("n_routed_experts") or s.get("num_local_experts") or 0
    a = s.get("num_experts_per_tok") or s.get("num_experts_per_token") or 0
    x = i * 2.0
    ax.add_patch(plt.Rectangle((x, 0), 1.6, 2.6, fill=False, lw=1.4, ec="#8a8f98"))
    ax.text(x + 0.8, 2.75, c.get("repo", "?").split("/")[-1][:18], ha="center", fontsize=9, weight="bold")
    rows_txt = [f"d = {s.get('hidden_size')}", f"layers = {s.get('num_hidden_layers')}",
                f"heads {s.get('num_attention_heads')} / kv {s.get('num_key_value_heads')}",
                (f"MoE {a} of {e}" if e else "dense FFN"),
                f"ffn {s.get('moe_intermediate_size') or s.get('intermediate_size')}",
                f"ctx {(s.get('max_position_embeddings') or 0) // 1024}K",
                f"act {s.get('hidden_act', '?')}"]
    for j, t in enumerate(rows_txt):
        ax.text(x + 0.8, 2.3 - j * 0.33, t, ha="center", fontsize=8, color="#333")
    if e:                                                          # show the sparsity as a filled bar
        frac = a / e
        ax.add_patch(plt.Rectangle((x + 0.15, -0.35), 1.3, 0.18, color="#e7eaef"))
        ax.add_patch(plt.Rectangle((x + 0.15, -0.35), 1.3 * frac, 0.18, color="#0b6cff"))
        ax.text(x + 0.8, -0.62, f"{100*frac:.1f}% of experts active", ha="center", fontsize=7, color="#555")
ax.set_xlim(-0.4, max(2.0 * len(rows), 2)); ax.set_ylim(-0.9, 3.1)
p = pathlib.Path("docs/papers/kimi-k3/models/arch_diagram.png"); fig.savefig(p, dpi=150); plt.close(fig)
print("architecture diagram written to", p)
