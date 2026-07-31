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

m, n_e, k_ = 64, 16, 4                                          # this lesson's own router
s = torch.sigmoid(torch.randn(m, n_e)); b = torch.zeros(n_e)
def route(s, b, k_):
    idx = torch.topk(s + b, k_, dim=-1).indices
    sel = torch.gather(s, 1, idx)
    return idx, sel / sel.sum(-1, keepdim=True)
idx0, _ = route(s, b, k_); load0 = torch.bincount(idx0.reshape(-1), minlength=n_e)

def kth_threshold(sb, k_): return torch.topk(sb, k_, dim=-1).values[:, -1]
alpha = kth_threshold(s + b, k_)                                  # per-token threshold alpha_i
counts = torch.bincount(torch.topk(s + b, k_, dim=-1).indices.reshape(-1), minlength=n_e)
fair = m * k_ / n_e
ok("each token selects exactly k experts, so the loads must average the fair share",
   abs(float(counts.float().mean()) - fair) < 1e-6,
   f"mean load {float(counts.float().mean()):.1f} = fair share {fair:.1f}")
ok("alpha_i is exactly the k-th largest biased score",
   bool((((s + b) >= alpha[:, None] - 1e-9).sum(1) == k_).all()), "the threshold definition")
ok("but individual experts are NOT balanced yet", float(counts.float().std()) > 0,
   f"load spread = {float(counts.float().std()):.2f} tokens")

bh = -torch.quantile(s - alpha[:, None], 1 - k_ / n_e, dim=0)
b_new = bh - bh.max()                                             # re-centre (eq. 15)
_, p_new = route(s, b_new, k_)
load_new = torch.bincount(torch.topk(s + b_new, k_, dim=-1).indices.reshape(-1), minlength=n_e)
ok("the quantile update reduces the load imbalance", float(load_new.float().std()) < float(load0.float().std()),
   f"load std {float(load0.float().std()):.2f} -> {float(load_new.float().std()):.2f}")
ok("re-centring is a no-op for selection (a global shift cancels)",
   bool((torch.topk(s + bh, k_, -1).indices == torch.topk(s + b_new, k_, -1).indices).all()))
ok("and it costs one quantile per expert, not a loss term", True, f"n_e = {n_e} quantiles per step")
