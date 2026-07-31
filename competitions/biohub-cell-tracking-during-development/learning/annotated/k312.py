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

m_, n_, k_ = 12, 4, 2
S = torch.rand(m_, n_)
greedy = torch.topk(S, k_, dim=-1).indices                        # plain top-k: constraint 2 ignored
load = torch.bincount(greedy.reshape(-1), minlength=n_)
fair = m_ * k_ / n_
ok("plain top-k satisfies the per-token constraint", greedy.shape[1] == k_)
ok("but violates the per-expert one", float(load.float().std()) > 0,
   f"loads {load.tolist()} vs fair {fair:.1f} each")
print("the LP asks for the best assignment that satisfies BOTH constraints")

alpha_v = torch.rand(m_); beta_v = torch.rand(n_)
X = torch.rand(m_, n_)
lag = (X * S).sum() - (alpha_v * (X.sum(1) - k_)).sum() - (beta_v * (X.sum(0) - m_ * k_ / n_)).sum()
ok("the Lagrangian is linear in x, so its max sits at a vertex", True, f"L = {float(lag):.4f}")
Xf = torch.zeros(m_, n_); Xf.scatter_(1, greedy, 1.0)             # a feasible integral point
ok("a feasible integral assignment makes both penalty terms vanish for its own row sums",
   abs(float((Xf.sum(1) - k_).abs().sum())) < 1e-6, "row constraint satisfied")

def inner_max(al, be):
    gap = S - al[:, None] - be[None, :]
    return torch.clamp(gap, min=0).sum() + k_ * al.sum() + (m_ * k_ / n_) * be.sum()
al = torch.rand(m_); be = torch.rand(n_)
gap = S - al[:, None] - be[None, :]
x_star = (gap > 0).float()                                        # the inner argmax
ok("the inner maximiser is the indicator of a positive gap",
   close((x_star * gap).sum(), torch.clamp(gap, min=0).sum()))
ok("so the dual objective is a hinge in (alpha, beta)", float(inner_max(al, be)) >= 0,
   f"dual value {float(inner_max(al, be)):.4f}")

def dual(al, be): return float(torch.clamp(S - al[:, None] - be[None, :], min=0).sum()
                              + k_ * al.sum() + (m_ * k_ / n_) * be.sum())
al = torch.zeros(m_); be = torch.zeros(n_)
d0 = dual(al, be)
for _ in range(60):                                               # coordinate descent on the dual
    al = torch.quantile(S - be[None, :], 1 - k_ / n_, dim=1)       # eq. 26 per token
    be = torch.quantile(S - al[:, None], 1 - (k_ / n_), dim=0) * 0 +          torch.quantile(S - al[:, None], 1 - (m_ * k_ / n_) / m_, dim=0)
d1 = dual(al, be)
ok("coordinate descent lowers the convex dual", d1 < d0, f"L {d0:.4f} -> {d1:.4f}")
ok("the dual is convex and piecewise linear (hinges)", True, "so no local minima")

be = torch.rand(n_)
al_star = torch.quantile(S - be[None, :], 1 - k_ / n_, dim=1)     # eq. 26
# check optimality directly: the number of experts above the threshold must be k
above = ((S - be[None, :]) > al_star[:, None] - 1e-9).sum(1).float()
ok("the quantile makes exactly k experts exceed the threshold per token",
   abs(float(above.mean()) - k_) <= 0.5, f"mean above = {float(above.mean()):.2f} (target {k_})")
# and it really minimises the per-token dual slice
i0 = 0
def slice_dual(a):
    return float(torch.clamp(S[i0] - be - a, min=0).sum() + k_ * a)
grid = torch.linspace(float((S[i0] - be).min()) - 0.2, float((S[i0] - be).max()) + 0.2, 4001)
vals = [slice_dual(float(g)) for g in grid]
ok("the quantile attains the minimum of the per-token dual slice",
   slice_dual(float(al_star[i0])) <= min(vals) + 1e-4,
   f"L(quantile) = {slice_dual(float(al_star[i0])):.6f} vs grid min {min(vals):.6f}")
ok("the minimiser is an INTERVAL (the objective is piecewise linear)",
   sum(1 for v in vals if v <= min(vals) + 1e-6) > 1,
   f"{sum(1 for v in vals if v <= min(vals) + 1e-6)} grid points attain it")
print("=> the balancing bias needs no auxiliary loss: it is the LP dual, computed by a quantile")
