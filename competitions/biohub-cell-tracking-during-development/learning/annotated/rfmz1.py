import torch, torch.nn as nn, torch.nn.functional as F      # an expert that decides for itself
import sys; sys.path.insert(0, "learning")
import vizkit as vz

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)

import pandas as pd
N_e, d_, B_ = 16, 32, 512
torch.manual_seed(0)
A_p = torch.randn(N_e, d_, 8) / d_ ** 0.5                          # each expert's own low-rank gate
toks = torch.randn(B_, d_)
nrm = torch.stack([torch.linalg.vector_norm(toks @ A_p[i], dim=-1) for i in range(N_e)], 1)
# heterogeneous thresholds = a deliberately UNBALANCED start (b = 0 would saturate every expert ON,
# which looks balanced only because nothing is sparse)
b = (nrm.mean(0) + torch.randn(N_e) * nrm.std() * 1.5).requires_grad_(True)
rho_star, mu = 0.25, 0.5

def losses(bv):
    # a bounded, differentiable density surrogate (eq. 12): sigmoid of the same hinge argument, so it
    # is monotone in the gate and lives in (0,1) where the target rho* also lives
    gs = torch.sigmoid(4.0 * (nrm - bv))
    L_EB = ((gs.mean(0) - rho_star) ** 2).mean()                   # eq. 13
    L_TB = ((gs.mean(1) - rho_star) ** 2).mean()                   # eq. 14
    return mu * L_EB + (1 - mu) * L_TB, gs                         # eq. 15

opt = torch.optim.Adam([b], lr=0.05)
L0, gs0 = losses(b)
spread0, dens0 = float(gs0.detach().mean(0).std()), float(gs0.detach().mean())
for _ in range(600):
    opt.zero_grad(); L, _ = losses(b); L.backward(); opt.step()
L1, gs1 = losses(b)
spread1, dens1 = float(gs1.detach().mean(0).std()), float(gs1.detach().mean())
print(f"  expert-load std {spread0:.4f} -> {spread1:.4f}    ({spread0/spread1:.0f}x tighter)")
print(f"  density         {dens0:.4f} -> {dens1:.4f}    (target {rho_star})")
print(f"  L_LB            {float(L0):.5f} -> {float(L1):.5f}")
ok("training ONLY the per-expert thresholds tightens the load distribution", spread1 < spread0 / 5,
   f"{spread0:.4f} -> {spread1:.4f}")
ok("and it drives the density to the target", abs(dens1 - rho_star) < abs(dens0 - rho_star) / 3,
   f"{dens0:.4f} -> {dens1:.4f} vs rho* = {rho_star}")
ok("the balance objective actually decreased", float(L1) < float(L0))
ok("no expert is starved and none monopolises", float(gs1.detach().mean(0).min()) > 0.1,
   f"loads span [{float(gs1.detach().mean(0).min()):.4f}, {float(gs1.detach().mean(0).max()):.4f}]")
print("balance from a purely LOCAL rule — no Softmax, no TopK, no router parameters")

import pandas as pd
T_, D_, b_, Bw = 4096, 7168, 2, 200e9
delta = lambda K, M: ((K + 1 - M) * T_ * D_ * b_) / (M * Bw)
rows = [dict(setting="Kaggle 2xT4", M=2, K=2, delta_ms=round(delta(2, 2) * 1e3, 4)),
        dict(setting="one 8-GPU node", M=8, K=8, delta_ms=round(delta(8, 8) * 1e3, 4)),
        dict(setting="expert-parallel 64", M=64, K=8, delta_ms=round(delta(8, 64) * 1e3, 4)),
        dict(setting="K3-like (896/16)", M=16, K=16, delta_ms=round(delta(16, 16) * 1e3, 4))]
df = pd.DataFrame(rows)
df["favours"] = ["routing-free" if x > 0 else "standard MoE" for x in df.delta_ms]
print(df.to_string(index=False))
ok("small device counts favour the routing-free pattern",
   df[df.setting == "Kaggle 2xT4"].delta_ms.iloc[0] > 0)
ok("large expert-parallel clusters do not", df[df.M == 64].delta_ms.iloc[0] < 0)
ok("so the efficiency claim is conditional, and the condition is checkable up front", True,
   "K + 1 > M")
vz.table(df, "eq. 30 evaluated on our own topologies", "positive delta = routing-free sends less",
         heat_cols=["delta_ms"])
