import torch, torch.nn as nn, torch.nn.functional as F      # bit allocation is one Lagrange multiplier
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
def quantize(x, bits):
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().max() / qmax
    return torch.round(x / s).clamp(-qmax - 1, qmax) * s

N = 24
heads = [torch.randn(2048) for _ in range(N)]
w = torch.distributions.LogNormal(0.0, 1.3).sample((N,))         # importance
LEGAL = torch.tensor([2.0, 3.0, 4.0, 8.0])                       # what kernels actually exist
budget = 4.0 * N

lw = torch.log(w)
raw = budget / N + (lw - lw.mean()) / torch.log(torch.tensor(4.0))
snap = LEGAL[(raw[:, None] - LEGAL[None, :]).abs().argmin(1)]     # snap to legal widths
while float(snap.sum()) > budget:                                # give bits back, cheapest first
    cand = (snap > LEGAL.min())
    idx = int((w * cand.float() + (~cand).float() * 1e9).argmin())
    snap[idx] = LEGAL[max(int((LEGAL == snap[idx]).nonzero()[0, 0]) - 1, 0)]

def real_distortion(bits):
    return float(sum(w[i] * ((quantize(heads[i], int(bits[i])) - heads[i]) ** 2).mean()
                     for i in range(N)))
d_alloc = real_distortion(snap)
d_unif = real_distortion(torch.full((N,), 4.0))
am, gm = float(w.mean()), float(torch.exp(torch.log(w).mean()))
print(pd.DataFrame({"head": range(N), "importance": w.round(decimals=3).tolist(),
                    "bits": snap.int().tolist()}).head(8).to_string(index=False))
print(f"  budget {float(snap.sum()):.0f}/{budget:.0f} bits · AM/GM predicted gain {am/gm:.2f}x")
ok("the allocator stays inside the budget", float(snap.sum()) <= budget + 1e-6)
ok("only hardware-legal bit-widths are used", bool(torch.isin(snap, LEGAL).all()),
   f"used {sorted(set(snap.int().tolist()))}")
ok("measured distortion beats uniform at the same budget", d_alloc < d_unif,
   f"weighted MSE {d_unif:.4e} -> {d_alloc:.4e} ({d_unif/d_alloc:.2f}x)")
ok("important heads got more bits", float(torch.corrcoef(torch.stack([torch.log(w), snap]))[0, 1]) > 0.5,
   f"corr(log importance, bits) = {float(torch.corrcoef(torch.stack([torch.log(w), snap]))[0,1]):+.3f}")
