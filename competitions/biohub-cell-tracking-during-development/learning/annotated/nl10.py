import torch, torch.nn as nn, torch.nn.functional as F      # the whole paper is linear algebra + autograd

import sys; sys.path.insert(0, "learning")
import vizkit as vz                                            # the shared visual + explainability layer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)                                  # so EVERY tensor/module below is on DEV
# These cells PROVE matrix identities, so they need full fp32: TF32 truncates the mantissa to 10 bits
# and an identity that holds to 1e-6 in fp32 only holds to ~1e-3 in TF32. Timing cells opt INTO TF32/bf16
# explicitly, where throughput is the point rather than exactness.
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):                                  # a lesson's PROOF prints PASS/FAIL, never prose
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):                                     # float-safe equality for matrix identities
    return torch.allclose(a, b, atol=tol, rtol=tol)

def newton_schulz(G, steps=5, eps=1e-7):                       # the orthogonalisation used by Muon/M3
    a, b, c = 3.4445, -4.7750, 2.0315                          # the standard quintic coefficients
    X = G / (G.norm() + eps)
    tall = X.shape[0] > X.shape[1]
    if tall: X = X.T
    for _ in range(steps):
        A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tall else X

import pandas as pd
d = 8
rows = []
for n in (2, 4, 8, 16, 32, 64):
    K = F.normalize(torch.randn(d, n), dim=0); V = torch.randn(d, n)
    M = V @ torch.linalg.pinv(K)                                # the OPTIMAL memory for these pairs
    rows.append(dict(pairs=n, capacity=d, residual=round(float((M @ K - V).pow(2).mean()), 4)))
df = pd.DataFrame(rows)
ok("residual is ~0 while pairs <= capacity", df[df.pairs <= d].residual.max() < 1e-6,
   f"{df[df.pairs <= d].residual.tolist()}")
ok("and grows once pairs exceed capacity (forgetting is forced)",
   df[df.pairs > d].residual.is_monotonic_increasing, f"{df[df.pairs > d].residual.tolist()}")
print("more levels buy more capacity at different time-scales; they do not repeal the pigeonhole principle")
df
