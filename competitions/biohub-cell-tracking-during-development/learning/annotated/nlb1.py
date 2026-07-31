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

d_k = d_v = 4                                                  # tiny so every number is visible
K = torch.eye(d_k)                                             # orthonormal keys k1..k4 (the identity's columns)
v1, v2 = torch.tensor([1., 2., 3., 4.]), torch.tensor([-1., 0., 5., 2.])   # two values to store

M = torch.zeros(d_v, d_k)                                      # the memory starts empty
M = M + torch.outer(v1, K[:, 0])                               # write (k1, v1):  M <- M + v1 k1^T
M = M + torch.outer(v2, K[:, 1])                               # write (k2, v2)

y1, y2 = M @ K[:, 0], M @ K[:, 1]                              # read with q = k1 and q = k2
ok("read(k1) == v1", close(y1, v1), f"{y1.tolist()}")
ok("read(k2) == v2", close(y2, v2), f"{y2.tolist()}")
ok("unwritten key reads 0", close(M @ K[:, 2], torch.zeros(d_v)))
M

Kr = F.normalize(torch.randn(d_k, 8), dim=0)                   # 8 random unit keys in 4-D -> overlapping
Vr = torch.randn(d_v, 8)                                       # their values
M_heb = Vr @ Kr.T                                              # Hebbian: write everything, sum of outer products
err_heb = (M_heb @ Kr - Vr).pow(2).mean().item()               # how wrong is the read-back?
M_ls = Vr @ torch.linalg.pinv(Kr)                              # the LEAST-SQUARES memory (the delta rule's fixed point)
err_ls = (M_ls @ Kr - Vr).pow(2).mean().item()
ok("Hebbian read-back has crosstalk error", err_heb > 1e-3, f"MSE={err_heb:.4f}")
ok("least-squares memory is strictly better", err_ls < err_heb, f"MSE={err_ls:.4f}")
print(f"crosstalk cost of Hebbian vs L2-optimal: {err_heb / max(err_ls, 1e-12):.1f}x")
