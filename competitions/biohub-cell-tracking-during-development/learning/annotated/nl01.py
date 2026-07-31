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

d = 8
Q_, _ = torch.linalg.qr(torch.randn(d, d))                      # orthonormal keys -> exact recall
ctx = [(Q_[:, i], torch.randn(d)) for i in range(5)]            # 5 in-context (key, value) facts
probe_k, probe_v = ctx[2]                                       # ask about the 3rd fact afterwards

W_frozen = torch.randn(d, d)                                    # "MLP after pre-training": frequency 0
err_frozen = (W_frozen @ probe_k - probe_v).norm().item()

M = torch.zeros(d, d)                                           # a memory with frequency > 0
for k, v in ctx:                                                # it WRITES while it reads (delta rule)
    M = M @ (torch.eye(d) - torch.outer(k, k)) + torch.outer(v, k)
err_adaptive = (M @ probe_k - probe_v).norm().item()

ok("frozen weights cannot recall a context fact", err_frozen > 1.0, f"err={err_frozen:.3f}")
ok("an updating memory recalls it (to solver precision)", err_adaptive < 1e-2,
   f"err={err_adaptive:.2e} vs frozen {err_frozen:.3f}")
print(f"same shapes, same FLOPs class - the only difference is update frequency "
      f"(0 vs {len(ctx)} writes)")
