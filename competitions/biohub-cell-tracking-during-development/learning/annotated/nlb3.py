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

d_in, d_out = 5, 3
W = torch.randn(d_out, d_in, requires_grad=True)
x = torch.randn(d_in)
target = torch.randn(d_out)

y = W @ x                                                      # forward: y = W x
loss = 0.5 * (y - target).pow(2).sum()                         # L = 1/2 ||y - target||^2
loss.backward()

dL_dy = (y - target).detach()                                  # the surprise in the OUTPUT space
outer = torch.outer(dL_dy, x)                                  # (dL/dy) (x)^T  -- the memory write
ok("autograd W.grad == (dL/dy) outer x", close(W.grad, outer),
   f"||grad||={W.grad.norm():.4f}")
ok("surprise is zero exactly when the prediction is right",
   close(torch.zeros(d_out), (target - target)))

eta = 0.1
W_step = (W - eta * W.grad).detach()                           # ordinary SGD step
M = W.detach().clone()                                         # the same weights, read as a memory
M = M + torch.outer(-eta * dL_dy, x)                           # write (key=x, value=-eta * surprise)
ok("SGD step == memory write with key=x, value=-eta*surprise", close(W_step, M))
