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

L, d = 7, 5                                                    # 7 tokens, width 5
K, V = torch.randn(L, d), torch.randn(L, d)
q = torch.randn(d)

s = torch.exp(K @ q / d ** 0.5)                                # the kernel s(k_i, q)
M_closed = (s[:, None] * V).sum(0) / s.sum()                   # the Nadaraya-Watson estimator (eq. 62)
M_attn = F.softmax(K @ q / d ** 0.5, dim=0) @ V                # standard softmax attention
ok("Nadaraya-Watson == softmax attention", close(M_closed, M_attn))

Mfit = torch.zeros(d, requires_grad=True)                      # and it really is the argmin: fit it
opt = torch.optim.LBFGS([Mfit], max_iter=200)
def closure():
    opt.zero_grad()
    obj = (s * (V - Mfit).pow(2).sum(-1)).sum()                # sum_i s_i ||v_i - M||^2
    obj.backward(); return obj
opt.step(closure)
ok("fitted argmin == the closed form", close(Mfit.detach(), M_closed, 1e-4))

c, t = 3, 6                                                    # window of 3, current position 6
idx = slice(t - c + 1, t + 1)
sw = torch.exp(K[idx] @ q / d ** 0.5)
swa_closed = (sw[:, None] * V[idx]).sum(0) / sw.sum()
swa_attn = F.softmax(K[idx] @ q / d ** 0.5, dim=0) @ V[idx]
ok("windowed NW == sliding-window attention", close(swa_closed, swa_attn))
print("full-context vs window differ (they compress different contexts):",
      not close(swa_closed, M_closed))
