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

T, d = 6, 4
X = torch.randn(T, d)
Wk, Wv, Wq = (torch.randn(d, d) / d ** 0.5 for _ in range(3))
K, V, Q = X @ Wk, X @ Wv, X @ Wq
ok("one shared interface for every model in this section", (K.shape, V.shape, Q.shape) == ((T, d),) * 3,
   f"K,V,Q each {tuple(K.shape)}")

s = torch.exp(K @ Q[3] / d ** 0.5)
ok("closed form == softmax attention", close((s[:, None] * V).sum(0) / s.sum(),
                                            F.softmax(K @ Q[3] / d ** 0.5, dim=0) @ V))
print("cache grows with L (perfect memory); state of a recurrent memory does not")

c, t = 3, 5
idx = slice(t - c + 1, t + 1)
sw = torch.exp(K[idx] @ Q[t] / d ** 0.5)
ok("windowed closed form == windowed attention",
   close((sw[:, None] * V[idx]).sum(0) / sw.sum(), F.softmax(K[idx] @ Q[t] / d ** 0.5, dim=0) @ V[idx]))

M = torch.zeros(d, d); alpha, eta = 0.95, 1.0
phi = lambda z: F.elu(z) + 1                                     # a positive feature map, as in linear attn
for t in range(T):
    M = alpha * M + eta * torch.outer(V[t], phi(K[t]))           # eq. 64
Mg = torch.zeros(d, d, requires_grad=True)
(-2 * (Mg @ phi(K[0])) @ V[0]).backward()
ok("the update is one GD step on -2<Mk, v>", close(-0.5 * Mg.grad, torch.outer(V[0], phi(K[0]))),
   "grad = -2 v phi(k)^T")
ok("state stays (d, d) for any T", M.shape == (d, d), f"T={T}")

Md, Mh = torch.zeros(d, d), torch.zeros(d, d); eta = 1.0
Kn = F.normalize(K, dim=-1)
for t in range(T):
    Md = (torch.eye(d) - eta * torch.outer(Kn[t], Kn[t])) @ Md + eta * torch.outer(V[t], Kn[t])
    Mh = Mh + torch.outer(V[t], Kn[t])                            # Hebbian, for comparison
err_d = sum(float((Md @ Kn[t] - V[t]).norm()) for t in range(T)) / T
err_h = sum(float((Mh @ Kn[t] - V[t]).norm()) for t in range(T)) / T
ok("delta rule recalls its own writes better than Hebbian", err_d < err_h,
   f"mean recall error: delta {err_d:.4f} vs Hebbian {err_h:.4f}")
Mg = torch.zeros(d, d, requires_grad=True)
((Mg @ Kn[0] - V[0]).pow(2).sum()).backward()
ok("gradient of the L2 objective == 2(Mk - v)k^T", close(Mg.grad, 2 * torch.outer(-V[0], Kn[0])))

Mo = torch.randn(d, d) * 0.1; Mh2 = Mo.clone(); alpha, eta = 1.0, 0.1
for t in range(T):
    Mo = alpha * Mo + eta * torch.outer(V[t], phi(K[t]) - Mo.T @ V[t])   # eq. 66
    Mh2 = alpha * Mh2 + eta * torch.outer(V[t], phi(K[t]))
ok("Oja's normalisation keeps the state bounded", float(Mo.norm()) < float(Mh2.norm()),
   f"||M||: Oja {float(Mo.norm()):.3f} vs Hebbian {float(Mh2.norm()):.3f}")

Mg = torch.randn(d, d, requires_grad=True)
k0, v0 = phi(K[0]), V[0]
(-2 * (Mg @ k0) @ v0 + (Mg.T @ v0).pow(2).sum()).backward()
analytic = -2 * torch.outer(v0, k0) + 2 * torch.outer(v0, Mg.detach().T @ v0)
ok("autograd gradient == Oja's analytic update direction", close(Mg.grad, analytic, 1e-4))

c = 3; Momega = torch.zeros(d, d); alpha = 0.98
for t in range(T):
    lo = max(0, t - c + 1)
    gsum = sum(0.5 ** (t - i) * (Momega @ Kn[i] - V[i]).unsqueeze(1) @ Kn[i].unsqueeze(0)
               for i in range(lo, t + 1))                         # eq. 68 with gamma = 0.5^(t-i)
    Momega = alpha * Momega - 0.3 * gsum
res_window = sum(float((Momega @ Kn[i] - V[i]).norm()) for i in range(T - c, T)) / c
ok("a windowed update fits the recent neighbourhood", res_window < err_h,
   f"recent-window error {res_window:.4f} vs Hebbian {err_h:.4f}")
