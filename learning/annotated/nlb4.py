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

d_k = d_v = 4
M0 = torch.randn(d_v, d_k)
k  = F.normalize(torch.randn(d_k), dim=0)                      # a unit key (the paper L2-normalises k, q)
v  = torch.randn(d_v)
eta = 0.3

M = M0.clone().requires_grad_(True)                            # --- objective 1: dot-product similarity
(-(M @ k) @ v).backward()                                      # L = -<M k, v>
hebb_auto = (M - eta * M.grad).detach()
hebb_form = M0 + eta * torch.outer(v, k)                       # closed form: + eta v k^T
ok("dot-product objective -> Hebbian write", close(hebb_auto, hebb_form))

M = M0.clone().requires_grad_(True)                            # --- objective 2: L2 regression
(0.5 * (M @ k - v).pow(2).sum()).backward()                    # L = 1/2 ||M k - v||^2
delta_auto = (M - eta * M.grad).detach()
delta_form = (torch.eye(d_v) - eta * torch.outer(k, k) if d_v == d_k else None)
delta_form = M0 @ (torch.eye(d_k) - eta * torch.outer(k, k)) + eta * torch.outer(v, k)
ok("L2 objective -> delta rule", close(delta_auto, delta_form))
print("the delta rule ERASES the old value at this key before writing:",
      f"forget factor along k = {1 - eta:.2f}")

kk = F.normalize(torch.randn(d_k), dim=0)
va, vb = torch.randn(d_v), torch.randn(d_v)
H = torch.zeros(d_v, d_k); D = torch.zeros(d_v, d_k)
for val in (va, vb):                                           # same key, two conflicting values
    H = H + torch.outer(val, kk)                                            # Hebbian
    D = D @ (torch.eye(d_k) - torch.outer(kk, kk)) + torch.outer(val, kk)   # delta (eta = 1)
ok("Hebbian read is the SUM (corrupted)", close(H @ kk, va + vb))
ok("delta read is the LATEST value (clean)", close(D @ kk, vb))

d = 12
k = F.normalize(torch.randn(d), dim=0); v = torch.randn(d)
M0 = torch.randn(d, d) * 0.25
eta = 1.0                                                       # eta = 1 -> a full overwrite at this key
M1 = M0 @ (torch.eye(d) - eta * torch.outer(k, k)) + eta * torch.outer(v, k)   # the delta-rule write

vz.heat(M0, "learning/assets/nested-learning/xai_M_before.png", "M before the write")
vz.heat(M1, "learning/assets/nested-learning/xai_M_after.png", "M after writing (k, v)")
vz.heat(M1 - M0, "learning/assets/nested-learning/xai_M_delta.png", "the change: a rank-1 stripe along k")

change = M1 - M0
proj = torch.outer(k, k)                                        # the subspace the rule is allowed to touch
in_k = float((change @ proj).norm()); out_k = float((change - change @ proj).norm())
ok("the write only touches the k-direction (that is what 'rank-1 erase' means)",
   in_k > 20 * out_k, f"||change along k|| {in_k:.3f} vs orthogonal {out_k:.2e}")
ok("and the memory now returns v at that key", close(M1 @ k, v, 1e-4),
   f"read error {(M1 @ k - v).norm():.2e}")
vz.tensor_view(change, "the write, as a tensor you can fold open and hover")
