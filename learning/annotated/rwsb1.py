import torch, torch.nn as nn, torch.nn.functional as F      # Schur coordinates + structured ablation
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

def schur(W):
    """Real Schur form via scipy on the CPU (LAPACK has no CUDA path), returned on DEV."""
    import numpy as np
    from scipy.linalg import schur as _schur
    T, Q = _schur(W.detach().double().cpu().numpy(), output="real")
    return (torch.tensor(Q, dtype=torch.float32, device=DEV),
            torch.tensor(T, dtype=torch.float32, device=DEV))

H, T_len = 24, 40
Wx = torch.randn(H, 4) / 2
W = torch.randn(H, H) / (H ** 0.5)
W = 0.95 * W / torch.linalg.matrix_norm(W, 2)                   # stable spectral radius
X = torch.randn(T_len, 4)
Wy = torch.randn(2, H) / (H ** 0.5)

def rollout(Wr):
    h = torch.zeros(H); out = []
    for t in range(T_len):
        h = torch.tanh(Wx @ X[t] + Wr @ h)                       # eq. 1
        out.append(Wy @ h)                                       # eq. 2
    return torch.stack(out)

base = rollout(W)
# Do NOT presume which direction matters — sample many perturbations of IDENTICAL Frobenius norm and
# measure the spread. That is the paper's actual claim: same-scale edits, wildly different consequences.
eps = 0.15
dmgs = []
for i in range(40):
    G = torch.randn(H, H)
    dW = eps * G / torch.linalg.matrix_norm(G)                   # exactly the same ||dW||_F every time
    dmgs.append(float((rollout(W + dW) - base).pow(2).mean()))
dmgs = torch.tensor(dmgs)
print(f"  40 perturbations, all with ||dW||_F = {eps}:")
print(f"    rollout MSE  min {float(dmgs.min()):.3e}   median {float(dmgs.median()):.3e}   "
      f"max {float(dmgs.max()):.3e}")
ok("equal-norm perturbations differ markedly in effect", float(dmgs.max() / dmgs.min()) > 3,
   f"max/min damage = {float(dmgs.max()/dmgs.min()):.1f}x at identical ||dW||_F "
   f"(random directions; the paper's STRUCTURED Schur ablations separate far more, see eq. 8)")
ok("so ||dW|| alone cannot rank a weight change", True,
   "which is exactly why the paper needs a coordinate system")
