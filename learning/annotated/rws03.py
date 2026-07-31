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

H, T_len, d_in, d_out = 24, 40, 4, 2                              # this lesson's own setup
Wx = torch.randn(H, d_in) / 2; Wy = torch.randn(d_out, H) / (H ** 0.5)
W = torch.randn(H, H) / (H ** 0.5); W = 0.95 * W / torch.linalg.matrix_norm(W, 2)
X = torch.randn(T_len, d_in)
def states(Wr):
    h = torch.zeros(H); hs = []
    for t in range(T_len):
        h = torch.tanh(Wx @ X[t] + Wr @ h)
        hs.append(h)
    return torch.stack(hs)
rollout = lambda Wr: states(Wr) @ Wy.T
Q, T_ = schur(W)
ok("Q is orthogonal", close(Q.T @ Q, torch.eye(H), 1e-4),
   f"max |QᵀQ - I| = {float((Q.T @ Q - torch.eye(H)).abs().max()):.2e}")
ok("the factorisation is exact", close(Q @ T_ @ Q.T, W, 1e-3),
   f"max |QTQᵀ - W| = {float((Q @ T_ @ Q.T - W).abs().max()):.2e}")
ok("T is upper quasi-triangular (only the 2x2 subdiagonal survives)",
   float(torch.tril(T_, -2).abs().max()) < 1e-3,
   f"max below the first subdiagonal = {float(torch.tril(T_, -2).abs().max()):.2e}")
ok("and the spectrum is preserved (a similarity transform)",
   close(torch.linalg.eigvals(W).abs().sort().values,
         torch.linalg.eigvals(T_).abs().sort().values, 1e-3))

def split_bn(T_, tol=1e-3):
    n = T_.shape[0]
    B = torch.zeros_like(T_); i = 0
    while i < n:
        if i + 1 < n and abs(float(T_[i + 1, i])) > tol:          # a 2x2 complex block
            B[i:i + 2, i:i + 2] = T_[i:i + 2, i:i + 2]; i += 2
        else:
            B[i, i] = T_[i, i]; i += 1
    return B, T_ - B

B, N = split_bn(T_)
ok("B + N reconstructs T exactly", close(B + N, T_, 1e-6))
ok("N is strictly upper (a pure coupling term)", float(torch.tril(N).abs().max()) < 1e-3,
   f"max lower-triangular entry of N = {float(torch.tril(N).abs().max()):.2e}")
ok("B alone carries the spectrum", close(torch.linalg.eigvals(B).abs().sort().values,
                                        torch.linalg.eigvals(T_).abs().sort().values, 1e-3))
ok("and N is exactly the NONNORMALITY (zero for a normal matrix)",
   float(N.abs().sum()) > 0,
   f"||N||_F / ||T||_F = {float(torch.linalg.matrix_norm(N)/torch.linalg.matrix_norm(T_)):.3f}")

lam = torch.linalg.eigvals(W).abs()
rho = float(lam.max())
for a in (0.9, 0.7, 0.5, 0.3):
    R = int((lam >= a * rho).sum())
    print(f"  alpha = {a}:  |R| = {R:>2} retained modes, |C| = {H - R:>2} complement")
alpha = 0.7
R_size = int((lam >= alpha * rho).sum())
ok("the retained set grows as alpha falls", int((lam >= 0.3 * rho).sum()) >= R_size)
ok("R and C partition the modes", R_size + (H - R_size) == H, f"|R|={R_size}, |C|={H-R_size}")
ok("the spectral radius is below 1 (a stable RNN)", rho < 1.0, f"rho(W) = {rho:.4f}")

r = R_size
blocks = {"T_RR": (slice(0, r), slice(0, r)), "T_C->R": (slice(0, r), slice(r, H)),
          "T_CC": (slice(r, H), slice(r, H))}
for name, (rs, cs) in blocks.items():
    sub = N[rs, cs]
    print(f"  {name:8s} shape {tuple(sub.shape)}  ||.||_F = {float(torch.linalg.matrix_norm(sub)):.4f}")
ok("the three coupling blocks tile the upper part of N",
   abs(float(torch.linalg.matrix_norm(N) ** 2)
       - sum(float(torch.linalg.matrix_norm(N[rs, cs]) ** 2) for rs, cs in blocks.values())) < 1e-2,
   "no coupling is double-counted or missed")
ok("B is block-diagonal by construction", float((B - torch.block_diag(B[:r, :r], B[r:, r:])).abs().max())
   < 1e-5)
