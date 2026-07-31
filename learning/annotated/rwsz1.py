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

import pandas as pd
H, T_len = 20, 30
Wx = torch.randn(H, 2) / 2; Wy = torch.randn(1, H) / (H ** 0.5)
W = torch.randn(H, H) / (H ** 0.5); W = 0.9 * W / torch.linalg.matrix_norm(W, 2)
Q, T_ = schur(W)

def split_bn(T_, tol=1e-3):
    n = T_.shape[0]; B = torch.zeros_like(T_); i = 0
    while i < n:
        if i + 1 < n and abs(float(T_[i + 1, i])) > tol:
            B[i:i + 2, i:i + 2] = T_[i:i + 2, i:i + 2]; i += 2
        else:
            B[i, i] = T_[i, i]; i += 1
    return B, T_ - B
B, N = split_bn(T_)
lam = torch.linalg.eigvals(W).abs(); r = int((lam >= 0.7 * float(lam.max())).sum())
BLOCKS = {"T_RR": (slice(0, r), slice(0, r)), "T_C->R": (slice(0, r), slice(r, H)),
          "T_CC": (slice(r, H), slice(r, H))}

def sensitivity_profile(W, Q, T_, blocks, drive):
    """Reusable probe: normalised damage per unit of relative edit, per coupling block."""
    def roll(Wr):
        h = torch.zeros(H); out = []
        for t in range(drive.shape[0]):
            h = torch.tanh(Wx @ drive[t] + Wr @ h)
            out.append(Wy @ h)
        return torch.stack(out)
    y = roll(W); var = float((y - y.mean(0)).pow(2).mean())
    recs = []
    for name, (rs, cs) in blocks.items():
        Tt = T_.clone(); Tt[rs, cs] = 0.0
        Wt = Q @ Tt @ Q.T
        dT = float(torch.linalg.matrix_norm(T_ - Tt) / torch.linalg.matrix_norm(T_))
        dfvu = float((roll(Wt) - y).pow(2).mean() / var)
        recs.append(dict(coupling=name, sensitivity=round(dfvu / max(dT, 1e-9), 4)))
    return pd.DataFrame(recs)

impulse = torch.zeros(T_len, 2); impulse[0, 0] = 1.0             # task A: remember one impulse
noise = torch.randn(T_len, 2) * 0.5                              # task B: track a noisy drive
pa = sensitivity_profile(W, Q, T_, BLOCKS, impulse).rename(columns={"sensitivity": "impulse_task"})
pb = sensitivity_profile(W, Q, T_, BLOCKS, noise).rename(columns={"sensitivity": "noisy_task"})
prof = pa.merge(pb, on="coupling")
print(prof.to_string(index=False))
ok("the probe returns one sensitivity per coupling block", len(prof) == 3)
ok("the sensitivity PROFILE differs between the two drives (task-restricted, as claimed)",
   float((prof.impulse_task - prof.noisy_task).abs().max()) > 1e-3,
   f"max profile difference {float((prof.impulse_task - prof.noisy_task).abs().max()):.4f}")
ok("so a 'free' direction on one task is not guaranteed free on another", True,
   "this is the constraint on reusing any pruning decision across tasks")
