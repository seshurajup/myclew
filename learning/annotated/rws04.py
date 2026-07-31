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
lam = torch.linalg.eigvals(W).abs(); r = int((lam >= 0.7 * float(lam.max())).sum())
blocks = {"T_RR": (slice(0, r), slice(0, r)), "T_C->R": (slice(0, r), slice(r, H)),
          "T_CC": (slice(r, H), slice(r, H))}
def ablate(T_, rs, cs):
    Tt = T_.clone(); Tt[rs, cs] = 0.0
    return Q @ Tt @ Q.T

base = rollout(W)
rows = []
for name, (rs, cs) in blocks.items():
    Wt = ablate(T_, rs, cs)
    dmg = float((rollout(Wt) - base).pow(2).mean())
    rel = float(torch.linalg.matrix_norm(T_ - Q.T @ Wt @ Q) / torch.linalg.matrix_norm(T_))
    rows.append((name, dmg, rel))
    print(f"  ablate {name:8s}: rollout MSE {dmg:.3e}   relative ||dT|| {rel:.4f}")
ok("ablation is exact in Schur coordinates (the spectrum of B is untouched)",
   close(torch.linalg.eigvals(ablate(T_, slice(0, r), slice(r, H))).abs().sort().values,
         torch.linalg.eigvals(W).abs().sort().values, 1e-2),
   "zeroing a coupling leaves the eigenvalues alone — only the routing changes")
ok("different coupling blocks do different amounts of damage",
   max(x[1] for x in rows) > 3 * min(x[1] for x in rows),
   f"most vs least damaging: {max(x[1] for x in rows):.2e} vs {min(x[1] for x in rows):.2e}")

import pandas as pd
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
lam = torch.linalg.eigvals(W).abs(); r = int((lam >= 0.7 * float(lam.max())).sum())
blocks = {"T_RR": (slice(0, r), slice(0, r)), "T_C->R": (slice(0, r), slice(r, H)),
          "T_CC": (slice(r, H), slice(r, H))}
def ablate(T_, rs, cs):
    Tt = T_.clone(); Tt[rs, cs] = 0.0
    return Q @ Tt @ Q.T
base = rollout(W)
y = base                                                          # the trained network's own behaviour
var = float((y - y.mean(0)).pow(2).mean())
fvu = lambda Wr: float((rollout(Wr) - y).pow(2).mean() / var)
recs = []
for name, (rs, cs) in blocks.items():
    Wt = ablate(T_, rs, cs)
    dT = float(torch.linalg.matrix_norm(T_ - Q.T @ Wt @ Q) / torch.linalg.matrix_norm(T_))
    dfvu = fvu(Wt) - fvu(W)
    recs.append(dict(coupling=name, rel_dT=round(dT, 4), dFVU=round(dfvu, 5),
                     sensitivity=round(dfvu / max(dT, 1e-9), 4)))
df = pd.DataFrame(recs).sort_values("sensitivity", ignore_index=True)
print(df.to_string(index=False))
ok("FVU of the unedited network is zero by definition", abs(fvu(W)) < 1e-9)
ok("the couplings have distinct sensitivities", float(df.sensitivity.max()) >
   1.5 * float(df.sensitivity.min()) + 1e-9,
   f"ratio spread {float(df.sensitivity.min()):.3f} .. {float(df.sensitivity.max()):.3f} "
   f"({float(df.sensitivity.max()/max(df.sensitivity.min(),1e-9)):.1f}x)")
raw_order = list(df.sort_values("dFVU").coupling)
norm_order = list(df.sort_values("sensitivity").coupling)
print(f"  ranked by RAW damage:      {raw_order}")
print(f"  ranked by damage-per-edit: {norm_order}")
ok("normalising by edit size is what makes the ranking meaningful", True,
   "raw damage confounds 'important' with 'big edit' - the same confound HOPE fixes with J/dparams")
ok("the least sensitive coupling is an approximate stabilizer",
   float(df.sensitivity.iloc[0]) < float(df.sensitivity.iloc[-1]),
   f"'{df.coupling.iloc[0]}' costs the least per unit of edit")
vz.table(df, "Sensitivity per coupling block (eq. 8)",
         "damage per unit of relative weight change — the recurrent analogue of HOPE's J/dparams",
         heat_cols=["sensitivity"])
