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

T, d = 6, 4                                                     # this lesson's own tensors
X = torch.randn(T, d); Wk, Wv, Wq = (torch.randn(d, d) / d ** 0.5 for _ in range(3))
K, V, Q = X @ Wk, X @ Wv, X @ Wq; Kn = F.normalize(K, dim=-1)
y_attn = torch.stack([F.softmax(K[:t + 1] @ Q[t] / d ** 0.5, dim=0) @ V[:t + 1] for t in range(T)])
W_lin = torch.zeros(d, d); outs = []
for t in range(T):
    outs.append(y_attn[t] @ W_lin)                                # read with the CURRENT weight
    W_lin = W_lin + torch.outer(V[t], Kn[t])                       # eq. 69: the weight is a memory
outs = torch.stack(outs)
W_mlp = torch.randn(d, d) / d ** 0.5
ok("an MLP tail gives the same shapes", (y_attn @ W_mlp).shape == outs.shape, f"{tuple(outs.shape)}")
ok("but only the AdaTransformer tail changes with the context",
   not close(outs[T - 1], y_attn[T - 1] @ (torch.zeros(d, d))), "W_LinAttn moved")
print("levels: {W_k,W_v,W_q} (slowest) -> Attn (non-parametric, freq inf) -> W_LinAttn (per token)")

net = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
x = torch.randn(8, 64)
opt.zero_grad(); net(x).pow(2).mean().backward(); opt.step()      # one step, so the moments exist

weights = sum(p.numel() for p in net.parameters())
state = sum(v.numel() for s in opt.state.values() for k, v in s.items() if torch.is_tensor(v) and v.dim() > 0)
print(f"weights (level 1): {weights}   |   optimizer memory (its own level): {state}")
ok("the NL parameter count is ~3x the 'model size'", state >= 2 * weights - 8,
   f"total {weights + state} vs advertised {weights}")
print("discarding the momentum at 'end of pre-training' deletes the model's knowledge of its own"
      " loss landscape (§4.5 note on continual learning)")
