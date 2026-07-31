import torch, torch.nn as nn, torch.nn.functional as F      # three levels: server, client, memory
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

def unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)

d = 16                                                          # this lesson's own setup
C = 4
Ks, Vs, Bs = unit(C, d), torch.randn(C, d), torch.rand(C) * 0.5 + 0.3
def step(S, t):
    return S + Bs[t] * torch.outer(Vs[t] - S @ Ks[t], Ks[t])
S_seq = torch.zeros(d, d)
for t in range(C):
    S_seq = step(S_seq, t)
# compose the chunk's affine map: W = prod (I - b k k^T)^T acting on the right, U = accumulated writes
W = torch.eye(d); U = torch.zeros(d, d)
for t in range(C):
    P = torch.eye(d) - Bs[t] * torch.outer(Ks[t], Ks[t])
    W = W @ P                                                   # right-acting projection product
    U = U @ P + Bs[t] * torch.outer(Vs[t], Ks[t])
S_affine = torch.zeros(d, d) @ W + U
ok("the chunk composes into ONE affine map S -> S W + U", close(S_seq, S_affine, 1e-5),
   f"max|diff| = {(S_seq - S_affine).abs().max():.2e}")
S_rand = torch.randn(d, d) * 0.1
S_seq2 = S_rand.clone()
for t in range(C):
    S_seq2 = step(S_seq2, t)
ok("and it holds for any carried state (so chunks chain)", close(S_seq2, S_rand @ W + U, 1e-5))
ok("inference memory is constant in sequence length", tuple(S_affine.shape) == (d, d))
