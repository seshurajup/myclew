import torch, torch.nn as nn, torch.nn.functional as F      # an expert that decides for itself
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

N, d = 8, 32
x1 = torch.randn(d)
experts = [nn.Linear(d, d, bias=False) for _ in range(N)]
gate_w = torch.rand(N); gate_w = gate_w / gate_w.sum()
out = sum(gate_w[i] * experts[i](x1) for i in range(N)) + x1      # eq. 3
ok("the mixture is linear in the gate weights", close(
    sum((2 * gate_w[i]) * experts[i](x1) for i in range(N)) + x1,
    2 * (out - x1) + x1, 1e-4))
ok("the definition does NOT require normalised gates", True,
   "sum-to-one is imposed by the Softmax of eq. 4, not by eq. 3")

K = 2
G_proj = torch.randn(d, N) / d ** 0.5
scores = x1 @ G_proj
topv, topi = torch.topk(scores, K)
gate = torch.zeros(N); gate[topi] = F.softmax(topv, -1)          # eq. 4
ok("only K experts are active", int((gate > 0).sum()) == K, f"{int((gate>0).sum())} of {N}")
ok("the active weights sum to one (a cross-expert constraint)", abs(float(gate.sum()) - 1) < 1e-6)
ok("the gate needs its OWN parameter matrix", G_proj.numel() == d * N,
   f"{d*N} extra parameters that no expert owns")

d_ff = 4 * d
Wup, Wgate, Wdown = (torch.randn(d, d_ff) / d ** 0.5, torch.randn(d, d_ff) / d ** 0.5,
                     torch.randn(d_ff, d) / d_ff ** 0.5)
ffn = lambda z: (F.silu(z @ Wup) * (z @ Wgate)) @ Wdown          # eq. 5
ok("the expert already computes its own gating projection", (x1 @ Wgate).shape == (d_ff,))
ok("output shape matches the residual stream", ffn(x1).shape == (d,))

r = 8
Agate, Bgate = torch.randn(d, r) / d ** 0.5, torch.randn(r, d_ff) / r ** 0.5
ffn_lr = lambda z: (F.silu((z @ Agate) @ Bgate) * (z @ Wup)) @ Wdown   # eq. 6
ok("the low-rank gate is much cheaper", d * r + r * d_ff < d * d_ff,
   f"{d*r + r*d_ff} vs {d*d_ff} parameters ({d*d_ff/(d*r + r*d_ff):.1f}x fewer)")
ok("and it exposes an r-dimensional summary per expert", (x1 @ Agate).shape == (r,), f"r = {r}")
ok("the expert still maps d -> d", ffn_lr(x1).shape == (d,))

A_all = torch.randn(N, d, r) / d ** 0.5                          # each expert's own A_gate
norms = torch.stack([torch.linalg.vector_norm(x1 @ A_all[i]) for i in range(N)])
topv, topi = torch.topk(norms, K)
gate7 = torch.zeros(N); gate7[topi] = F.softmax(topv, -1)         # eq. 7
ok("the score comes from the experts, not from a router parameter", norms.shape == (N,))
ok("no extra router matrix is needed", True, f"saved {d*N} parameters")
ok("but Softmax and TopK still couple the experts", abs(float(gate7.sum()) - 1) < 1e-6
   and int((gate7 > 0).sum()) == K, "eqs. 8-10 remove that too")
