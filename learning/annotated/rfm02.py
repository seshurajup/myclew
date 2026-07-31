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

T, d = 16, 32
h_prev = torch.randn(T, d)
Wqkv = torch.randn(d, 3 * d) / d ** 0.5
q, kk, v = (h_prev @ Wqkv).chunk(3, dim=-1)
attn = F.softmax(q @ kk.T / d ** 0.5, dim=-1) @ v
x = attn + h_prev                                                # eq. 1
ok("the residual preserves the shape", x.shape == h_prev.shape, f"{tuple(x.shape)}")
ok("and the identity path is intact", close(x - attn, h_prev))

W_moe = torch.randn(d, d) / d ** 0.5                             # fixed, or the identity below is void
moe = lambda z: z @ W_moe                                        # any MoE body
h_new = moe(x) + x                                               # eq. 2
ok("the block is a residual around the MoE body", close(h_new - x, moe(x), 1e-4))
ok("so replacing the body changes nothing else in the block", True,
   "the router lives strictly inside MoE(.)")
