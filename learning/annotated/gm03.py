import torch, torch.nn as nn, torch.nn.functional as F      # a memory is the argmin of a write loss
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

d, c_len, q_len = 32, 24, 4
C = torch.randn(c_len, d); Q = torch.randn(q_len, d)
prompt = torch.cat([C, Q], 0)                                   # eq. 1
ok("conditioning = concatenation along the sequence axis", prompt.shape == (c_len + q_len, d),
   f"{tuple(C.shape)} + {tuple(Q.shape)} -> {tuple(prompt.shape)}")
ok("attention cost is quadratic in the concatenated length",
   (c_len + q_len) ** 2 > c_len ** 2 + q_len ** 2, f"{(c_len+q_len)**2} vs {c_len**2 + q_len**2}")
