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

K_ = 4
N = torch.tensor([120.0, 80.0, 300.0, 500.0])
wts = N / N.sum()
losses = torch.tensor([0.9, 1.4, 0.7, 0.5])
obj = float((wts * losses).sum())
ok("the objective is a convex combination of client losses",
   abs(float(wts.sum()) - 1.0) < 1e-6 and min(losses) <= obj <= max(losses), f"J = {obj:.4f}")
ok("a big client dominates it", int(wts.argmax()) == int(N.argmax()),
   f"weights {[round(float(w), 3) for w in wts]}")
print("in NL terms: the SLOWEST level, one update per round, context = every client's data")
