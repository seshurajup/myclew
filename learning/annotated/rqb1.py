import torch, torch.nn as nn, torch.nn.functional as F      # bit allocation is one Lagrange multiplier
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

def quantize(x, bits):
    qmax = 2 ** (bits - 1) - 1
    s = x.abs().max() / qmax
    return torch.round(x / s).clamp(-qmax - 1, qmax) * s

x = torch.randn(1 << 16)
bits = torch.arange(2, 9)
mse = torch.tensor([float(((quantize(x, int(b)) - x) ** 2).mean()) for b in bits])
logD = torch.log(mse)
A = torch.stack([torch.ones_like(bits.float()), -bits.float()], 1)
coef = torch.linalg.lstsq(A, logD.unsqueeze(1)).solution.squeeze(1)
alpha, beta = float(torch.exp(coef[0])), float(torch.exp(coef[1]))
pred = A @ coef
r2 = float(1 - ((logD - pred) ** 2).sum() / ((logD - logD.mean()) ** 2).sum())
for b, m in zip(bits.tolist(), mse.tolist()):
    print(f"  {b} bits -> MSE {m:.3e}")
ok("the distortion curve is exponential in the bit-width", r2 > 0.995, f"R^2 = {r2:.5f}")
ok("and the base is about 4 (one bit = a quarter of the squared error)", 3.0 < beta < 5.0,
   f"fitted alpha = {alpha:.3e}, beta = {beta:.3f}")

N = 32
w = torch.distributions.LogNormal(0.0, 1.4).sample((N,))         # importance, heavy-tailed as observed
am, gm = float(w.mean()), float(torch.exp(torch.log(w).mean()))
print(f"  {N} heads: importance min {float(w.min()):.3e}  max {float(w.max()):.3e}  "
      f"ratio {float(w.max()/w.min()):.0f}x")
print(f"  arithmetic mean {am:.4f} vs geometric mean {gm:.4f}  ->  AM/GM = {am/gm:.3f}")
ok("importance spans orders of magnitude", float(w.max() / w.min()) > 20)
ok("so AM/GM > 1, which is exactly the suboptimality of uniform bits (eq. 4)", am / gm > 1.0,
   f"uniform costs {am/gm:.2f}x the optimum")
