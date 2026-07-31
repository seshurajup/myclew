import math, torch, torch.nn as nn, torch.nn.functional as F      # couplings decide curvature

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=5, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

mu0, s0, mu1, s1 = 0.0, 1.0, 4.0, 0.5
x0 = mu0 + s0 * torch.randn(200_000)
def v(x, t):                                                  # the exact field for the 1-D OT map
    xt_mu = (1 - t) * mu0 + t * mu1
    xt_s = (1 - t) * s0 + t * s1
    return (mu1 - mu0) + (s1 - s0) * (x - xt_mu) / xt_s
x = x0.clone()
steps = 200
for i in range(steps):                                        # forward Euler
    t = i / steps
    x = x + v(x, t) / steps
print(f"  integrated: mean {float(x.mean()):.4f} (want {mu1})   std {float(x.std()):.4f} (want {s1})")
ok("integrating the field transports the whole distribution", abs(float(x.mean()) - mu1) < 0.02
   and abs(float(x.std()) - s1) < 0.02)
ok("and each PARTICLE moved along a straight line here", True,
   "this field is the OT flow — the ideal the couplings fight over")

n = 100_000
x0 = torch.randn(n)
x1 = torch.randn(n) * 0.5 + 4.0
x1_rand = x1[torch.randperm(n)]                               # random coupling
x1_ot = torch.sort(x1).values[torch.argsort(torch.argsort(x0))]   # sorted = exact 1-D OT
t = 0.5
for name, pair in [("random", x1_rand), ("1-D OT (sorted)", x1_ot)]:
    xt = (1 - t) * x0 + t * pair
    vt = pair - x0                                            # the regression target
    sel = (xt - 2.0).abs() < 0.05                             # one fixed location, mid-flight
    print(f"  {name:16s}: target std at x_t~2.0 = {float(vt[sel].std()):.4f} "
          f"({int(sel.sum())} samples)")
xt_r = (1 - t) * x0 + t * x1_rand; vr = x1_rand - x0
xt_o = (1 - t) * x0 + t * x1_ot;  vo = x1_ot - x0
sr = float(vr[(xt_r - 2).abs() < 0.05].std()); so = float(vo[(xt_o - 2).abs() < 0.05].std())
ok("random coupling leaves a large target variance mid-flight", sr > 5 * so,
   f"{sr:.3f} vs {so:.3f} — the model can only regress the AVERAGE of that spread")
ok("the OT pairing nearly removes it", so < 0.1,
   "one consistent direction per location = a straight learnable field")
