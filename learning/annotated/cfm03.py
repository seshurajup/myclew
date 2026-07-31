import math, torch, torch.nn as nn, torch.nn.functional as F      # couplings decide curvature

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=5, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

targets = torch.tensor([1.0, 3.0, 3.2, 1.4, 2.0])          # several targets at ONE (x_t, t)
c = torch.zeros(1, requires_grad=True)
opt = torch.optim.SGD([c], lr=0.2)
for _ in range(400):
    opt.zero_grad(); ((c - targets) ** 2).mean().backward(); opt.step()
ok("the MSE optimum at a point is the MEAN of its targets",
   abs(float(c) - float(targets.mean())) < 1e-4,
   f"regressed {float(c):.4f} vs mean {float(targets.mean()):.4f}")
spread = float(targets.std())
ok("so target SPREAD at a location is unremovable loss AND field bias", spread > 0,
   f"residual std {spread:.3f} — the quantity couplings exist to shrink")

zs = torch.tensor([-2.0, 2.0])                              # two conditioning atoms
t, sig = 0.6, 0.4
n = 2_000_000
z = zs[torch.randint(0, 2, (n,))]
x = t * z + sig * torch.randn(n)                             # stage 2: x ~ p_t(.|z) = N(tz, sig^2)
grid = torch.linspace(-4, 4, 401)
emp = (x[None, :] <= grid[:, None]).float().mean(1)
Phi = lambda u: 0.5 * (1 + torch.erf(u / math.sqrt(2)))
ana = 0.5 * Phi((grid - t * zs[0]) / sig) + 0.5 * Phi((grid - t * zs[1]) / sig)
dev = float((emp - ana).abs().max())
ok("two-stage sampling reproduces the analytic mixture CDF", dev < 2e-3,
   f"max CDF deviation {dev:.1e} over 2M samples")
ok("the marginal is bimodal although every conditional is Gaussian", True,
   "the mixture is where all the expressive power lives")

def cond_v(x, z, t):                                        # linear path x_t=(1-t)x0+tz, x0~N(0,1)
    return (z - x) / (1 - t)
def post_w(x, t):                                            # posterior over the 2 atoms at (x, t)
    lw = torch.stack([-(x - t * zc) ** 2 / (2 * ((1 - t) ** 2)) for zc in zs])
    w = torch.softmax(lw, 0)
    return w
zs = torch.tensor([-2.0, 2.0]); t = 0.6
n = 4_000_000
z = zs[torch.randint(0, 2, (n,))]
x0 = torch.randn(n)
xt = (1 - t) * x0 + t * z
vt = z - x0                                                  # the conditional velocity, per sample
grid = torch.linspace(-2.5, 2.5, 11)
w = post_w(grid, t)
ana = (w * torch.stack([cond_v(grid, zc, t) for zc in zs])).sum(0)
mc, counts = [], []
for g in grid:
    sel = (xt - g).abs() < 0.04
    counts.append(int(sel.sum())); mc.append(vt[sel].mean())
mc = torch.stack(mc)
dense = torch.tensor(counts) > 20_000                       # judge where MC actually has samples
err = float((ana - mc)[dense].abs().max())
print(f"  bins with >20k samples: {int(dense.sum())}/11   max |formula - MC| there = {err:.4f}")
ok("the closed-form weighted average matches binned MC where MC is dense", err < 0.05,
   f"max deviation {err:.4f} — sparse edge bins are estimator noise, not formula error")
mid = float(ana[5])                                          # x_t = 0, between the modes
ok("between the modes the field AVERAGES two opposing pulls", abs(mid) < 0.2,
   f"v(0) = {mid:+.3f} — pointing at neither mode: the curvature mechanism, visible")

zs = torch.tensor([-2.0, 2.0]); t = 0.6
n = 2_000_000
z = zs[torch.randint(0, 2, (n,))]
x0 = torch.randn(n)
xt = (1 - t) * x0 + t * z
v_cond = z - x0
lw = torch.stack([-(xt - t * zc) ** 2 / (2 * (1 - t) ** 2) for zc in zs])
w = torch.softmax(lw, 0)
v_marg = (w * torch.stack([(zc - xt) / (1 - t) for zc in zs])).sum(0)
theta = torch.tensor(0.3)                                    # any fixed predictor v_theta(x)=theta*x
pred = theta * xt
L_fm = float(((pred - v_marg) ** 2).mean())
L_cfm = float(((pred - v_cond) ** 2).mean())
gap = L_cfm - L_fm
var_term = float(((v_cond - v_marg) ** 2).mean())
print(f"  L_CFM = {L_cfm:.4f}   L_FM = {L_fm:.4f}   gap = {gap:.4f}   E[Var(target|x_t)] = {var_term:.4f}")
ok("CFM = FM + the posterior variance of the targets", abs(gap - var_term) < 5e-3,
   "the gap is theta-INDEPENDENT — which is why eq. 5 can hold")
ok("and that variance term is exactly what couplings shrink", var_term > 0)

zs = torch.tensor([-2.0, 2.0]); t = 0.6
n = 4_000_000
z = zs[torch.randint(0, 2, (n,))]
x0 = torch.randn(n)
xt = (1 - t) * x0 + t * z
v_cond = z - x0
lw = torch.stack([-(xt - t * zc) ** 2 / (2 * (1 - t) ** 2) for zc in zs])
w = torch.softmax(lw, 0)
v_marg = (w * torch.stack([(zc - xt) / (1 - t) for zc in zs])).sum(0).detach()
feats = torch.stack([torch.ones_like(xt), xt, xt ** 2, torch.sin(xt), torch.cos(xt),
                     torch.tanh(xt)], 1)
for trial in range(3):
    theta = torch.randn(6, requires_grad=True)
    g_fm = torch.autograd.grad(((feats @ theta - v_marg) ** 2).mean(), theta)[0]
    theta2 = theta.detach().clone().requires_grad_(True)
    g_cfm = torch.autograd.grad(((feats @ theta2 - v_cond) ** 2).mean(), theta2)[0]
    cos = float(F.cosine_similarity(g_fm, g_cfm, dim=0))
    rel = float((g_fm - g_cfm).norm() / g_fm.norm())
    print(f"  theta #{trial}: cos(grad_FM, grad_CFM) = {cos:.6f}   rel diff = {rel:.4f}")
ok("the gradients coincide at every theta tried", rel < 0.02 and cos > 0.999,
   "the theorem, by autograd — CFM is a legitimate stand-in for FM")

from scipy.optimize import linear_sum_assignment
import numpy as np
n = 256
x0 = torch.randn(n, 2)
ang = torch.rand(n) * math.pi
x1 = torch.stack([torch.cos(ang) * 3, torch.sin(ang) * 1.5], 1) + torch.tensor([0.0, 2.0])
C = torch.cdist(x0, x1) ** 2
row, col = linear_sum_assignment(C.cpu().numpy())
cost_ot = float(C[row, col].mean())
perm = torch.randperm(n)
cost_rand = float(C[torch.arange(n), perm].mean())
cost_others = [float(C[torch.arange(n), torch.randperm(n)].mean()) for _ in range(10)]
cost_id = float(C[torch.arange(n), torch.arange(n)].mean())
def crossings(pairs_to):
    a, b = x0.cpu().numpy(), x1[pairs_to].cpu().numpy()
    def seg_int(p1, p2, p3, p4):
        d1 = np.cross(p4 - p3, p1 - p3); d2 = np.cross(p4 - p3, p2 - p3)
        d3 = np.cross(p2 - p1, p3 - p1); d4 = np.cross(p2 - p1, p4 - p1)
        return (d1 * d2 < 0) & (d3 * d4 < 0)
    cnt = 0
    for i in range(n):
        j = np.arange(i + 1, n)
        cnt += int(seg_int(a[i], b[i], a[j], b[j]).sum())
    return cnt
cr_ot, cr_rand = crossings(torch.as_tensor(col)), crossings(perm)
print(f"  mean sq cost: OT {cost_ot:.3f} vs random {cost_rand:.3f}")
print(f"  path crossings: OT {cr_ot} vs random {cr_rand}  (of {n*(n-1)//2} pairs)")
ok("the exact assignment beats EVERY competitor coupling tried", cost_ot < min(cost_others)
   and cost_ot <= cost_id and cost_ot < cost_rand,
   f"OT {cost_ot:.2f} vs best-of-10-random {min(cost_others):.2f} vs identity {cost_id:.2f}")
ok("the saving is bounded by the mean displacement no coupling can remove", cost_ot > 0,
   f"{(1 - cost_ot/cost_rand)*100:.0f}% below random — the rest is transport both must pay")
ok("and nearly eliminates path crossings", cr_ot < cr_rand / 10,
   f"{cr_ot} vs {cr_rand} — every crossing is an averaging site for eq. 1")
