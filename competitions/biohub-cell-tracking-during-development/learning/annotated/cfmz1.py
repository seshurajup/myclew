import math, torch, torch.nn as nn, torch.nn.functional as F      # couplings decide curvature

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=5, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

torch.manual_seed(0)
n_data = 4096
th = torch.rand(n_data) * math.pi
modes = (torch.rand(n_data) < 0.5)
x1_all = torch.stack([torch.cos(th) * 2 - 1, torch.sin(th)], 1)
x1_all[modes] = -x1_all[modes] + torch.tensor([0.0, 0.5])      # two interleaved moons
lab = modes.long()                                             # the cluster identity (k-means would find it)

class VNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(3, 128), nn.SiLU(), nn.Linear(128, 128), nn.SiLU(),
                               nn.Linear(128, 2))
    def forward(self, x, t):
        return self.f(torch.cat([x, t[:, None]], 1))

from scipy.optimize import linear_sum_assignment
mu_k = torch.stack([x1_all[lab == k].mean(0) for k in range(2)]) * 0.8   # per-cluster sources
def make_batch(kind, bs=512):
    idx = torch.randint(0, n_data, (bs,))
    x1 = x1_all[idx]
    if kind == "random":
        x0 = torch.randn(bs, 2) * 0.6
    else:                                                      # cluster-OT
        k = lab[idx]
        x0 = mu_k[k] + torch.randn(bs, 2) * 0.35               # each cluster's OWN source
        for kk in range(2):
            m = (k == kk).nonzero().flatten()
            if len(m) > 1:
                Cm = torch.cdist(x0[m], x1[m]) ** 2
                r, c = linear_sum_assignment(Cm.cpu().numpy())
                x1[m] = x1[m][torch.as_tensor(c)]              # exact OT within the cluster
    return x0, x1

def train(kind, steps=1500):
    torch.manual_seed(1)
    net = VNet()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    for _ in range(steps):
        x0, x1 = make_batch(kind)
        t = torch.rand(len(x0))
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        loss = ((net(xt, t) - (x1 - x0)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return net.eval()

net_rand = train("random")
net_cot = train("cot")
print("both models trained: same net, same optimiser, same 1500 steps — only the coupling differs")
ok("training completed for both", True)

def sample(net, kind, n=4096, steps=1):
    if kind == "random":
        x = torch.randn(n, 2) * 0.6
    else:
        k = torch.randint(0, 2, (n,))
        x = mu_k[k] + torch.randn(n, 2) * 0.35
    traj = [x.clone()]
    with torch.no_grad():
        for i in range(steps):
            t = torch.full((n,), i / steps)
            x = x + net(x, t) / steps
            traj.append(x.clone())
    return x, traj

def curvature(net, kind, steps=64):
    xT, traj = sample(net, kind, n=2048, steps=steps)
    P = torch.stack(traj)                                      # (steps+1, n, 2)
    chord = P[-1] - P[0]
    tgrid = torch.linspace(0, 1, steps + 1)[:, None, None]
    straight = P[0][None] + tgrid * chord[None]
    return float((P - straight).norm(dim=-1).mean())

def quality(x):                                                # mean NN distance to the true manifold
    d = torch.cdist(x, x1_all)
    return float(d.min(1).values.mean())

c_r, c_c = curvature(net_rand, "random"), curvature(net_cot, "cot")
q1_r = quality(sample(net_rand, "random", steps=1)[0])
q1_c = quality(sample(net_cot, "cot", steps=1)[0])
q64_r = quality(sample(net_rand, "random", steps=64)[0])
q64_c = quality(sample(net_cot, "cot", steps=64)[0])
ref = quality(torch.randn(4096, 2))
print(f"  curvature (mean deviation from straight): random {c_r:.4f}   cluster-OT {c_c:.4f}")
print(f"  1-step quality (NN dist, lower=better)  : random {q1_r:.4f}   cluster-OT {q1_c:.4f}")
print(f"  64-step quality                          : random {q64_r:.4f}   cluster-OT {q64_c:.4f}")
print(f"  (untrained reference: {ref:.4f})")
ok("cluster-OT trains a measurably straighter flow", c_c < c_r * 0.7,
   f"{c_c:.4f} vs {c_r:.4f} — {(1-c_c/c_r)*100:.0f}% less bending")
ok("which pays exactly where the paper says: ONE-step generation", q1_c < q1_r * 0.8,
   f"{q1_c:.4f} vs {q1_r:.4f}")
ok("with many steps both are healthy (the gap is a few-step gap)", q64_r < ref / 3 and q64_c < ref / 3,
   f"{q64_r:.4f} / {q64_c:.4f} vs untrained {ref:.4f}")
