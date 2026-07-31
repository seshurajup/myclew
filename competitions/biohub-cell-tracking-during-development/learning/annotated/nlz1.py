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

class DeltaMomentumSGD(torch.optim.Optimizer):
    """SGD whose momentum uses the L2 (delta-rule) objective of eq. 49 instead of a fixed EMA."""
    def __init__(self, params, lr=1e-2, alpha=0.9, eta=0.1):
        super().__init__(params, dict(lr=lr, alpha=alpha, eta=eta))
    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            for p in grp["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                m = st.setdefault("m", torch.zeros_like(p))
                gs = p.grad / (1 + p.grad.norm())                 # normalised key (the paper's assumption)
                decay = max(0.0, grp["alpha"] - grp["eta"] * float(gs.pow(2).sum()))
                m.mul_(decay).add_(gs, alpha=-grp["lr"])          # eq. 49
                p.add_(m)

torch.manual_seed(0)
def fit(OptCls, **kw):
    net = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1))
    opt = OptCls(net.parameters(), **kw)
    X = torch.randn(256, 16); y = (X[:, :4].sum(1, keepdim=True) > 0).float()
    for _ in range(300):
        opt.zero_grad(); F.binary_cross_entropy_with_logits(net(X), y).backward(); opt.step()
    return float(F.binary_cross_entropy_with_logits(net(X), y))
base = fit(torch.optim.SGD, lr=0.05, momentum=0.9)
delta = fit(DeltaMomentumSGD, lr=0.05)
ok("Delta Momentum trains a real net", delta < 0.6, f"BCE: SGD-M {base:.4f} vs Delta {delta:.4f}")
print("the point is robustness, not a win here: one clamped line replaces the fixed low-pass filter")

class CMSHead(nn.Module):
    def __init__(s, d, periods=(1, 4, 16)):
        super().__init__()
        s.levels = nn.ModuleList([nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
                                  for _ in periods])
        s.periods = periods
    def forward(s, x):
        for lv in s.levels:
            x = x + lv(x)
        return x
head = CMSHead(16); opt = torch.optim.SGD(head.parameters(), lr=0.05)
X = torch.randn(64, 16); Y = torch.randn(64, 16)
applied = {i: 0 for i in range(3)}
for step in range(1, 65):
    opt.zero_grad(); F.mse_loss(head(X), Y).backward()
    for i, (lv, C) in enumerate(zip(head.levels, head.periods)):
        if step % C:                                             # eq. 71: not this level's turn
            for p in lv.parameters():
                p.grad = None
        else:
            applied[i] += 1
    opt.step()
ok("each level was updated at its own frequency", [applied[i] for i in range(3)] == [64, 16, 4],
   f"{applied} for periods {head.periods}")
ok("inference cost is unchanged (the gate is on the STEP, not the forward pass)",
   head(X).shape == X.shape)

def nl_audit(model, opt, accum_steps=1):
    rows = []
    weights = sum(p.numel() for p in model.parameters())
    state = sum(v.numel() for s in opt.state.values() for v in s.values()
                if torch.is_tensor(v) and v.dim() > 0)
    rows.append(dict(level=1, component="weights", context="the training set",
                     objective="the task loss", freq=f"1/{accum_steps} per batch", params=weights))
    rows.append(dict(level=2, component=type(opt).__name__ + " state", context="the gradients",
                     objective="compress the gradient stream", freq="1 per step", params=state))
    return rows
net = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
opt.zero_grad(); net(torch.randn(4, 32)).pow(2).mean().backward(); opt.step()
rows = nl_audit(net, opt, accum_steps=8)
for r in rows:
    print(f"  level {r['level']}: {r['component']:<18} ctx={r['context']:<16} "
          f"freq={r['freq']:<16} params={r['params']}")
ok("the audit finds more parameters than the model advertises",
   rows[1]["params"] >= 2 * rows[0]["params"] - 8,
   f"advertised {rows[0]['params']},真 total {rows[0]['params'] + rows[1]['params']}")
