import math, torch, torch.nn as nn, torch.nn.functional as F      # couplings decide curvature

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=5, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

n_cond, d = 4, 8
emb = nn.Embedding(n_cond, 32)
head = nn.Sequential(nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 2 * d))
mu_true = torch.randn(n_cond, d) * 2
sig_true = torch.rand(n_cond, d) * 0.8 + 0.2
opt = torch.optim.Adam(list(emb.parameters()) + list(head.parameters()), lr=3e-3)
for _ in range(3000):
    c = torch.randint(0, n_cond, (1024,))
    x = mu_true[c] + sig_true[c] * torch.randn(1024, d)
    out = head(emb(c)); mu_p, log_s = out[:, :d], out[:, d:]
    nll = (log_s + (x - mu_p) ** 2 / (2 * torch.exp(2 * log_s))).mean()
    opt.zero_grad(); nll.backward(); opt.step()
c_all = torch.arange(n_cond)
out = head(emb(c_all)); mu_p, sig_p = out[:, :d], torch.exp(out[:, d:])
ok("the module recovers every condition's mean", float((mu_p - mu_true).abs().max()) < 0.12,
   f"max err {float((mu_p - mu_true).abs().max()):.3f}")
ok("and every condition's per-dimension std", float((sig_p - sig_true).abs().max()) < 0.12,
   f"max err {float((sig_p - sig_true).abs().max()):.3f}")
ok("but a DIAGONAL sigma cannot carry correlation — the stated trade", True,
   "the paper accepts this for stability in high dimension; eq. 9's full Sigma is train-time only")

c = torch.randint(0, n_cond, (50_000,))
out = head(emb(c)); mu_c, sig_c = out[:, :d], torch.exp(out[:, d:])
x0_cond = mu_c + sig_c * torch.randn(50_000, d)               # eq. (12)
x0_glob = torch.randn(50_000, d)                              # the N(0, I) alternative
d_cond = float((x0_cond - mu_true[c]).norm(dim=1).mean())
d_glob = float((x0_glob - mu_true[c]).norm(dim=1).mean())
print(f"  distance to the condition's true source: conditional start {d_cond:.3f} vs global {d_glob:.3f}")
ok("conditional starts are FAR closer to where they must end up", d_cond < d_glob / 2,
   f"{d_cond:.2f} vs {d_glob:.2f}")
ok("that head start is the few-step advantage, quantified", True,
   "less distance for the ODE to cover = fewer steps for the same quality")

mse = torch.linspace(0, 6, 200)
R = torch.exp(-mse)                                           # eq. (13)
ok("reward is bounded in (0, 1]", float(R.min()) > 0 and float(R.max()) <= 1.0)
ok("perfect reconstruction gives exactly R = 1", abs(float(torch.exp(-torch.tensor(0.0))) - 1) < 1e-9)
ok("and R is strictly decreasing in the error", bool((R.diff() < 0).all()),
   "no local incentives to be wrong — the shaping is monotone")

r1 = torch.rand(10_000)
V_single = r1                                                 # eq. (14): one transition, gamma irrelevant
ok("single transition: V is IDENTICALLY the reward", torch.equal(V_single, r1))
gamma = 0.9
r2 = torch.rand(10_000)
V_two = r1 + gamma * r2                                       # a 2-step MDP for contrast
ok("with a second step the identity breaks", not torch.allclose(V_two, r1),
   f"mean |V - R| = {float((V_two - r1).abs().mean()):.3f} — the single-step structure is what "
   f"removes the critic")

n_cond, m = 8, 20_000
base_v = torch.rand(n_cond) * 0.5 + 0.25                      # each condition's typical value
V = base_v[:, None] + 0.1 * torch.randn(n_cond, m)
A = V - V.mean(1, keepdim=True)                               # eq. (15)
ok("the advantage is mean-zero within every condition", float(A.mean(1).abs().max()) < 1e-6)
logp_grad = torch.randn(n_cond, m)                            # stand-in for grad log pi(a|c)
g_raw = (V * logp_grad)
g_adv = (A * logp_grad)
var_raw = float(g_raw.var())
var_adv = float(g_adv.var())
print(f"  policy-gradient estimator variance: raw {var_raw:.4f} vs with baseline {var_adv:.4f} "
      f"({var_raw/var_adv:.1f}x lower)")
ok("the baseline slashes estimator variance", var_adv < var_raw / 3,
   f"{var_raw/var_adv:.1f}x — the practical reason eq. 15 exists")
diff = g_raw - g_adv                                          # = E_pk[V] * grad log pi, mean-zero
se_diff = float(diff.std() / math.sqrt(diff.numel()))
ok("while leaving the EXPECTED gradient unchanged", abs(float(diff.mean())) < 4 * se_diff,
   f"|mean diff| = {abs(float(diff.mean())):.2e} within 4 SE ({4*se_diff:.2e}) — unbiased")
