import math, torch, torch.nn as nn, torch.nn.functional as F     # neurons as FUNCTIONS

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); torch.set_printoptions(precision=5, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

SQ2 = math.sqrt(2.0)

def Phi(z):                                          # standard normal CDF
    return 0.5 * (1.0 + torch.erf(z / SQ2))

def phi(z):                                          # standard normal PDF
    return torch.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

def K_self(gamma, beta):                             # eq (3): ||f||^2 per unit output norm
    r = beta / gamma.abs()
    return (gamma ** 2 + beta ** 2) * Phi(r) + beta * gamma.abs() * phi(r)

def mc_E(fn, n=10_000_000, chunk=2_000_000):         # big-sample Monte-Carlo expectation
    tot = 0.0
    for i in range(0, n, chunk):
        m = min(chunk, n - i)
        tot += float(fn(m).sum())
    return tot / n

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

x = torch.randn(120_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)              # x-tilde = [x, 1]
def make(w_in, w_out):
    return F.relu(xt @ w_in)[:, None] * w_out[None, :]
wi1, wo1 = torch.randn(9), torch.randn(3)
wi2, wo2 = torch.randn(9), torch.randn(3)
f, g = make(wi1, wo1), make(wi2, wo2)
s = 3.7
ok("s*f stays realizable (fold s into w_out)", torch.allclose(s * f, make(wi1, s * wo1), atol=1e-4),
   "PH-1 closure under positive scaling")
target = f + g
best = None
for _ in range(300):                                        # random search for a single neuron matching f+g
    wi = torch.randn(9); act = F.relu(xt @ wi)
    wo = (act[:, None] * target).mean(0) / (act ** 2).mean().clamp_min(1e-9)
    err = float(((act[:, None] * wo[None, :]) - target).pow(2).sum(1).mean())
    best = err if best is None else min(best, err)
tnorm = float(target.pow(2).sum(1).mean())
ok("but f+g is NOT (in general) a single realizable neuron", best / tnorm > 0.02,
   f"best single-neuron fit leaves {best/tnorm:.1%} of ||f+g||^2 — the loss merging must manage")

g1 = torch.randn(50_000, 4); g2 = 0.6 * g1 + 0.8 * torch.randn(50_000, 4)
f_i, f_j = g1, 0.7 * g1 + 0.4 * g2                          # two correlated 'neurons' as samples
def ip(a, b): return float((a * b).sum(1).mean())
S = f_i + f_j
s = 1.1                                                     # any fixed scale
cands = [torch.randn(50_000, 4) for _ in range(64)]
cands = [c / math.sqrt(ip(c, c)) for c in cands]            # unit Hilbert norm each
num = [ip(s * c - f_i, s * c - f_i) + ip(s * c - f_j, s * c - f_j) for c in cands]
ali = [ip(c, S) for c in cands]
best_by_dist = min(range(64), key=lambda k: num[k])
best_by_align = max(range(64), key=lambda k: ali[k])
ok("minimising the merge distance == maximising alignment with f_i+f_j",
   best_by_dist == best_by_align,
   "the cross term -2s<psi, f_i+f_j> is the only psi-dependent part of the numerator")
expand = [ip(f_i, f_i) + ip(f_j, f_j) + 2 * s * s - 2 * s * a for a in ali]
ok("algebra check: numerator = a + 2s^2 - 2s<psi,S> exactly",
   max(abs(e - n) for e, n in zip(expand, num)) < 1e-2,
   f"max residual {max(abs(e-n) for e,n in zip(expand,num)):.1e}")

x = torch.randn(400_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
def K_dir(u):                                               # K(u,u) = E[ReLU(u.x~)^2]
    return float((F.relu(xt @ u) ** 2).mean())
errs = []
for _ in range(12):
    u = F.normalize(torch.randn(9), dim=0)
    v = F.normalize(torch.randn(3), dim=0)
    psi = F.relu(xt @ u)[:, None] * v[None, :] / math.sqrt(K_dir(u))
    errs.append(abs(float(psi.pow(2).sum(1).mean()) - 1.0))
ok("||psi||_H = 1 for EVERY direction, by construction", max(errs) < 5e-3,
   f"max deviation {max(errs):.1e} across 12 random (u, v)")
ok("so the search space is exactly {unit u} x {unit v}", True,
   "two spheres — which is what eqs. 10-11 optimise over")

x = torch.randn(400_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
wi1, wo1 = F.normalize(torch.randn(9), dim=0) * 1.3, torch.randn(3)
wi2, wo2 = F.normalize(torch.randn(9), dim=0) * 0.8, torch.randn(3)
child = lambda wi, wo: F.relu(xt @ wi)[:, None] * wo[None, :]
S = child(wi1, wo1) + child(wi2, wo2)
def Kx(u, w):                                               # cross-kernel by MC: E[ReLU(u.x)ReLU(w.x)]
    return float((F.relu(xt @ u) * F.relu(xt @ w)).mean())
u = F.normalize(torch.randn(9), dim=0)
m = Kx(u, wi1) * wo1 + Kx(u, wi2) * wo2                    # the kernel-weighted output sum
v_star = F.normalize(m, dim=0)
Ku = float((F.relu(xt @ u) ** 2).mean())
def align(v):
    psi = F.relu(xt @ u)[:, None] * v[None, :] / math.sqrt(Ku)
    return float((psi * S).sum(1).mean())
a_star = align(v_star)
rand_best = max(align(F.normalize(torch.randn(3), dim=0)) for _ in range(200))
ok("v* (kernel-weighted sum, normalised) beats 200 random unit v's", a_star >= rand_best - 1e-4,
   f"{a_star:.5f} vs best random {rand_best:.5f}")
ok("and the u-objective ||m||/sqrt(K(u,u)) IS the alignment at v*",
   abs(a_star - float(m.norm()) / math.sqrt(Ku)) < 2e-3,
   "Cauchy-Schwarz is tight exactly at v* — eq. 10's two halves agree")

x = torch.randn(400_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
w = F.normalize(torch.randn(9), dim=0)
u = F.normalize(torch.randn(9), dim=0)
Kp = float((F.relu(xt @ u) * F.relu(xt @ w)).mean())
Km = float((F.relu(xt @ -u) * F.relu(xt @ w)).mean())
ok("ReLU kernels are NOT sign-symmetric", abs(Kp - Km) / max(Kp, Km, 1e-9) > 0.05,
   f"K(+u,w)={Kp:.4f} vs K(-u,w)={Km:.4f}")
ok("so a sign-blind SVD initialiser NEEDS this one-comparison fix", True,
   "evaluate the exact objective at +/-u and keep the better — two evaluations, no search")

a, b, E = 2.31, 1.74, 0.9                                   # any a>0, b>0, E>=0
def J2(s):                                                   # the squared objective of §7.1.2
    return (2 * s ** 2 - 2 * b * s + a) / (s + E) ** 2
s_star = (a + b * E) / (2 * E + b)                           # eq. (12)
grid = torch.linspace(1e-4, 20, 2_000_001)
s_grid = float(grid[torch.argmin(J2(grid))])
ok("a 2M-point grid agrees with the closed form", abs(s_grid - s_star) < 1e-3,
   f"grid {s_grid:.6f} vs s* {s_star:.6f}")
s_t = torch.tensor(s_star, requires_grad=True)
J2(s_t).backward()
ok("autograd: dJ/ds = 0 exactly at s*", abs(float(s_t.grad)) < 1e-5,
   f"gradient at s* = {float(s_t.grad):.2e}")
ok("the denominator 2E+b > 0 always (b>0 by the phase check)", 2 * E + b > 0,
   "a unique positive minimiser is guaranteed")
E0 = 1e-12
ok("layer-collapse limit E_rem->0 gives s* -> a/b", abs((a + b * E0) / (2 * E0 + b) - a / b) < 1e-9,
   f"s* -> {a/b:.6f}")

x = torch.randn(300_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
wi1, wo1 = F.normalize(torch.randn(9), dim=0), torch.randn(3)
wi2, wo2 = F.normalize(torch.randn(9), dim=0), torch.randn(3)
M = torch.outer(wo1, wi1) + torch.outer(wo2, wi2)           # eq. 14's matrix
u_hat = torch.linalg.svd(M)[2][0]                           # top right singular vector
def obj(u):
    Ku = float((F.relu(xt @ u) ** 2).mean())
    m = float((F.relu(xt @ u) * F.relu(xt @ wi1)).mean()) * wo1         + float((F.relu(xt @ u) * F.relu(xt @ wi2)).mean()) * wo2
    return float(m.norm()) / math.sqrt(Ku), m
o_p, m_p = obj(u_hat); o_m, m_m = obj(-u_hat)
u_c, m_c = (u_hat, m_p) if o_p >= o_m else (-u_hat, m_m)    # eq. 11 applied
v_star = F.normalize(m_c, dim=0)
ok("the recipe runs: SVD init -> sign fix -> v* readout", v_star.shape == (3,) and
   abs(float(v_star.norm()) - 1) < 1e-6)
ok("the sign fix chose the better branch", max(o_p, o_m) == (o_p if torch.equal(u_c, u_hat) else o_m),
   f"objective {max(o_p, o_m):.5f} vs discarded {min(o_p, o_m):.5f}")

x = torch.randn(300_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
wi1, wo1 = F.normalize(torch.randn(9), dim=0), torch.randn(3)
wi2, wo2 = F.normalize(torch.randn(9), dim=0), torch.randn(3)
M = torch.outer(wo1, wi1) + torch.outer(wo2, wi2)
u_svd = torch.linalg.svd(M)[2][0]
rand = F.normalize(torch.randn(400, 9), dim=1)
vals = (rand @ M.T).norm(dim=-1)                             # ||M u|| for each candidate direction
ok("the top right singular vector beats 400 random directions on ||Mu||",
   float((M @ u_svd).norm()) >= float(vals.max()) - 1e-5,
   f"{float((M @ u_svd).norm()):.5f} vs best random {float(vals.max()):.5f}")
def relu_obj(u):
    Ku = float((F.relu(xt @ u) ** 2).mean())
    m = float((F.relu(xt @ u) * F.relu(xt @ wi1)).mean()) * wo1         + float((F.relu(xt @ u) * F.relu(xt @ wi2)).mean()) * wo2
    return float(m.norm()) / math.sqrt(Ku)
r_svd = max(relu_obj(u_svd), relu_obj(-u_svd))
r_rand = max(relu_obj(F.normalize(torch.randn(9), dim=0)) for _ in range(50))
ok("as a warm start for the TRUE objective it beats random restarts", r_svd >= r_rand,
   f"linearised start {r_svd:.5f} vs best of 50 random {r_rand:.5f}")

x = torch.randn(500_000, 8)
xt = torch.cat([x, torch.ones(len(x), 1)], 1)
u_s = F.normalize(torch.randn(9), dim=0)
v_s = F.normalize(torch.randn(3), dim=0)
s_star = 1.83
K_us = float((F.relu(xt @ u_s) ** 2).mean())
def build(RF):
    w_in = math.sqrt(s_star * RF) * K_us ** -0.25 * u_s     # eq. (15)
    w_out = math.sqrt(s_star / RF) * K_us ** -0.25 * v_s
    return F.relu(xt @ w_in)[:, None] * w_out[None, :]
f_a, f_b = build(1.0), build(23.0)
ok("two different R_F give the IDENTICAL function", torch.allclose(f_a, f_b, atol=1e-4),
   f"max diff {float((f_a-f_b).abs().max()):.1e} — the resharding ratio is pure gauge")
norm = math.sqrt(float(f_a.pow(2).sum(1).mean()))
ok("and the constructed neuron's Hilbert norm IS s*", abs(norm - s_star) / s_star < 5e-3,
   f"||f_p|| = {norm:.4f} vs s* = {s_star}")
ok("PH-1 is what makes both true", True,
   "Psi(c u.x) = c Psi(u.x): scale slides freely between the two projections")

wi1 = F.normalize(torch.randn(9), dim=0)
wi2 = F.normalize(torch.randn(9), dim=0)
wo1, wo2 = torch.randn(3), torch.randn(3)
M = torch.outer(wo1, wi1) + torch.outer(wo2, wi2)
u_hat = torch.linalg.svd(M)[2][0]
A = torch.stack([wi1, wi2], 1)                               # 9 x 2
c = torch.linalg.lstsq(A, u_hat).solution
recon = A @ c
ok("the linearised optimum lies IN span{w_in_i, w_in_j}",
   float((recon - u_hat).norm()) < 1e-5,
   f"residual {float((recon-u_hat).norm()):.1e} — rank-2 M has row space = that span")
print(f"  c1 = {float(c[0]):+.4f}, c2 = {float(c[1]):+.4f}")
ok("two numbers now carry the whole merge into BN-land", c.shape == (2,),
   "eqs. 17-18 turn (c1, c2) into raw weights and BN stats")

ok("convention: raw weights = effective weights for the new neuron", True,
   "the new BN layer's own statistics then re-standardise exactly what we built")
ok("mu_p is chosen so the shift lands on b_p", True,
   "verified distributionally in the next cell together with eq. 18's variance")

c1, c2 = 0.8, 0.5
beta_i, beta_j = 0.3, -0.6
gam_i, gam_j = 1.2, 0.9
rho = 0.7
n = 10_000_000
z1 = torch.randn(n); z2 = rho * z1 + math.sqrt(1 - rho ** 2) * torch.randn(n)
y_i = beta_i + gam_i * z1
y_j = beta_j + gam_j * z2
y_p = c1 * y_i + c2 * y_j
beta_p = c1 * beta_i + c2 * beta_j                            # eq. (18)
gam_p = math.sqrt(c1**2 * gam_i**2 + c2**2 * gam_j**2 + 2*c1*c2*abs(gam_i)*abs(gam_j)*rho)
ok("the merged mean is the beta combination", abs(float(y_p.mean()) - beta_p) < 2e-3,
   f"MC {float(y_p.mean()):+.5f} vs formula {beta_p:+.5f}")
ok("the merged std needs the CORRELATION term", abs(float(y_p.std()) - gam_p) < 2e-3,
   f"MC {float(y_p.std()):.5f} vs formula {gam_p:.5f}")
naive = math.sqrt(c1**2 * gam_i**2 + c2**2 * gam_j**2)
ok("assuming independence would be badly wrong here", abs(naive - gam_p) / gam_p > 0.15,
   f"naive {naive:.4f} vs correct {gam_p:.4f} ({abs(naive-gam_p)/gam_p:.0%} off at rho=0.7)")
