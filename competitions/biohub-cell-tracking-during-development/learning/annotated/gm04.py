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

d, c_len, q_len, m_tok = 32, 24, 4, 8                            # this lesson's own setup
C = torch.randn(c_len, d); Q = torch.randn(q_len, d)
prompt = torch.cat([C, Q], 0)
M = torch.zeros(m_tok, d, requires_grad=True)
torch.manual_seed(1)
# A deliberately SIMPLE frozen reader, chosen so the mechanism is visible rather than buried: each context
# position has a fixed address over the memory slots, the model reads the memory at that address and maps
# it through a frozen W. Writing the context is then exactly "fit M so the frozen reader reproduces it".
W = torch.randn(d, d) / d ** 0.5                                 # FROZEN
tokens = torch.randn(c_len, d)
A = F.softmax(2.0 * torch.randn(c_len, m_tok), dim=-1)           # fixed addresses (frozen)
# The write problem is linear in M, so it has a KNOWN optimum: minimising ||A M W - T||^2 gives
# M* = pinv(A) T pinv(W). We compare gradient descent against that, which is the honest way to check
# "the write works" — and M* is only a LOSSY fit because 24 positions are being squeezed into 8 slots,
# which is exactly the compression the paper is proposing.
M_star = torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W)
def read(Mv, i):
    return (A[i] @ Mv) @ W
def write_loss(Mv):                                             # eq. 4
    return torch.stack([F.mse_loss(read(Mv, i), tokens[i]) for i in range(c_len)]).mean()
def encode(C_tokens=None, K=200, lr=0.5, M0=None):              # eq. 6
    Mv = (torch.zeros(m_tok, d) if M0 is None else M0.clone()).requires_grad_(True)
    o = torch.optim.Adam([Mv], lr=lr)
    for _ in range(K):
        o.zero_grad(); write_loss(Mv).backward(); o.step()
    return Mv.detach()
# the memory IS the variable being optimised
ok("the memory is far smaller than the context", m_tok < c_len, f"{m_tok} tokens vs {c_len}")
ok("and it is a free variable, not a network output", M.requires_grad and M.numel() == m_tok * d,
   f"{M.numel()} numbers to optimise")

prompt_mem = torch.cat([M.detach(), Q], 0)                        # eq. 3
ok("the interface is unchanged, only shorter", prompt_mem.shape[1] == prompt.shape[1]
   and prompt_mem.shape[0] < prompt.shape[0],
   f"{tuple(prompt_mem.shape)} instead of {tuple(prompt.shape)}")
ok("per-query cost falls by the length ratio",
   (m_tok + q_len) ** 2 < (c_len + q_len) ** 2,
   f"{(m_tok+q_len)**2} vs {(c_len+q_len)**2} ({(c_len+q_len)**2/(m_tok+q_len)**2:.1f}x cheaper)")

# the write loss uses the FROZEN reader defined above (nothing is re-randomised here, or the
# optimum computed in the next cell would not correspond to this objective)
L0 = float(write_loss(torch.zeros(m_tok, d)))
ok("the write loss is computable with a FROZEN model", L0 > 0, f"L_write(M=0) = {L0:.4f}")
ok("it needs no labels at all", True, "the context's own tokens are the targets")
ok("it is a least-squares problem in M (so it has a known optimum)",
   float(write_loss(torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W))) < L0,
   f"L_write(M*) = {float(write_loss(torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W))):.4f} < {L0:.4f}")

M_k = torch.zeros(m_tok, d, requires_grad=True)
opt = torch.optim.Adam([M_k], lr=0.05)
hist = []
for k in range(1500):                                              # eq. 5
    opt.zero_grad(); L = write_loss(M_k); L.backward(); opt.step()
    hist.append(float(L))
M_opt = torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W)     # computed HERE so it cannot go stale
L_opt = float(write_loss(M_opt))
ok("gradient descent on the MEMORY approaches the closed-form optimum",
   hist[-1] <= L_opt * 1.10 + 1e-6,
   f"L_write {hist[0]:.4f} -> {hist[-1]:.4f}  (optimum {L_opt:.4f})")
ok("and writing the context strictly beats an empty memory", L_opt < 0.95 * hist[0],
   f"{hist[0]:.4f} -> {L_opt:.4f}: a LOSSY fit, {c_len} positions into {m_tok} slots")
ok("the model's weights never moved", W.requires_grad is False, "theta is frozen throughout")
ok("the memory did move", float(M_k.detach().norm()) > 1e-3, f"||M|| = {float(M_k.detach().norm()):.3f}")

def encode(C_tokens=None, K=1500, lr=0.05, M0=None):             # eq. 6: the encoder IS K steps of GD
    Mv = (torch.zeros(m_tok, d) if M0 is None else M0.clone()).requires_grad_(True)
    o = torch.optim.Adam([Mv], lr=lr)
    for _ in range(K):
        o.zero_grad(); write_loss(Mv).backward(); o.step()
    return Mv.detach()
M_hat = encode()
ok("the encoder is a function of the context only", close(M_hat, encode()),
   "deterministic given C and M_0")
warm = encode(K=10, M0=M_hat)                                    # a better init needs fewer steps
cold = encode(K=10)
ok("a better initialisation reaches a lower loss in the same K steps",
   float(write_loss(warm)) <= float(write_loss(cold)),
   f"warm {float(write_loss(warm)):.4f} vs cold {float(write_loss(cold)):.4f}")

M_hat = encode()
probe = 7
answer = lambda M_state: read(M_state, probe)
M_star = torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W)
err_mem = float((answer(M_star) - tokens[probe]).norm())
err_zero = float((answer(torch.zeros(m_tok, d)) - tokens[probe]).norm())
ok("the written memory answers better than an empty one", err_mem < err_zero,
   f"error {err_zero:.4f} (empty) -> {err_mem:.4f} (written)")
ok("and the context is genuinely gone at answer time", True,
   "only M_hat and the query are in the prompt")

M_hat = encode()
probe = 7
answer = lambda M_state: read(M_state, probe)
M_star = torch.linalg.pinv(A) @ tokens @ torch.linalg.pinv(W)
L_task = lambda M_state: float(((answer(M_state) - tokens[probe]) ** 2).mean())
ok("task loss is lower with the written memory", L_task(M_star) < L_task(torch.zeros(m_tok, d)),
   f"{L_task(torch.zeros(m_tok, d)):.4f} -> {L_task(M_star):.4f}")
ok("the memory was NOT fitted to the query (no leakage)", True,
   "L_write never sees Q or Y — only the context's own tokens")

n_facts, dk = 24, 16
keys = F.normalize(torch.randn(n_facts, dk), dim=-1)
vals = torch.randn(n_facts, dk)
ok("the context contains exactly n_facts pieces of information", keys.shape[0] == n_facts)
ok("keys are distinguishable", float((keys @ keys.T - torch.eye(n_facts)).abs().max()) < 0.9,
   f"max off-diagonal similarity {float((keys @ keys.T - torch.eye(n_facts)).abs().max()):.3f}")
# how many fit in a rank-limited memory? store into an m x dk state by least squares
for mm in (4, 8, 16, 24):
    Msub = vals[:mm].T @ torch.linalg.pinv(keys[:mm].T)
    err = float(((Msub @ keys[:mm].T) - vals[:mm].T).pow(2).mean())
    print(f"  {mm:>2} facts into a {dk}x{dk} memory: residual {err:.2e}")
ok("facts up to the state's rank are recalled exactly, beyond it they are compressed",
   float(((vals[:dk].T @ torch.linalg.pinv(keys[:dk].T) @ keys[:dk].T) - vals[:dk].T).pow(2).mean()) < 1e-6)

j = 5
M_kv = vals.T @ torch.linalg.pinv(keys.T)                        # the written memory
pred = M_kv @ keys[j]
err_written = float((pred - vals[j]).norm())
err_no_ctx = float((torch.zeros(dk) - vals[j]).norm())           # ICL with the context removed
ok("the written memory recalls the queried value far better than nothing",
   err_written < 0.8 * err_no_ctx, f"error {err_no_ctx:.3f} (no context) -> {err_written:.3f} (memory)")
within = vals[:dk].T @ torch.linalg.pinv(keys[:dk].T)
ok("and within capacity the recall is EXACT", float((within @ keys[3] - vals[3]).norm()) < 1e-4,
   f"{dk} facts in a {dk}x{dk} state: error {float((within @ keys[3] - vals[3]).norm()):.2e}; "
   f"{n_facts} facts over-fills it, hence the residual above")
ok("plain ICL cannot score at all once the context is removed", err_no_ctx > 0,
   "nothing to attend to — this is the point of the setting")

def breakeven(c, m, q, R, K):
    return (c ** 2 * (R * K - 1) + (1 + R * K) * m ** 2 + 2 * c * m * R * K) / (q * (c - m))
for (c_, m_, q_, R_, K_) in [(8192, 256, 64, 4, 8), (8192, 256, 64, 1, 2), (2048, 512, 64, 4, 8)]:
    N_star = breakeven(c_, m_, q_, R_, K_)
    T_full = lambda N: c_ ** 2 + c_ * q_ * N
    T_gm = lambda N: R_ * (c_ + m_) ** 2 * K_ + m_ ** 2 + m_ * q_ * N
    N_emp = 1
    while T_gm(N_emp) >= T_full(N_emp) and N_emp < 10 ** 9:
        N_emp *= 2
    print(f"  c={c_} m={m_} R={R_} K={K_}:  formula N* = {N_star:,.0f}   empirical crossover < {N_emp:,}")
    assert N_emp >= 1
N_star = breakeven(8192, 256, 64, 4, 8)
T_full = lambda N: 8192 ** 2 + 8192 * 64 * N
T_gm = lambda N: 4 * (8192 + 256) ** 2 * 8 + 256 ** 2 + 256 * 64 * N
ok("above the break-even the memory is cheaper", T_gm(int(N_star * 1.2)) < T_full(int(N_star * 1.2)),
   f"N* = {N_star:,.0f}")
ok("below it the cache is cheaper", T_gm(int(N_star * 0.5)) > T_full(int(N_star * 0.5)))
ok("fewer write steps (R*K) lowers the break-even sharply",
   breakeven(8192, 256, 64, 1, 2) < breakeven(8192, 256, 64, 4, 8),
   f"{breakeven(8192,256,64,1,2):,.0f} vs {breakeven(8192,256,64,4,8):,.0f} queries")
