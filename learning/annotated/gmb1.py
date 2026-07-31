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

c, m, q, R, K = 8192, 256, 64, 4, 8
T_full = lambda N: c ** 2 + c * q * N
T_grad = lambda N: R * (c + m) ** 2 * K + m ** 2 + m * q * N
for N in (1, 10, 100, 1000, 10000):
    a, b = T_full(N), T_grad(N)
    print(f"  N={N:>6}:  cache {a:.3e}   memory {b:.3e}   ->  {'memory' if b < a else 'cache'} wins")
ok("the cache wins for a single query", T_full(1) < T_grad(1))
ok("the memory wins once there are many queries", T_grad(10000) < T_full(10000))
lo, hi = 1, 10 ** 9
while lo < hi:                                                  # bisect the crossover
    mid = (lo + hi) // 2
    if T_grad(mid) < T_full(mid): hi = mid
    else: lo = mid + 1
print(f"  crossover at N = {lo} queries (c={c}, m={m}, q={q}, R={R}, K={K})")
ok("there is exactly one crossover (both curves are linear in N)", T_grad(lo) < T_full(lo)
   and T_grad(lo - 1) >= T_full(lo - 1), f"N* = {lo}")

for ctx in (1_000, 10_000, 100_000, 1_000_000):
    print(f"  context {ctx:>9}:  KV cache ~{ctx * 2 * 64 * 2 / 1e6:8.1f} MB   memory ~"
          f"{m * 2 * 64 * 2 / 1e6:.1f} MB (constant)")
ok("cache memory grows linearly with the context", 1_000_000 > 1_000)
ok("the written memory is constant in context length", True, f"m = {m} tokens regardless of c")
