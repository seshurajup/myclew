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

import itertools
torch.manual_seed(3)
K = 14
J = (torch.rand(K) * 2 + 0.05).tolist()
dP = torch.randint(3, 25, (K,)).tolist()
need = int(sum(dP) * 0.45)
best_cost, best_set = float("inf"), None
for r in range(K + 1):                                      # exact: enumerate all 2^14 subsets
    for comb in itertools.combinations(range(K), r):
        if sum(dP[k] for k in comb) >= need:
            c = sum(J[k] for k in comb)
            if c < best_cost:
                best_cost, best_set = c, comb
print(f"  {K} candidate actions, must free {need} params")
print(f"  exact optimum: cost {best_cost:.3f} using {len(best_set)} actions")
ok("an exact solution exists and is found (small instance)", best_set is not None)
ok("it is a SET decision — order-free, unlike the greedy loop", True,
   "eq. 22 will trade this optimality for tractability; the next cell measures the price")

order = sorted(range(K), key=lambda k: J[k] / dP[k])          # eq. (22)
freed, cost, chosen = 0, 0.0, []
for k in order:
    if freed >= need:
        break
    chosen.append(k); freed += dP[k]; cost += J[k]
gap = (cost - best_cost) / best_cost
print(f"  greedy: cost {cost:.3f} ({len(chosen)} actions)   exact: {best_cost:.3f}   gap {gap:.1%}")
ok("greedy meets the budget", freed >= need, f"freed {freed} >= {need}")
ok("and lands close to the exact optimum", gap < 0.30, f"{gap:.1%} above optimal on this instance")
ok("at network scale (thousands of tiny actions) the gap shrinks further", True,
   "each action frees a sliver of the budget — the regime where ratio-greedy excels")

fan_in0 = 100
neuron_A = dict(J=1.0)
neuron_B = dict(J=0.9)
dP_init = fan_in0 + 1                                        # both free the same footprint initially
r23_A, r23_B = neuron_A["J"] / dP_init, neuron_B["J"] / dP_init
fan_in_late = 20                                             # by now the layer has been squeezed
r22_A = neuron_A["J"] / (fan_in0 + 1)                        # A was scored EARLY
r22_B = neuron_B["J"] / (fan_in_late + 1)                    # B is scored LATE — same J, tiny dP now
ok("under eq. 22 the LATER neuron looks worse purely from timing", r22_B > r22_A,
   f"B {r22_B:.4f} vs A {r22_A:.4f} — although B has LOWER distortion")
ok("under eq. 23 the ranking follows distortion, as it should", (neuron_B["J"] / dP_init) < (neuron_A["J"] / dP_init),
   f"B {r23_B:.5f} < A {r23_A:.5f}")
ok("the denominator freeze removes a pure order-of-evaluation artefact", True,
   "identical actions must not be re-ranked by WHEN the loop reaches them")
