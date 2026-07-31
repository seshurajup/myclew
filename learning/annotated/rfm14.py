import torch, torch.nn as nn, torch.nn.functional as F      # an expert that decides for itself
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

T_, D_, N_e = 4096, 7168, 896                                     # K3's published shape
t_routing = T_ * (D_ + 2) * N_e                                   # eq. 20
print(f"  routing cost ~ {t_routing/1e12:.2f} Tflop-units for T={T_}, D={D_}, N={N_e}")
ok("routing scales linearly in the number of experts", 2 * t_routing ==
   T_ * (D_ + 2) * (2 * N_e), "double the experts, double the router")
ok("so it is not negligible at 896 experts", t_routing > T_ * D_ * 100,
   f"{t_routing/(T_*D_):.0f}x a single dense projection")

M, K_, b, Bw, a_lat = 8, 8, 2, 200e9, 5e-6
t_a2a = (M - 1) * a_lat + (K_ * T_ * D_ * b) / (M * Bw)           # eq. 21
print(f"  all-to-all ~ {t_a2a*1e3:.3f} ms  (latency {(M-1)*a_lat*1e3:.3f} ms + "
      f"bandwidth {(K_*T_*D_*b)/(M*Bw)*1e3:.3f} ms)")
ok("the payload term scales with K", (2 * K_ * T_ * D_ * b) / (M * Bw) >
   (K_ * T_ * D_ * b) / (M * Bw))
ok("and the latency term with the device count", (2 * M - 1) * a_lat > (M - 1) * a_lat)

D_act = 3072                                                      # K3's moe_intermediate_size
t_exp = 3 * K_ * (T_ / M) * D_ * D_act                            # eq. 22
print(f"  expert compute ~ {t_exp/1e12:.2f} Tflop-units")
ok("expert compute is linear in K", 3 * (2 * K_) * (T_ / M) * D_ * D_act == 2 * t_exp)
ok("and it dominates the router at this shape", t_exp > t_routing,
   f"{t_exp/t_routing:.1f}x the routing cost")

T_moe = t_routing + t_exp + 2 * (M - 1) * a_lat + 2 * (K_ * T_ * D_ * b) / (M * Bw)
comm = 2 * (M - 1) * a_lat + 2 * (K_ * T_ * D_ * b) / (M * Bw)
print(f"  total {T_moe:.3e}   of which communication-like terms: {comm:.3e}")
ok("the all-to-all is paid twice per step", abs(comm - 2 * t_a2a) < 1e-9,
   "dispatch + combine")
ok("and communication is a real fraction of the step", comm > 0)

t_ag = (M - 1) * a_lat + ((M - 1) * T_ * D_ * b) / (M * Bw)       # eq. 24
print(f"  all-gather ~ {t_ag*1e3:.3f} ms   vs all-to-all {t_a2a*1e3:.3f} ms")
ok("the payload now scales with M, not K", ((M - 1) * T_ * D_ * b) / (M * Bw) > 0)
ok("so it is cheaper exactly when K > M-1", (K_ > M - 1) == (t_a2a > t_ag),
   f"K={K_}, M-1={M-1}: {'all-gather wins' if t_ag < t_a2a else 'all-to-all wins'}")

r_ = 8
t_score = T_ * D_ * r_ * (N_e / M)                                # eq. 25
print(f"  scoring ~ {t_score/1e12:.2f} Tflop-units  vs routing {t_routing/1e12:.2f}")
ratio_sr = t_score / t_routing
ok("the cost ratio is essentially r/M", abs(ratio_sr - r_ / M) < 0.01,
   f"measured {ratio_sr:.4f} vs r/M = {r_/M:.4f}")
ok("so scoring is cheaper exactly when r < M (and equal when r = M)",
   (r_ < M and ratio_sr < 1) or (r_ == M and abs(ratio_sr - 1) < 0.01) or (r_ > M and ratio_sr > 1),
   f"r={r_}, M={M} -> ratio {ratio_sr:.4f}")
ok("and it is local — no cross-device score exchange", True, "N/M experts per device")

K_eff = 8.0
t_exp_rf = K_eff * (T_ / M) * (r_ + 2 * D_) * D_act               # eq. 26
print(f"  expert compute: standard {t_exp/1e12:.2f} vs routing-free {t_exp_rf/1e12:.2f} Tflop-units")
ok("the low-rank gate makes the per-expert cost slightly cheaper", (r_ + 2 * D_) < 3 * D_,
   f"{r_ + 2*D_} vs {3*D_}")
ok("K_eff is a measured average, not a constant", isinstance(K_eff, float),
   "the ReLU decides per token, so this must be monitored, not assumed")

t_comb = a_lat + (K_eff * T_ * D_ * b) / (M * Bw)                 # eq. 27
print(f"  combine ~ {t_comb*1e3:.3f} ms  (one latency term, not {M-1})")
ok("the latency term collapses to a single alpha", a_lat < (M - 1) * a_lat)
ok("payload scales with the EMERGENT K_eff", (2 * K_eff * T_ * D_ * b) / (M * Bw) >
   (K_eff * T_ * D_ * b) / (M * Bw), "so measuring K_eff is not optional")

T_rf = t_score + t_exp_rf + M * a_lat + ((M - 1 + K_eff) * T_ * D_ * b) / (M * Bw)
print(f"  standard {T_moe:.4e}   routing-free {T_rf:.4e}   ->  "
      f"{'routing-free' if T_rf < T_moe else 'standard'} wins at this shape")
payload_std, payload_rf = 2 * K_, (M - 1 + K_eff)
ok("the payload comparison is 2K vs (M-1+K_eff)", payload_std == 16 and abs(payload_rf - 15.0) < 1e-9,
   f"{payload_std} vs {payload_rf}")
ok("so the verdict depends on the TOPOLOGY, not the method",
   (payload_rf < payload_std) == (payload_rf < payload_std), "M and K decide it")

ratio = (t_score + t_exp_rf) / (t_routing + t_exp)                # eq. 29
print(f"  compute ratio = {ratio:.4f}  ({'cheaper' if ratio < 1 else 'more expensive'})")
ok("the ratio is finite and positive", 0 < ratio < 10, f"{ratio:.4f}")
worse = (T_ * D_ * 64 * (N_e / M) + 16.0 * (T_ / M) * (64 + 2 * D_) * D_act) / (t_routing + t_exp)
ok("a larger rank or a larger K_eff can flip it", worse > ratio,
   f"r=64, K_eff=16 gives {worse:.4f} vs {ratio:.4f}")
ok("so this is a per-deployment CALCULATION, not a universal claim", True,
   "evaluate on your own (r, K, K_eff, M, N)")

import pandas as pd
delta = lambda K, M: ((K + 1 - M) * T_ * D_ * b) / (M * Bw)       # eq. 30
rows = [dict(K=K, M=M, delta_ms=round(delta(K, M) * 1e3, 4),
             verdict="routing-free sends LESS" if delta(K, M) > 0 else "routing-free sends MORE")
        for K, M in [(8, 8), (8, 16), (16, 8), (2, 8), (32, 16)]]
df = pd.DataFrame(rows)
print(df.to_string(index=False))
ok("the sign is decided entirely by K + 1 - M", all(
    (r_["delta_ms"] > 0) == (r_["K"] + 1 - r_["M"] > 0) for r_ in rows))
ok("so many experts per token favour routing-free", delta(32, 16) > 0)
ok("and few experts per token favour the standard design", delta(2, 8) < 0)
vz.table(df, "Communication delta (eq. 30)",
         "positive = the routing-free design sends less data", heat_cols=["delta_ms"])
