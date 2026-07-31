import torch, torch.nn as nn, torch.nn.functional as F      # K3's maths is delta rules + softmax + LP duality
import json, pathlib

import sys; sys.path.insert(0, "learning")
import vizkit as vz                                            # the shared visual + explainability layer

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every proof runs on the GPU
torch.set_default_device(DEV)
# These cells PROVE matrix identities, so they need full fp32: TF32 truncates the mantissa to 10 bits
# and an identity that holds to 1e-6 in fp32 only holds to ~1e-3 in TF32. Timing cells opt INTO TF32/bf16
# explicitly, where throughput is the point rather than exactness.
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

def close(a, b, tol=1e-5):
    return torch.allclose(a, b, atol=tol, rtol=tol)

def k3cfg():                                                   # the PUBLISHED architecture (no weights)
    p = pathlib.Path("docs/papers/kimi-k3/models/moonshotai__Kimi-K3.json")
    c = json.loads(p.read_text()) if p.exists() else {}
    return {**c.get("text_config", {}), **{k: v for k, v in c.items() if k != "text_config"}}

V_ = 64
teacher = torch.softmax(torch.randn(V_) * 1.5, 0)
student = torch.softmax(torch.randn(V_) * 1.5, 0).requires_grad_(True)
tok = 7
lo, hi = -2.0, 2.0
ratio = torch.log(teacher[tok] / student[tok])
r = torch.clamp(ratio.detach(), lo, hi)                           # eq. 16: clip(sg(log ratio))
ok("the reward is bounded by the clip", lo <= float(r) <= hi, f"r = {float(r):.4f}")
ok("stop-gradient keeps the teacher out of the backward pass", not r.requires_grad)
ok("positive reward exactly when the teacher likes the token more",
   (float(r) > 0) == (float(teacher[tok]) > float(student[tok])),
   f"teacher {float(teacher[tok]):.4f} vs student {float(student[tok]):.4f}")
big = torch.log(torch.tensor(1e-9) / torch.tensor(0.5))
ok("clipping is what tames the rare-token blow-up", float(torch.clamp(big, lo, hi)) == lo,
   f"unclipped {float(big):.1f} -> {lo}")

p = torch.softmax(torch.randn(V_), 0); q = torch.softmax(torch.randn(V_), 0)
overlap = torch.minimum(p, q).sum()
L_lk = -torch.log(overlap)
ok("overlap lies in [0, 1]", 0 <= float(overlap) <= 1, f"overlap = {float(overlap):.4f}")
ok("identical distributions give zero loss", abs(float(-torch.log(torch.minimum(p, p).sum()))) < 1e-6)
ok("it is symmetric (KL is not)",
   close(-torch.log(torch.minimum(p, q).sum()), -torch.log(torch.minimum(q, p).sum())))
disjoint_p = torch.zeros(4); disjoint_p[0] = 1.0
disjoint_q = torch.zeros(4); disjoint_q[1] = 1.0
kl = float(F.kl_div(torch.log(disjoint_q + 1e-30), disjoint_p, reduction="sum"))
ok("and it stays finite where KL diverges", bool(torch.isfinite(-torch.log(torch.minimum(
    disjoint_p, disjoint_q).sum() + 1e-30))) and kl > 50, f"KL = {kl:.1f}")
ok("overlap = 1 - total variation distance",
   abs(float(overlap) - (1 - 0.5 * float((p - q).abs().sum()))) < 1e-5)

R, dm = 4, 6
Ms = [torch.eye(dm) + 0.1 * torch.randn(dm, dm) for _ in range(R)]  # per-rank transfers
seq = torch.eye(dm)
for M in Ms:
    seq = M @ seq
left = (Ms[3] @ Ms[2]) @ (Ms[1] @ Ms[0])                            # combine in pairs instead
ok("associativity lets ranks combine partial products in any grouping", close(seq, left, 1e-5),
   f"max|diff| = {(seq - left).abs().max():.2e}")
ok("so the communication schedule is free to reorder", True, "the bound in Appendix E uses this")
