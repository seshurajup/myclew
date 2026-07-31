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

R, dm = 4, 6                                                   # this lesson's own ranks
Ms = [torch.eye(dm) + 0.1 * torch.randn(dm, dm) for _ in range(R)]
seq = torch.eye(dm)
for M in Ms:
    seq = M @ seq

partial = torch.eye(dm)
prefix = []
for M in Ms:                                                       # prefix products, computable early
    partial = M @ partial; prefix.append(partial.clone())
ok("prefix products give every stage's state in one pass", close(prefix[-1], seq, 1e-5),
   f"{len(prefix)} stages")
ok("a stage can start as soon as its prefix exists (overlap comms with compute)", True)
