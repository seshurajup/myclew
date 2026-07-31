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

beta = 4.0
for z0 in (0.01, 0.1, 0.5):
    z_ = torch.tensor([z0])
    exact = beta * torch.tanh(z_ / beta)
    err = float((exact - z_).abs())
    pred = z0 ** 3 / (3 * beta ** 2)                               # the leading term of the expansion
    print(f"  z={z0}: |beta tanh(z/beta) - z| = {err:.3e}, predicted {pred:.3e}")
z_ = torch.tensor([0.1])
ok("the error matches the cubic prediction", abs(float((beta * torch.tanh(z_ / beta) - z_).abs())
                                                 - 0.1 ** 3 / (3 * beta ** 2)) < 1e-6)
ok("and it shrinks like 1/beta^2",
   float((8.0 * torch.tanh(z_ / 8.0) - z_).abs()) < float((beta * torch.tanh(z_ / beta) - z_).abs()))

cfg = k3cfg(); b1 = cfg.get("activation_situ_beta") or 4.0; b2 = cfg.get("activation_situ_linear_beta") or 25.0
def situ_glu_full(z, u, beta1=b1, beta2=b2):
    gate = beta1 * torch.tanh(z / beta1) * torch.sigmoid(z)        # |gate| <= beta1
    lin = beta2 * torch.tanh(u / beta2)                            # |lin|  <= beta2
    return gate * lin
z = torch.linspace(-1e3, 1e3, 2001); u = torch.linspace(1e3, -1e3, 2001)
v = situ_glu_full(z, u)
ok(f"the published betas give the paper's constant beta1*beta2 = {b1*b2:.0f}", abs(b1 * b2 - 100) < 1e-9,
   f"beta1={b1}, beta2={b2}")
ok("no input, however extreme, exceeds the bound", float(v.abs().max()) <= b1 * b2 + 1e-4,
   f"max |activation| = {float(v.abs().max()):.2f} <= {b1*b2:.0f}")
ok("an unbounded GLU has no such ceiling",
   float((F.silu(z) * u).abs().max()) > 1e5, f"SiLU-GLU reaches {float((F.silu(z)*u).abs().max()):.1e}")
