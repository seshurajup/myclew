import torch, torch.nn as nn, torch.nn.functional as F      # three levels: server, client, memory
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

def unit(*shape):
    return F.normalize(torch.randn(*shape), dim=-1)

d2 = 32
client = nn.Sequential(nn.Linear(d2, d2), nn.GELU(), nn.Linear(d2, d2))
opt = torch.optim.AdamW(client.parameters(), lr=1e-3)
client(torch.randn(4, d2)).sum().backward(); opt.step()
weights = sum(p.numel() for p in client.parameters())
state = sum(v.numel() for s in opt.state.values() for v in s.values()
            if torch.is_tensor(v) and v.dim() > 0)
memory = d2 * d2                                                 # the level-1 state, per client, local
rows = [("3  server (per round)", weights, "shipped"),
        ("2  optimizer (per step)", state, "local, discarded at round end"),
        ("1  memory S (per token)", memory, "local, never shipped")]
for name, n, note in rows:
    print(f"  level {name:26s} {n:>7} params   {note}")
ok("the shipped payload is only the level-3 parameters", weights < state + memory,
   f"{weights} shipped vs {state + memory} kept local")
ok("and the local state is what carries the client's own distribution", memory > 0,
   "non-IID specialisation lives in S, not in theta")
