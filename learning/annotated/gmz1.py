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

import pandas as pd
def breakeven(c, m, q, R, K):
    return (c ** 2 * (R * K - 1) + (1 + R * K) * m ** 2 + 2 * c * m * R * K) / (q * (c - m))
rows = []
for c_ in (2048, 8192, 32768):
    for RK in ((1, 2), (4, 8)):
        rows.append(dict(context=c_, R=RK[0], K=RK[1], memory=256,
                         breakeven_queries=round(breakeven(c_, 256, 64, RK[0], RK[1]))))
df = pd.DataFrame(rows)
print(df.to_string(index=False))
ok("a longer context RAISES the break-even (writing costs more up front)",
   df[df.R == 4].breakeven_queries.is_monotonic_increasing,
   "so writing pays off soonest for medium contexts asked many questions")
ok("cheaper writes (small R*K) always lower the break-even",
   bool((df[df.R == 1].breakeven_queries.values < df[df.R == 4].breakeven_queries.values).all()))
vz.table(df, "GradMem break-even (eq. 13)", "queries needed before writing beats caching",
         heat_cols=["breakeven_queries"], lower_better=["breakeven_queries"])
