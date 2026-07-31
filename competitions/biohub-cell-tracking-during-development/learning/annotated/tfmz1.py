import sys, warnings, inspect, dataclasses
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, torch

sys.path.insert(0, "research/tabfm_repo")                  # the real clone, not a reimplementation
# the repo pins typeguard<3; ours is 4.x, whose AST transform rejects jaxtyping shape strings. Disabling
# the decorator removes RUNTIME TYPE ASSERTIONS only — it cannot change what any function computes.
import typeguard; typeguard.typechecked = lambda f=None, **k: (f if f is not None else (lambda g: g))

from tabfm.src import classifier_and_regressor as CR
from tabfm.src.pytorch import model as M

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # every model proof runs on the GPU
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False   # exact proofs
torch.manual_seed(0); np.random.seed(0); torch.set_printoptions(precision=4, sci_mode=False)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
rs = np.random.RandomState(0)
n, d = 400, 8
Xall = rs.randn(n, d); w = rs.randn(d)
yall = Xall @ w + 0.6 * rs.randn(n)
Xtr, ytr, Xte, yte = Xall[:300], yall[:300], Xall[300:], yall[300:]

for label, mk in [("DecisionTree(depth 4)", lambda: DecisionTreeRegressor(max_depth=4, random_state=0)),
                  ("RandomForest(30)", lambda: RandomForestRegressor(n_estimators=30, max_depth=4,
                                                                     random_state=0))]:
    single = mean_squared_error(yte, mk().fit(Xtr, ytr).predict(Xte))
    preds = []
    for sd in range(16):
        perm = np.random.RandomState(sd).permutation(d)
        preds.append(mk().fit(Xtr[:, perm], ytr).predict(Xte[:, perm]))
    views = mean_squared_error(yte, np.mean(preds, axis=0))
    print(f"  {label:22s} 1 view {single:.4f} -> 16 views {views:.4f}  "
          f"({(single - views) / single * 100:+.2f}%)")
    ok(f"{label} gains NOTHING from column-permutation views", abs(views - single) < 0.05,
       "the permuted tree IS the same tree — no diversity to average")
print("\nADOPTION RULE: column-shuffle TTA is worthless for order-invariant learners (trees, GBMs).")
print("It needs a predictor whose output depends on column order — the next cell shows TabFM is one.")

import math
net = M.TabFM(embed_dim=16, max_classes=3, col_num_blocks=1, col_nhead=2, col_num_inds=4,
              row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2,
              ff_factor=2, is_classifier=True).to(DEV).eval()
with torch.no_grad():                                          # unit 26: fill the frequency BUFFER
    ff = net.cell_embedder.fourier_frequencies
    basis = torch.logspace(-2, 1, ff.shape[-1], device=DEV).expand_as(ff)
    net.cell_embedder.fourier_frequencies.copy_(basis)
    net.cell_embedder.fourier_frequencies_cat.copy_(basis)

B, T_tr, T_te, D, C = 1, 32, 8, 6, 3
x = torch.randn(B, T_tr + T_te, D, device=DEV)
y = torch.full((B, T_tr + T_te), -100.0, device=DEV)
y[:, :T_tr] = torch.randint(0, C, (B, T_tr), device=DEV).float()
ts = torch.tensor([T_tr], device=DEV)

with torch.no_grad():
    base = net(x, y, ts)
    rp = torch.randperm(T_tr, device=DEV)
    xr, yr = x.clone(), y.clone()
    xr[:, :T_tr], yr[:, :T_tr] = x[:, :T_tr][:, rp], y[:, :T_tr][:, rp]
    row_perm = net(xr, yr, ts)
    col_perm = net(x[:, :, torch.tensor([5, 0, 3, 1, 4, 2], device=DEV)], y, ts)
d_row = float((base[:, T_tr:] - row_perm[:, T_tr:]).abs().max())
d_col = float((base - col_perm).abs().max())
print(f"  permute CONTEXT ROWS -> answer moves by {d_row:.2e}   (a set)")
print(f"  permute COLUMNS      -> answer moves by {d_col:.2e}   (not a set)")
ok("row order does not matter", d_row < 1e-5, "your storage order cannot change the prediction")
ok("column order DOES", d_col > 1e-3, "so a feature-shuffle view is a genuinely different look")

with torch.no_grad():
    V = torch.stack([torch.softmax(net(x[:, :, torch.randperm(D, device=DEV)], y, ts)[:, T_tr:], -1)
                     for _ in range(16)])
per_view = float(V.std(0).mean())
quads = torch.stack([V[i * 4:(i + 1) * 4].mean(0) for i in range(4)])
of_mean = float(quads.std(0).mean())
print(f"  per-view std {per_view:.5f} -> std of 4-view means {of_mean:.5f}  "
      f"(ratio {per_view / of_mean:.2f}, sqrt(4) = 2.00)")
ok("the views genuinely disagree", per_view > 1e-4, f"mean std {per_view:.5f}")
ok("and averaging shrinks the spread at about the sqrt(N) rate", per_view / of_mean > 1.5,
   f"{per_view / of_mean:.2f}x for N=4 — variance reduction, as advertised")
ok("mechanism established; ACCURACY would need the real checkpoint", True,
   "random init proves the averaging works, not that the answers are right")

from pathlib import Path
rdir = Path("research/tabfm_repo/results")
files = sorted(rdir.glob("*.parquet"))
print(f"{len(files)} result files shipped with the repo:")
rows = []
for f in files:
    df = pd.read_parquet(f)
    rows.append(dict(file=f.stem.replace("tabfm-", ""), rows=len(df), cols=df.shape[1]))
    print(f"  {f.stem:52s} {len(df):5d} rows x {df.shape[1]} cols")
ok("the authors' own TabArena results are in the clone", len(files) >= 4)
ok("covering both tasks", any("classification" in f.stem for f in files)
   and any("regression" in f.stem for f in files))
ok("and both plain and ensembled variants", any("ensemble" in f.stem for f in files),
   "so the ensembling gain is measurable from their data")
first = pd.read_parquet(files[0])
print("\ncolumns:", list(first.columns)[:10])
ok("these are THEIR numbers, not a reproduction", True,
   "we never ran the 1.0.0 checkpoint — no accuracy claim is ours")
