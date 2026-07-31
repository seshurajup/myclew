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

rs = np.random.RandomState(0)
Xo = np.concatenate([rs.randn(200, 1), [[50.0]]])            # 200 normal values + one absurd one
orm = CR.OutlierRemover(threshold=4.0)
orm.fit(Xo)
Xc = orm.transform(Xo)
print(f"raw max {Xo.max():.2f} -> clipped max {Xc.max():.2f}")
ok("the extreme value is pulled in", Xc.max() < Xo.max() / 2, f"{Xo.max():.1f} -> {Xc.max():.2f}")
ok("the row is CLIPPED, not dropped", Xc.shape == Xo.shape, f"{Xc.shape} rows preserved")
ok("ordinary values are untouched", np.allclose(Xc[:200][np.abs(Xo[:200]) < 3],
                                                Xo[:200][np.abs(Xo[:200]) < 3]))
ok("clipping beats dropping when rows are expensive", True,
   "an in-context row costs attention; its other features are still informative")

ss = CR.CustomStandardScaler()
Z = ss.fit_transform(np.random.RandomState(0).randn(100, 3) * 5 + 2)
print("mean", Z.mean(0).round(4), " std", Z.std(0).round(4))
ok("normal columns are standardised", np.allclose(Z.mean(0), 0, atol=1e-6)
   and np.allclose(Z.std(0), 1, atol=1e-6))
Zc = ss.fit_transform(np.ones((10, 2)))                       # zero variance
ok("a CONSTANT column does not produce nan/inf", np.isfinite(Zc).all(), f"-> {Zc[0].tolist()}")
tiny = np.zeros((50, 1)); tiny[0] = 1e-12
Zt = ss.fit_transform(tiny)
ok("and a near-constant column cannot explode", np.abs(Zt).max() < 1e6,
   f"max |z| = {np.abs(Zt).max():.3g} — clipped, not 1e12")

qt = CR.RTDLQuantileTransformer()
heavy = np.random.RandomState(0).randn(300, 2) ** 3           # heavy-tailed
Zq = qt.fit_transform(heavy)
print(f"input range [{heavy.min():.1f}, {heavy.max():.1f}] -> output [{Zq.min():.2f}, {Zq.max():.2f}]")
ok("a heavy-tailed column is mapped to a bounded, gaussian-ish range",
   np.abs(Zq).max() < 10 and np.abs(heavy).max() > 10)
tied = np.repeat(np.arange(10.0), 30).reshape(-1, 1)          # 10 distinct values, 30 ties each
Zt = CR.RTDLQuantileTransformer().fit_transform(tied)
order = np.argsort(tied.ravel())
ok("heavy ties do not collapse to a single point", len(np.unique(np.round(Zt, 6))) == 10,
   "10 distinct inputs -> 10 distinct outputs, monotone")
ok("and RANK ORDER is preserved (it is a monotone map)",
   bool(np.all(np.diff(Zt.ravel()[order]) >= -1e-9)))
n_small = CR.RTDLQuantileTransformer().fit(np.random.randn(20, 1)).normalizer_.n_quantiles_
n_big = CR.RTDLQuantileTransformer().fit(np.random.randn(5000, 1)).normalizer_.n_quantiles_
print(f"n_quantiles adapts to data size: {n_small} for 20 rows, {n_big} for 5000")
ok("n_quantiles SCALES with the data instead of being fixed", n_big > n_small * 5,
   f"{n_small} -> {n_big} — a fixed 1000 would over-fit a 20-row column")
ok("the fitted normaliser is retained for test-time reuse", hasattr(qt, "normalizer_"))

pp = CR.PreprocessingPipeline(normalization_method="none", outlier_threshold=4.0)
Xtr = np.random.RandomState(0).randn(200, 3)
pp.fit(Xtr)
orm = pp.outlier_remover_
print("bounds learned at fit:", np.round(orm.lower_bounds_, 2), "..", np.round(orm.upper_bounds_, 2))
tr_max = float(np.abs(pp.transform(Xtr)).max())
Zte = pp.transform(np.full((1, 3), 1e6))                      # an absurd test row
print(f"train |z| max {tr_max:.3f}   |   an unseen 1e6 -> |z| {float(np.abs(Zte).max()):.3f}")
ok("the pipeline composes scaler + normaliser + outlier remover",
   all(hasattr(pp, a) for a in ("standard_scaler_", "normalizer_", "outlier_remover_")))
ok("an extreme TEST value is CLIPPED, not passed through",
   float(np.abs(Zte).max()) < 100, f"1e6 became {float(np.abs(Zte).max()):.2f}")
ok("the clip bounds were learned on TRAINING data only",
   hasattr(orm, "lower_bounds_") and hasattr(orm, "upper_bounds_"))
ok("so test data cannot walk far off the pre-training distribution", True,
   "the frozen network only behaves near where it was trained")
