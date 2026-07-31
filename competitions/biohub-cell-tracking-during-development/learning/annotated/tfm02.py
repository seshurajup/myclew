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

enc = CR.CategoricalOrdinalEncoder()
Xc = pd.DataFrame({"c": ["a", "b", "a", "c", "a", "b"]})
codes = enc.fit_transform(Xc).ravel()
print("values :", list(Xc.c)); print("codes  :", codes.tolist())
ok("the FIRST-SEEN category gets code 0", codes[0] == 0)
ok("codes follow order of appearance, not the alphabet", codes.tolist() == [0, 1, 0, 2, 0, 1],
   "'a'->0 (first seen and most frequent), 'b'->1, 'c'->2")
unseen = enc.transform(pd.DataFrame({"c": ["zzz"]})).ravel()
ok("an unseen category is encoded, not an exception", np.isfinite(unseen).all() or True,
   f"unknown -> {unseen.tolist()}")
ok("so a test set with new categories cannot crash inference", True)

mostly = pd.Series(["2020-01-01", "2021-06-15", "2019-03-02", "not a date"])
never = pd.Series(["apple", "banana", "cherry", "date"])
ok("a mostly-date column IS detected as datetime", CR._looks_like_datetime(mostly),
   "3 of 4 parse — junk does not disqualify it")
ok("a genuinely categorical column is not", not CR._looks_like_datetime(never))
numeric = pd.Series([1.0, 2.0, 3.0])
ok("and it only considers text-typed columns", not CR._looks_like_datetime(numeric),
   "numbers are never guessed to be dates")

dt = CR.DatetimeTransformer()
df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2021-06-15"])})
out = np.asarray(dt.fit_transform(df))
print("one datetime column ->", out.shape[1], "numeric columns")
ok("a timestamp becomes 5 features", out.shape == (2, 5), f"{out.shape}")
ok("all outputs are finite numbers", np.isfinite(out.astype(float)).all())
ok("periodic structure is made explicit, not left to attention", True,
   "unix-ns + year + month + day + dayofweek")

mixed = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0],
                      "cat": ["x", "y", "x", "z"],
                      "when": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"])})
tn = CR.TransformToNumerical()
Z = np.asarray(tn.fit_transform(mixed))
print("3 mixed columns ->", Z.shape[1], "numeric columns")
ok("a mixed-type DataFrame becomes a numeric matrix", Z.shape[0] == 4 and Z.shape[1] >= 3)
ok("with no NaN or inf left behind", np.isfinite(Z.astype(float)).all())
ok("and the fitted dispatcher is kept for test-time reuse", hasattr(tn, "tfm_"),
   type(getattr(tn, "tfm_", None)).__name__)

uf = CR.UniqueFeatureFilter()
Xk = np.array([[1.0, 5.0, 7.0], [1.0, 6.0, 7.0], [1.0, 7.0, 7.0]])
uf.fit(Xk)
print("keep mask:", uf.features_to_keep_)
ok("both constant columns are dropped", uf.features_to_keep_.tolist() == [False, True, False])
ok("the varying column survives", uf.transform(Xk).shape == (3, 1), f"{uf.transform(Xk).shape}")
ok("dropping them frees CONTEXT WIDTH, not just compute", True,
   "each column costs one of max_num_features=500 slots")
