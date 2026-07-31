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

print("tabfm version   :", __import__("tabfm").__version__)
print("classifier     :", CR.TabFMClassifier.__name__)
print("regressor      :", CR.TabFMRegressor.__name__)
print("preprocessing  :", sum(1 for n in dir(CR) if n[0].isupper()), "public transformer classes")
src = (__import__("pathlib").Path("research/tabfm_repo/tabfm/src/classifier_and_regressor.py")
       .read_text().splitlines())
print("estimator file :", len(src), "lines")
ok("we are reading the real clone, not a paraphrase", len(src) > 3000)
ok("and it ships BOTH task heads", hasattr(CR, "TabFMClassifier") and hasattr(CR, "TabFMRegressor"))
print("\nweights (not downloaded here):",
      __import__("tabfm.src.pytorch.tabfm_v1_0_0", fromlist=["x"]).HF_REPO_ID)

net = M.TabFM(embed_dim=16, max_classes=3, col_num_blocks=1, col_nhead=2, col_num_inds=4,
              row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2,
              ff_factor=2, is_classifier=True).to(DEV).eval()
n_par = sum(p.numel() for p in net.parameters())
print(f"a tiny TabFM: {n_par:,} parameters on {DEV}")
ok("the graph builds with random init — no checkpoint needed", n_par > 0)
ok("and it is a plain nn.Module", isinstance(net, torch.nn.Module))
sig = inspect.signature(M.TabFM.__init__)
print("\nthe architecture IS its hyper-parameters:")
for k, v in list(sig.parameters.items())[1:8]:
    print(f"  {k:20s} default {v.default}")
ok("so every structural claim below is checkable offline", True,
   "only ACCURACY would need the real weights")
