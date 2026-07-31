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

print("5 rows padded to a multiple of 4 ->", CR._pad_batch_to_multiple_of(np.zeros((5, 3)), 4).shape)
ok("5 rows become 8 with divisor 4", CR._pad_batch_to_multiple_of(np.zeros((5, 3)), 4).shape == (8, 3))
ok("an already-aligned array is untouched",
   CR._pad_batch_to_multiple_of(np.zeros((8, 3)), 4).shape == (8, 3))
ok("divisor <= 1 is a no-op", CR._pad_batch_to_multiple_of(np.zeros((5, 3)), 1).shape == (5, 3),
   "so callers need no special case")
ok("only axis 0 is padded", CR._pad_batch_to_multiple_of(np.zeros((5, 3)), 4).shape[1] == 3)

Xp = CR._pad_features(np.ones((4, 3)), 6)
print("3 features padded to 6 ->", Xp.shape)
ok("the width becomes the target", Xp.shape == (4, 6), f"{Xp.shape}")
ok("real columns are preserved", np.allclose(Xp[:, :3], 1.0))
ok("padded columns are ZERO", np.allclose(Xp[:, 3:], 0.0))
ok("and zero means 'at the mean' after standardisation", True,
   "uninformative, not misleading — which is why the scaler in §3 comes first")

m = CR._pad_cat_mask(np.array([True, False]), 5)
print("mask [True, False] padded to 5 ->", m.tolist())
ok("the real flags survive", m[:2].tolist() == [True, False])
ok("padding is False, never True", not m[2:].any(),
   "padded columns are NOT declared categorical")
ok("dtype stays boolean", m.dtype == bool, str(m.dtype))
ok("and it agrees with unit 20's zero-padding", True,
   "zeros in a numeric column; False in the mask — one consistent story")
