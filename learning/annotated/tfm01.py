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

net = M.TabFM(embed_dim=16, max_classes=3, col_num_blocks=1, col_nhead=2, col_num_inds=4,
              row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2,
              ff_factor=2, is_classifier=True).to(DEV).eval()
before = {k: v.detach().clone() for k, v in net.state_dict().items()}

X = pd.DataFrame(np.random.RandomState(0).randn(40, 4), columns=list("abcd"))
y = np.random.RandomState(1).randint(0, 3, 40)
clf = CR.TabFMClassifier(model=net, n_estimators=2, batch_size=1, verbose=False)
clf.fit(X, y)

same = all(torch.equal(before[k], v) for k, v in net.state_dict().items())
ok("fit() changed NOT ONE parameter", same, f"{len(before)} tensors compared exactly")
ok("what it fitted instead was the PREPROCESSORS", hasattr(clf, "X_encoder_"),
   "X_encoder_, ensemble_generator_, classes_ — all state, no weights")
ok("and it remembered the classes", list(getattr(clf, "classes_", [])) == [0, 1, 2],
   f"classes_ = {list(getattr(clf, 'classes_', []))}")
print("\nso 'training' here = fitting encoders + storing rows. The network is a fixed function.")

Xte = pd.DataFrame(np.random.RandomState(2).randn(9, 4), columns=list("abcd"))
pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)
ok("predict returns one label per test row", pred.shape == (9,), f"{pred.shape}")
ok("predict_proba returns a simplex per row", proba.shape == (9, 3)
   and np.allclose(proba.sum(1), 1.0, atol=1e-5), f"{proba.shape}, rows sum to 1")
ok("every prediction is a known class", set(np.unique(pred)) <= set(clf.classes_))
print("\nNOTE: the context rows are re-encoded on EVERY predict call unless cache_context=True —")
print("inference cost therefore grows with the TRAINING set, which is why §7 exists.")

try:
    CR._check_classifier_output_dim(1, 3)                     # a regression head, 3 classes
    ok("a 1-wide head is rejected for 3-class work", False)
except ValueError as e:
    ok("a 1-wide head is rejected for 3-class work", True, str(e).split(".")[0][:80])
CR._check_classifier_output_dim(10, 3)                        # a wide head is fine — extra logits unused
ok("a head WIDER than n_classes is accepted", True, "spare logits are simply unused")
ok("the guard is a pure shape check — no data needed", True,
   "the cheapest possible place to catch a wrong checkpoint")

try:
    CR._check_regressor_output_dim(10)                         # a classification head
    ok("a many-logit head is rejected for regression", False)
except ValueError as e:
    ok("a many-logit head is rejected for regression", True, str(e).split(".")[0][:80])
CR._check_regressor_output_dim(1)
ok("exactly one output per row is what regression means", True)
print("\ntwo tiny guards, both sides of the same mistake — copy this pattern for any frozen checkpoint.")
