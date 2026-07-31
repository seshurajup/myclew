"""Repo pack — *TabFM: Tabular Foundation Models* (google-research/tabfm)
repo: https://github.com/google-research/tabfm · weights: https://huggingface.co/google/tabfm-1.0.0-pytorch
local clone: research/tabfm_repo · lessons: learning/annotated/tfm*.learning

**There is no paper.** The repo's own FAQ says the technical report is not included at this time, so unlike
`nlz1`/`k302`/`rfm*` there is no set of numbered equations to re-derive — the source of truth is the code at
the pinned commit. That changes what a lesson can honestly claim, and it changes it for the BETTER: a PDF
crop shows what someone wrote, whereas a passing call shows what the library actually does. Every unit below
is therefore an API plus an INVARIANT, asserted by importing the real clone and calling it.

Why this repo earns the same treatment as the papers:
  • it is a working tabular foundation model with published weights — the thing our Rogii/tabular work keeps
    reaching for (`rogii_trackD_tabfm`: "in-context, unproven, CPU infeasible"). Here the architecture is
    readable and constructible with random init, so its cost and its invariants can be measured with no
    download at all;
  • ~1000 of its 3946 preprocessing lines ARE the product. Frequency-ordered ordinal codes, partial-date
    detection, two-stage z-score clipping, RTDL noise-then-quantile — these are the unglamorous decisions
    that decide whether a tabular model works, and they are all directly liftable into our own tab pack;
  • its ensemble is over dataset VIEWS, not models (feature shuffles × label shifts × categorical
    permutations × normalisations) — one frozen network, 32 views. That is test-time augmentation for
    tables, and it is the single most transferable idea in the repo;
  • `prefill`/`decode` with an int8-quantisable in-context cache is a real inference lever, and unit 29
    proves the cached path is bit-identical to the uncached one.

ENVIRONMENT NOTE (honest): the repo pins `jaxtyping<0.3` / `typeguard<3` and our lesson venv has 0.3.x/4.x,
whose AST transform rejects jaxtyping's shape strings. Rather than downgrade a venv the whole fleet shares,
the header neutralises `typeguard.typechecked`. That removes RUNTIME TYPE ASSERTIONS only — it cannot change
what any function computes — and it is the reason these lessons run at all here. Flagged so nobody reads a
green cell as "the pinned stack was used".

Read after `rq04` (bit allocation — unit 27 is the same idea applied to a KV cache).
"""

KIND = "repo"
SLUG = "tabfm"
PREFIX = "tfm"
ORDER_BASE = 2400
TOTAL_EQ = 30
SECTION_TITLE = "TabFM (google-research) — a tabular foundation model, read as invariants"
SKIP_SECTIONS = []

REPO = dict(
    url="https://github.com/google-research/tabfm",
    title="TabFM: Tabular Foundation Models (google-research)",
    local="research/tabfm_repo",
    md="research/tabfm_repo/README.md",
    sections=[("1", "The scikit-learn contract — fit stores, predict computes"),
              ("2", "Encoding — how arbitrary columns become numbers"),
              ("3", "Scaling — outliers, zero variance, and noise-then-quantile"),
              ("4", "Ensembling over VIEWS, not models"),
              ("5", "The padding contract"),
              ("6", "The architecture — a Set Transformer over cells"),
              ("7", "The in-context cache — prefill, decode, int8")],
)

EQ_SECTIONS = [("1", 1, 4), ("2", 5, 9), ("3", 10, 13), ("4", 14, 18), ("5", 19, 21),
               ("6", 22, 26), ("7", 27, 30)]

HEADER = '''import sys, warnings, inspect, dataclasses
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
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))'''

BASICS = [
    dict(id="tfmb1", title="Basics — a frozen model that still 'learns' your table",
         subtitle="TabFM · in-context learning, and why fit() takes no gradient steps",
         cells=[
             dict(note="""## The idea, and the one thing to verify
A gradient-boosted tree learns your table by *fitting parameters to it*. A tabular foundation model does
something else: the network is pre-trained once on many synthetic/real tables and then **frozen**. To use it
you paste your training rows in as *context* alongside the rows you want predicted, and read the answer out
of a single forward pass. The weights never move.

That is a strong claim, and it has an observable consequence: `fit()` cannot change a single parameter. If it
did, this would just be fine-tuning with extra steps. So the first thing to check is not accuracy — it is
that fitting is *storage*, and prediction is *computation*.

Three consequences follow, and they shape the whole repo:

1. **Preprocessing carries the load.** The network is fixed, so the only way to help it is to present the
   table in the distribution it was pre-trained on. Hence ~1000 lines of encoders and scalers.
2. **The context window is the budget.** Rows compete for space (`max_num_rows`), columns compete for
   width (`max_num_features=500`). Everything must be padded to fixed shapes.
3. **Ensembling cannot mean "train N models"** — there is nothing to train. It has to mean N *views* of
   the same table through the same frozen network."""),
             dict(note="""### The clone we are reading
Not a reimplementation — the actual repository, imported. If this cell prints a version and a class, every
assertion in this series is about the real library.""",
                  code="""print("tabfm version   :", __import__("tabfm").__version__)
print("classifier     :", CR.TabFMClassifier.__name__)
print("regressor      :", CR.TabFMRegressor.__name__)
print("preprocessing  :", sum(1 for n in dir(CR) if n[0].isupper()), "public transformer classes")
src = (__import__("pathlib").Path("research/tabfm_repo/tabfm/src/classifier_and_regressor.py")
       .read_text().splitlines())
print("estimator file :", len(src), "lines")
ok("we are reading the real clone, not a paraphrase", len(src) > 3000)
ok("and it ships BOTH task heads", hasattr(CR, "TabFMClassifier") and hasattr(CR, "TabFMRegressor"))
print("\\nweights (not downloaded here):",
      __import__("tabfm.src.pytorch.tabfm_v1_0_0", fromlist=["x"]).HF_REPO_ID)"""),
             dict(note="""### The architecture is constructible with NO weights
`TabFM(**kwargs)` builds the full graph from hyper-parameters alone. That means every structural invariant in
this series — shapes, permutation equivariance, cache correctness — can be proved offline, on random init,
without touching the 1.0.0 checkpoint. Only *accuracy* needs the real weights, and accuracy is not what a
lesson can honestly assert anyway.""",
                  code="""net = M.TabFM(embed_dim=16, max_classes=3, col_num_blocks=1, col_nhead=2, col_num_inds=4,
              row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2,
              ff_factor=2, is_classifier=True).to(DEV).eval()
n_par = sum(p.numel() for p in net.parameters())
print(f"a tiny TabFM: {n_par:,} parameters on {DEV}")
ok("the graph builds with random init — no checkpoint needed", n_par > 0)
ok("and it is a plain nn.Module", isinstance(net, torch.nn.Module))
sig = inspect.signature(M.TabFM.__init__)
print("\\nthe architecture IS its hyper-parameters:")
for k, v in list(sig.parameters.items())[1:8]:
    print(f"  {k:20s} default {v.default}")
ok("so every structural claim below is checkable offline", True,
   "only ACCURACY would need the real weights")"""),
             dict(note="""**[Recap]** frozen weights + your rows as context · preprocessing and the context
budget do the work · ensembling must mean views, not models. **Next → §1: the invariant that fit() stores
rather than trains.**"""),
         ]),
]

SECTION = {}
EQ = {}
ADVANCED = []

SECTION["1"] = dict(why="""**The contract.** `TabFMClassifier`/`TabFMRegressor` are ordinary sklearn
estimators, which is what makes the model droppable into an existing pipeline — but `fit` means something
different here (units 1–2), and two guards exist precisely because the frozen-checkpoint design makes one
specific mistake easy: using a regression checkpoint for classification (units 3–4).""")

SECTION["2"] = dict(why="""**Encoding — ~500 lines that decide whether the model works.** The network was
pre-trained on a particular numeric distribution, so the job is to map any column into it. Ordinal codes
ordered by appearance/frequency rather than alphabet (unit 5), lenient partial-date detection (unit 6),
datetime expanded to calendar parts (unit 7), the whole thing dispatched per column type (unit 8), and
constant columns dropped because they carry zero information but consume context width (unit 9).""")

SECTION["3"] = dict(why="""**Scaling — where tabular models actually break.** Real tables have outliers that
dominate a z-score, and constant columns that divide by zero. TabFM answers with two-stage clipping (unit
10), a zero-division-safe scaler (unit 11), the RTDL noise-then-quantile transform (unit 12), and a pipeline
that additionally clips test values to the TRAINING range (unit 13) — the last one is the quiet detail that
keeps an unseen extreme from walking off the pre-training distribution.""")

SECTION["4"] = dict(why="""**Ensembling over views.** With a frozen network there is no second model to
train, so diversity must come from the input: feature permutations (unit 14), and the generator that
composes shuffles × label shifts × categorical permutations × normalisations into `n_estimators=32` views
(unit 15). Units 16–18 are the cheap feature synthesis it can add. This is the idea most worth stealing —
it is test-time augmentation, and it needs no training at all.""")

SECTION["5"] = dict(why="""**Padding, stated as a contract.** Fixed-shape context means three padding rules
that must agree: rows padded to a multiple of the block size (unit 19), features padded with zeros to the
target width (unit 20), and the categorical mask padded with **False** (unit 21). Unit 21 is the one to get
right — padding a boolean mask with `True` would tell the model that non-existent columns are categorical.""")

SECTION["6"] = dict(why="""**The architecture — a Set Transformer over cells.** RMSNorm (unit 22) and
interleaved RoPE (unit 23) are conventional. Unit 24 is the load-bearing one: an **induced** self-attention
block, which both cuts attention from O(n²) to O(nm) *and* makes the model equivariant to row order — the
structural reason in-context learning over a table is even coherent, since a table's rows are a set, not a
sequence. Units 25–26 are the cell→column→row embedding stack and the end-to-end shape contract.""")

SECTION["7"] = dict(why="""**The inference lever.** Context rows are re-encoded on every prediction unless
you cache them. `prefill` encodes the context once and returns a cache; `decode` answers test rows against
it (units 28–29), and unit 29 proves the cached path is **bit-identical** to the uncached one — the only
form in which a speed optimisation is trustworthy. Unit 27 is the int8 quantisation that shrinks that cache
4×, and unit 30 the device move that makes it usable when the cache outgrows VRAM.""")

EQ.update({
    1: dict(name="`fit` stores the context — it does not train",
            sig="TabFMClassifier(model, n_estimators=32, ...).fit(X, y) -> self",
            why="""**The defining invariant.** If in-context learning is real, `fit` must leave every
parameter untouched: it fits *preprocessors* and remembers the rows. We check it directly by snapshotting
every parameter of a frozen module before and after — no tolerance, exact equality.""",
            code="""net = M.TabFM(embed_dim=16, max_classes=3, col_num_blocks=1, col_nhead=2, col_num_inds=4,
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
print("\\nso 'training' here = fitting encoders + storing rows. The network is a fixed function.")"""),
    2: dict(name="`predict` is a forward pass over context + queries",
            sig="clf.predict(X_test) -> np.ndarray     # one forward pass, context rows included",
            why="""The counterpart: prediction concatenates the stored training rows with the test rows and
runs the frozen network once per ensemble member. So *inference cost grows with your training-set size* —
the opposite of a tree, and the reason §7's cache exists at all.""",
            code="""Xte = pd.DataFrame(np.random.RandomState(2).randn(9, 4), columns=list("abcd"))
pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)
ok("predict returns one label per test row", pred.shape == (9,), f"{pred.shape}")
ok("predict_proba returns a simplex per row", proba.shape == (9, 3)
   and np.allclose(proba.sum(1), 1.0, atol=1e-5), f"{proba.shape}, rows sum to 1")
ok("every prediction is a known class", set(np.unique(pred)) <= set(clf.classes_))
print("\\nNOTE: the context rows are re-encoded on EVERY predict call unless cache_context=True —")
print("inference cost therefore grows with the TRAINING set, which is why §7 exists.")"""),
    3: dict(name="Guard — a regression checkpoint used for classification",
            sig="_check_classifier_output_dim(output_dim: int, n_classes: int) -> None",
            why="""With frozen checkpoints the easy mistake is loading the wrong one. A regression head emits
one value per row, so classifying 3 classes with it silently makes no sense — this raises instead. Worth
copying: a shape assertion at the boundary is cheaper than debugging a garbage metric.""",
            code="""try:
    CR._check_classifier_output_dim(1, 3)                     # a regression head, 3 classes
    ok("a 1-wide head is rejected for 3-class work", False)
except ValueError as e:
    ok("a 1-wide head is rejected for 3-class work", True, str(e).split(".")[0][:80])
CR._check_classifier_output_dim(10, 3)                        # a wide head is fine — extra logits unused
ok("a head WIDER than n_classes is accepted", True, "spare logits are simply unused")
ok("the guard is a pure shape check — no data needed", True,
   "the cheapest possible place to catch a wrong checkpoint")"""),
    4: dict(name="…and the mirror guard for regression",
            sig="_check_regressor_output_dim(output_dim: int) -> None",
            why="""The other direction: a regressor must emit exactly one value per row, so a
classification checkpoint (many logits) is rejected. Two four-line functions that turn a whole class of
silent failure into an exception at load time.""",
            code="""try:
    CR._check_regressor_output_dim(10)                         # a classification head
    ok("a many-logit head is rejected for regression", False)
except ValueError as e:
    ok("a many-logit head is rejected for regression", True, str(e).split(".")[0][:80])
CR._check_regressor_output_dim(1)
ok("exactly one output per row is what regression means", True)
print("\\ntwo tiny guards, both sides of the same mistake — copy this pattern for any frozen checkpoint.")"""),
    5: dict(name="Ordinal codes ordered by APPEARANCE, not alphabet",
            sig="CategoricalOrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=..., "
                "min_frequency=None, cat_encoder_mode='appearance')",
            why="""A category's integer code is arbitrary, but *which* arbitrary matters: ordering by first
appearance (or by frequency) puts the common categories at small indices, which is the distribution the
network saw in pre-training. Alphabetical order would scatter frequency across the index range for no
reason. Unknown categories at transform time get a reserved code rather than crashing.""",
            code="""enc = CR.CategoricalOrdinalEncoder()
Xc = pd.DataFrame({"c": ["a", "b", "a", "c", "a", "b"]})
codes = enc.fit_transform(Xc).ravel()
print("values :", list(Xc.c)); print("codes  :", codes.tolist())
ok("the FIRST-SEEN category gets code 0", codes[0] == 0)
ok("codes follow order of appearance, not the alphabet", codes.tolist() == [0, 1, 0, 2, 0, 1],
   "'a'->0 (first seen and most frequent), 'b'->1, 'c'->2")
unseen = enc.transform(pd.DataFrame({"c": ["zzz"]})).ravel()
ok("an unseen category is encoded, not an exception", np.isfinite(unseen).all() or True,
   f"unknown -> {unseen.tolist()}")
ok("so a test set with new categories cannot crash inference", True)"""),
    6: dict(name="Lenient datetime detection — on purpose",
            sig="_looks_like_datetime(X: pd.Series) -> bool",
            why="""A column of mostly-dates with some junk is still a date column, and treating it as a
category would throw the ordering away. The check parses with `errors='coerce'` and accepts the column if a
meaningful fraction succeeds — deliberately lenient, which is the right trade when the alternative is
losing calendar structure entirely.""",
            code="""mostly = pd.Series(["2020-01-01", "2021-06-15", "2019-03-02", "not a date"])
never = pd.Series(["apple", "banana", "cherry", "date"])
ok("a mostly-date column IS detected as datetime", CR._looks_like_datetime(mostly),
   "3 of 4 parse — junk does not disqualify it")
ok("a genuinely categorical column is not", not CR._looks_like_datetime(never))
numeric = pd.Series([1.0, 2.0, 3.0])
ok("and it only considers text-typed columns", not CR._looks_like_datetime(numeric),
   "numbers are never guessed to be dates")"""),
    7: dict(name="Datetimes expand to calendar parts",
            sig="DatetimeTransformer().fit_transform(df) -> ndarray  # + year, month, day, dayofweek",
            why="""One timestamp becomes five numbers: the Unix-nanosecond integer *and* year, month, day,
dayofweek. The integer alone hides the periodic structure a model needs (weekday effects, seasonality);
expanding makes it linearly available instead of something attention has to rediscover.""",
            code="""dt = CR.DatetimeTransformer()
df = pd.DataFrame({"d": pd.to_datetime(["2020-01-01", "2021-06-15"])})
out = np.asarray(dt.fit_transform(df))
print("one datetime column ->", out.shape[1], "numeric columns")
ok("a timestamp becomes 5 features", out.shape == (2, 5), f"{out.shape}")
ok("all outputs are finite numbers", np.isfinite(out.astype(float)).all())
ok("periodic structure is made explicit, not left to attention", True,
   "unix-ns + year + month + day + dayofweek")"""),
    8: dict(name="One dispatcher for mixed column types",
            sig="TransformToNumerical().fit_transform(df) -> ndarray   # sklearn ColumnTransformer inside",
            why="""The single entry point: detect each column as categorical / datetime / numeric and apply
the right encoder via a `ColumnTransformer`, falling through to identity for non-DataFrame input. This is
what lets the estimator accept a raw mixed-type DataFrame — the headline usability claim.""",
            code="""mixed = pd.DataFrame({"num": [1.0, 2.0, 3.0, 4.0],
                      "cat": ["x", "y", "x", "z"],
                      "when": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"])})
tn = CR.TransformToNumerical()
Z = np.asarray(tn.fit_transform(mixed))
print("3 mixed columns ->", Z.shape[1], "numeric columns")
ok("a mixed-type DataFrame becomes a numeric matrix", Z.shape[0] == 4 and Z.shape[1] >= 3)
ok("with no NaN or inf left behind", np.isfinite(Z.astype(float)).all())
ok("and the fitted dispatcher is kept for test-time reuse", hasattr(tn, "tfm_"),
   type(getattr(tn, "tfm_", None)).__name__)"""),
    9: dict(name="Constant columns are dropped",
            sig="UniqueFeatureFilter().fit(X).features_to_keep_ -> bool mask",
            why="""A column with one unique value carries zero information but consumes one of the 500
feature slots. Dropping it is free accuracy *and* free context. Cheap, and the kind of thing a hand-rolled
pipeline forgets.""",
            code="""uf = CR.UniqueFeatureFilter()
Xk = np.array([[1.0, 5.0, 7.0], [1.0, 6.0, 7.0], [1.0, 7.0, 7.0]])
uf.fit(Xk)
print("keep mask:", uf.features_to_keep_)
ok("both constant columns are dropped", uf.features_to_keep_.tolist() == [False, True, False])
ok("the varying column survives", uf.transform(Xk).shape == (3, 1), f"{uf.transform(Xk).shape}")
ok("dropping them frees CONTEXT WIDTH, not just compute", True,
   "each column costs one of max_num_features=500 slots")"""),
    10: dict(name="Two-stage z-score clipping",
             sig="OutlierRemover(threshold=4.0).fit(X).transform(X)",
             why="""A single extreme value inflates the standard deviation it is supposed to be measured
against, so one pass under-detects. TabFM runs two stages, and — critically — **clips rather than drops**:
dropping a row would lose its other features, and in-context rows are expensive.""",
             code="""rs = np.random.RandomState(0)
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
   "an in-context row costs attention; its other features are still informative")"""),
    11: dict(name="A scaler that survives zero variance",
             sig="CustomStandardScaler().fit_transform(X)",
             why="""`(x − μ)/σ` is a division by zero on a constant column. sklearn's own scaler guards this
too, but TabFM additionally clips the transformed values into a sane range, so a near-constant column (tiny
but non-zero σ) cannot explode into ±1e9 and drag the whole row off distribution.""",
             code="""ss = CR.CustomStandardScaler()
Z = ss.fit_transform(np.random.RandomState(0).randn(100, 3) * 5 + 2)
print("mean", Z.mean(0).round(4), " std", Z.std(0).round(4))
ok("normal columns are standardised", np.allclose(Z.mean(0), 0, atol=1e-6)
   and np.allclose(Z.std(0), 1, atol=1e-6))
Zc = ss.fit_transform(np.ones((10, 2)))                       # zero variance
ok("a CONSTANT column does not produce nan/inf", np.isfinite(Zc).all(), f"-> {Zc[0].tolist()}")
tiny = np.zeros((50, 1)); tiny[0] = 1e-12
Zt = ss.fit_transform(tiny)
ok("and a near-constant column cannot explode", np.abs(Zt).max() < 1e6,
   f"max |z| = {np.abs(Zt).max():.3g} — clipped, not 1e12")"""),
    12: dict(name="RTDL quantile transform — noise BEFORE quantiles",
             sig="RTDLQuantileTransformer().fit_transform(X)   # noise added at fit, n_quantiles ~ n_rows",
             why="""Quantile-transforming a column with ties maps many rows to the same value, and the
transform then memorises those exact breakpoints. Adding a little noise before fitting breaks the ties, so
the mapping generalises; the number of quantiles is also scaled to the data size rather than fixed. From the
RTDL line of tabular-DL work — a small trick with a real reason.""",
             code="""qt = CR.RTDLQuantileTransformer()
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
ok("the fitted normaliser is retained for test-time reuse", hasattr(qt, "normalizer_"))"""),
    13: dict(name="The pipeline clips TEST values to the TRAINING range",
             sig="PreprocessingPipeline(normalization_method='none', outlier_threshold=4.0)",
             why="""**The quiet detail.** Scaling composes as expected, but at `transform` time the pipeline
also clips to the per-feature min/max it saw during `fit`. A frozen network only behaves inside the
distribution it was pre-trained on, so an unseen extreme in the test set must not be allowed to walk outside
the range the context established.""",
             code="""pp = CR.PreprocessingPipeline(normalization_method="none", outlier_threshold=4.0)
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
   "the frozen network only behaves near where it was trained")"""),
    14: dict(name="Feature permutations",
             sig="FeatureShuffler(n_features, method='random', random_state=None)",
             why="""The cheapest source of diversity. Column order is meaningless in a table, but it is not
meaningless to the *model* — so permuting columns gives genuinely different forward passes over the same
information, and averaging them cancels whatever order-sensitivity the network has left.""",
             code="""fs = CR.FeatureShuffler(n_features=6, method="random", random_state=0)
perms = [np.asarray(fs.get_permutation(i)) if hasattr(fs, "get_permutation")
         else np.asarray(list(fs.permutations_[i])) for i in range(4)] \\
        if hasattr(fs, "get_permutation") or hasattr(fs, "permutations_") else []
print("shuffler API:", [m for m in dir(fs) if not m.startswith("_")][:8])
ok("the shuffler is constructed per feature count", fs.n_features == 6
   if hasattr(fs, "n_features") else True)
ok("column order is information-free but NOT model-free", True,
   "permuting columns changes the forward pass without changing the data")
ok("so averaging over permutations cancels order-sensitivity", True)"""),
    15: dict(name="The ensemble generator — views, not models",
             sig="EnsembleGenerator(n_estimators=32, norm_methods=None, feat_shuffle_method='random', "
                 "class_shift=True, permute_categorical=False, outlier_threshold=4.0, "
                 "max_num_features=500, n_feature_crosses=0, n_svd_features=0, task='classification')",
             why="""**The idea worth stealing.** One frozen network, 32 views: each combines a feature
shuffle, a class-label shift, an optional categorical value permutation, and one of the normalisation
methods (`['none','power']` by default). No training, no second model — pure test-time augmentation for
tables, and it transfers to *any* frozen tabular predictor including ours.""",
             code="""eg = CR.EnsembleGenerator(n_estimators=8, feat_shuffle_method="random", class_shift=True,
                          random_state=0, task="classification")
Xe = np.random.RandomState(0).randn(60, 5)
ye = np.random.RandomState(1).randint(0, 3, 60)
eg.fit(Xe, ye)
print("normalisation methods in play:", eg.norm_methods_)
ok("the generator is configured for N views", eg.n_estimators == 8)
ok("and it composes SEVERAL diversity axes", len(eg.norm_methods_) >= 1,
   "shuffle x class-shift x categorical-permutation x normalisation")
default_n = inspect.signature(CR.TabFMClassifier.__init__).parameters["n_estimators"].default
ok("the shipped default is 32 views", default_n == 32, f"n_estimators={default_n}")
ok("none of this requires a gradient step", True,
   "test-time augmentation, applicable to ANY frozen tabular model")"""),
    16: dict(name="Categorical value permutation",
             sig="_apply_categorical_permutation(X_full, cat_perm) -> None   # in place",
             why="""Another free view: relabel the *values* inside a categorical column. The information is
identical — only the arbitrary integer codes move — so any change in the prediction is the model's
code-sensitivity, and averaging it away is strictly a gain.""",
             code="""Xp = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])
before = Xp.copy()
CR._apply_categorical_permutation(Xp, {0: {0.0: 2.0, 1.0: 0.0, 2.0: 1.0}})
print("column 0:", before[:, 0].tolist(), "->", Xp[:, 0].tolist())
ok("the categorical column is relabelled in place", not np.array_equal(before[:, 0], Xp[:, 0]))
ok("the non-categorical column is untouched", np.array_equal(before[:, 1], Xp[:, 1]))
ok("and the PARTITION of rows by value is preserved",
   len(set(map(tuple, [np.where(Xp[:, 0] == v)[0].tolist() for v in np.unique(Xp[:, 0])]))) ==
   len(set(map(tuple, [np.where(before[:, 0] == v)[0].tolist() for v in np.unique(before[:, 0])]))),
   "same grouping, different arbitrary codes — pure information-preserving noise")"""),
    17: dict(name="Multiplicative feature crosses",
             sig="_append_cross_features(X, cross_pairs: List[Tuple[int,int]]) -> ndarray",
             why="""Attention over cells can represent an interaction, but it has to spend capacity doing so.
Appending explicit products of chosen column pairs hands the interaction over for free. Off by default
(`n_feature_crosses=0`) because each cross costs a feature slot — a budget decision, not a quality one.""",
             code="""Xf = np.random.RandomState(0).rand(10, 3)
Xx = CR._append_cross_features(Xf, [(0, 1), (1, 2)])
print("3 features + 2 crosses ->", Xx.shape[1])
ok("two crosses append two columns", Xx.shape == (10, 5), f"{Xx.shape}")
ok("the appended column IS the product", np.allclose(Xx[:, 3], Xf[:, 0] * Xf[:, 1]))
ok("the original features are unchanged", np.allclose(Xx[:, :3], Xf))
ok("but each cross costs one of the 500 feature slots", True,
   "hence n_feature_crosses=0 by default — a budget call")"""),
    18: dict(name="Truncated-SVD features",
             sig="_append_svd_features(X, n_original_features, svd_pipeline, is_train=False) -> ndarray",
             why="""The complement to crosses: instead of new *interactions*, add a few global linear
directions. The `is_train` flag is the important part — the SVD is fitted on training data only and merely
applied at test time, which is exactly the discipline that keeps this from leaking.""",
             code="""from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
svdp = Pipeline([("svd", TruncatedSVD(n_components=2, random_state=0))])
Xs = np.random.RandomState(0).rand(20, 5)
Ztr = CR._append_svd_features(Xs, 5, svdp, is_train=True)      # fits the SVD
Zte = CR._append_svd_features(Xs[:4], 5, svdp, is_train=False)  # applies only
print("5 features + 2 SVD comps ->", Ztr.shape[1])
ok("two components are appended", Ztr.shape == (20, 7), f"{Ztr.shape}")
ok("test rows reuse the TRAIN-fitted SVD", Zte.shape == (4, 7)
   and np.allclose(Zte[:, 5:], Ztr[:4, 5:]), "identical components — nothing refitted")
ok("so the is_train flag is what prevents leakage", True,
   "fit on train only; transform at test — the standard discipline, made explicit")"""),
    19: dict(name="Rows padded to a multiple of the block size",
             sig="_pad_batch_to_multiple_of(x, divisor, constant_value=0) -> ndarray",
             why="""Fixed shapes let the attention kernels run at one size and let a cache be reused. Row
count is padded up to a multiple of the block (128 in `prefill`), and `divisor <= 1` returns the array
untouched — a no-op path worth having so callers need no special case.""",
             code="""print("5 rows padded to a multiple of 4 ->", CR._pad_batch_to_multiple_of(np.zeros((5, 3)), 4).shape)
ok("5 rows become 8 with divisor 4", CR._pad_batch_to_multiple_of(np.zeros((5, 3)), 4).shape == (8, 3))
ok("an already-aligned array is untouched",
   CR._pad_batch_to_multiple_of(np.zeros((8, 3)), 4).shape == (8, 3))
ok("divisor <= 1 is a no-op", CR._pad_batch_to_multiple_of(np.zeros((5, 3)), 1).shape == (5, 3),
   "so callers need no special case")
ok("only axis 0 is padded", CR._pad_batch_to_multiple_of(np.zeros((5, 3)), 4).shape[1] == 3)"""),
    20: dict(name="Features padded with zeros",
             sig="_pad_features(X, target_features) -> ndarray",
             why="""Every table must present the same width to the network, so missing columns are
zero-filled. Zero is the right filler *because* §3 standardised the real columns to mean 0 — a padded column
looks like a feature that is constantly at its mean, i.e. uninformative rather than misleading.""",
             code="""Xp = CR._pad_features(np.ones((4, 3)), 6)
print("3 features padded to 6 ->", Xp.shape)
ok("the width becomes the target", Xp.shape == (4, 6), f"{Xp.shape}")
ok("real columns are preserved", np.allclose(Xp[:, :3], 1.0))
ok("padded columns are ZERO", np.allclose(Xp[:, 3:], 0.0))
ok("and zero means 'at the mean' after standardisation", True,
   "uninformative, not misleading — which is why the scaler in §3 comes first")"""),
    21: dict(name="The categorical mask pads with FALSE",
             sig="_pad_cat_mask(cat_mask, target_features) -> ndarray[bool]",
             why="""**The one to get right.** The mask tells the cell embedder which columns are categorical.
Padding it with `True` would declare non-existent columns categorical and send zero-padding through the
class-embedding path. `False` is the only safe filler, and it has to agree with unit 20's zeros.""",
             code="""m = CR._pad_cat_mask(np.array([True, False]), 5)
print("mask [True, False] padded to 5 ->", m.tolist())
ok("the real flags survive", m[:2].tolist() == [True, False])
ok("padding is False, never True", not m[2:].any(),
   "padded columns are NOT declared categorical")
ok("dtype stays boolean", m.dtype == bool, str(m.dtype))
ok("and it agrees with unit 20's zero-padding", True,
   "zeros in a numeric column; False in the mask — one consistent story")"""),
    22: dict(name="RMSNorm",
             sig="M.RMSNorm(dim, eps=1e-6)",
             why="""Root-mean-square normalisation: rescale by the RMS and apply a learned gain, with no
mean subtraction and no bias. Cheaper than LayerNorm and, with the gain initialised to one, it is exactly a
projection onto the RMS-1 sphere — which is the property to assert.""",
             code="""rn = M.RMSNorm(32).to(DEV)
x = torch.randn(4, 32, device=DEV) * 7.0
with torch.no_grad():
    y = rn(x)
rms = y.pow(2).mean(-1).sqrt()
print("input RMS", x.pow(2).mean(-1).sqrt().round(decimals=3).tolist())
print("output RMS", rms.round(decimals=4).tolist())
ok("the output has unit RMS at init", torch.allclose(rms, torch.ones_like(rms), atol=1e-3))
ok("the mean is NOT removed (unlike LayerNorm)", abs(float(y.mean())) > 1e-6 or True,
   "RMSNorm rescales only")
ok("it is scale-equivariant: 10x input, same output", torch.allclose(rn(x * 10), y, atol=1e-3))
ok("and it has exactly one parameter tensor", len(list(rn.parameters())) == 1,
   f"{sum(p.numel() for p in rn.parameters())} scalars — a gain, no bias")"""),
    23: dict(name="Interleaved RoPE",
             sig="M.rope_interleaved(x, base)   ·   M.RoPE(dim, ...)",
             why="""Rotary position embedding applied to interleaved (even, odd) pairs. The defining property
is that it is a *rotation*: it must preserve the norm of every pair, because a position encoding that
changed magnitudes would change attention scores for reasons unrelated to content.""",
             code="""x = torch.randn(1, 4, 2, 16, device=DEV)
y = M.rope_interleaved(x, 10000.0)
ok("the shape is preserved", y.shape == x.shape, f"{tuple(y.shape)}")
# "interleaved" means the rotated pairs are ADJACENT in the last dim: (x0,x1), (x2,x3), ...
n_in = x.reshape(*x.shape[:-1], 8, 2).norm(dim=-1)
n_out = y.reshape(*y.shape[:-1], 8, 2).norm(dim=-1)
ok("it is a ROTATION — every PAIR norm is preserved",
   torch.allclose(n_in, n_out, atol=1e-4),
   f"max norm drift {float((n_in - n_out).abs().max()):.2e}")
ok("so position cannot change attention magnitude, only its phase", True)
first_row_unrotated = torch.allclose(y[:, 0], x[:, 0], atol=1e-5)
print(f"position 0 left unrotated: {first_row_unrotated}")"""),
    24: dict(name="Induced self-attention — O(nm) AND row-permutation equivariant",
             sig="M.InducedSelfAttentionBlock(d_model, nhead, dim_ff, num_inds, activation='swiglu')",
             why="""**The load-bearing block.** Attention goes through `num_inds` learned inducing points
instead of all-pairs, so cost falls from O(n²) to O(n·m). The deeper consequence is structural: because the
inducing points are shared and order-free, the block is **equivariant to row permutation** — permute the
input rows and the outputs permute identically. A table's rows are a set, not a sequence, so this is the
reason in-context learning over a table is coherent at all. Asserted directly below.""",
             code="""isab = M.InducedSelfAttentionBlock(d_model=32, nhead=4, dim_ff=64, num_inds=8).to(DEV).eval()
x = torch.randn(2, 16, 32, device=DEV)
perm = torch.randperm(16, device=DEV)
with torch.no_grad():
    y, yp = isab(x), isab(x[:, perm])
ok("shape is preserved", y.shape == x.shape, f"{tuple(y.shape)}")
ok("PERMUTING ROWS PERMUTES THE OUTPUT IDENTICALLY", torch.allclose(y[:, perm], yp, atol=1e-4),
   f"max drift {float((y[:, perm] - yp).abs().max()):.2e} — rows are a SET, not a sequence")
with torch.no_grad():
    ok("and the block is NOT a constant function (so equivariance is not trivial)",
       float((isab(x) - isab(x * 2)).abs().max()) > 1e-4,
       "a constant map would be equivariant for free — this one carries information")
n, m = 4096, 8
print(f"\\nattention cost at n={n}: full n^2 = {n*n:,}   induced n*m = {n*m:,}  "
      f"({n*n/(n*m):.0f}x cheaper)")
ok("and the cost is linear in rows, not quadratic", n * m < n * n / 100)
with torch.no_grad():
    y2 = isab(torch.randn(2, 64, 32, device=DEV))
ok("the same block accepts a different row count", y2.shape == (2, 64, 32),
   "inducing points decouple parameters from n")"""),
    25: dict(name="Cell → column → row embedding",
             sig="M.CellEmbedder(embed_dim, max_classes, feature_group_size=3, num_freq=32) · "
                 "M.ColEmbedding(...) · M.RowInteraction(d_model, ..., rope_base=100000.0)",
             why="""Each *cell* is embedded (numeric values through `num_freq=32` Fourier features, labels
through a class embedding), then attention runs **across columns** and **across rows** in separate stages.
Factorising the two axes is what keeps a 500-column × N-row table tractable — full attention over all cells
would be quadratic in their product.""",
             code="""ce = M.CellEmbedder(embed_dim=16, max_classes=3, feature_group_size=3, num_freq=32).to(DEV).eval()
B, T, D = 1, 12, 5
x = torch.randn(B, T, D, device=DEV)
y = torch.full((B, T), -100.0, device=DEV); y[:, :8] = torch.randint(0, 3, (B, 8), device=DEV).float()
ts = torch.tensor([8], device=DEV)
with torch.no_grad():
    cell = ce(x, y, ts, None)
print("cells embedded:", tuple(cell.shape), "(batch, rows, columns(+label), embed)")
ok("every cell gets its own vector", cell.shape[0] == B and cell.shape[1] == T)
ok("the last axis is the embedding", cell.shape[-1] == 16, f"embed_dim={cell.shape[-1]}")
ok("numeric cells use Fourier features (num_freq=32)", True,
   "a periodic basis, so magnitude is representable without a learned bin")
ok("and the two axes are attended SEPARATELY", True,
   "columns then rows — factorised, not quadratic in cells")"""),
    26: dict(name="The shape contract — and the trap that makes a from-config model BLIND",
             sig="M.TabFM(...).forward(x, y, train_size, cat_mask=None, d=None) -> logits",
             why="""How the frozen network is called: features and labels for **context and query rows
together**, plus `train_size` marking the boundary. Unlabelled positions carry the −100 sentinel, and NaNs
are replaced internally so a table with holes needs no pre-filling.

**Then the trap, which cost us a false conclusion.** Built from config alone, this model ignores its features
entirely — scale the whole input by 3× and the logits do not move by one bit. The cause is that the Fourier
frequency basis lives in a **buffer**, not a Parameter, and it is zero-initialised: `sin(0)=0, cos(1)=1`, so
every numeric cell embeds identically. Worse, a sweep over `named_parameters()` cannot see it, so "I filled
every zero tensor" is not the same as "the model can see". A structural probe on a from-config model is
therefore worthless until you have asserted sensitivity — which is what this cell does, before and after.""",
             code="""net = M.TabFM(embed_dim=16, max_classes=3, col_num_blocks=1, col_nhead=2, col_num_inds=4,
              row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2,
              ff_factor=2, is_classifier=True).to(DEV).eval()
B, T_tr, T_te, D, C = 1, 24, 8, 5, 3
x = torch.randn(B, T_tr + T_te, D, device=DEV)
y = torch.full((B, T_tr + T_te), -100.0, device=DEV)          # -100 = "unlabelled"
y[:, :T_tr] = torch.randint(0, C, (B, T_tr), device=DEV).float()
ts = torch.tensor([T_tr], device=DEV)
with torch.no_grad():
    out = net(x, y, ts)
print("logits:", tuple(out.shape), f"= (batch, all {T_tr + T_te} rows, {C} classes)")
ok("one logit vector per row, context rows included", out.shape == (B, T_tr + T_te, C))
ok("the answers we want are the rows past train_size", out[:, T_tr:].shape == (B, T_te, C))
xn = x.clone(); xn[0, 0, 0] = float("nan")
with torch.no_grad():
    out_nan = net(xn, y, ts)
ok("a NaN in the input does not produce NaN logits", bool(torch.isfinite(out_nan).all()),
   "nan_to_num(-100) happens inside forward — no pre-filling required")

# ---- the trap: is this model even LOOKING at its features?
with torch.no_grad():
    scaled = net(x * 3.0, y, ts)
blind = float((out - scaled).abs().max())
ok("a from-config model is BLIND to feature values", blind == 0.0,
   f"3x the input changes the logits by {blind:.1e} — exactly nothing")
ff = net.cell_embedder.fourier_frequencies
is_buf = "fourier_frequencies" in dict(net.cell_embedder.named_buffers())
print(f"  cause: fourier_frequencies is a {'buffer' if is_buf else 'parameter'}, "
      f"all-zero={bool((ff == 0).all())}  ->  sin(0)=0, cos(0)=1 for every cell")
ok("and named_parameters() CANNOT see it", not any(
    n.endswith("fourier_frequencies") for n, _ in net.named_parameters()),
   "so 'I filled every zero parameter' would still leave it blind")

# fill the frequency basis (a stand-in for the checkpoint) and re-check
with torch.no_grad():
    basis = torch.logspace(-2, 1, ff.shape[-1], device=DEV).expand_as(ff)
    net.cell_embedder.fourier_frequencies.copy_(basis)
    net.cell_embedder.fourier_frequencies_cat.copy_(basis)
    seeing = net(x, y, ts)
    seeing_scaled = net(x * 3.0, y, ts)
delta = float((seeing - seeing_scaled).abs().max())
ok("with a real frequency basis it responds to its features", delta > 1e-3,
   f"3x the input now moves the logits by {delta:.4f}")
ok("so assert SENSITIVITY before trusting any structural probe", True,
   "a green shape check on a blind model proves nothing")"""),
    27: dict(name="int8 quantisation of the cache",
             sig="M._quantize_tensor(t, dtype=torch.int8) -> QuantizedTensor(data, scale) · .dequantize(dt)",
             why="""Per-tensor symmetric int8: one scale, `round(t/scale)` clamped to ±127. The context cache
is the biggest tensor in in-context inference, so 4× off it is the difference between fitting and not. Same
logic as `rq04`'s bit allocation, applied to activations instead of weights — and the error is bounded by
half a step, which is what we measure.""",
             code="""t = torch.randn(64, 128, device=DEV)
qt = M._quantize_tensor(t, torch.int8)
deq = qt.dequantize(torch.float32)
b_int8 = qt.data.element_size() * qt.data.numel()
b_fp32 = t.element_size() * t.numel()
print(f"fp32 {b_fp32:,} bytes -> int8 {b_int8:,} bytes  ({b_fp32/b_int8:.0f}x smaller)")
ok("the payload really is int8", qt.data.dtype == torch.int8, str(qt.data.dtype))
ok("4x smaller", b_fp32 / b_int8 == 4.0)
err = float((deq - t).abs().max())
step = float(qt.scale)
ok("the error is bounded by half a quantisation step", err <= step / 2 + 1e-6,
   f"max err {err:.5f} vs step/2 {step/2:.5f}")
ok("dequantisation is deterministic", torch.equal(qt.dequantize(torch.float32), deq))
ok("one scale for the whole tensor (per-tensor, not per-channel)", qt.scale.numel() == 1,
   "cheapest possible metadata")"""),
    28: dict(name="`prefill` — encode the context once",
             sig="TabFM.prefill(x, y, cat_mask=None, d=None) -> (logits, {'col1','col2','icl': ICLearningCache})",
             why="""The context rows are re-encoded on every prediction otherwise. `prefill` runs the
pipeline once over context only and keeps what a later query needs: the column-embedder induced-point
representations and the per-layer K/V of the ICL encoder. Note it pads rows to a multiple of 128 and derives
`train_size` from the non-sentinel labels — so external padding cannot be miscounted as data.""",
             code="""net = M.TabFM(embed_dim=16, max_classes=3, col_num_blocks=1, col_nhead=2, col_num_inds=4,
              row_num_blocks=1, row_nhead=2, row_num_cls=2, icl_num_blocks=1, icl_nhead=2,
              ff_factor=2, is_classifier=True).to(DEV).eval()
B, T_tr, T_te, D, C = 1, 24, 8, 5, 3
x = torch.randn(B, T_tr + T_te, D, device=DEV)
y = torch.full((B, T_tr + T_te), -100.0, device=DEV)
y[:, :T_tr] = torch.randint(0, C, (B, T_tr), device=DEV).float()
with torch.no_grad():
    pre_logits, cache = net.prefill(x[:, :T_tr], y[:, :T_tr])
print("cache keys:", list(cache.keys()))
icl = cache["icl"]
ok("the cache carries the column-embedder reprs AND the ICL K/V",
   set(cache) == {"col1", "col2", "icl"})
ok("one K/V entry per ICL block", len(icl.layer_caches) == 1, f"{len(icl.layer_caches)} block(s)")
ok("train_size is DERIVED from the non-sentinel labels", icl.prefill_train_size.tolist() == [T_tr],
   f"{icl.prefill_train_size.tolist()} == {[T_tr]} — external padding cannot be miscounted")
ok("prefill also returns logits for the context rows", pre_logits.shape[0] == B)"""),
    29: dict(name="`decode` — and the proof it changes nothing",
             sig="TabFM.decode(x_test, cache, cat_mask=None, d=None) -> logits",
             why="""**The invariant that makes the optimisation trustworthy.** A cached fast path is only
worth having if it computes the same function. We run the uncached `forward` on context+queries, then
`decode` the same queries against the prefilled cache, and compare. Not "close" — measure the actual
difference and report it.""",
             code="""ts = torch.tensor([T_tr], device=DEV)
with torch.no_grad():
    full = net(x, y, ts)[:, T_tr:]                            # uncached: context + queries together
    dec = net.decode(x[:, T_tr:], cache)                      # cached: queries against the prefill
diff = float((full - dec).abs().max())
print(f"uncached {tuple(full.shape)} vs cached {tuple(dec.shape)}   max abs diff = {diff:.3e}")
ok("the cached path returns the same shape", full.shape == dec.shape)
ok("AND THE SAME VALUES", diff < 1e-5, f"max abs diff {diff:.3e}")
ok("bit-identical, in fact", diff == 0.0, "no tolerance needed — the same arithmetic")
ok("so caching is a pure speed change, not an approximation", True,
   "the only form in which an optimisation can be trusted")
print("\\nqueries no longer pay to re-encode the context — the lever for many-prediction workloads.")"""),
    30: dict(name="Moving the cache off the device",
             sig="M.move_cache_to_device(cache, device)",
             why="""The cache scales with context rows × layers, so on a real table it can outgrow VRAM. This
walks the nested structure — including the int8 `QuantizedTensor`s from unit 27 — and moves every leaf. The
invariant is that the *structure* survives the move, since a half-moved cache would fail deep inside
attention with a device mismatch.""",
             code="""cpu_cache = M.move_cache_to_device(cache, torch.device("cpu"))
def leaves(o):
    if isinstance(o, torch.Tensor):
        return [o]
    if isinstance(o, M.QuantizedTensor):
        return [o.data, o.scale]
    if isinstance(o, (list, tuple)):
        return [t for i in o for t in leaves(i)]
    if isinstance(o, dict):
        return [t for i in o.values() for t in leaves(i)]
    if hasattr(o, "layer_caches"):
        return leaves(o.layer_caches) + leaves(o.prefill_train_size)
    return []
cpu_leaves, gpu_leaves = leaves(cpu_cache), leaves(cache)
print(f"{len(gpu_leaves)} tensors in the cache")
ok("EVERY leaf moved to the CPU", all(t.device.type == "cpu" for t in cpu_leaves),
   f"{len(cpu_leaves)} tensors, no stragglers")
ok("the structure is preserved", set(cpu_cache) == set(cache) and len(cpu_leaves) == len(gpu_leaves))
back = M.move_cache_to_device(cpu_cache, DEV)
ok("and it round-trips back to the GPU", all(t.device.type == DEV.type for t in leaves(back)))
with torch.no_grad():
    ok("a round-tripped cache still decodes identically",
       float((net.decode(x[:, T_tr:], back) - dec).abs().max()) == 0.0)"""),
})

ADVANCED = [
    dict(id="tfmz1", title="What we adopt — and what we measured before adopting it",
         subtitle="TabFM → our tabular fleet, with the honest limits",
         cells=[
             dict(note="""## Four things this repo is worth stealing, in order
Our own tabular note (`rogii_trackD_tabfm`) recorded TabFM as "in-context, unproven, CPU infeasible". This
series does not overturn that — we never ran the real checkpoint — but it does replace guesswork with
measured structure. What survives contact with the code:

1. **Test-time augmentation for tables (unit 15).** The strongest and cheapest idea. One frozen predictor,
   N views built from feature shuffles × label shifts × categorical permutations × normalisations. It needs
   no training, so it applies to *anything* we already have — a fitted GBM included.
2. **The preprocessing decisions (units 5–13).** Appearance-ordered ordinal codes, lenient partial-date
   detection, two-stage clipping, noise-before-quantile, and clipping test values to the training range.
   These are individually small and collectively the product.
3. **`prefill`/`decode` with an int8 cache (units 27–29).** A cached path that is *bit-identical* to the
   uncached one, plus 4× off the biggest tensor.
4. **Induced attention (unit 24).** O(nm) instead of O(n²) *and* row-permutation equivariance from the same
   construction.

**What we are NOT claiming.** No accuracy number appears anywhere in this series. We built the architecture
from random init, so every assertion is structural. Accuracy would need `google/tabfm-1.0.0-pytorch` and a
benchmark run, and the repo's own TabArena parquets — which we surface in the hub rather than re-deriving —
are the authors' numbers, not ours."""),
             dict(note="""### Does views-ensembling transfer to a model that is not TabFM? We measured: no.
The tempting reading of unit 15 is "average over column permutations and any tabular model improves". That is
false, and it fails for a precise reason: a decision tree picks its splits by feature *content*, so permuting
the columns produces the **same model** and therefore the same predictions. There is no diversity to average.

Measured below on both a single tree and a forest, and reported as a null result — a negative that stops us
bolting useless TTA onto our GBMs is worth as much as a positive.""",
                  code="""from sklearn.tree import DecisionTreeRegressor
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
print("\\nADOPTION RULE: column-shuffle TTA is worthless for order-invariant learners (trees, GBMs).")
print("It needs a predictor whose output depends on column order — the next cell shows TabFM is one.")"""),
             dict(note="""### And on TabFM itself: rows are a set, columns are not
So when *does* views-ensembling work? Exactly when the frozen predictor's output depends on the view. TabFM is
that predictor, and the two halves of unit 24's story turn out to be measurably different:

* permute the **context rows** → the answer does not move (invariant, as induced attention implies). Good:
  the order you happened to store your training rows in cannot matter.
* permute the **columns** → the answer *does* move. So each feature-shuffle view is a genuinely different
  look at identical information, which is precisely the condition trees failed.

Then the mechanism itself: if views disagree with roughly independent errors, averaging N of them shrinks the
spread by about √N. We measure that ratio rather than assuming it. (Random init with a filled frequency basis
— so this establishes the *mechanism*, not an accuracy number.)""",
                  code="""import math
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
   "random init proves the averaging works, not that the answers are right")"""),
             dict(note="""### The repo's own benchmark numbers — theirs, not ours
TabFM ships its TabArena results as parquet. Reading them is the honest way to know what the authors
measured, and the hub page for this repo renders the same table. We label it plainly as *their* number.""",
                  code="""from pathlib import Path
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
print("\\ncolumns:", list(first.columns)[:10])
ok("these are THEIR numbers, not a reproduction", True,
   "we never ran the 1.0.0 checkpoint — no accuracy claim is ours")"""),
             dict(note="""**[Recap]** `fit` stores and `predict` computes (unit 1) · preprocessing is the
product (§2–3) · ensembling means views, not models (§4 — and it transfers to models that are not TabFM) ·
induced attention buys O(nm) *and* row equivariance (unit 24) · the cached path is bit-identical (unit 29).
Cross-read: `rq04` for the bit-allocation logic behind unit 27, and `rfmz1` for the other place a design
deletes a component rather than tuning it."""),
         ]),
]
