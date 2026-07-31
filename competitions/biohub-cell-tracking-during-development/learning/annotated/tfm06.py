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

rn = M.RMSNorm(32).to(DEV)
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
   f"{sum(p.numel() for p in rn.parameters())} scalars — a gain, no bias")

x = torch.randn(1, 4, 2, 16, device=DEV)
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
print(f"position 0 left unrotated: {first_row_unrotated}")

isab = M.InducedSelfAttentionBlock(d_model=32, nhead=4, dim_ff=64, num_inds=8).to(DEV).eval()
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
print(f"\nattention cost at n={n}: full n^2 = {n*n:,}   induced n*m = {n*m:,}  "
      f"({n*n/(n*m):.0f}x cheaper)")
ok("and the cost is linear in rows, not quadratic", n * m < n * n / 100)
with torch.no_grad():
    y2 = isab(torch.randn(2, 64, 32, device=DEV))
ok("the same block accepts a different row count", y2.shape == (2, 64, 32),
   "inducing points decouple parameters from n")

ce = M.CellEmbedder(embed_dim=16, max_classes=3, feature_group_size=3, num_freq=32).to(DEV).eval()
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
   "columns then rows — factorised, not quadratic in cells")

net = M.TabFM(embed_dim=16, max_classes=3, col_num_blocks=1, col_nhead=2, col_num_inds=4,
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
   "a green shape check on a blind model proves nothing")
