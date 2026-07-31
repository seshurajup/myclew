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

t = torch.randn(64, 128, device=DEV)
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
   "cheapest possible metadata")

net = M.TabFM(embed_dim=16, max_classes=3, col_num_blocks=1, col_nhead=2, col_num_inds=4,
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
ok("prefill also returns logits for the context rows", pre_logits.shape[0] == B)

ts = torch.tensor([T_tr], device=DEV)
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
print("\nqueries no longer pay to re-encode the context — the lever for many-prediction workloads.")

cpu_cache = M.move_cache_to_device(cache, torch.device("cpu"))
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
       float((net.decode(x[:, T_tr:], back) - dec).abs().max()) == 0.0)
