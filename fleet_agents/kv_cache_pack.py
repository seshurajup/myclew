"""kv_cache_pack — the LONG-CONTEXT / KV-CACHE lever from Gemma 4 (arXiv 2607.02770, §2 "Long-context
efficiency"): the KV cache grows linearly with context length AND layer count, so at 32k/128k/256k it
dominates memory. Gemma 4 attacks it three ways, all captured here as a pure memory calculator:

  • kv-cache-longctx  — model KV-cache bytes vs context length under (1) a local:global attention ratio
                        with a sliding-window cap on local layers, (2) KV-cache sharing across global
                        layers, and (3) reusing keys as values in global layers (values=keys). Reports
                        bytes and the % reduction vs a naive full-global K+V cache — the report's "up to
                        37.5% global KV cache" figure is reproducible from these two global-layer levers.

KV cache bytes for one layer at sequence length L, with h kv-heads of dim d, in `dtype_bytes`:
    per_layer = tensors · L · h · d · dtype_bytes         (tensors=2 for separate K and V)
Local layers cap L at the sliding window w; global layers use full L, can share cache (÷share), and can
set values=keys (tensors 2→1). Naive baseline = every layer global, K+V, full L.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- core math
def kv_cache_bytes(seq_len, n_layers, n_kv_heads, head_dim, *, dtype_bytes=1,
                   local_global_ratio=5, window=4096, global_share=1.0, values_as_keys=False,
                   reuse_fraction=1.0):
    """Total KV-cache bytes across all layers under the Gemma-4 long-context scheme.
    local_global_ratio r: r local layers per 1 global layer (Gemma-4 uses 5, or 4 for E2B).
    window: sliding-window size capping the seq dim of LOCAL layers.
    global_share: divide global-layer cache by this KV-sharing factor (>=1).
    values_as_keys: if True, global layers reuse keys as values, eliminating `reuse_fraction` of the
      value tensor (reuse_fraction=1.0 → the full V half is dropped, the 50% bound; the report's
      p-RoPE global design corresponds to ~0.75 → a 37.5% global-KV cut, see global_kv_reduction)."""
    L = int(seq_len); n = int(n_layers); h = int(n_kv_heads); d = int(head_dim)
    r = max(0, int(local_global_ratio)); db = float(dtype_bytes)
    n_global = max(1, round(n / (r + 1))) if r >= 0 else n
    n_local = n - n_global
    Lloc = min(L, int(window))
    local_bytes = n_local * 2 * Lloc * h * d * db
    gtensors = (2.0 - float(reuse_fraction)) if values_as_keys else 2.0
    global_bytes = n_global * gtensors * L * h * d * db / max(1.0, float(global_share))
    return {"total_bytes": float(local_bytes + global_bytes),
            "local_bytes": float(local_bytes), "global_bytes": float(global_bytes),
            "n_global": int(n_global), "n_local": int(n_local)}


def naive_kv_bytes(seq_len, n_layers, n_kv_heads, head_dim, dtype_bytes=1):
    """Baseline: every layer global, separate K and V, full sequence length (no window/share/reuse)."""
    return float(2 * int(n_layers) * int(seq_len) * int(n_kv_heads) * int(head_dim) * float(dtype_bytes))


def global_kv_reduction(global_share=1.0, values_as_keys=True, reuse_fraction=1.0):
    """% reduction of the GLOBAL-layer KV cache from the global-only levers vs plain K+V.
    values=keys eliminates `reuse_fraction` of the value tensor (of the 2-tensor K+V), then KV-sharing
    divides by `global_share` (>=1). Returns a fraction in [0,1].
      • reuse_fraction=1.0 (drop the whole V) → 0.50 cap from reuse alone.
      • reuse_fraction=0.75 → 0.375, i.e. the report's stated 37.5% global-KV cut for its p-RoPE global
        design (only ~3/4 of the value tensor's cost is removed once positions are handled)."""
    saved = 0.5 * float(reuse_fraction) if values_as_keys else 0.0   # fraction of the 2-tensor K+V removed
    kept = (1.0 - saved) / max(1.0, float(global_share))
    return float(np.clip(1.0 - kept, 0.0, 1.0))


def reduction_vs_naive(seq_len, n_layers, n_kv_heads, head_dim, *, dtype_bytes=1, **kw):
    """Fraction by which the Gemma-4 scheme cuts TOTAL KV cache vs the naive all-global baseline."""
    opt = kv_cache_bytes(seq_len, n_layers, n_kv_heads, head_dim, dtype_bytes=dtype_bytes, **kw)["total_bytes"]
    base = naive_kv_bytes(seq_len, n_layers, n_kv_heads, head_dim, dtype_bytes)
    return float(1.0 - opt / base) if base > 0 else 0.0


# ---------------------------------------------------------------- agent
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class KVCacheLongCtx(_B):
    name = "kv-cache-longctx"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("seq_len", "n_layers", "n_kv_heads", "head_dim") if k not in s]
        if missing:
            return self.escalate(worker, "leader",
                f"kv-cache-longctx needs spec keys {missing} — none provided")
        kw = dict(dtype_bytes=float(s.get("dtype_bytes", 1)),
                  local_global_ratio=int(s.get("local_global_ratio", 5)),
                  window=int(s.get("window", 4096)),
                  global_share=float(s.get("global_share", 1.0)),
                  values_as_keys=bool(s.get("values_as_keys", True)))
        res = kv_cache_bytes(int(s["seq_len"]), int(s["n_layers"]), int(s["n_kv_heads"]), int(s["head_dim"]), **kw)
        red = reduction_vs_naive(int(s["seq_len"]), int(s["n_layers"]), int(s["n_kv_heads"]), int(s["head_dim"]),
                                 dtype_bytes=kw["dtype_bytes"], local_global_ratio=kw["local_global_ratio"],
                                 window=kw["window"], global_share=kw["global_share"], values_as_keys=kw["values_as_keys"])
        gred = global_kv_reduction(kw["global_share"], kw["values_as_keys"])
        gb = res["total_bytes"] / 1e9
        msg = (f"kv-cache-longctx: {gb:.3f} GB @ {int(s['seq_len'])} ctx "
               f"({res['n_global']} global / {res['n_local']} local layers); "
               f"global-KV cut {gred*100:.1f}%, total cut {red*100:.1f}% vs naive")
        data = {"total_gb": gb, "total_bytes": res["total_bytes"], "global_kv_reduction": gred,
                "total_reduction_vs_naive": red, "n_global": res["n_global"], "n_local": res["n_local"]}
        self.log(msg, kind="finding",
                 recommendation="raise local:global ratio or shrink the window to fit longer context; values=keys + KV-sharing shrink the few global layers (Gemma-4: up to 37.5%)")
        return self.done(data, msg)


_KV = KVCacheLongCtx()


def run_kv(q, worker): return _KV.run(q, worker)
