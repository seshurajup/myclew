"""gpu_patterns — the parallel-programming PRIMITIVES that srush/GPU-Puzzles teaches (map, zip, broadcast,
reduce, prefix-scan, tiled/blocked matmul), ported as correct reference implementations, PLUS a roofline
arithmetic-intensity cost model that turns "which kernel pattern / tiling" into a compute-vs-memory-bound
decision. GPU-Puzzles itself is a set of numba.cuda teaching exercises (needs a GPU), so we lift the concepts,
not the kernels: (a) reference semantics for each pattern so a fleet trainer can reason about / test a fused
kernel, and (b) the roofline model that says whether an op is memory- or compute-bound and what its ceiling is
— the actual lever for kernel/tiling choices (pairs with hardware_tune's measured peaks).

Primitives (numpy, deps = numpy):
  • prefix_scan(x)                — inclusive parallel prefix sum (Blelloch semantics) == np.cumsum.
  • segment_reduce(x, seg)        — reduce within segments (the reduce puzzle).
  • tiled_matmul(A, B, tile)      — blocked matmul (the shared-memory-tiling puzzle) == A@B.
  • arithmetic_intensity(flops, bytes)          — FLOP/byte.
  • roofline(flops, bytes, peak_flops, peak_bw) — attainable FLOP/s + 'compute'|'memory' bound + which tiling helps.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- parallel primitives (reference semantics)
def prefix_scan(x):
    """Inclusive prefix sum (Blelloch scan semantics). Reference for a parallel-scan kernel; == np.cumsum."""
    x = np.asarray(x, float).copy()
    n = len(x)
    # up-sweep + down-sweep would be the parallel form; the result equals a sequential inclusive scan.
    out = np.empty_like(x); acc = 0.0
    for i in range(n):
        acc += x[i]; out[i] = acc
    return out


def segment_reduce(x, seg_ids, op="sum"):
    """Reduce values within contiguous segment ids (the reduce puzzle). Returns one value per segment id."""
    x = np.asarray(x, float); seg = np.asarray(seg_ids)
    ids = np.unique(seg)
    f = {"sum": np.sum, "max": np.max, "min": np.min, "mean": np.mean}[op]
    return {int(s): float(f(x[seg == s])) for s in ids}


def tiled_matmul(A, B, tile=16):
    """Blocked/tiled matmul (the shared-memory tiling puzzle): accumulate C in `tile`×`tile` blocks. Numerically
    equals A @ B; the tiling is what a GPU kernel does to reuse operands from shared memory."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    M, K = A.shape; K2, N = B.shape
    assert K == K2, "inner dims must match"
    C = np.zeros((M, N))
    t = int(tile)
    for i0 in range(0, M, t):
        for j0 in range(0, N, t):
            for k0 in range(0, K, t):
                C[i0:i0+t, j0:j0+t] += A[i0:i0+t, k0:k0+t] @ B[k0:k0+t, j0:j0+t]
    return C


# ---------------------------------------------------------------- roofline cost model
def arithmetic_intensity(flops, bytes_moved):
    """FLOP per byte — the x-axis of the roofline. High AI → compute-bound (tiling/fusion helps); low AI →
    memory-bound (reduce bytes moved / fuse to avoid round-trips)."""
    return float(flops) / max(float(bytes_moved), 1e-12)


def roofline(flops, bytes_moved, peak_flops, peak_bw):
    """Roofline: attainable FLOP/s = min(peak_flops, AI·peak_bw); bound = 'memory' if AI < ridge else 'compute'.
    ridge_ai = peak_flops/peak_bw is the intensity where the machine flips from memory- to compute-bound."""
    ai = arithmetic_intensity(flops, bytes_moved)
    ridge = peak_flops / max(peak_bw, 1e-12)
    attainable = min(float(peak_flops), ai * float(peak_bw))
    return {"arithmetic_intensity": ai, "ridge_point": ridge,
            "bound": "compute" if ai >= ridge else "memory",
            "attainable_flops": attainable, "pct_peak": attainable / peak_flops,
            "lever": "tile/fuse for compute reuse" if ai >= ridge else "cut bytes moved / fuse to avoid HBM round-trips"}


# ---------------------------------------------------------------- agent
class GPUPatterns(BaseAgent):
    name = "gpu-patterns"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        # 5090-ish peaks (bf16 ~ 209 TFLOP/s dense, ~1.8 TB/s HBM) — overridable via spec.
        peak_flops = float(s.get("peak_flops", 209e12)); peak_bw = float(s.get("peak_bw", 1.8e12))
        M = int(s.get("m", 4096))
        # a big matmul (compute-bound) vs an elementwise add (memory-bound)
        mm_flops = 2.0 * M**3; mm_bytes = 3 * M * M * 2                       # bf16 A,B,C
        add_flops = M * M; add_bytes = 3 * M * M * 2
        rmm = roofline(mm_flops, mm_bytes, peak_flops, peak_bw)
        radd = roofline(add_flops, add_bytes, peak_flops, peak_bw)
        msg = (f"gpu-patterns: roofline @ {peak_flops/1e12:.0f}TFLOP/s,{peak_bw/1e12:.1f}TB/s (ridge AI="
               f"{rmm['ridge_point']:.0f}) — {M}³ matmul AI={rmm['arithmetic_intensity']:.0f}→{rmm['bound']}-bound "
               f"({rmm['pct_peak']*100:.0f}% peak, lever: {rmm['lever']}); elementwise-add AI="
               f"{radd['arithmetic_intensity']:.2f}→{radd['bound']}-bound. Reference primitives (scan/reduce/"
               f"tiled-matmul) for verifying fused kernels (GPU-Puzzles)")
        self.log(msg, kind="finding",
                 recommendation="classify an op's roofline first: compute-bound → tile/fuse for reuse; "
                                "memory-bound → cut bytes / fuse to avoid HBM round-trips (don't over-optimize FLOPs)")
        return self.done({"matmul_bound": rmm["bound"], "add_bound": radd["bound"],
                          "matmul_ai": rmm["arithmetic_intensity"]}, msg)


_AGENT = GPUPatterns()


def run_gpupatterns(q, worker):
    return _AGENT.run(q, worker)
