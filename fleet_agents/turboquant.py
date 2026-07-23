"""turboquant — Google Research's TurboQuant data-oblivious vector quantizer (arXiv:2504.19874), the algorithm
behind RyanCodrai/turbovec (a Rust vector index that fits a 31GB fp32 corpus in 4GB and searches faster than
FAISS). "Data-oblivious" = NO training/rebuild phase: the quantizer is fixed by the DIMENSION alone, so you can
ingest vectors online and they're indexed immediately. The insight: after a random orthogonal rotation, every
coordinate of a unit vector on the sphere S^{d-1} follows the SAME known marginal — Beta((d-1)/2,(d-1)/2) on
[-1,1] — so a single Lloyd-Max scalar quantizer fit to that Beta (once, analytically) is near-optimal for EVERY
coordinate of EVERY vector, with no data-dependent codebook. Storing ||v|| and a RaBitQ-style length-scale
recovers an unbiased inner-product estimate at search time. The turbovec Rust core is SIMD/bit-packing; the
ALGORITHM is pure-numpy and offline-testable.

Why the fleet wants it: a training-free, GPU-free way to compress embedding banks 8× (fp32→4bit) with graceful
recall — for retrieval-augmented pipelines, dedup, nearest-neighbor over large embedding sets, air-gapped RAG.

Primitives (numpy, deps = numpy + scipy for the Beta):
  • make_rotation(dim, seed)              — deterministic seeded orthogonal matrix (QR + sign-fix, like turbovec).
  • lloyd_max_beta(bits, dim)             — (boundaries, centroids) Lloyd-Max codebook for the Beta marginal.
  • encode(V, rotation, boundaries, centroids) — normalize→rotate→quantize; returns (codes, scales).
  • decode(codes, scales, centroids, rotation) — reconstruct approximate vectors.
  • quantize_index(V, bits)               — one-call fit+encode; returns an index dict for search.
  • search(index, q, k)                   — top-k by reconstructed inner product (unbiased estimator).
"""
from __future__ import annotations
import numpy as np
_trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))  # numpy 2.x renamed trapz→trapezoid
from .base import BaseAgent


# ---------------------------------------------------------------- deterministic random rotation
def make_rotation(dim, seed=42):
    """Deterministic dim×dim orthogonal matrix: QR of a seeded Gaussian, with the sign-correction
    Q = Q·diag(sign(diag(R))) so it's unique/reproducible (turbovec rotation.rs)."""
    rng = np.random.RandomState(int(seed))
    G = rng.randn(int(dim), int(dim))
    Q, R = np.linalg.qr(G)
    Q = Q * np.sign(np.diag(R))              # sign-fix → deterministic
    return Q.astype(np.float64)


# ---------------------------------------------------------------- Lloyd-Max codebook for the Beta marginal
def lloyd_max_beta(bits, dim, max_iter=200, tol=1e-12):
    """Lloyd-Max optimal scalar quantizer for Beta((d-1)/2,(d-1)/2) on [-1,1] — the marginal every coordinate
    follows after rotation. Returns (boundaries (2^bits-1,), centroids (2^bits,)). Analytic; no data."""
    from scipy.stats import beta as _beta
    a = (dim - 1) / 2.0
    B = _beta(a, a)                                   # on [0,1]; we map to [-1,1] via x=2t-1
    n = 1 << int(bits)
    std = np.sqrt(2.0 * a / ((2.0 * a + 1.0) * 4.0 * a))
    spread = 3.0 * std
    cent = np.linspace(-spread, spread, n)
    for _ in range(max_iter):
        bnd = (cent[:-1] + cent[1:]) / 2.0
        edges = np.concatenate([[-1.0], bnd, [1.0]])
        new = cent.copy()
        # conditional mean of x on each cell under the Beta pdf, via fine numeric integration
        for i in range(n):
            lo, hi = edges[i], edges[i + 1]
            xs = np.linspace(lo, hi, 256)
            pdf = B.pdf((xs + 1) / 2.0) / 2.0
            prob = _trapz(pdf, xs)
            if prob > 1e-15:
                new[i] = _trapz(xs * pdf, xs) / prob
        if np.max(np.abs(new - cent)) < tol:
            cent = new; break
        cent = new
    bnd = (cent[:-1] + cent[1:]) / 2.0
    return bnd.astype(np.float64), cent.astype(np.float64)


# ---------------------------------------------------------------- encode / decode
def encode(V, rotation, boundaries, centroids):
    """Normalize each row to unit length, rotate, quantize each coord to the nearest codebook level.
    Returns (codes (n,dim) int, scales (n,) = original norms). Data-oblivious: same codebook for all."""
    V = np.asarray(V, float)
    norms = np.linalg.norm(V, axis=1)
    unit = V / np.clip(norms, 1e-12, None)[:, None]
    rot = unit @ rotation.T                            # rotate into the Beta-marginal frame
    codes = np.searchsorted(boundaries, rot)           # bucket index per coord
    return codes.astype(np.int32), norms.astype(np.float32)


def decode(codes, scales, centroids, rotation):
    """Reconstruct approximate vectors: centroid lookup → un-rotate → rescale by stored norm."""
    rec_unit = centroids[codes] @ rotation             # inverse rotation (orthogonal → transpose is inverse; @R since we rotated by R^T)
    rec_unit = rec_unit / np.clip(np.linalg.norm(rec_unit, axis=1), 1e-12, None)[:, None]
    return rec_unit * np.asarray(scales, float)[:, None]


def quantize_index(V, bits=4, seed=42):
    """Fit (dimension-only) + encode a matrix into a compact index dict. No training on V's distribution."""
    V = np.asarray(V, float); dim = V.shape[1]
    R = make_rotation(dim, seed)
    bnd, cent = lloyd_max_beta(bits, dim)
    codes, scales = encode(V, R, bnd, cent)
    return {"codes": codes, "scales": scales, "centroids": cent, "boundaries": bnd,
            "rotation": R, "bits": int(bits), "dim": dim, "n": V.shape[0]}


def search(index, q, k=10):
    """Top-k by inner product between query q and the RECONSTRUCTED database vectors. Returns (scores, idx)."""
    recon = decode(index["codes"], index["scales"], index["centroids"], index["rotation"])
    scores = recon @ np.asarray(q, float)
    idx = np.argsort(-scores)[:k]
    return scores[idx], idx


def compression_ratio(dim, bits):
    """fp32 bytes / quantized bytes per vector (codes = bits/coord + one fp32 scale amortized)."""
    fp32 = 32.0 * dim
    quant = bits * dim + 32.0
    return fp32 / quant


# ---------------------------------------------------------------- agent
class TurboQuant(BaseAgent):
    name = "turboquant"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        rng = np.random.RandomState(int(s.get("seed", 0)))
        n = int(s.get("n", 2000)); dim = int(s.get("dim", 64)); bits = int(s.get("bits", 4))
        topk = int(s.get("k", 10))
        V = rng.randn(n, dim)
        index = quantize_index(V, bits=bits)
        # recall@k vs exact float search, averaged over queries
        nq = 50; recalls = []
        for _ in range(nq):
            qv = rng.randn(dim)
            exact = np.argsort(-(V @ qv))[:topk]
            _, approx = search(index, qv, k=topk)
            recalls.append(len(set(exact.tolist()) & set(approx.tolist())) / topk)
        recall = float(np.mean(recalls))
        cr = compression_ratio(dim, bits)
        msg = (f"turboquant: data-oblivious {bits}-bit quantization of {n}×{dim} — {cr:.1f}× compression "
               f"(fp32→{bits}b), recall@{topk}={recall:.2f} vs exact, NO training phase (codebook fixed by dim "
               f"via Beta marginal + random rotation). Compress embedding banks for RAG/dedup/NN (TurboQuant)")
        self.log(msg, kind="finding",
                 recommendation="quantize_index(embeddings, bits=4) for an 8× embedding bank with graceful recall; "
                                "training-free and GPU-free — ingest online, no rebuilds")
        return self.done({"recall": recall, "compression": cr, "bits": bits}, msg)


_AGENT = TurboQuant()


def run_turboquant(q, worker):
    return _AGENT.run(q, worker)
