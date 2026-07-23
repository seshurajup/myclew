"""turboquant_test — data-wise verifier for the TurboQuant data-oblivious quantizer (turbovec).

Core properties:
  1. make_rotation is orthogonal (Q Qᵀ = I) and deterministic for a fixed seed.
  2. After rotation, unit-vector coords follow the Beta((d-1)/2,(d-1)/2) marginal (mean ≈ 0, symmetric).
  3. lloyd_max_beta returns 2^bits centroids, sorted, symmetric about 0, in [-1,1].
  4. Reconstruction error decreases with more bits; recall@k beats random and rises with bits.
  5. Compression ratio ≈ 8× at 4-bit for reasonable dim.
  6. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import turboquant as TQ


def _run():
    print("=== TURBOQUANT VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}
    dim = 64

    # 1. rotation orthogonal + deterministic
    R = TQ.make_rotation(dim, 42)
    checks["rotation_orthogonal"] = np.allclose(R @ R.T, np.eye(dim), atol=1e-8)
    checks["rotation_deterministic"] = np.allclose(R, TQ.make_rotation(dim, 42))

    # 2. rotated coords ~ symmetric zero-mean marginal
    U = rng.randn(4000, dim); U = U / np.linalg.norm(U, axis=1, keepdims=True)
    rot = U @ R.T
    checks["marginal_zero_mean"] = abs(float(rot.mean())) < 0.02
    checks["marginal_bounded"] = bool(np.all(np.abs(rot) <= 1.0 + 1e-6))

    # 3. codebook
    bnd, cent = TQ.lloyd_max_beta(4, dim)
    checks["codebook_size"] = len(cent) == 16 and len(bnd) == 15
    checks["codebook_sorted"] = bool(np.all(np.diff(cent) > 0))
    checks["codebook_symmetric"] = abs(float(cent.mean())) < 1e-6
    checks["codebook_in_range"] = bool(cent.min() > -1 and cent.max() < 1)
    print(f"  -> 4-bit centroids (first 4): {np.round(cent[:4],3)}")

    # 4. more bits → lower reconstruction error; recall rises
    V = rng.randn(1500, dim)
    err = {}
    for b in (2, 4):
        idx = TQ.quantize_index(V, bits=b)
        rec = TQ.decode(idx["codes"], idx["scales"], idx["centroids"], idx["rotation"])
        err[b] = float(np.linalg.norm(rec - V, axis=1).mean())
    print(f"  -> mean recon error: 2-bit={err[2]:.3f}  4-bit={err[4]:.3f}")
    checks["more_bits_lower_err"] = err[4] < err[2]

    idx4 = TQ.quantize_index(V, bits=4); topk = 10; rec = []
    for _ in range(60):
        qv = rng.randn(dim)
        exact = set(np.argsort(-(V @ qv))[:topk].tolist())
        _, ap = TQ.search(idx4, qv, k=topk)
        rec.append(len(exact & set(ap.tolist())) / topk)
    recall = float(np.mean(rec))
    print(f"  -> recall@{topk} (4-bit) = {recall:.2f}  (random ≈ {topk/len(V):.3f})")
    checks["recall_beats_random"] = recall > 0.3
    checks["recall_meaningful"] = recall > 0.5

    # 5. compression
    cr = TQ.compression_ratio(dim, 4)
    checks["compression_8x"] = 6.0 < cr < 9.0
    print(f"  -> 4-bit compression ratio (dim {dim}) = {cr:.1f}x")

    # 6. agent
    st, dta, to, msg = TQ.run_turboquant({"spec": {"n": 1500, "dim": 64, "bits": 4}}, "t")
    checks["agent_done"] = st == "done" and dta["recall"] > 0.4

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== turboquant: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
