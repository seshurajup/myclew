"""volumetric_detection_pack_test — data-wise verifier for the 3D heatmap detection primitives (offline, CPU).

Builds a synthetic 3D volume with known keypoints, checks the encode→tile→decode round-trip actually recovers
the planted points (recall≈1, low FP), stitch reconstruction is exact, blob mode filters by size, and every
agent contract returns done()."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import volumetric_detection_pack as V


def _run():
    print("=== VOLUMETRIC DETECTION PACK VERIFIER (synthetic 3D) ===")
    checks = {}; rng = np.random.RandomState(0)
    shape = (24, 48, 48); n = 15
    pts = np.stack([rng.uniform(5, shape[d] - 5, n) for d in range(3)], 1)

    # --- encode: peaks ~1 at each planted point ---
    hm = V.gaussian_heatmap(shape, pts, sigma=1.6, n_classes=1)[0]
    checks["encode_shape"] = hm.shape == shape
    checks["encode_peak_near_1"] = 0.9 <= hm.max() <= 1.0001
    vals = [hm[tuple(np.round(p).astype(int))] for p in pts]
    checks["encode_hot_at_points"] = float(np.mean(vals)) > 0.6

    # --- tiling covers the whole volume, minimal + fixed overlap ---
    for ov in (None, 8):
        coords = V.tile_coords(shape, (16, 32, 32), (ov, ov, ov) if ov else None)
        patches = [np.ones((16, 32, 32)) for _ in coords]
        cov = (V.stitch(shape, patches, coords, (16, 32, 32)) > 0).mean()
        checks[f"tiling_full_coverage_ov{ov}"] = cov == 1.0

    # --- stitch reconstructs an arbitrary volume exactly (overlap-averaged) ---
    vol = rng.rand(*shape)
    coords = V.tile_coords(shape, (16, 32, 32))
    patches = [vol[c[0]:c[0]+16, c[1]:c[1]+32, c[2]:c[2]+32] for c in coords]
    recon = V.stitch(shape, patches, coords, (16, 32, 32))
    checks["stitch_exact"] = np.abs(recon - vol).max() < 1e-9

    # --- peak decode recovers planted points (high recall, controlled FP) ---
    noisy = hm + rng.rand(*shape) * 0.05
    det = V.decode_peaks(noisy, threshold=0.3, min_distance=3, mode="peak", subpixel=True)
    tp, prec, rec = V.match_points(det[:, :3], pts, tol=2.0)
    print(f"  -> peak decode: {len(det)} dets, recall={rec:.3f} precision={prec:.3f}")
    checks["peak_recall_high"] = rec >= 0.85
    checks["peak_precision_ok"] = prec >= 0.7
    checks["peak_subpixel_noninteger"] = np.any(np.abs(det[:, :3] - np.round(det[:, :3])) > 1e-6) if len(det) else False

    # --- radius NMS collapses a tight cluster to one ---
    cl = np.array([[10, 10, 10], [10, 10, 11], [10, 11, 10]], float)
    keep = V.radius_nms(cl, np.array([0.9, 0.8, 0.7]), radius=3.0)
    checks["nms_collapses_cluster"] = len(keep) == 1

    # --- blob decode + voxel-count size filter drops a tiny speck ---
    bl = V.gaussian_heatmap(shape, pts[:5], sigma=2.0, n_classes=1)[0]
    bl[2, 2, 2] = 1.0                                            # 1-voxel speck
    big = V.decode_peaks(bl, threshold=0.3, mode="blob", size_min=5)
    small = V.decode_peaks(bl, threshold=0.3, mode="blob", size_min=0)
    checks["blob_size_filter_drops_speck"] = len(big) < len(small)

    # --- 2D works too (competition-agnostic dimensionality) ---
    p2 = np.array([[10.0, 20.0], [30.0, 40.0]])
    hm2 = V.gaussian_heatmap((50, 60), p2, sigma=1.5, n_classes=1)[0]
    d2 = V.decode_peaks(hm2, threshold=0.3, min_distance=2, mode="peak")
    _, _, r2 = V.match_points(d2[:, :2], p2, tol=2.0)
    checks["works_in_2d"] = r2 == 1.0

    # --- agent contracts ---
    st, d, to, msg = V.run_encode({"spec": {"shape": [20, 32, 32], "n": 8, "sigma": 1.5}}, "t")
    checks["encode_agent"] = st == "done" and d["peak"] > 0.9
    st, d, to, msg = V.run_patch({"spec": {"shape": [30, 64, 64], "patch": [16, 32, 32]}}, "t")
    checks["patch_agent"] = st == "done" and d["coverage"] == 1.0 and d["recon_err"] < 1e-6
    st, d, to, msg = V.run_decode({"spec": {"seed": 1, "mode": "peak"}}, "t")
    checks["decode_agent"] = st == "done" and d["recall"] >= 0.7

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== volumetric-detection-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
