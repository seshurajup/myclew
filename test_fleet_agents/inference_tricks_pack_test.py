"""inference_tricks_pack_test — data-wise verifier for the WBF/snapshot/TTA/BN primitives (offline, pure numpy).

  • box-WBF merges two near-duplicate boxes into ONE with averaged coords.
  • point-WBF clusters nearby points and keeps far ones separate.
  • snapshot-average of identical probs == the same probs; rank-average is monotone.
  • multi-TTA on an identity predict_fn round-trips (inverse ∘ forward == id) → fused == input.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import inference_tricks_pack as I


def _run():
    print("=== INFERENCE-TRICKS PACK VERIFIER ===")
    checks = {}

    # ---- box-WBF: two near-duplicate boxes from 2 models → one fused box with averaged coords
    b1 = [[0.10, 0.10, 0.20, 0.20]]
    b2 = [[0.12, 0.12, 0.22, 0.22]]
    boxes, sc = I.weighted_boxes_fusion([b1, b2], [[0.9], [0.8]], iou_thr=0.4)
    checks["wbf_merges_to_one"] = len(boxes) == 1
    exp = (np.array(b1[0]) * 0.9 + np.array(b2[0]) * 0.8) / (0.9 + 0.8)
    checks["wbf_avg_coords"] = np.allclose(boxes[0], exp, atol=1e-4)
    print(f"  -> WBF {len(boxes)} box, coords {np.round(boxes[0],4)} vs exp {np.round(exp,4)}")

    # two FAR boxes (low IoU) stay separate
    far, _ = I.weighted_boxes_fusion([[[0.0, 0.0, 0.1, 0.1]], [[0.8, 0.8, 0.9, 0.9]]],
                                     [[0.9], [0.9]], iou_thr=0.4)
    checks["wbf_keeps_far_separate"] = len(far) == 2

    # ---- point-WBF: near points cluster, far point separate (3D)
    p_a = [[10.0, 10.0, 5.0], [50.0, 50.0, 5.0]]
    p_b = [[10.5, 10.2, 5.1]]                               # near the first point of model A
    pts, psc = I.weighted_points_fusion([p_a, p_b], [[0.9, 0.7], [0.8]], dist_thr=3.0)
    checks["pwbf_clusters"] = len(pts) == 2                 # {near-pair fused} + {far point}
    # the fused near-cluster centroid is a conf-weighted mean near (10,10,5)
    near = pts[np.argmin(np.linalg.norm(pts - np.array([10, 10, 5]), axis=1))]
    checks["pwbf_centroid"] = np.linalg.norm(near - np.array([10.17, 10.09, 5.05])) < 0.5
    print(f"  -> point-WBF {len(pts)} pts, near-centroid {np.round(near,3)}")

    # ---- snapshot-average: identical probs → same; weighted mean sanity
    probs = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3]])
    same = I.snapshot_average([probs, probs, probs], mode="prob")
    checks["snap_identity"] = np.allclose(same, probs)
    mix = I.snapshot_average([np.array([0.0, 1.0]), np.array([1.0, 0.0])], weights=[3, 1], mode="prob")
    checks["snap_weighted"] = np.allclose(mix, [0.25, 0.75])

    # rank-average is monotone: preserves order of a monotone-increasing vector
    a = np.array([0.1, 0.4, 0.9, 0.95]); b = np.array([-5.0, 0.0, 2.0, 3.0])
    r = I.snapshot_average([a, b], mode="rank")
    checks["snap_rank_monotone"] = bool(np.all(np.diff(r) > 0)) and r.min() >= 0 and r.max() <= 1
    print(f"  -> rank-avg {np.round(r,3)} monotone={checks['snap_rank_monotone']}")

    # ---- multi-TTA round-trip on identity predict_fn (2D, with a leading channel dim)
    x = np.random.RandomState(0).rand(2, 8, 6)
    tfms = I.tta_transforms_2d()
    for _n, fwd, inv in tfms:                               # each transform is invertible
        assert np.allclose(inv(fwd(x)), x), f"transform {_n} not invertible"
    out = I.multi_tta(x, lambda arr: arr, tfms, fuse="mean")
    checks["tta_roundtrip"] = np.allclose(out, x, atol=1e-8)
    # a non-identity predict_fn (add constant) still fuses to input+const under identity-equivariance
    out2 = I.multi_tta(x, lambda arr: arr + 1.0, tfms, fuse="mean")
    checks["tta_equivariant"] = np.allclose(out2, x + 1.0, atol=1e-8)
    print(f"  -> multi-TTA {len(tfms)} transforms, round-trip max-err {np.max(np.abs(out-x)):.1e}")

    # ---- agent contracts
    st, d, to, msg = I.run_wbf({"spec": {"mode": "box", "boxes_list": [b1, b2],
                                         "scores_list": [[0.9], [0.8]], "iou_thr": 0.4}}, "t")
    checks["wbf_agent"] = st == "done" and len(d["boxes"]) == 1
    st, d, to, msg = I.run_wbf({"spec": {"mode": "point", "points_list": [p_a, p_b],
                                         "scores_list": [[0.9, 0.7], [0.8]], "dist_thr": 3.0}}, "t")
    checks["pwbf_agent"] = st == "done" and len(d["points"]) == 2
    st, d, to, msg = I.run_snapshot({"spec": {"outputs": [probs.tolist(), probs.tolist()], "mode": "prob"}}, "t")
    checks["snap_agent"] = st == "done" and np.allclose(np.array(d["averaged"]), probs)
    st, d, to, msg = I.run_tta({"spec": {"x": x.tolist(), "fuse": "mean"}}, "t")
    checks["tta_agent"] = st == "done" and d["round_trip_err"] < 1e-8

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== inference-tricks-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
