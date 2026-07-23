"""misc_domain_pack_test — verifier for the domain post-processing / FE agents (offline)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import misc_domain_pack as M


def _run():
    print("=== MISC DOMAIN PACK VERIFIER ===")
    checks = {}

    # hierarchy: parent(0) must be >= children(1,2). Start with parent low, child high → propagates up.
    probs = np.array([[0.2, 0.9, 0.1], [0.5, 0.3, 0.7]])
    edges = [(1, 0), (2, 0)]  # (child, parent)
    P = M.propagate_hierarchy(probs, edges)
    checks["hierarchy_parent_ge_child"] = np.all(P[:, 0] >= np.maximum(P[:, 1], P[:, 2]) - 1e-9)
    checks["hierarchy_lifts"] = P[0, 0] == 0.9  # lifted to max child

    # invariance: same shape rotated by an angle → same normalized coords (up to numeric)
    pts = np.array([[1.0, 0.0], [0.0, 1.0]])
    n1 = M.egocentric_normalize(pts, origin=[0, 0], axis_angle=0.0)
    ang = 0.7
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pts_rot = pts @ R.T
    n2 = M.egocentric_normalize(pts_rot, origin=[0, 0], axis_angle=ang)
    checks["invariance_matches"] = np.allclose(n1, n2, atol=1e-9)

    # template retrieval: nearest template returned first
    q = np.array([1.0, 0.0]); templates = np.array([[0.9, 0.1], [-1, 0], [0, 1.0]])
    ranked = M.retrieve_rerank(q, templates, k=3)
    checks["template_nearest_first"] = ranked[0] == 0

    # calendar: a known weekend date flagged
    X, names = M.calendar_features(["2025-01-04"])  # 2025-01-04 is a Saturday
    checks["calendar_weekend"] = X[0, names.index("is_weekend")] == 1.0
    X2, _ = M.calendar_features(["2025-01-06"])  # Monday
    checks["calendar_weekday"] = X2[0, names.index("is_weekend")] == 0.0

    # annotation-error: planted wrong labels flagged
    rng = np.random.RandomState(0); y = rng.rand(200); pred = y + rng.normal(0, 0.02, 200)
    y[10] += 1.0; y[50] -= 1.0  # two clear errors
    idx, thr = M.flag_label_errors(y, pred, z=3.0)
    checks["annot_flags_errors"] = 10 in idx and 50 in idx

    # binary compressor: repetitive data compresses
    res = M.compress_artifact(b"ABCD" * 500, cap=100)
    checks["bincompress"] = res["compressed_bytes"] < res["orig_bytes"] and res["under_cap"] is True

    # knn-feature agent
    Xk = rng.rand(300, 4); yk = (Xk[:, 0] > 0.5).astype(float)
    st, d, to, msg = M.run_knnfeat({"spec": {"X": Xk.tolist(), "y": yk.tolist()}}, "t")
    checks["knn_feature_agent"] = st == "done" and "_feat" in d

    # agent contracts
    st, d, to, msg = M.run_hier({"spec": {"probs": probs.tolist(), "edges": [list(e) for e in edges]}}, "t")
    checks["hier_agent"] = st == "done"
    st, d, to, msg = M.run_annoterr({"spec": {"y": y.tolist(), "oof_pred": pred.tolist()}}, "t")
    checks["annot_agent"] = st == "done" and len(d["flagged_idx"]) >= 2

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== misc-domain-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
