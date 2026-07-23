"""gnn_probe_test — plant tracks with locally-COHERENT flow (so neighbourhood context is informative);
assert the agent computes both AUCs and context does not hurt (its core comparison is sound)."""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
import pandas as pd
from fleet_agents import gnn_probe


def _make_tracks(path):
    rng = np.random.RandomState(0)
    n = 120
    pos = rng.rand(n, 3) * np.array([10, 60, 60])
    rows = []
    for t in range(8):
        # locally-coherent velocity field: velocity depends on position (neighbours move alike)
        vel = np.stack([np.sin(pos[:, 1] / 20), np.cos(pos[:, 2] / 20), 0.2 * np.ones(n)], axis=1)
        pos = pos + vel + rng.randn(n, 3) * 0.2
        for i in range(n):
            rows.append({"track_id": i, "t": t, "z": pos[i, 0], "y": pos[i, 1], "x": pos[i, 2]})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_gnn_probe_compares_context_vs_pairwise():
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "coh_test.csv"); _make_tracks(f)
        s, res, to, msg = gnn_probe.report(
            {"question": "t", "spec": {"tracks_glob": os.path.join(d, "*.csv"), "file_filter": [],
                                       "n_frames": 6, "radius_um": 8.0, "k_neigh": 8, "vox": [1, 1, 1]}}, "test")
        assert s == "done", msg
        assert "auc_pairwise" in res and "auc_context" in res, f"AUCs not computed: {res}"
        assert 0.4 <= res["auc_pairwise"] <= 1.0 and 0.4 <= res["auc_context"] <= 1.0, f"bad AUCs: {res}"
        assert res["auc_context"] >= res["auc_pairwise"] - 0.02, f"context should not hurt on coherent flow: {res}"
        return {"aucs_computed": True, "context_not_worse": res["auc_context"] >= res["auc_pairwise"] - 0.02}


def _run():
    print("=== GNN-PROBE DATA-WISE VERIFIER ===")
    try:
        r = test_gnn_probe_compares_context_vs_pairwise()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== gnn-probe: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
