"""tracker_consensus_test — plant clean node detections; assert consensus links > 0 and consensus ⊆ union."""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import pandas as pd
import numpy as np
from fleet_agents import tracker_consensus


def _make_nodes(path):
    # 20 cells moving smoothly over 6 frames → trackers should agree on the obvious links
    rng = np.random.RandomState(0)
    base = rng.rand(20, 3) * np.array([5, 40, 40])
    rows, nid = [], 1
    for t in range(6):
        pos = base + t * 0.5
        for i in range(20):
            rows.append({"node_id": nid, "t": t, "z": pos[i, 0], "y": pos[i, 1], "x": pos[i, 2]}); nid += 1
    pd.DataFrame(rows).to_csv(path, index=False)


def test_tracker_consensus_produces_agreed_links():
    with tempfile.TemporaryDirectory() as d:
        _make_nodes(os.path.join(d, "emb_test.csv"))
        s, res, to, msg = tracker_consensus.run(
            {"question": "t", "spec": {"nodes_dir": d, "max_embryos": 1, "agreement_k": 3}}, "test")
        assert s == "done", msg
        assert res["consensus"] > 0, f"no consensus links found: {res}"
        assert res["consensus"] <= res["union"], f"consensus {res['consensus']} > union {res['union']} (impossible)"
        return {"consensus_positive": res["consensus"] > 0, "consensus_subset_union": res["consensus"] <= res["union"]}


def _run():
    print("=== TRACKER-CONSENSUS DATA-WISE VERIFIER ===")
    try:
        r = test_tracker_consensus_produces_agreed_links()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== tracker-consensus: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
