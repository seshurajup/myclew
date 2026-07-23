"""ext_label_stats_test — plant tracks with KNOWN node/link/division counts; assert the agent counts them."""
import os, sys, tempfile
from pathlib import Path
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import pandas as pd
from fleet_agents import ext_label_stats


def _make_tracks(path):
    # 3 tracks × 5 frames = 15 nodes; consecutive links = 3×4 = 12; 1 division (track 3 has parent = track 1)
    rows = []
    for tid, parent in [(1, -1), (2, -1), (3, 1)]:
        for t in range(5):
            rows.append({"track_id": tid, "t": t, "z": t * 1.0, "y": t * 1.0, "x": t * 1.0, "parent_track_id": parent})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_ext_label_stats_counts_match_ground_truth():
    with tempfile.TemporaryDirectory() as d:
        _make_tracks(os.path.join(d, "ZSNS_test_tracks.csv"))
        old = ext_label_stats.TRACKS
        ext_label_stats.TRACKS = Path(d)
        try:
            s, res, to, msg = ext_label_stats.report({"question": "t", "spec": {}}, "test")
        finally:
            ext_label_stats.TRACKS = old
        assert s == "done", msg
        assert res["nodes"] == 15, f"expected 15 nodes, got {res['nodes']}"
        assert res["links"] == 12, f"expected 12 consecutive links, got {res['links']}"
        return {"nodes_correct": res["nodes"] == 15, "links_correct": res["links"] == 12}


def _run():
    print("=== EXT-LABEL-STATS DATA-WISE VERIFIER ===")
    try:
        r = test_ext_label_stats_counts_match_ground_truth()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== ext-label-stats: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
