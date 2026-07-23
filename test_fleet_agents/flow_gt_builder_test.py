"""flow_gt_builder_test — plant tracks with a KNOWN division; assert flow vectors + division label built."""
import os, sys, tempfile
from pathlib import Path
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import pandas as pd
from fleet_agents import flow_gt_builder


def _make_tracks(path):
    # parent track 1 (frames 0-2) splits into tracks 2 & 3 (start frame 3) → 1 division at (track1, t2)
    rows = []
    for t in range(3):
        rows.append({"track_id": 1, "t": t, "z": t, "y": t, "x": t, "parent_track_id": -1})
    for tid in (2, 3):
        for t in range(3, 6):
            rows.append({"track_id": tid, "t": t, "z": t, "y": t + tid, "x": t, "parent_track_id": 1})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_flow_gt_builder_builds_flow_and_division():
    with tempfile.TemporaryDirectory() as d:
        _make_tracks(os.path.join(d, "ZSNS_test_tracks.csv"))
        old_t, old_out = flow_gt_builder.TRACKS, flow_gt_builder.OUT
        flow_gt_builder.TRACKS = Path(d); flow_gt_builder.OUT = Path(d) / "out"
        try:
            s, res, to, msg = flow_gt_builder.build({"question": "t", "spec": {}}, "test")
        finally:
            flow_gt_builder.TRACKS, flow_gt_builder.OUT = old_t, old_out
        assert s == "done", msg
        assert res["rows"] > 0, f"no flow vectors built: {res}"
        assert res["divisions"] >= 1, f"missed the planted division: {res}"
        return {"flow_built": res["rows"] > 0, "division_found": res["divisions"] >= 1}


def _run():
    print("=== FLOW-GT-BUILDER DATA-WISE VERIFIER ===")
    try:
        r = test_flow_gt_builder_builds_flow_and_division()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== flow-gt-builder: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
