"""box_sample_test — plant a DENSE embryo (2000 cells/frame); assert box-sample produces boxes whose median
cell count is near the competition target (~250), not the dense original. Density-matching, ground-truth."""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np, pandas as pd
from fleet_agents import box_sample


def _make_dense_gt(path):
    rng = np.random.RandomState(0); rows = []
    for t in range(20):
        n = 2000                                       # DENSE — like ZSNS
        div = (rng.rand(n) < 0.01).astype(int)
        z = rng.rand(n)*10; y = rng.rand(n)*500; x = rng.rand(n)*500
        for i in range(n):
            rows.append({"embryo": "DENSE", "t": t, "z": z[i], "y": y[i], "x": x[i],
                         "dz": rng.randn()*0.3, "dy": rng.randn()*0.3, "dx": rng.randn()*0.3, "is_division": int(div[i])})
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_box_sample_matches_competition_density():
    with tempfile.TemporaryDirectory() as d:
        gt = os.path.join(d, "dense.parquet"); out = os.path.join(d, "boxed.parquet")
        _make_dense_gt(gt)
        s, res, to, msg = box_sample.run({"question": "t", "spec": {
            "gt_path": gt, "target_cells": 250, "boxes_per_frame": 4, "n_boxes": 12, "target_frames": 8, "out_path": out}}, "test")
        assert s == "done", msg
        med = res["median_cells_per_box"]
        assert 100 <= med <= 500, f"boxed density {med} not near competition target 250 (dense original was 2000)"
        assert res["boxes"] >= 5, f"too few boxes: {res['boxes']}"
        # COMPLETE PATHS: each box must be a spatio-temporal crop spanning many frames (not per-frame fragments)
        boxed = pd.read_parquet(out)
        frames_per_box = boxed.groupby("embryo")["t"].nunique()
        # windowed to ~8 frames AND multi-frame (complete paths within window), not fragmented, not full-movie
        assert 4 <= frames_per_box.median() <= 12, f"track length not windowed to ~8: median {frames_per_box.median()}"
        return {"density_matched": 100 <= med <= 500, "produced_boxes": res["boxes"] >= 5,
                "complete_paths": 4 <= frames_per_box.median() <= 12}


def test_label_drop_keeps_complete_tracks():
    """The competition author labels a FEW COMPLETE (unbroken) tracks. Assert box-sample's label-drop
    (_keep_complete_tracks) keeps k WHOLE lineages (all frames, no gaps), NOT random per-frame nodes."""
    import numpy as np, pandas as pd
    from fleet_agents.box_sample import _keep_complete_tracks
    rng = np.random.RandomState(0); rows = []
    for _ in range(5):                                        # 5 cells moving by a fixed flow across 10 frames
        z, y, x = rng.rand(3) * np.array([60, 250, 250]); vz, vy, vx = rng.randn(3) * 2
        for t in range(10):
            rows.append({"t": t, "z": z, "y": y, "x": x, "dz": vz, "dy": vy, "dx": vx, "is_division": 0})
            z, y, x = z + vz, y + vy, x + vx
    kept = _keep_complete_tracks(pd.DataFrame(rows), 2, np, rng)
    per = kept.groupby("t").size()
    unbroken = kept["t"].nunique() == 10 and int(per.max()) <= 2      # all frames present, ~2 tracks
    return {"keeps_2_complete_tracks": unbroken, "no_frame_gaps": kept["t"].nunique() == 10}


def _run():
    print("=== BOX-SAMPLE DATA-WISE VERIFIER ===")
    try:
        r = test_box_sample_matches_competition_density()
        r.update(test_label_drop_keeps_complete_tracks())     # track-integrity (competition pattern match)
        for k, v in r.items(): print(f"  {'OK' if v else 'X'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  X FAILED: {e}"); ok = False
    print(f"=== box-sample: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
