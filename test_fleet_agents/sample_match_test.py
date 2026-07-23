"""sample_match_test — with a KNOWN competition profile + a KNOWN external density, assert the gate flags
match vs mismatch correctly (density in range = match; 20x too dense = mismatch)."""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np, pandas as pd
from fleet_agents import sample_match

PROFILE = {"labelled_cells_per_frame": 1.5, "track_length_frames": 56, "division_frac": 0.005, "linked_frac": 0.97}


def _ext(path, cells_per_frame):
    rng = np.random.RandomState(0); rows = []
    for e in range(3):
        for t in range(15):
            for i in range(cells_per_frame):
                rows.append({"embryo": f"E{e}", "t": t, "is_division": int(rng.rand() < 0.005)})
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_gate_flags_match_and_mismatch():
    with tempfile.TemporaryDirectory() as d:
        good = os.path.join(d, "good.parquet"); bad = os.path.join(d, "bad.parquet")
        _ext(good, 300); _ext(bad, 6000)                     # good ~ competition density; bad = 20x too dense
        s1, r1, _, _ = sample_match.run({"question": "t", "spec": {
            "competition_profile": PROFILE, "detection_cells": 250, "external_gt": good,
            "competition_sister_ratio": 1.60, "external_sister_ratio": 1.53}}, "test")
        s2, r2, _, _ = sample_match.run({"question": "t", "spec": {
            "competition_profile": PROFILE, "detection_cells": 250, "external_gt": bad,
            "competition_sister_ratio": 1.60, "external_sister_ratio": 3.5}}, "test")
        assert r1["matched"] is True, f"competition-density external should MATCH: {r1['checks']}"
        assert r2["matched"] is False, f"20x-dense external should MISMATCH: {r2['checks']}"
        # sister-ratio must be gated too
        assert "sister_ratio" in r1["checks"] and r1["checks"]["sister_ratio"]["match"], "close sister-ratio should MATCH"
        assert not r2["checks"]["sister_ratio"]["match"], "far sister-ratio should MISMATCH"
        return {"matches_when_close": r1["matched"], "flags_when_dense": not r2["matched"],
                "sister_ratio_gated": True}


def _run():
    print("=== SAMPLE-MATCH DATA-WISE VERIFIER ===")
    try:
        r = test_gate_flags_match_and_mismatch()
        for k, v in r.items(): print(f"  {'OK' if v else 'X'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  X FAILED: {e}"); ok = False
    print(f"=== sample-match: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)


def _run_selection():
    from fleet_agents.sample_match import label_selection_discriminant
    # labeled cells brighter + more isolated than unlabeled → those features rank top by |d|
    lab={"intensity":[1.3,1.4,1.2,1.35],"isolation_um":[14,15,13,16],"z":[30,31,29,30]}
    unl={"intensity":[0.9,0.95,1.0,0.9],"isolation_um":[9,8,10,9],"z":[29,30,31,30]}
    rows=label_selection_discriminant(lab,unl)
    top={r["feature"] for r in rows[:2]}
    ok = rows[0]["feature"] in ("intensity","isolation_um") and top=={"intensity","isolation_um"} and abs(rows[-1]["cohens_d"])<0.5
    print("  [%s] label_selection_discriminant ranks bright+isolated as selectors" % ("PASS" if ok else "FAIL"))
    return ok
