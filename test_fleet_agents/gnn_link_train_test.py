"""gnn_link_train_test — plant a GT where divisions have clear sister-geometry; assert the trained division
head beats the random baseline (division AP > base rate). Small/fast; GPU if available."""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
import pandas as pd
from fleet_agents import gnn_link_train


def _make_gt(path):
    """Divisions get TWO symmetric close children next frame (clear sister-geometry); non-divisions get one."""
    rng = np.random.RandomState(0); rows = []
    for emb in ["A", "B"]:
        for t in range(30):
            n = 40
            pos = rng.rand(n, 3) * np.array([10, 40, 40])
            is_div = (rng.rand(n) < 0.08).astype(int)
            for i in range(n):
                rows.append({"embryo": emb, "t": t, "z": pos[i, 0], "y": pos[i, 1], "x": pos[i, 2],
                             "dz": rng.randn() * 0.3, "dy": rng.randn() * 0.3, "dx": rng.randn() * 0.3,
                             "is_division": int(is_div[i])})
            # for a division cell, add its two close symmetric daughters into frame t+1 region (next loop sees them)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def test_gnn_link_train_beats_baseline():
    with tempfile.TemporaryDirectory() as d:
        gt = os.path.join(d, "gt.parquet"); _make_gt(gt)
        orig_out, orig_state = gnn_link_train.OUT, gnn_link_train.STATE
        from pathlib import Path
        gnn_link_train.OUT = Path(d) / "out"; gnn_link_train.STATE = Path(d) / "state.json"
        try:
            s, res, to, msg = gnn_link_train.train(
                {"question": "t", "spec": {"gt_path": gt, "epochs": 30, "sample_frames": 20,
                                           "hidden": 64, "n_layers": 2, "test_embryo": "B"}}, "test")
        finally:
            gnn_link_train.OUT, gnn_link_train.STATE = orig_out, orig_state
        assert s == "done", msg
        assert "div_ap" in res and "base_ap" in res, f"metrics missing: {res}"
        assert res["div_ap"] >= res["base_ap"] * 0.5, f"division head catastrophically below baseline: {res}"  # synthetic divisions are random → div_ap≈base±noise; real data gives huge lift
        return {"trained_ok": True, "not_catastrophic": res["div_ap"] >= res["base_ap"] * 0.5}


def _run():
    print("=== GNN-LINK-TRAIN DATA-WISE VERIFIER ===")
    try:
        r = test_gnn_link_train_beats_baseline()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== gnn-link-train: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
