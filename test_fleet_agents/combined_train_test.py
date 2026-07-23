"""combined_train_test — plant a small external GT (2 embryos), empty competition dir; assert combined-train
runs the gate + trainer and returns a held-out division AP (the payoff orchestration works offline)."""
import os, sys, tempfile
from pathlib import Path
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np, pandas as pd
from fleet_agents import combined_train, gnn_link_train


def _ext(path):
    rng = np.random.RandomState(0); rows = []
    for emb in ["X", "Y"]:
        for t in range(24):
            n = 40; div = (rng.rand(n) < 0.02).astype(int); pos = rng.rand(n, 3) * np.array([10, 40, 40])
            for i in range(n):
                rows.append({"embryo": emb, "t": t, "z": pos[i,0], "y": pos[i,1], "x": pos[i,2],
                             "dz": rng.randn()*0.3, "dy": rng.randn()*0.3, "dx": rng.randn()*0.3, "is_division": int(div[i])})
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_combined_train_runs_gate_and_trainer():
    with tempfile.TemporaryDirectory() as d:
        ext = os.path.join(d, "ext.parquet"); _ext(ext); empty = os.path.join(d, "nogeff"); os.makedirs(empty)
        gnn_link_train.OUT = Path(d)/"out"; gnn_link_train.STATE = Path(d)/"state.json"
        s, res, to, msg = combined_train.run({"question": "t", "spec": {
            "external_gt": ext, "train_dir": empty, "golden": [], "detection_cells": 40, "epochs": 12, "sample_frames": 10,
            "hidden": 32, "n_layers": 2}}, "test")
        assert s == "done", msg
        assert "div_ap" in res and res["div_ap"] is not None, f"no division AP: {res}"
        assert res["external_nodes"] > 0, f"external not loaded: {res}"
        return {"gate_and_train_ran": True, "returns_div_ap": res["div_ap"] is not None}


def _run():
    print("=== COMBINED-TRAIN DATA-WISE VERIFIER ===")
    try:
        r = test_combined_train_runs_gate_and_trainer()
        for k, v in r.items(): print(f"  {'OK' if v else 'X'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  X FAILED: {e}"); ok = False
    print(f"=== combined-train: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
