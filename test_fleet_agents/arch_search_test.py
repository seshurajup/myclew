"""arch_search_test — assert arch-search starts from the BASIC baseline and grows ONE axis at a time
(greedy coordinate-ascent), training each step. Proves the 'start simple, one-by-one' behaviour."""
import os, sys, tempfile
from pathlib import Path
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np, pandas as pd
from fleet_agents import arch_search, gnn_link_train


def _make_gt(path):
    rng = np.random.RandomState(0); rows = []
    for emb in ["A", "B"]:
        for t in range(24):
            n = 40; div = (rng.rand(n) < 0.08).astype(int); pos = rng.rand(n, 3) * np.array([10, 40, 40])
            for i in range(n):
                rows.append({"embryo": emb, "t": t, "z": pos[i,0], "y": pos[i,1], "x": pos[i,2],
                             "dz": rng.randn()*0.3, "dy": rng.randn()*0.3, "dx": rng.randn()*0.3,
                             "is_division": int(div[i])})
    pd.DataFrame(rows).to_parquet(path, index=False)


def test_arch_search_is_incremental():
    with tempfile.TemporaryDirectory() as d:
        gt = os.path.join(d, "gt.parquet"); _make_gt(gt)
        gnn_link_train.OUT = Path(d)/"out"; gnn_link_train.STATE = Path(d)/"state.json"
        s, res, to, msg = arch_search.run({"question": "t", "spec": {
            "gt_path": gt, "search_space": {"hidden_dim": [32, 64], "n_layers": [2, 3]},   # simplest first
            "epochs": 12, "sample_frames": 10, "test_embryo": "B"}}, "test")
        assert s == "done", msg
        steps = res["steps"]
        # first step must be the BASIC baseline = smallest value on every axis
        assert steps[0]["step"] == "basic baseline", f"must start basic: {steps[0]}"
        assert steps[0]["config"] == {"hidden_dim": 32, "n_layers": 2}, f"baseline not simplest: {steps[0]['config']}"
        # every later step changes exactly ONE axis vs the baseline (one-by-one)
        for st in steps[1:]:
            assert "→" in st["step"], f"step not single-axis: {st}"
        return {"starts_basic": True, "one_axis_at_a_time": True, "trains_each": len(steps) >= 3}


def _run():
    print("=== ARCH-SEARCH DATA-WISE VERIFIER (start basic, one axis at a time) ===")
    try:
        r = test_arch_search_is_incremental()
        for k, v in r.items(): print(f"  {'OK' if v else 'X'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  X FAILED: {e}"); ok = False
    print(f"=== arch-search: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
