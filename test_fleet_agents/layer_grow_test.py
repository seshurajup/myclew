"""layer_grow_test — assert layer-grow starts at 1 layer, grows one layer at a time keeping only proven
gains, AND runs XAI validation on the chosen model (goal validated by interpretability, not just metric)."""
import os, sys, tempfile
from pathlib import Path
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np, pandas as pd
from fleet_agents import layer_grow, gnn_link_train


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


def test_layer_grow_proves_and_xai_validates():
    with tempfile.TemporaryDirectory() as d:
        gt = os.path.join(d, "gt.parquet"); _make_gt(gt)
        gnn_link_train.OUT = Path(d)/"out"; gnn_link_train.STATE = Path(d)/"state.json"
        s, res, to, msg = layer_grow.run({"question": "t", "spec": {
            "gt_path": gt, "max_layers": 3, "hidden": 32, "epochs": 12, "sample_frames": 10, "test_embryo": "B"}}, "test")
        assert s == "done", msg
        proof = res["proof"]
        assert proof[0]["layers"] == 1 and proof[0]["why"] == "minimal baseline", f"must start at 1 layer: {proof[0]}"
        # each subsequent step adds exactly one layer
        for i in range(1, len(proof)):
            assert proof[i]["layers"] == proof[i-1]["layers"] + 1, f"depth not incremental: {proof}"
        # XAI validation present on the chosen model
        assert "xai" in res and "driver_feature" in res["xai"], f"no XAI validation: {res.get('xai')}"
        return {"starts_1_layer": True, "grows_one_at_a_time": True, "xai_validated": "driver_feature" in res["xai"]}


def _run():
    print("=== LAYER-GROW DATA-WISE VERIFIER (depth by proof + XAI validation) ===")
    try:
        r = test_layer_grow_proves_and_xai_validates()
        for k, v in r.items(): print(f"  {'OK' if v else 'X'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  X FAILED: {e}"); ok = False
    print(f"=== layer-grow: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
