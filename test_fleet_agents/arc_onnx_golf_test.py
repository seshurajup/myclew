"""arc_onnx_golf_test — data-wise verifier for the arc-onnx-golf TOOL (must actually emit + verify + cost
real ONNX under the OFFICIAL one-hot [1,10,30,30] semantics). Synthetic grids, no Kaggle. Skips gracefully
(PASS) if onnx/onnxruntime are unavailable (matches how guarded/heavy agents self-skip under the test venv).
"""
import os
import sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

try:
    import onnx  # noqa
    import onnxruntime  # noqa
    import numpy as np  # noqa
    _HAVE = True
except Exception:  # noqa: BLE001
    _HAVE = False


def _run():
    print("=== ARC-ONNX-GOLF DATA-WISE VERIFIER ===")
    if not _HAVE:
        print("  ~ onnx/onnxruntime not in this env — SKIP (agent import stays lazy so the fleet loads)")
        print("=== arc-onnx-golf: PASS (skipped) ===")
        return True
    from fleet_agents import arc_onnx_golf as G
    checks = {}

    # 1) transpose (diagonal reflection) — a 3x3 non-symmetric grid → cost 0, score 25
    g = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    gt = [list(r) for r in zip(*g)]
    r = G.golf([(g, gt)], test=[(g, gt)])
    checks["transpose_solved"] = r["solved"] is True
    checks["transpose_cost0"] = r["cost"] == 0
    checks["transpose_score25"] = abs(r["score"] - 25.0) < 1e-6

    # 2) palette bijection recolor: 1<->2 swap, 0 fixed → Gather cost 10, score 22.697415
    def recolor(grid, m):
        return [[m.get(c, c) for c in row] for row in grid]
    m = {1: 2, 2: 1}
    src = [[0, 1, 2], [1, 2, 0], [2, 0, 1]]
    r2 = G.golf([(src, recolor(src, m))], test=[([[1, 1], [2, 0]], recolor([[1, 1], [2, 0]], m))])
    checks["recolor_solved"] = r2["solved"] is True
    checks["recolor_transform"] = r2["transform"] == "recolor"
    checks["recolor_cost10"] = r2["cost"] == 10
    checks["recolor_score"] = abs(r2["score"] - 22.697415) < 1e-4

    # 3) many-to-one colour map (2->1 AND 1->1) → not channel-bijective → conv fallback, cost 100
    m2 = {2: 1}
    src3 = [[0, 1, 2], [2, 1, 0], [1, 2, 0]]
    r3 = G.golf([(src3, recolor(src3, m2))], test=[([[2, 2], [1, 0]], recolor([[2, 2], [1, 0]], m2))])
    checks["conv_solved"] = r3["solved"] is True
    checks["conv_transform"] = r3["transform"] == "recolor_conv"
    checks["conv_cost100"] = r3["cost"] == 100

    # 4) official validity + ban-list gate present, and a WRONG model is honestly reported unsolved
    bad = G.golf([(g, recolor(g, {0: 5}))], test=[(g, recolor(g, {0: 5}))], transform="transpose")
    checks["wrong_model_unsolved"] = bad["solved"] is False

    # 5) empty-spec agent path returns the emitter catalogue (contract tuple)
    status, res, to, msg = G.run({"question": "smoke", "spec": {}}, "test")
    checks["catalogue_path"] = status == "done" and isinstance(res, dict) and res.get("n_emitters", 0) >= 5

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    print(f"  -> transpose cost={r['cost']} score={r['score']}; recolor cost={r2['cost']} score={r2['score']}; "
          f"conv cost={r3['cost']} score={r3['score']}")
    ok = all(checks.values())
    print(f"=== arc-onnx-golf: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
