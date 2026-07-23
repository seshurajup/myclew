"""perf_choice_test — assert the benchmark ranks vectorised/GPU FASTER than the per-node Python loop (the
anti-pattern), so the agent always recommends against per-node loops."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import perf_choice


def test_recommends_vectorised_over_loop():
    s, d, to, m = perf_choice.run({"question": "t", "spec": {"n_frames": 60, "n_cells": 300, "k": 2}}, "test")
    assert s == "done", m
    tim = d["timings"]
    assert d["best"] != "per_node_loop", f"per-node loop should NEVER be the winner: {tim}"
    assert tim["vectorised_cpu"] < tim["per_node_loop"], f"vectorised must beat per-node loop: {tim}"
    assert d["speedup_vs_loop"] > 1, f"no speedup measured: {d}"
    return {"loop_not_best": d["best"] != "per_node_loop", "vectorised_faster": tim["vectorised_cpu"] < tim["per_node_loop"]}


def _run():
    print("=== PERF-CHOICE DATA-WISE VERIFIER ===")
    try:
        r = test_recommends_vectorised_over_loop()
        for k, v in r.items(): print(f"  {'OK' if v else 'X'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  X FAILED: {e}"); ok = False
    print(f"=== perf-choice: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
