"""metric_probe_test — DATA-WISE verifier for the metric-probe agent (no heavy deps, fast).

Plants tiny SYNTHETIC metrics with KNOWN degeneracies and asserts the prober (1) DETECTS each exploit,
(2) QUANTIFIES its delta correctly, (3) does NOT false-flag a perturbation the metric is immune to, and
(4) satisfies the (status,data,to,message) agent contract on an empty spec (synthetic demo path).
"""
import os
import sys

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import metric_probe as MP


def _run():
    print("=== METRIC-PROBE DATA-WISE VERIFIER ===")
    checks = {}

    # ---- 1. global-reachability-credit degeneracy (the toy analogue of the real biohub division bug) ----
    score, _ = MP._toy_reachability_metric(n_real=6)
    base = MP._toy_baseline(n_real=6)                       # 6 disconnected real nodes → detect 1.0, credit 0
    rep = MP.probe(score, base, [MP.hub_unify_fpfree(), MP.garbage_edges(n=3)])
    by = {r["name"]: r for r in rep["results"]}
    checks["base_score_is_1.0"] = abs(rep["base_score"] - 1.0) < 1e-9
    hub = by["hub_unify_fpfree"]
    checks["hub_detected_exploit"] = hub["exploit"] is True
    checks["hub_delta_quantified_+0.1"] = abs(hub["delta"] - 0.1) < 1e-9   # EXACT known degeneracy magnitude
    checks["hub_bug_class"] = hub["bug_class"] == "global-reachability-credit"
    # garbage edges (unmatched, don't touch real-node connectivity) must NOT move THIS metric → not false-flagged
    checks["garbage_not_false_flagged"] = by["garbage_edges(n=3)"]["exploit"] is False
    print(f"  hub_unify delta={hub['delta']:+.4f} (want +0.1000)  garbage delta={by['garbage_edges(n=3)']['delta']:+.4f}")

    # ---- 2. under-prediction-bonus degeneracy (size-ratio adjustment rewards predicting FEWER nodes) ----
    n_est = 100.0

    def under_metric(g):
        ratio = (g.num_nodes() - n_est) / n_est
        return 0.9 * (1 - 0.1 * ratio)                     # edge-J saturated at 0.9; only N_pred matters

    base2 = MP.DictGraph({i: {} for i in range(6)}, [])
    rep2 = MP.probe(under_metric, base2, [MP.node_sparsify(frac=0.5, seed=0)])
    sp = rep2["results"][0]
    checks["sparsify_detected_exploit"] = sp["exploit"] is True
    checks["sparsify_delta_positive"] = sp["delta"] is not None and sp["delta"] > 0   # score goes UP on fewer nodes
    checks["sparsify_bug_class"] = sp["bug_class"] == "under-prediction-bonus"
    print(f"  node_sparsify delta={sp['delta']:+.5f} (want > 0, under-prediction bonus)")

    # ---- 3. a ROBUST metric (counts real correctness only) must yield NO exploit ----
    def robust_metric(g):
        return sum(1 for n in g.node_ids() if n < 6) / 6.0  # only real nodes count; injections are inert
    rep3 = MP.probe(robust_metric, MP._toy_baseline(6), MP.graph_perturbation_suite())
    checks["robust_metric_no_exploit"] = rep3["n_exploits"] == 0
    print(f"  robust metric: n_exploits={rep3['n_exploits']} (want 0), verdict='{rep3['verdict'][:40]}...'")

    # ---- 4. agent contract on empty spec (synthetic demo path) ----
    status, data, to, msg = MP.run({"question": "metric-probe smoke", "spec": {}}, "test")
    checks["contract_status_done"] = status == "done"
    checks["contract_finds_exploit"] = isinstance(data, dict) and data.get("n_exploits", 0) >= 1
    checks["contract_reports_bug_class"] = "global-reachability-credit" in (data.get("bug_classes") or [])

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== metric-probe: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)
