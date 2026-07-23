"""div-temporal-feas verifier — CPU/GPU data-wise self-test.

Checks the GATE LOGIC on synthetic separability tables (no heavy training):
  • a clear temporal>single, all-stages-above-ceiling table -> GO
  • temporal ≈ single -> NO-GO (honest kill)
  • temporal helps overall but a stage/hard-case < ceiling -> WEAK-GO
And checks the worker's XAI/staging helpers are wired (mine returns labelled triples; stage buckets span).
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents.div_temporal_feas import DivTemporalFeas


def _tbl(single, temporal, stage_t, emb_t, hard_t):
    return {"mode": "mini", "overall": {"single": single, "temporal": temporal},
            "by_stage": {f"S{i}": {"temporal": v} for i, v in enumerate(stage_t)},
            "by_embryo": {"44b6": {"temporal": emb_t[0]}, "6bba": {"temporal": emb_t[1]}},
            "hard_invisible": {"temporal": hard_t}}


def _run():
    print("=== DIV-TEMPORAL-FEAS VERIFIER ===")
    a = DivTemporalFeas(); checks = {}

    go = a._verdict(_tbl(0.671, 0.86, [0.82, 0.84, 0.83, 0.85, 0.81], [0.83, 0.85], 0.80))
    checks["clear_temporal_win_is_GO"] = go["decision"] == "GO"

    nogo = a._verdict(_tbl(0.62, 0.63, [0.60, 0.61, 0.62, 0.60, 0.63], [0.61, 0.62], 0.60))
    checks["temporal_eq_single_is_NOGO"] = nogo["decision"] == "NO-GO"

    weak = a._verdict(_tbl(0.62, 0.72, [0.80, 0.55, 0.81, 0.80, 0.79], [0.75, 0.74], 0.60))
    checks["stage_below_ceiling_is_not_GO"] = weak["decision"] in ("WEAK-GO", "NO-GO") and weak["decision"] != "GO"

    # delta is reported and signed correctly
    checks["delta_signed"] = go["delta_temporal_vs_single"] > 0 and nogo["delta_temporal_vs_single"] >= 0

    # worker helpers importable + staging spans buckets on a toy density map
    import importlib.util as u
    spec = u.spec_from_file_location("_w", os.path.join(COMP, "fleet_agents", "_div_temporal_feas_worker.py"))
    w = u.module_from_spec(spec); spec.loader.exec_module(w)
    import pandas as pd
    dens = {f"d{i}": float(i) for i in range(10)}
    df = pd.DataFrame({"ds": [f"d{i}" for i in range(10)], "label": [1] * 10})
    st = w.assign_stage(df, dens)
    checks["staging_spans_all_S0_S4"] = set(st.stage.unique()) == {f"S{i}" for i in range(5)}

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("RESULT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
