"""feasibility-gate / feasibility-map verifier — DATA-WISE self-test (no heavy agents, no real ledger writes).

Proves the orchestration + the Lever Feasibility Map insight:
  • verdict canonicalization (GO / NO-GO / WEAK-GO / CV-GO-LB-neutral)
  • record_gate → feasibility_gates round-trip (via the LEDGER, isolated by monkeypatch) + dedup latest-wins + ranked order
  • verdict DERIVED from measured evidence (Δ+significance+LB note) — honest GO/NO-GO/flag
  • math-master paired significance on a clear shift
  • official-score patched-metric reading (edge + 0.1·div) from a real JSON artifact
  • the map renders a ranked table; this SESSION's backfill spans GO + NO-GO + CV-GO/LB-neutral
"""
import os, sys, json, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import insights, feasibility_gate as FG


class _FakeLedger:
    """In-memory stand-in for the ledger decision store (isolates the test from the real journal)."""
    def __init__(self): self.rows = []
    def log(self, agent, summary, detail="", kind="finding", recommendation=None, **kw):
        e = {"agent": agent, "summary": summary, "finding": summary, "detail": detail,
             "kind": kind, "recommendation": recommendation, "ts": f"t{len(self.rows)}"}
        # mimic ledger.log dedup on (agent, summary, recommendation)
        for p in self.rows:
            if (p["agent"], p["finding"], p["recommendation"]) == (agent, summary, recommendation):
                return p
        self.rows.append(e); return e
    def decisions(self): return list(self.rows)
    def entries(self): return []                    # build_md needs this; no scored rows in the isolated test


def _run():
    print("=== FEASIBILITY-GATE VERIFIER ===")
    checks = {}

    # 1) verdict canonicalization
    checks["norm_go"] = insights._norm_verdict("adopt") == "GO"
    checks["norm_nogo"] = insights._norm_verdict("NO-GO") == "NO-GO" and insights._norm_verdict("killed") == "NO-GO"
    checks["norm_weak"] = insights._norm_verdict("WEAK-GO") == "WEAK-GO"
    checks["norm_flag"] = insights._norm_verdict("CV-GO/LB-neutral") == "CV-GO/LB-NEUTRAL"

    # 2) record_gate → feasibility_gates round-trip through the (isolated) ledger
    fake = _FakeLedger()
    real = insights.ledger
    real_out = insights.OUT
    _tmp_out = tempfile.NamedTemporaryFile(suffix="_INSIGHTS.md", delete=False)
    _tmp_out.close()
    insights.ledger = fake
    insights.OUT = __import__("pathlib").Path(_tmp_out.name)   # don't touch the real docs/INSIGHTS.md
    try:
        insights.record_gate("gap_fill", "GO", "bridges missed edges under 7µm", delta="+0.029", evidence="EXP_9")
        insights.record_gate("temporal-division", "NO-GO", "temporal≈single", delta="0.709 vs 0.690", evidence="EXP_5")
        insights.record_gate("edge-consensus", "CV-GO/LB-neutral", "CV over-credits", delta="+0.0026", evidence="EXP_7")
        # latest-wins dedup: re-record gap_fill with a changed mechanism
        insights.record_gate("gap_fill", "GO", "bridges missed edges (updated)", delta="+0.031", evidence="EXP_9b")
        gates = insights.feasibility_gates()
        names = [g["lever_name"] for g in gates]
        checks["roundtrip_3_levers"] = len(gates) == 3 and set(names) == {"gap_fill", "temporal-division", "edge-consensus"}
        gap = next(g for g in gates if g["lever_name"] == "gap_fill")
        checks["latest_wins"] = "updated" in gap["mechanism"] and gap["delta"] == "+0.031"
        # ranked order: GO before CV-GO/LB-NEUTRAL before NO-GO
        checks["ranked_order"] = names.index("gap_fill") < names.index("edge-consensus") < names.index("temporal-division")
        md = insights.feasibility_map_md()
        checks["map_renders_table"] = ("Lever Feasibility Map" in md and "| lever |" in md
                                       and "gap_fill" in md and "⛔" in md and "✅" in md)
        checks["map_counts"] = "1 GO" in md and "1 NO-GO" in md

        # 6) backfill spans all three verdict buckets, through the isolated ledger
        fake.rows.clear()
        st, res, to, msg = FG.backfill("tester")
        checks["backfill_ok"] = st == "done" and res["backfilled"] == len(FG.SESSION_GATES)
        checks["backfill_has_go"] = res["n_go"] >= 1
        checks["backfill_has_nogo"] = res["n_nogo"] >= 1
        checks["backfill_has_flag"] = res["n_cv_go_lb_neutral"] >= 1
        # the three named honesty gates from the task are present with the right verdicts
        bg = {g["lever_name"]: g["verdict"] for g in insights.feasibility_gates()}
        checks["temporal_is_nogo"] = bg.get("temporal-division") == "NO-GO"
        checks["detector_is_go"] = bg.get("detector-signal") == "GO"
        checks["consensus_is_flag"] = bg.get("edge-consensus") == "CV-GO/LB-NEUTRAL"
    finally:
        insights.ledger = real
        insights.OUT = real_out
        try: os.unlink(_tmp_out.name)
        except OSError: pass

    # 3) verdict derivation from measured evidence
    checks["derive_go"] = FG._derive_verdict({"44b6": 0.02}, {"mean_delta": 0.02, "significant": True}, None) == "GO"
    checks["derive_nogo"] = FG._derive_verdict({"44b6": -0.011}, {"mean_delta": -0.011, "significant": False}, None) == "NO-GO"
    checks["derive_weak"] = FG._derive_verdict({"44b6": 0.005}, {"mean_delta": 0.005, "significant": False}, None) == "WEAK-GO"
    checks["derive_flag"] = FG._derive_verdict({"44b6": 0.003}, {"mean_delta": 0.003, "significant": True}, "LB 0.888 neutral") == "CV-GO/LB-NEUTRAL"

    # 4) math-master paired significance on a clear positive shift
    before = [0.50, 0.52, 0.48, 0.51, 0.49, 0.50, 0.53]
    after = [0.55, 0.58, 0.54, 0.57, 0.56, 0.55, 0.59]
    txt, stats = FG._significance(before, after)
    checks["significance_positive_and_sig"] = stats["mean_delta"] > 0 and stats["significant"]

    # 5) official-score patched metric read (edge + 0.1·div) from a real JSON artifact
    with tempfile.TemporaryDirectory() as td:
        jp = os.path.join(td, "official_score.json")
        with open(jp, "w") as f:
            json.dump({"overall": {"edge_jaccard": 0.80, "division_jaccard": 0.50},
                       "per_embryo": {"44b6": {"edge_jaccard": 0.88, "division_jaccard": 0.40},
                                      "6bba": {"edge_jaccard": 0.78, "division_jaccard": 0.60}}}, f)
        pat = FG._official_patched(jp)
        checks["patched_metric_math"] = (abs(pat["44b6"] - (0.88 + 0.1 * 0.40)) < 1e-6
                                         and abs(pat["overall"] - (0.80 + 0.1 * 0.50)) < 1e-6)

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  [{'ok' if v else 'XX'}] {k}")
    print("RESULT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
