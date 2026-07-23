"""base_record_test — verify the BaseAgent.record() feature actually WORKS (writes a CV-ranked experiment
row to the journal ledger), not just that the method exists. This is the feature that lets training/scoring
agents put a real golden-CV row in the journal's sorted table (vs only findings). Cleans up its test row."""
import json
import os
import sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import fleet_agents as fa

LEDGER = os.path.join(COMP, "docs", "experiment_ledger.jsonl")
MARK = "base-record-selftest-XYZ"


def _rows():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER) if l.strip()]


def test_record_writes_a_cv_row():
    before = len(_rows())
    agent = fa.AGENTS["xai"]                       # any BaseAgent
    row = agent.record(change=MARK, cv=0.9123, description="record self-test", train_set="golden12")
    try:
        after = _rows()
        assert len(after) >= before, "ledger did not grow"
        mine = [r for r in after if r.get("change") == MARK]
        assert mine, f"no row with change={MARK} was written"
        assert abs(float(mine[-1].get("cv")) - 0.9123) < 1e-6, f"cv not recorded: {mine[-1].get('cv')}"
        assert mine[-1].get("trn_set") == "golden12", f"trn_set not recorded: {mine[-1].get('trn_set')}"
        ok = True
    finally:
        # CLEAN UP: remove the self-test row so we don't pollute the real journal
        rows = [l for l in open(LEDGER) if l.strip() and MARK not in l]
        open(LEDGER, "w").writelines(rows)
    return {"row_written": ok, "cv_recorded": True}


def _run():
    print("=== BASEAGENT.record() DATA-WISE VERIFIER ===")
    try:
        r = test_record_writes_a_cv_row()
        for k, v in r.items(): print(f"  {'OK' if v else 'X'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  X FAILED: {e}"); ok = False
    print(f"=== base-record: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
