"""sub_journal_test — DATA-WISE verifier for the sub-journal agent. Fully OFFLINE: the Kaggle CLI is
monkeypatched to return a small fixed submissions CSV, and the ledger is redirected to a temp dir + its
Postgres/insights side-writes are disabled — so NOTHING touches the real Kaggle API or the real journal.

Asserts:
  • parse_submissions + parse_cv are correct (cv-in-description parsed, no-cv → None, no-score dropped),
  • the agent RECORDS new entries with cv parsed from the description,
  • it writes the docs/sub_<ref>.json provenance artifacts,
  • BOTH public and private are set on each journaled entry,
  • it is IDEMPOTENT — a 2nd run journals 0 new,
  • a duplicate ref already present in a hand-written entry is ENRICHED, not double-recorded.
"""
import os
import sys
import tempfile
from pathlib import Path

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

from fleet_agents import sub_journal as SJ
from fleet_agents import ledger as L


# a small fixed submissions CSV: a cv-in-description row, a no-cv row, a no-score row (dropped), and a row
# whose ref is ALREADY represented by a pre-seeded hand-written journal entry (must enrich, not duplicate).
CSV = (
    "ref,fileName,date,description,status,publicScore,privateScore\n"
    "20000003,submission.csv,2026-07-16 03:00:00,climb model cv0.75 run,SubmissionStatus.COMPLETE,0.90000,0.91000\n"
    "20000002,submission.csv,2026-07-16 02:00:00,plain baseline no cv here,SubmissionStatus.COMPLETE,0.80000,0.82000\n"
    "20000009,submission.csv,2026-07-16 02:30:00,still scoring,SubmissionStatus.COMPLETE,,\n"
    "20000001,submission.csv,2026-07-16 01:00:00,pre-existing hand-written run,SubmissionStatus.COMPLETE,0.70000,0.72000\n"
)


def _run():
    print("=== SUB-JOURNAL DATA-WISE VERIFIER ===")
    checks = {}

    # ---- pure parsers ----
    subs = SJ.parse_submissions(CSV)
    checks["parse_only_complete"] = len(subs) == 4                        # all 4 are COMPLETE (one has no score)
    by_ref = {s["ref"]: s for s in subs}
    checks["parse_public_private"] = (by_ref["20000003"]["public"] == 0.9
                                      and by_ref["20000003"]["private"] == 0.91)
    checks["parse_cv_in_desc"] = SJ.parse_cv("climb model cv0.75 run") == 0.75
    checks["parse_cv_absent"] = SJ.parse_cv("plain baseline no cv here") is None
    checks["expand_abbrev_refs"] = SJ._expand_refs("subs 54761848/50/51") == {"54761848", "54761850", "54761851"}

    # ---- isolate the ledger to a temp dir; disable PG + insights side writes ----
    tmp = Path(tempfile.mkdtemp(prefix="subjournal_test_"))
    docs = tmp / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    L._comp_docs = lambda slug=None: docs                                 # redirect ledger reads/writes
    L._pg_journal = lambda rows: None                                     # no Postgres in the test
    L._pg_decisions = lambda rows: None
    os.environ["LEDGER_NO_INSIGHTS"] = "1"                                # no insights refresh

    # pre-seed a hand-written entry that already covers ref 20000001 (mentions it in its observation)
    L.record(change="hand-written-baseline", script="manual",
             description="the original baseline, sub 20000001 on the LB", observation="")

    spec = {"spec": {"competition": "test-subjournal-comp", "rows": CSV, "out_dir": str(docs)}}

    # ---- 1st run ----
    status, data, to, msg = SJ.run(spec, "tester")
    checks["run_ok"] = status == "done"
    checks["journaled_two_new"] = data.get("journaled") == 2              # 20000003 + 20000002 (no-score dropped)
    checks["enriched_one_existing"] = data.get("skipped") == 1            # 20000001 already present → enriched

    entries = {e.get("change"): e for e in L.entries()}
    e3 = entries.get("ksub:20000003")
    e2 = entries.get("ksub:20000002")
    checks["recorded_cv_from_desc"] = e3 is not None and e3.get("cv") == 0.75
    checks["recorded_no_cv_is_none"] = e2 is not None and e2.get("cv") is None
    checks["both_public_private_new"] = (e3.get("public") == 0.9 and e3.get("private") == 0.91
                                         and e2.get("public") == 0.8 and e2.get("private") == 0.82)
    # the pre-existing entry got ENRICHED with both scores (not duplicated)
    hand = next((e for e in L.entries() if e.get("change") == "hand-written-baseline"), None)
    checks["existing_enriched"] = hand is not None and hand.get("public") == 0.7 and hand.get("private") == 0.72
    checks["no_dup_for_existing_ref"] = "ksub:20000001" not in entries    # never double-recorded

    # provenance artifacts written for every processed submission
    checks["artifact_written"] = (docs / "sub_20000003.json").exists() and (docs / "sub_20000002.json").exists()
    import json as _j
    art = _j.loads((docs / "sub_20000003.json").read_text())
    checks["artifact_has_scores_and_source"] = (art.get("public") == 0.9 and art.get("private") == 0.91
                                                and art.get("cv") == 0.75
                                                and art.get("source") == "kaggle competitions submissions")

    n_after_first = len(L.entries())

    # ---- 2nd run: IDEMPOTENT ----
    status2, data2, _, _ = SJ.run(spec, "tester")
    checks["idempotent_zero_new"] = data2.get("journaled") == 0
    checks["idempotent_no_growth"] = len(L.entries()) == n_after_first

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("    1st run:", {k: data.get(k) for k in ("journaled", "skipped")},
          "| 2nd run journaled:", data2.get("journaled"))
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
