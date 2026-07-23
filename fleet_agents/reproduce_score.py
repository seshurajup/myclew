"""reproduce-score — golden-12 score a public notebook's pipeline (Python, no leader).

Learned-graph notebooks are already anchored by notebook-sync (shared pilkwang pipeline = 0.8708).
This handles the RULE-BASED DoG family: it runs the DoG params through our own golden-CV rule-based
scorer (experiments/pipeline/e28_rulebased_goldencv.py) and writes the real golden-12 CV to the journal.

Honest: if that scorer can't take the notebook's params non-interactively yet, we do NOT fake a number —
we leave the row LB-only and say so. (Wiring e28 to accept arbitrary DoG params is the remaining step.)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
E28 = COMP / "experiments" / "pipeline" / "e28_rulebased_goldencv.py"
PY = COMP / "research" / "cellmot_venv" / "bin" / "python"


def score(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}   # tolerate a missing/None spec
    ref = spec.get("ref", "?")
    exp = spec.get("exp")
    # can we run the rule-based golden-CV scorer at all?
    if not E28.exists() or not PY.exists():
        return ("done", {"ref": ref, "cv": None}, "all",
                f"[{worker}] reproduce-score {ref}: rule-based scorer not available → row stays LB-only (honest).")
    # NOTE: parameterized DoG reproduction is not yet wired non-interactively — do not fake a CV.
    # When e28 accepts --params, run it here and ledger.set_scores(exp, cv=<golden_cv>).
    return ("done", {"ref": ref, "family": "rule-based-DoG", "cv": None, "status": "reproduction-pending"}, "all",
            f"[{worker}] reproduce-score {ref}: rule-based DoG params captured; golden-12 run pending "
            f"(e28 param-wiring is the remaining step). Row stays LB-only until then — no faked CV.")
