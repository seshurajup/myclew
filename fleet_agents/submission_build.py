"""submission-build — assemble the Kaggle submission from the best predictions (HUMAN submits).

Deterministic: takes the chosen inference predictions (+ division post-proc) and writes the submission
file in the competition format. NEVER submits to Kaggle (human-gated). Escalates if the required output
format isn't wired yet — no fake file.
"""
from __future__ import annotations

from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
OUT = COMP / "submissions"


def build(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    method = spec.get("method")
    if not method:
        return ("escalated", {}, "leader", f"[{worker}] submission-build: which scored method to package?")
    split = str(spec.get("split", "split_0"))               # split: prediction split subdir to package (default split_0)
    preds = COMP / "research" / "official_repo" / "predictions" / "seshu" / str(method) / split
    if not preds.exists():
        return ("escalated", {"preds": str(preds)}, "researcher",
                f"[{worker}] submission-build: no predictions for '{method}' — run inference first.")
    OUT.mkdir(parents=True, exist_ok=True)
    # NOTE: the exact Kaggle output packaging (geff/zip layout) is competition-specific — wire it here.
    return ("escalated", {"method": method, "preds": str(preds)}, "leader",
            f"[{worker}] submission-build: predictions for '{method}' are ready at {preds}. The exact Kaggle "
            f"output packaging is the one step to wire. NEVER auto-submit — human submits after review.")
