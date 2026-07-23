"""Learner agent — capture NEW knowledge as a Pattern-B lesson (.py pure code + .learning) and refresh.

The rule (user): whenever the team learns something new — a kept experiment, a finding, or a user
'teach me X' request — it becomes a lesson in the established Pattern B: a `.py` (PURE runnable code,
no prose) next to a `.learning` (all explanation + the code shown + REAL captured outputs), same
basename. This agent writes both and refreshes the `.learning` via learning/lessonkit.py so the :7777
hub shows it. Deterministic (no LLM writes the lesson content — it's handed in via the spec).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
LEARN_DIR = COMP / "learning" / "fleet_lessons"
LESSONKIT = COMP / "learning" / "lessonkit.py"
KV = "/home/seshu/miniconda3/envs/kaggle_vision/bin/python"


def add_lesson(lesson_id: str, title: str, note: str, code: str) -> dict:
    """Write the Pattern-B pair (<id>.py pure code + <id>.learning) and refresh the .learning."""
    LEARN_DIR.mkdir(parents=True, exist_ok=True)
    py = LEARN_DIR / f"{lesson_id}.py"
    learning = LEARN_DIR / f"{lesson_id}.learning"
    py.write_text(code if code.endswith("\n") else code + "\n")   # PURE code only
    learning.write_text(f"# {title}\n\n{note}\n\n--- code\n{code}\n--- output\n")  # explanation + code + (output filled by refresh)
    refreshed = False
    if LESSONKIT.exists():
        r = subprocess.run([KV, str(LESSONKIT), str(learning)], cwd=str(COMP),
                           capture_output=True, text=True, timeout=180)
        refreshed = (r.returncode == 0)
    return {"py": str(py), "learning": str(learning), "refreshed": refreshed}


def learn(q, worker):
    """Fleet handler — add a lesson if the spec carries one; else report the learner is ready."""
    spec = q.get("spec", {})
    if spec.get("title") and spec.get("code"):
        res = add_lesson(spec.get("id", "fleet_lesson"), spec["title"], spec.get("note", ""), spec["code"])
        return ("done", res, "all",
                f"[{worker}] LEARNER: added lesson '{spec['title']}' → {Path(res['learning']).name} "
                f"(Pattern B: pure .py + .learning{', refreshed' if res['refreshed'] else ''}; shows on :7777).")
    n = len(list(LEARN_DIR.glob("*.learning"))) if LEARN_DIR.exists() else 0
    return ("done", {"lessons": n}, "all",
            f"[{worker}] LEARNER ready: any NEW finding → a Pattern-B lesson (pure .py + .learning with real "
            f"outputs, refreshed via lessonkit, shown on :7777). {n} fleet lesson(s) so far. "
            f"Hand me {{id,title,note,code}} and I'll write the pair.")
