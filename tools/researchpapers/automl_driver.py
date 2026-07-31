"""automl_driver — the goal-oriented Python-leader heartbeat for start_automl.sh.

Drives the FULL deterministic agent portfolio at sensible cadences (guarded against pile-up), so every
Python agent contributes toward the goal WITHOUT the Claude leader/researcher:

  every cycle : orchestrate (leader picks next experiment) · combo-search (golden-12 param grid)
  periodic    : pre-analysis (diagnose weakest link) · scorer (CV trajectory) · post-analysis (verdict)
  assemble    : best-config · pipeline-run (combined golden-CV) — turn learnings into a scored config
  report      : insights (handoff) · ledger (journal)
  discover    : kaggle-scout (LB) · notebook-sync (daily → feeds verify-cv with real params)

Each agent is only (re)enqueued when NONE of its kind is open/claimed, so the board never piles up.
Cadence is in cycles; one cycle = AUTOML_TICK seconds (default 120; --fast sets 60).
"""
import os
import sqlite3
import sys
import time

sys.path.insert(0, os.environ.get("FLEET_ROOT", os.getcwd()))
from researchpapers.fleet import board  # noqa: E402

TICK = int(os.environ.get("AUTOML_TICK", "120"))
GOAL = os.environ.get("AUTOML_GOAL", "keep improving golden-CV toward the public LB bar (0.885+)")

# (thread, kind, every_n_cycles, question) — the goal-oriented portfolio.
PORTFOLIO = [
    ("S", "scoreboard",     1,  "automl: refresh the live golden-CV leaderboard table"),
    ("S", "heal",           2,  "automl: on any training failure, escalate to Claude to fix"),
    ("S", "orchestrate",    1,  "automl: pick + enqueue the next experiment (Python leader)"),
    ("S", "combo-search",   1,  "automl: grid public-notebook post-proc on golden-12 (no Claude)"),
    ("S", "fullconfig-search", 1, "automl: WIDE 8-axis search over yaroslav-v4 full ILP config (0.8803) → beat public"),
    ("S", "config-ablate",  3,  "automl: leave-one-block-out ablation of the yaroslav-v4 config → load-bearing map"),
    ("A", "ext-label-stats",16, "automl: inventory external Zebrahub dense labels (flow prior + division count)"),
    ("S", "flow-gt-build",  24, "automl: build per-node flow+division GT from external tracks (affinity supervision)"),
    ("S", "block-synth",    3,  "automl: diff notebooks → compose NEW post-proc code-block recipes"),
    ("S", "combine-winners", 4, "automl: GM trick — stack the best of each lever into one recipe"),
    ("S", "ablate-best",    2,  "automl: GM trick — 'same as X but ONE change' from the best"),
    ("A", "pre-analysis",   4,  "automl: diagnose the current weakest link before the next experiment"),
    ("A", "post-analysis",  5,  "automl: verdict on the last experiment (delta / kept / rejected)"),
    ("A", "scorer",         6,  "automl: report the CV trajectory across runs"),
    ("S", "best-config",    8,  "automl: assemble the best inference config from public learnings"),
    ("S", "pipeline-run",   9,  "automl: score base + support models → combined golden-CV"),
    ("S", "insights",      12,  "automl: refresh docs/INSIGHTS.md handoff report"),
    ("S", "ledger",        14,  "automl: report the experiment ledger"),
    ("A", "eda-stats",     20,  "automl: data fingerprint (density/stage/motion/divisions)"),
    ("S", "kaggle-scout",  30,  "automl: pull the public LB + top notebooks (don't miss the floor)"),
    ("S", "notebook-sync", 360, "automl: DAILY pull new notebooks → real golden-12 verify-cv"),
]


def _open(kind: str) -> int:
    try:
        c = sqlite3.connect(board.DB, timeout=5)
        n = c.execute("SELECT count(*) FROM questions WHERE kind=? AND status IN ('open','claimed')",
                      (kind,)).fetchone()[0]
        c.close()
        return n
    except Exception:  # noqa: BLE001
        return 1  # on error assume busy → don't pile up


def main():
    print(f"[automl_driver] goal: {GOAL}", flush=True)
    print(f"[automl_driver] tick={TICK}s · driving {len(PORTFOLIO)} agent kinds", flush=True)
    board.add("S", "orchestrate", "automl start: Python leader — pick the next experiment", {})
    cycle = 0
    while True:
        enq = []
        for thread, kind, every, q in PORTFOLIO:
            if cycle % every == 0 and _open(kind) == 0:
                try:
                    board.add(thread, kind, q, {})
                    enq.append(kind)
                except Exception:  # noqa: BLE001
                    pass
        if enq:
            print(f"[automl_driver] cycle {cycle}: enqueued {', '.join(enq)}", flush=True)
        cycle += 1
        time.sleep(TICK)


if __name__ == "__main__":
    main()
