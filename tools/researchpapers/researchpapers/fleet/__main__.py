"""Fleet orchestrator (COMPETITION-AGNOSTIC) — 3..10 deterministic Python worker-agents.

  python -m researchpapers.fleet --workers 5      # daemon (3..10 workers)
  python -m researchpapers.fleet --once           # drain the board once, then exit
  python -m researchpapers.fleet --dry-run --once # validate; throwaway board, print, no chat writes
  python -m researchpapers.fleet --status         # board counts and exit

The competition's SEED (questions) and HANDLERS (kind -> deterministic fn) come from its
`fleet_agents` package via registry.load_competition(). This framework only: runs the worker loop,
posts insights into the runtime chat (sync), and escalates 'reason' questions to a Claude agent.
"""
from __future__ import annotations

import argparse
import re
import threading
import time

from . import board, post
from .registry import load_competition

# status chatter that is NOT useful to the leader (findings/results/escalations are important)
_ROUTINE_RE = re.compile(
    r"READY \(|dry-run mode|no scored runs|holding|picked up|queue busy|no jobs yet"
    r"|training healthy|no training running|SMOKE GREEN"
    r"|no NEW top notebooks|already synced|no new items", re.I)

HANDLERS: dict = {}  # populated in main() from the competition package (+ generic 'reason')
ON_RESULT = None     # optional competition hook: log EVERY agent's finding to its journal


def handle_reason(q, worker):
    """Generic escalation — the ONLY place a Claude agent is engaged (framework-level, not competition)."""
    to = q["spec"].get("agent", "researcher")
    return ("escalated", {"escalated_to": to}, to,
            f"[{worker}] REASONING NEEDED → {to}: {q['question']}")


def worker_loop(wid: int, once: bool, stop: threading.Event) -> None:
    name = f"fleet-{wid}"
    while not stop.is_set():
        q = board.claim_next(name)
        if not q:
            if once:
                return
            time.sleep(5)
            continue
        agent = q["kind"]  # the AGENT (its function) is the chat identity — not the anonymous pool worker
        try:
            handler = HANDLERS.get(q["kind"])
            if not handler:
                board.complete(q["id"], "failed", {"error": f"no handler for kind {q['kind']}"})
                post.post_thread(agent, "all", f"[{agent}] no handler for kind '{q['kind']}'")
            else:
                status, result, to, msg = handler(q, agent)  # handler names itself by function
                if ON_RESULT:  # competition hook — ALL agents contribute findings to the journal (deduped)
                    try:
                        ON_RESULT(agent, status, result, to, msg)
                    except Exception:  # noqa: BLE001
                        pass
                # 'holding' = waiting for the one GPU; retried silently so the chat isn't flooded.
                # '_posted' = the agent already posted+updated its OWN live message (e.g. verify-cv's
                # ⏳→✓ single-message progress) — don't double-post.
                if status != "holding" and not (isinstance(result, dict) and result.get("_posted")):
                    # findings/escalations = important (seen by leader); status chatter = routine (hidden by default)
                    routine = (to == "all") and bool(_ROUTINE_RE.search(msg))
                    # structured template: kind + data payload so agents can READ each other deterministically
                    post.post_thread(agent, to, msg, routine=routine, kind=q["kind"],
                                     data=result if isinstance(result, dict) else {})
                board.complete(q["id"], status, result)
        except Exception as exc:  # noqa: BLE001
            board.complete(q["id"], "failed", {"error": str(exc)})
            post.post_thread(agent, "all", f"[{agent}] FAILED {q['question'][:40]}: {exc}",
                             kind=q["kind"], data={"error": str(exc)})


def main() -> int:
    global HANDLERS, ON_RESULT
    ap = argparse.ArgumentParser(description="Deterministic research-fleet workers (competition-agnostic)")
    ap.add_argument("--workers", type=int, default=5, help="worker count (clamped to 3..10)")
    ap.add_argument("--once", action="store_true", help="drain open questions once, then exit")
    ap.add_argument("--dry-run", action="store_true", help="print actions, write nothing to the thread")
    ap.add_argument("--interval", type=float, default=5.0, help="idle poll seconds (daemon mode)")
    ap.add_argument("--status", action="store_true", help="print board stats and exit")
    a = ap.parse_args()

    comp = load_competition()
    HANDLERS = dict(comp.HANDLERS)
    HANDLERS.setdefault("reason", handle_reason)
    ON_RESULT = getattr(comp, "on_result", None)  # optional: journal contribution from every agent

    if a.status:
        print(f"competition: {getattr(comp, 'NAME', '?')} · board:",
              board.seed(comp.SEED), "questions ·", board.stats())
        return 0

    post.DRY = a.dry_run
    if a.dry_run:
        import tempfile
        from pathlib import Path
        board.DB = Path(tempfile.mkstemp(prefix="fleet-dry-", suffix=".db")[1])  # throwaway board
    n = max(3, min(10, a.workers))
    total = board.seed(comp.SEED)
    print(f"fleet up [{getattr(comp, 'NAME', '?')}]: {total} questions · {n} workers · "
          f"handlers={sorted(HANDLERS)} · dry_run={a.dry_run} · once={a.once}")

    stop = threading.Event()
    threads = [threading.Thread(target=worker_loop, args=(i + 1, a.once, stop), daemon=True) for i in range(n)]
    for t in threads:
        t.start()
    try:
        if a.once:
            for t in threads:
                t.join()
        else:
            reanalyze_every, last = 600.0, time.time()
            hold_every, last_hold = 30.0, time.time()
            daily_every, last_daily = 86400.0, 0.0   # notebook-sync: truly once a day (agent also self-gates 20h)
            while True:
                time.sleep(a.interval)
                if time.time() - last_daily >= daily_every:
                    board.reopen("notebook-sync")    # DAILY public-notebook sweep — not every 10 min
                    last_daily = time.time()
                if time.time() - last_hold >= hold_every:
                    board.reopen_status("holding")  # retry queue-held work every 30s — covers metrics-report
                    board.reopen("train-monitor")    # LIVE watchdog: re-sample the running training each cycle
                    board.reopen("scorer")           # SAFETY NET: backfill every golden_cv MLflow→journal each cycle
                    board.reopen("orchestrate")      # SELF-DRIVE: pick+enqueue the next experiment (no leader needed)
                    board.reopen("insights")         # HANDOFF: keep docs/INSIGHTS.md current for the super-agents
                    board.reopen("plan-ingest")      # HUMAN DIRECTION: pick up edits to docs/human_plan.yml
                    last_hold = time.time()          # NOTE: do NOT reopen 'metrics-report' when done → it re-posts
                if time.time() - last >= reanalyze_every:
                    n_re = board.reopen("analysis")
                    if n_re:
                        post.post_thread("orchestrator", "all",
                                         f"[orchestrator] re-opened {n_re} analysis question(s) to re-decompose fresh MLflow runs",
                                         kind="orchestrator")
                    last = time.time()
    except KeyboardInterrupt:
        stop.set()
    print("board:", board.stats())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
