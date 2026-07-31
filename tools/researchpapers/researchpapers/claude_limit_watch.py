"""claude-limit watchdog — hold the Claude agents (leader/researcher) on a usage-limit, resume on reset.

Claude Code (the leader/researcher tmux panes) pauses when it hits the usage limit. This watcher makes
that EXPLICIT and safe for the workflow:
  * detects the limit signature in the leader/researcher panes,
  * writes a `.claude_hold` flag (with the reset time) + posts ONE '⏸ ON HOLD' status to the thread,
  * when the panes clear (Claude auto-resumes at reset), posts '▶ RESUMED' and removes the flag.

The DETERMINISTIC Python fleet is Claude-independent, so it keeps running (analysis/scoring/journaling/
monitoring) the whole time; only GPU experiments (leader-driven) pause, since a limited leader can't POST
new ones — they resume automatically when the limit resets. The flag lets the idle-monitor + fleet avoid
nudging/spamming a limited leader.

  python -m researchpapers.claude_limit_watch --interval 60
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

SESSION = "research-runtime"
AGENT_WINDOWS = ("leader", "researcher")
TMUX = os.environ.get("TMUX_BIN", "/home/seshu/miniconda3/bin/tmux")
# Claude Code usage-limit signatures (broad — matches the pane text when rate-limited)
_LIMIT_RE = re.compile(
    r"usage limit reached|approaching (your )?usage limit|reached your usage limit|"
    r"5-hour limit|rate limit|limit will reset|resets at|try again (later|at)", re.I)
_RESET_RE = re.compile(r"reset[s]?\s*(?:at|in)?\s*([0-9:apm ]+)", re.I)


def _runtime_dir() -> Path:
    env = os.environ.get("RESEARCH_MVP_RUNTIME_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[1] / ".research-mvp-data" / "runtime"


FLAG = _runtime_dir() / ".claude_hold"


def _pane(window: str) -> str:
    try:
        return subprocess.run([TMUX, "capture-pane", "-t", f"{SESSION}:{window}", "-p", "-S", "-40"],
                              capture_output=True, text=True, timeout=6).stdout
    except Exception:  # noqa: BLE001
        return ""


def _limited():
    """(is_limited, which_agents, reset_hint) — scan the Claude panes for the limit signature."""
    hit, reset = [], ""
    for w in AGENT_WINDOWS:
        txt = _pane(w)
        if _LIMIT_RE.search(txt):
            hit.append(w)
            m = _RESET_RE.search(txt)
            if m and not reset:
                reset = m.group(1).strip()
    return bool(hit), hit, reset


def _post(content, kind):
    try:
        from researchpapers.fleet import post
        post.post_thread("claude-watch", "all", content, routine=False, kind=kind,
                         data={"hold": kind == "claude-hold"})
    except Exception:  # noqa: BLE001
        pass


def tick():
    limited, who, reset = _limited()
    held = FLAG.exists()
    if limited and not held:
        FLAG.parent.mkdir(parents=True, exist_ok=True)
        FLAG.write_text(f"agents={','.join(who)} reset={reset or '?'} since={int(time.time())}")
        _post(f"⏸ ON HOLD — Claude usage limit hit for {', '.join(who)} (resets {reset or 'soon'}). "
              f"The Python fleet keeps running deterministic work; GPU experiments resume when the leader is back.",
              "claude-hold")
    elif held and not limited:
        FLAG.unlink(missing_ok=True)
        _post("▶ RESUMED — Claude limit cleared; leader/researcher back online, experiments continue.",
              "claude-resume")


def main() -> int:
    ap = argparse.ArgumentParser(description="Hold Claude agents on usage-limit; resume on reset.")
    ap.add_argument("--interval", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    while True:
        tick()
        if a.once:
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    raise SystemExit(main())
