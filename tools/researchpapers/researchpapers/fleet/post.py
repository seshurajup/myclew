"""Post messages into the SAME runtime thread the Claude agents use.

Replicates researchpapers.runtime_cli.queue_message's on-disk format (thread.jsonl event +
per-recipient inbox file) so fleet workers show up in the :7788 board chat with their own
names, without being registered as Claude tmux agents. Set DRY=True to print instead of write.
"""
from __future__ import annotations

import datetime
import json
import os
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl  # POSIX file lock so concurrent workers don't clobber the thread on rewrite
except Exception:  # noqa: BLE001
    fcntl = None

DRY = False  # fleet sets this in --dry-run so nothing touches the real thread


@contextmanager
def _locked(path: Path):
    """Exclusive lock guarding read-modify-write of thread.jsonl (append is safe; rewrite needs this)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lp = path.with_suffix(path.suffix + ".lock")
    f = open(lp, "w")
    try:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        yield
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def runtime_dir() -> Path:
    env = os.environ.get("RESEARCH_MVP_RUNTIME_DIR")
    if env:
        return Path(env)
    # package lives at <repo>/researchpapers/fleet ; runtime state at <repo>/.research-mvp-data/runtime
    return Path(__file__).resolve().parents[2] / ".research-mvp-data" / "runtime"


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _msg_sig(text: str) -> str:
    """Normalised signature (whitespace-collapsed header+opening) so SAME or near-identical messages collapse."""
    import re
    return re.sub(r"\s+", " ", (text or "").strip().lower())[:160]


def _post_allowed(thread: Path, sender: str, content: str, routine: bool = False) -> bool:
    """False (skip) if this message repeats one already on the board. Two rules:
    (a) ROUTINE broadcasts only — same/SIMILAR signature as any of the last 3 entries → skip (kills the
        'same status every cycle' spam). NOT applied to non-routine msgs, so a LIVE-updating message's initial
        post is never suppressed (its follow-ups use update_thread, which bypasses this entirely).
    (b) ANY message — this sender's most-recent post had byte-identical content → skip.
    Fail-open on any error."""
    try:
        if not thread.exists():
            return True
        lines = thread.read_text(encoding="utf-8", errors="replace").splitlines()
        if routine:                                # (a) aggressive similar-dedup ONLY for routine spam
            sig = _msg_sig(content)
            for ln in lines[-3:]:
                try:
                    if _msg_sig(json.loads(ln).get("content", "")) == sig:
                        return False
                except Exception:  # noqa: BLE001
                    continue
        for ln in reversed(lines[-80:]):           # (b) this sender's latest identical post → skip (safe for all)
            try:
                e = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if e.get("from") == sender:
                return (e.get("content") or "") != content
        return True
    except Exception:  # noqa: BLE001
        return True


def post_thread(sender: str, to: str, content: str, routine: bool | None = None,
                kind: str | None = None, data: dict | None = None) -> str:
    """Append one message to the shared thread (+ inbox file if directed at one agent).

    MESSAGE TEMPLATE (every agent message follows this so agents can READ each other deterministically):
      from, to, kind (the agent/handler kind), content (human text), data (structured payload),
      routine (bool — status chatter hidden from the leader & board-by-default), timestamp, event_id.
    routine default: directed (to != 'all') = important, broadcast fleet status (to == 'all') = routine.
    """
    if routine is None:
        routine = (to == "all")
    event = {
        "timestamp": _now_iso(),
        "from": sender,
        "to": to,
        "type": "message",
        "kind": kind,
        "content": content,
        "data": data or {},
        "routine": bool(routine),
        "event_id": f"evt-{uuid.uuid4().hex[:12]}",
    }
    if DRY:
        print(f"[DRY post] {sender} -> {to}: {content[:100]}")
        return event["event_id"]
    rt = runtime_dir()
    thread = rt / "thread.jsonl"
    thread.parent.mkdir(parents=True, exist_ok=True)
    # GENERAL DEDUP (covers EVERY agent): if this sender's most-recent message has IDENTICAL content,
    # skip — kills the "same message repeated every cycle" spam (scorer/monitor/metrics-report/any).
    if not _post_allowed(thread, sender, content, routine):
        return event["event_id"]
    with _locked(thread):
        with open(thread, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    if to and to != "all":
        inbox = rt / "inbox" / to
        inbox.mkdir(parents=True, exist_ok=True)
        p = inbox / f"msg-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(event, f, ensure_ascii=False)
    return event["event_id"]


def update_thread(event_id: str, content: str, data: dict | None = None) -> bool:
    """UPDATE an existing message IN PLACE (same event_id) instead of posting a new one — so a long
    agent shows live progress on ONE line (⏳ running → progress → ✓ done) and /runtime stays clean.
    Race-safe via the thread lock. Returns True if the message was found and rewritten."""
    if DRY or not event_id:
        return False
    thread = runtime_dir() / "thread.jsonl"
    if not thread.exists():
        return False
    with _locked(thread):
        lines = thread.read_text(encoding="utf-8", errors="replace").splitlines()
        hit = False
        for i, ln in enumerate(lines):
            try:
                e = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if e.get("event_id") == event_id:
                e["content"] = content
                e["timestamp"] = _now_iso()      # bump so the board re-sorts it to the top
                if data is not None:
                    e["data"] = data
                lines[i] = json.dumps(e, ensure_ascii=False)
                hit = True
                break
        if hit:
            tmp = thread.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
            os.replace(tmp, thread)              # atomic swap
    return hit
