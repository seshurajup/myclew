"""Shared question board (SQLite, WAL) — the fleet's work queue.

Deterministic workers atomically claim the next open question, run it, and record the result.
SQLite with BEGIN IMMEDIATE gives us safe concurrent claims across worker threads/processes.
"""
from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path

def _db_path() -> Path:
    """fleet.db location. RP_COMP_ROOT set → <RP_COMP_ROOT>/.research-mvp-data/fleet/fleet.db (a second
    board instance serving another comp); UNSET → this package's fleet.db (byte-identical). Never raises."""
    import os
    try:
        v = (os.environ.get("RP_COMP_ROOT") or "").strip()
        if v:
            p = Path(v).expanduser().resolve() / ".research-mvp-data" / "fleet" / "fleet.db"
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
    except Exception:  # noqa: BLE001
        pass
    return Path(__file__).resolve().parent / "fleet.db"


DB = _db_path()

SCHEMA = """
CREATE TABLE IF NOT EXISTS questions(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  thread TEXT, kind TEXT, question TEXT,
  spec TEXT DEFAULT '{}',
  status TEXT DEFAULT 'open',      -- open | claimed | done | escalated | failed
  claimed_by TEXT DEFAULT '',
  result TEXT DEFAULT '',
  created TEXT, updated TEXT
);
"""


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB, timeout=30)
    c.execute("PRAGMA journal_mode=WAL;")
    c.execute(SCHEMA)
    return c


def seed(questions) -> int:
    """Insert any MISSING seed questions (idempotent, dedup on question text). This lets NEW agent
    types added to a competition's SEED show up on an existing board — not just on a fresh one.
    questions=[(thread,kind,q,spec),...]."""
    c = _conn()
    try:
        have = {r[0] for r in c.execute("SELECT question FROM questions").fetchall()}
        for thread, kind, q, spec in questions:
            if q not in have:
                c.execute(
                    "INSERT INTO questions(thread,kind,question,spec,created,updated) VALUES(?,?,?,?,?,?)",
                    (thread, kind, q, json.dumps(spec or {}), _now(), _now()),
                )
        c.commit()
        return c.execute("SELECT count(*) FROM questions").fetchone()[0]
    finally:
        c.close()


def claim_next(worker: str):
    """Atomically claim the lowest-id open question. Returns a dict (spec parsed) or None."""
    c = _conn()
    try:
        c.execute("BEGIN IMMEDIATE")
        cur = c.execute("SELECT * FROM questions WHERE status='open' ORDER BY id LIMIT 1")
        row = cur.fetchone()
        if not row:
            c.execute("COMMIT")
            return None
        cols = [d[0] for d in cur.description]
        d = dict(zip(cols, row))
        c.execute("UPDATE questions SET status='claimed',claimed_by=?,updated=? WHERE id=?",
                  (worker, _now(), d["id"]))
        c.execute("COMMIT")
        d["spec"] = json.loads(d.get("spec") or "{}")
        return d
    except Exception:
        c.execute("ROLLBACK")
        raise
    finally:
        c.close()


def complete(qid: int, status: str, result) -> None:
    c = _conn()
    try:
        c.execute("UPDATE questions SET status=?,result=?,updated=? WHERE id=?",
                  (status, json.dumps(result), _now(), qid))
        c.commit()
    finally:
        c.close()


def add(thread: str, kind: str, question: str, spec=None) -> None:
    """Add a follow-up question discovered during work (dedup on identical text)."""
    c = _conn()
    try:
        if c.execute("SELECT count(*) FROM questions WHERE question=?", (question,)).fetchone()[0] == 0:
            c.execute("INSERT INTO questions(thread,kind,question,spec,created,updated) VALUES(?,?,?,?,?,?)",
                      (thread, kind, question, json.dumps(spec or {}), _now(), _now()))
            c.commit()
    finally:
        c.close()


def reopen(kind: str) -> int:
    """Re-open finished questions of a kind so workers re-run them against fresh state
    (e.g. re-decompose the metric as new scored runs land in MLflow). Returns count re-opened."""
    if kind == "metrics-report":
        return 0  # metrics-report is ONE-SHOT (leader kept/reject table): never bulk-reopen — it re-broadcasts (flood).
    c = _conn()
    try:
        cur = c.execute("UPDATE questions SET status='open',claimed_by='' "
                        "WHERE kind=? AND status IN ('done','escalated','failed')", (kind,))
        c.commit()
        return cur.rowcount
    finally:
        c.close()


def reopen_status(status: str) -> int:
    """Re-open questions currently in `status` (e.g. 'holding' → 'open' when the GPU queue frees).
    Excludes metrics-report: it is one-shot and re-opening it re-broadcasts the kept/reject table (flood)."""
    c = _conn()
    try:
        cur = c.execute("UPDATE questions SET status='open',claimed_by='' "
                        "WHERE status=? AND kind != 'metrics-report'", (status,))
        c.commit()
        return cur.rowcount
    finally:
        c.close()


def stats() -> dict:
    c = _conn()
    try:
        return dict(c.execute("SELECT status,count(*) FROM questions GROUP BY status").fetchall())
    finally:
        c.close()
