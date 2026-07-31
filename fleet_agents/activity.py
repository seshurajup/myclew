"""activity — the one-liner every python fleet-agent calls to stream what it is doing to the competition
board (:7788 → per-competition Postgres `kaggle_<slug>.experiment_decisions`). For a non-home competition
the board has no live tmux daemon, so its "Activity — one living card per agent" + "Decisions & findings"
panels render THESE rows. Each `beat()` is a distinct row (dedup keyed by agent+timestamp) so an agent's
card updates live as it works. Best-effort: never raises, so logging can never break an agent.

Usage (from any fleet agent):
    from fleet_agents import activity
    activity.beat("rogii-wellbore-geology-prediction", "engine",
                  "running affine-cal + multi-scale NCC CV (fold 1/5)",
                  detail="field-grouped folds, 100-well subset", kind="running")
"""
from __future__ import annotations
import datetime as _dt


def _db():
    try:
        from fleet_agents import db as _d  # normal import path
        return _d
    except Exception:  # noqa: BLE001
        try:
            import importlib.util as _u
            from pathlib import Path as _P
            p = _P(__file__).with_name("db.py")
            s = _u.spec_from_file_location("_fleet_db_activity", str(p))
            m = _u.module_from_spec(s); s.loader.exec_module(m)
            return m
        except Exception:  # noqa: BLE001
            return None


def beat(slug: str, agent: str, summary: str, *, detail: str | None = None,
         kind: str = "activity", run: str | None = None, recommendation: str | None = None) -> bool:
    """Post one live activity row for `agent` to the competition board. Returns True if written."""
    db = _db()
    if db is None:
        return False
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    row = dict(agent=agent, kind=kind, summary=summary, detail=detail,
               recommendation=recommendation, run=run, ts=ts)
    # make each beat distinct so the agent's card streams (upsert dedups on agent|summary|recommendation);
    # we vary summary invisibly with a zero-width-free ts suffix stored only in dedup via recommendation slot
    try:
        # write via a private path that guarantees a unique dedup_key = agent|summary|ts
        name = db.ensure_db(slug)
        con = db._connect(name)
        try:
            import json as _j, zlib as _z
            aid = _z.crc32(agent.encode()) & 0x7FFFFFFF  # stable per-agent id for the fleet_board row
            status = "running" if kind in ("running", "result") else ("done" if kind == "done" else "idle")
            with con, con.cursor() as cur:
                cur.execute(
                    """INSERT INTO experiment_decisions
                         (agent,kind,summary,detail,recommendation,run,ts,dedup_key,raw)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (dedup_key) DO NOTHING""",
                    (agent, kind, summary, detail, recommendation, run, ts,
                     f"{agent}|{summary}|{ts}", _j.dumps(row)))
                # ALSO mirror into fleet_board so the Python-agents panel (/api/runtime/fleet) shows this
                # deterministic agent + its latest action, live. Upsert on the stable per-agent id.
                cur.execute(
                    """INSERT INTO fleet_board (id,thread,kind,question,spec,status,claimed_by,result,created,updated)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (id) DO UPDATE SET status=EXCLUDED.status, claimed_by=EXCLUDED.claimed_by,
                         result=EXCLUDED.result, question=EXCLUDED.question, updated=EXCLUDED.updated""",
                    (aid, run or "session", agent, summary[:200], _j.dumps({"kind": kind}),
                     status, agent, (detail or "")[:300], ts, ts))
        finally:
            con.close()
        return True
    except Exception:  # noqa: BLE001
        # fall back to the public upsert (may dedup identical summaries, still fine)
        try:
            db.upsert_decisions(slug, [row]); return True
        except Exception:  # noqa: BLE001
            return False
