"""db — per-competition PostgreSQL store (user 2026-07-12: "each competition, respective kaggle_... database
on ubuntu"). One database per competition, named `kaggle_<slug_with_underscores>`, on the local PG instance
(localhost / postgres / seshu — same box as mlflow_db, tokendb). Holds everything we download/find:
  • research_index   — models/papers the research-search agent finds (was docs/research_index.jsonl)
  • lb_snapshot      — per-sync leaderboard AGGREGATE (was docs/lb_history.jsonl)
  • lb_team          — per-sync PER-TEAM leaderboard rows (powers the :7777 LB page with full detail)

JSONL files stay as a redundant backup; PG is the queryable source. All access via this one helper so the
DB-name convention + connection live in a single place.
"""
from __future__ import annotations
import json
import os

PG = {"host": "localhost", "user": "postgres", "password": os.environ.get("PGPASSWORD", "seshu")}


def db_name(slug):
    """Competition slug → PG database name: kaggle_<slug with -/. → _>, lowercased, ≤63 chars."""
    import re
    s = re.sub(r"[^a-z0-9]+", "_", str(slug).lower()).strip("_")
    return ("kaggle_" + s)[:63]


def _connect(dbname):
    import psycopg2
    return psycopg2.connect(dbname=dbname, **PG)


def ensure_db(slug):
    """Create the competition's database if missing; return its name. Idempotent."""
    name = db_name(slug)
    import psycopg2
    con = psycopg2.connect(dbname="postgres", **PG); con.autocommit = True
    try:
        cur = con.cursor()
        cur.execute("SELECT 1 FROM pg_database WHERE datname=%s", (name,))
        if not cur.fetchone():
            cur.execute(f'CREATE DATABASE "{name}"')
    finally:
        con.close()
    init_schema(name)
    return name


DDL = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE TABLE IF NOT EXISTS research_index (
    id BIGSERIAL PRIMARY KEY,
    name TEXT, src TEXT, url TEXT UNIQUE, tags JSONB DEFAULT '[]'::jsonb,
    downloads BIGINT DEFAULT 0, likes BIGINT DEFAULT 0, date TEXT, summary TEXT,
    query TEXT, score DOUBLE PRECISION, ts TIMESTAMPTZ DEFAULT now()
);
-- Advanced search: a WEIGHTED generated tsvector (name=A > tags=B > summary/src/query=C), GIN-indexed, so
-- ranking runs INSIDE Postgres via ts_rank_cd (indexed, O(log N)) instead of pulling every row into Python.
ALTER TABLE research_index ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english', coalesce(name,'')), 'A') ||
    setweight(to_tsvector('english', coalesce(tags::text,'')), 'B') ||
    setweight(to_tsvector('english', coalesce(summary,'') || ' ' || coalesce(query,'') || ' ' || coalesce(src,'')), 'C')
) STORED;
CREATE INDEX IF NOT EXISTS research_index_tsv_idx ON research_index USING GIN (tsv);
CREATE INDEX IF NOT EXISTS research_index_trgm_idx ON research_index USING GIN (name gin_trgm_ops);
CREATE TABLE IF NOT EXISTS lb_snapshot (
    id BIGSERIAL PRIMARY KEY, utc TIMESTAMPTZ, top_score DOUBLE PRECISION, leader TEXT,
    n_teams INT, leader_last_sub_age_h DOUBLE PRECISION, median_last_sub_age_h DOUBLE PRECISION,
    active_24h INT, active_72h INT, top10_active_24h INT, raw JSONB
);
CREATE TABLE IF NOT EXISTS lb_team (
    id BIGSERIAL PRIMARY KEY, utc TIMESTAMPTZ, rank INT, team TEXT, team_id TEXT,
    score DOUBLE PRECISION, submission_utc TIMESTAMPTZ, age_h DOUBLE PRECISION
);
CREATE INDEX IF NOT EXISTS lb_team_utc_idx ON lb_team (utc);
-- EXPERIMENT JOURNAL — one row per experiment (the grandmaster ledger). Postgres is the queryable
-- source of truth per competition; the JSONL/markdown files stay as a dual-written backup. Keyed by
-- `exp` so re-writing the same experiment upserts (CV/LB backfill after scoring). `raw` holds the
-- full ledger dict so nothing is lost even as columns evolve.
CREATE TABLE IF NOT EXISTS experiment_journal (
    exp TEXT PRIMARY KEY,
    cv DOUBLE PRECISION, cv_text TEXT, lb DOUBLE PRECISION, lb_text TEXT,
    descr TEXT, change_key TEXT, parent TEXT, script TEXT, trn_set TEXT,
    stage TEXT, kept BOOLEAN, observation TEXT, git_hash TEXT,
    ts TIMESTAMPTZ DEFAULT now(), raw JSONB
);
CREATE INDEX IF NOT EXISTS experiment_journal_trnset_idx ON experiment_journal (trn_set);
-- DECISION / FINDING TRAIL (the story behind the numbers → powers /insights + the journal narrative).
CREATE TABLE IF NOT EXISTS experiment_decisions (
    id BIGSERIAL PRIMARY KEY,
    agent TEXT, kind TEXT, summary TEXT, detail TEXT, recommendation TEXT,
    run TEXT, git_hash TEXT, ts TIMESTAMPTZ DEFAULT now(), dedup_key TEXT UNIQUE, raw JSONB
);
-- FLEET BOARD mirror (the deterministic agent work-queue that lives in SQLite fleet.db). Dual-written
-- so the selected-competition board can read tasks/status from Postgres too. `id` mirrors the SQLite id.
CREATE TABLE IF NOT EXISTS fleet_board (
    id BIGINT PRIMARY KEY, thread TEXT, kind TEXT, question TEXT, spec JSONB,
    status TEXT, claimed_by TEXT, result TEXT, created TIMESTAMPTZ, updated TIMESTAMPTZ
);
"""


def init_schema(name):
    con = _connect(name)
    try:
        with con, con.cursor() as cur:
            cur.execute(DDL)
    finally:
        con.close()


def upsert_research(slug, rows):
    """Insert candidates (dedup on url). Returns count inserted. rows = research-search candidate dicts."""
    if not rows:
        return 0
    name = ensure_db(slug)
    con = _connect(name); n = 0
    try:
        with con, con.cursor() as cur:
            for r in rows:
                url = r.get("url") or r.get("name")
                if not url or str(r.get("name", "")).startswith("<"):
                    continue
                cur.execute(
                    """INSERT INTO research_index (name,src,url,tags,downloads,likes,date,summary,query,score)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (url) DO UPDATE SET downloads=EXCLUDED.downloads, likes=EXCLUDED.likes""",
                    (r.get("name"), r.get("src"), url, json.dumps(r.get("tags", [])),
                     int(r.get("downloads") or 0), int(r.get("likes") or 0), r.get("date"),
                     r.get("summary"), r.get("query"), r.get("score")))
                n += cur.rowcount
    finally:
        con.close()
    return n


def all_research(slug):
    """Every research_index row as a dict (for BM25 in the app / agent)."""
    name = db_name(slug)
    try:
        con = _connect(name)
    except Exception:
        return []
    try:
        with con, con.cursor() as cur:
            cur.execute("SELECT name,src,url,tags,downloads,likes,date,summary,query,score FROM research_index")
            cols = ["name", "src", "url", "tags", "downloads", "likes", "date", "summary", "query", "score"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]
    finally:
        con.close()


def search_research_fts(slug, query, top=15):
    """FAST search INSIDE Postgres: weighted FTS (ts_rank_cd over the GIN-indexed tsv) with a pg_trgm fuzzy
    fallback (typo-tolerant) when FTS finds nothing. Returns candidate dicts with a `rank` — no full-table
    scan, no Python-side ranking. This is the advanced/scalable path; the Python BM25 stays as a fallback."""
    if not query:
        return []
    name = db_name(slug)
    try:
        con = _connect(name)
    except Exception:
        return []
    cols = "name,src,url,tags,downloads,likes,date,summary"
    try:
        with con, con.cursor() as cur:
            cur.execute(
                f"""SELECT {cols}, ts_rank_cd(tsv, websearch_to_tsquery('english', %s)) AS rank
                    FROM research_index
                    WHERE tsv @@ websearch_to_tsquery('english', %s)
                    ORDER BY rank DESC, downloads DESC LIMIT %s""",
                (query, query, int(top)))
            rows = cur.fetchall()
            if not rows:                                    # FTS empty → trigram fuzzy (handles typos/partials)
                cur.execute(
                    f"""SELECT {cols}, similarity(name, %s) AS rank
                        FROM research_index
                        WHERE name %% %s ORDER BY rank DESC, downloads DESC LIMIT %s""",
                    (query, query, int(top)))
                rows = cur.fetchall()
        keys = cols.split(",") + ["rank"]
        return [dict(zip(keys, r)) for r in rows]
    except Exception:
        return []
    finally:
        con.close()


def insert_lb(slug, utc_iso, analysis, teams):
    """Store one LB sync: the aggregate snapshot + per-team rows (for the LB page)."""
    name = ensure_db(slug)
    con = _connect(name)
    try:
        with con, con.cursor() as cur:
            cur.execute(
                """INSERT INTO lb_snapshot (utc,top_score,leader,n_teams,leader_last_sub_age_h,
                   median_last_sub_age_h,active_24h,active_72h,top10_active_24h,raw)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (utc_iso, analysis.get("top_score"), analysis.get("leader"), analysis.get("n_teams"),
                 analysis.get("leader_last_sub_age_h"), analysis.get("median_last_sub_age_h"),
                 analysis.get("active_24h"), analysis.get("active_72h"), analysis.get("top10_active_24h"),
                 json.dumps(analysis)))
            for i, t in enumerate(teams):
                cur.execute(
                    """INSERT INTO lb_team (utc,rank,team,team_id,score,submission_utc,age_h)
                       VALUES (%s,%s,%s,%s,%s,%s,%s)""",
                    (utc_iso, i + 1, t.get("team"), t.get("team_id"), t.get("score"),
                     t.get("submission_utc"), t.get("age_h")))
    finally:
        con.close()


def latest_lb(slug):
    """Most-recent per-team LB rows (rank order) for the LB page, or []."""
    name = db_name(slug)
    try:
        con = _connect(name)
    except Exception:
        return []
    try:
        with con, con.cursor() as cur:
            cur.execute("SELECT max(utc) FROM lb_team")
            row = cur.fetchone()
            if not row or not row[0]:
                return []
            cur.execute("""SELECT rank,team,score,submission_utc,age_h FROM lb_team WHERE utc=%s ORDER BY rank""",
                        (row[0],))
            cols = ["rank", "team", "score", "submission_utc", "age_h"]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    finally:
        con.close()


# ----------------------------- experiment journal (per-competition PG) -----------------------------

def _num(v):
    """A finite float or None (status strings like 'bad'/'overfit'/'nan' → None numeric, kept in *_text)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if v == v and abs(v) != float("inf") else None


def upsert_journal(slug, rows):
    """Upsert experiment-journal rows (dedup on `exp`) into kaggle_<slug>.experiment_journal. rows = the
    ledger entry dicts (exp, cv, lb, desc, change, parent, script, trn_set, stage, kept, observation,
    git_hash). Best-effort; returns count written. The JSONL/markdown files remain the dual-write backup."""
    if not rows:
        return 0
    name = ensure_db(slug)
    con = _connect(name); n = 0
    try:
        with con, con.cursor() as cur:
            for e in rows:
                exp = e.get("exp")
                if not exp:
                    continue
                cur.execute(
                    """INSERT INTO experiment_journal
                         (exp,cv,cv_text,lb,lb_text,descr,change_key,parent,script,trn_set,stage,kept,
                          observation,git_hash,ts,raw)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (exp) DO UPDATE SET
                         cv=EXCLUDED.cv, cv_text=EXCLUDED.cv_text, lb=EXCLUDED.lb, lb_text=EXCLUDED.lb_text,
                         descr=EXCLUDED.descr, change_key=EXCLUDED.change_key, parent=EXCLUDED.parent,
                         script=EXCLUDED.script, trn_set=EXCLUDED.trn_set, stage=EXCLUDED.stage,
                         kept=EXCLUDED.kept, observation=EXCLUDED.observation, git_hash=EXCLUDED.git_hash,
                         raw=EXCLUDED.raw""",
                    (exp, _num(e.get("cv")), None if e.get("cv") is None else str(e.get("cv")),
                     _num(e.get("lb")), None if e.get("lb") is None else str(e.get("lb")),
                     e.get("desc"), e.get("change"), e.get("parent"), e.get("script"), e.get("trn_set"),
                     e.get("stage"), bool(e.get("kept")) if e.get("kept") is not None else None,
                     e.get("observation"), e.get("git_hash"), e.get("ts"), json.dumps(e)))
                n += 1
    finally:
        con.close()
    return n


def all_journal(slug):
    """Every experiment-journal row (the full ledger dict via `raw`) for a competition, or []. Ordered by ts."""
    name = db_name(slug)
    try:
        con = _connect(name)
    except Exception:
        return []
    try:
        with con, con.cursor() as cur:
            cur.execute("SELECT raw FROM experiment_journal ORDER BY ts NULLS LAST, exp")
            return [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        return []
    finally:
        con.close()


def upsert_decisions(slug, rows):
    """Upsert decision/finding rows (dedup on dedup_key = agent|summary|recommendation) → PG. Best-effort."""
    if not rows:
        return 0
    name = ensure_db(slug)
    con = _connect(name); n = 0
    try:
        with con, con.cursor() as cur:
            for e in rows:
                summ = e.get("summary") or e.get("finding") or ""
                key = f"{e.get('agent')}|{summ}|{e.get('recommendation')}"
                cur.execute(
                    """INSERT INTO experiment_decisions (agent,kind,summary,detail,recommendation,run,git_hash,ts,dedup_key,raw)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (dedup_key) DO NOTHING""",
                    (e.get("agent"), e.get("kind"), summ, e.get("detail"), e.get("recommendation"),
                     e.get("run"), e.get("git_hash"), e.get("ts"), key, json.dumps(e)))
                n += cur.rowcount
    finally:
        con.close()
    return n


def all_decisions(slug):
    """Every decision/finding row (full dict) for a competition, ts-ordered, or []."""
    name = db_name(slug)
    try:
        con = _connect(name)
    except Exception:
        return []
    try:
        with con, con.cursor() as cur:
            cur.execute("SELECT raw FROM experiment_decisions ORDER BY ts NULLS LAST, id")
            return [r[0] for r in cur.fetchall() if r[0]]
    except Exception:
        return []
    finally:
        con.close()


def sync_board(slug, rows):
    """Mirror the SQLite fleet board rows into PG (upsert on id). rows = dicts from the SQLite board."""
    if not rows:
        return 0
    name = ensure_db(slug)
    con = _connect(name); n = 0
    try:
        with con, con.cursor() as cur:
            for r in rows:
                if r.get("id") is None:
                    continue
                spec = r.get("spec")
                if isinstance(spec, str):
                    try:
                        spec = json.loads(spec)
                    except Exception:
                        spec = {}
                cur.execute(
                    """INSERT INTO fleet_board (id,thread,kind,question,spec,status,claimed_by,result,created,updated)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (id) DO UPDATE SET status=EXCLUDED.status, claimed_by=EXCLUDED.claimed_by,
                         result=EXCLUDED.result, updated=EXCLUDED.updated""",
                    (r.get("id"), r.get("thread"), r.get("kind"), r.get("question"), json.dumps(spec or {}),
                     r.get("status"), r.get("claimed_by"), r.get("result"), r.get("created"), r.get("updated")))
                n += 1
    finally:
        con.close()
    return n


def all_board(slug, limit=200):
    """Recent fleet-board rows (newest first) for a competition, or []."""
    name = db_name(slug)
    try:
        con = _connect(name)
    except Exception:
        return []
    try:
        with con, con.cursor() as cur:
            cur.execute("""SELECT id,thread,kind,question,spec,status,claimed_by,result,created,updated
                           FROM fleet_board ORDER BY id DESC LIMIT %s""", (int(limit),))
            cols = ["id","thread","kind","question","spec","status","claimed_by","result","created","updated"]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    except Exception:
        return []
    finally:
        con.close()
