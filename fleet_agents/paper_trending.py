"""paper-trending — standing, REFRESHABLE index of Hugging Face Daily Papers with day/week/month
highlight rollups, feeding the :7777 paper search engine. Design adopted from
yukyeongleee/paper-digest-bot (HF Daily as trending source, UTC last-completed-day, dedup state)
but deterministic code instead of prompt-driven, and Postgres instead of Slack/JSONL.

Reuses the per-"competition" Postgres pattern in db.py (DB `kaggle_papers`): each paper is upserted
into the standard `research_index` table (so the existing :7777 FTS/trgm search covers papers with
zero new UI), plus a `papers_trending` table holding per-day upvote/comment counts that powers the
famous-of-the-{day,week,month} rollups.

Usage:
    python paper_trending.py refresh [days]      # ingest last N completed UTC days (default 3)
    python paper_trending.py top day|week|month  # highlight rollup
    python paper_trending.py search "kv cache quantization"
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import ensure_db, _connect  # noqa: E402

SLUG = "papers"
API = "https://huggingface.co/api/daily_papers?date={date}"

TRENDING_SQL = """
CREATE TABLE IF NOT EXISTS papers_trending (
    arxiv_id TEXT, day DATE, title TEXT, upvotes BIGINT DEFAULT 0, comments BIGINT DEFAULT 0,
    url TEXT, ts TIMESTAMPTZ DEFAULT now(), PRIMARY KEY (arxiv_id, day)
);
CREATE INDEX IF NOT EXISTS papers_trending_day_idx ON papers_trending (day);
"""


def _cur(autocommit=True):
    con = _connect(ensure_db(SLUG)); con.autocommit = autocommit
    cur = con.cursor(); cur.execute(TRENDING_SQL)
    return con, cur


def _fetch_day(day):
    with urllib.request.urlopen(API.format(date=day.isoformat()), timeout=30) as r:
        return json.loads(r.read().decode())


def refresh(days=3):
    """Ingest the last N COMPLETED UTC days (today-UTC is still in progress — skip it)."""
    con, cur = _cur()
    last_done = dt.datetime.now(dt.timezone.utc).date() - dt.timedelta(days=1)
    n = 0
    for k in range(days):
        day = last_done - dt.timedelta(days=k)
        try:
            items = _fetch_day(day)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {day} fetch failed: {e}")
            continue
        for it in items:
            p = it.get("paper") or {}
            aid = p.get("id") or ""
            if not aid:
                continue
            title = (p.get("title") or "").strip().replace("\n", " ")
            up = int(p.get("upvotes") or 0); cm = int(it.get("numComments") or 0)
            url = f"https://arxiv.org/abs/{aid}"
            authors = ", ".join(a.get("name", "") for a in (p.get("authors") or [])[:8])
            summary = (p.get("summary") or "")[:1500]
            cur.execute(
                """INSERT INTO papers_trending (arxiv_id, day, title, upvotes, comments, url)
                   VALUES (%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (arxiv_id, day) DO UPDATE SET
                     upvotes=EXCLUDED.upvotes, comments=EXCLUDED.comments, ts=now()""",
                (aid, day, title, up, cm, url),
            )
            cur.execute(
                """INSERT INTO research_index (name, src, url, tags, downloads, likes, date, summary, query, score)
                   VALUES (%s,'hf-daily',%s,%s,%s,%s,%s,%s,'daily_papers',%s)
                   ON CONFLICT (url) DO UPDATE SET
                     likes=EXCLUDED.likes, downloads=EXCLUDED.downloads, summary=EXCLUDED.summary,
                     score=EXCLUDED.score, date=EXCLUDED.date, ts=now()""",
                (title, url, json.dumps([f"authors:{authors}"]), cm, up, day.isoformat(),
                 summary, float(up) + 0.5 * cm),
            )
            n += 1
        print(f"  {day}: {len(items)} papers")
    con.close()
    print(f"paper-trending refresh: upserted {n} rows over {days} day(s)")
    return n


def top(period="day", limit=10):
    """Most-famous papers of the last completed day / trailing 7 days / trailing 30 days.
    Week/month rank by the paper's MAX single-day upvotes (a paper relists across days)."""
    win = {"day": 1, "week": 7, "month": 30}[period]
    con, cur = _cur(autocommit=False)
    cur.execute(
        """SELECT arxiv_id, max(title), max(upvotes) AS up, max(comments), max(day)
           FROM papers_trending
           WHERE day > (SELECT max(day) FROM papers_trending) - %s::int
           GROUP BY arxiv_id ORDER BY up DESC LIMIT %s""",
        (win, limit),
    )
    rows = cur.fetchall()
    con.close()
    return rows


FIG_DIR = Path("/home/seshu/mlflow-artifacts/paper_figures")
_ARXIV_ID_RE = __import__("re").compile(r"^(\d{4}\.\d{4,5}(v\d+)?|[a-z][a-z-]*(\.[A-Z]{2})?/\d{7}(v\d+)?)$")


def figure(arxiv_id):
    """Concept-figure extraction ported from paper-digest-bot/scripts/extract_figure.py:
    arXiv-native HTML then ar5iv fallback; arxiv.org-domain allowlist enforced on fetch AND
    post-redirect (paper HTML is author-controlled content); 25MB cap. Returns saved path or None."""
    import re
    import urllib.parse
    if not _ARXIV_ID_RE.match(arxiv_id):
        raise ValueError(f"invalid arxiv id: {arxiv_id}")

    def allowed(u):
        p = urllib.parse.urlparse(u)
        h = (p.hostname or "").lower()
        return p.scheme in ("http", "https") and (h == "arxiv.org" or h.endswith(".arxiv.org"))

    def fetch(u, cap=25 * 1024 * 1024):
        if not allowed(u):
            raise ValueError(f"disallowed URL: {u}")
        req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (fleet paper-trending)"})
        with urllib.request.urlopen(req, timeout=60) as r:
            if not allowed(r.geturl()):
                raise ValueError(f"redirect to disallowed URL: {r.geturl()}")
            data = r.read(cap + 1)
            if len(data) > cap:
                raise ValueError("response too large")
            return data, r.geturl()

    for base in (f"https://arxiv.org/html/{arxiv_id}", f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"):
        try:
            raw, final = fetch(base)
        except Exception:  # noqa: BLE001
            continue
        html = raw.decode("utf-8", errors="replace")
        src = None
        for block in re.findall(r"<figure\b.*?</figure>", html, re.S | re.I):
            m = re.search(r"""<img[^>]+src=["']([^"']+)["']""", block, re.I)
            if m:
                src = m.group(1); break
        if not src:
            continue
        bt = re.search(r"""<base[^>]+href=["']([^"']+)["']""", html, re.I)
        join = urllib.parse.urljoin(final, bt.group(1)) if bt else final
        img_url = urllib.parse.urljoin(join, src)
        try:
            data, _ = fetch(img_url)
        except Exception:  # noqa: BLE001
            continue
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        ext = Path(img_url.split("?")[0]).suffix or ".png"
        out = FIG_DIR / f"{arxiv_id.replace('/', '_')}{ext}"
        out.write_bytes(data)
        return str(out)
    return None


def interest_top(period="week", limit=12, seeds=3):
    """Interest-relevant feed: rank trending papers by (upvote signal + interest bonus vs the
    user's profile from paper_interest.py), then guarantee `seeds` pure-trending exploration
    picks (highest raw upvotes, regardless of interest) so the profile keeps growing.

    Returns (picks, seed_rows): picks = [(aid, title, up, cm, day, summary, iscore, why)],
    seed_rows same shape (why=[]). Seeds are disjoint from picks."""
    from paper_interest import interest_score  # noqa: PLC0415
    con, cur = _cur(autocommit=False)
    win = {"day": 1, "week": 7, "month": 30}[period]
    cur.execute(
        """SELECT p.arxiv_id, max(p.title), max(p.upvotes) AS up, max(p.comments), max(p.day),
                  max(r.summary)
           FROM papers_trending p LEFT JOIN research_index r ON r.url = p.url
           WHERE p.day > (SELECT max(day) FROM papers_trending) - %s::int
           GROUP BY p.arxiv_id ORDER BY up DESC LIMIT 200""",
        (win,),
    )
    rows = cur.fetchall()
    con.close()
    scored = []
    for aid, title, up, cm, day, summary in rows:
        isc, why = interest_score(f"{title} {summary or ''}")
        # upvote signal on a compressed scale so a single viral paper can't bury everything relevant
        import math
        combined = isc + 4.0 * math.log1p(up)
        scored.append((combined, isc, why, (aid, title, up, cm, day, summary or "")))
    scored.sort(key=lambda x: -x[0])
    picks, seen = [], set()
    for combined, isc, why, base in scored:
        aid, title, up, cm, day, summary = base
        picks.append((aid, title, up, cm, day, summary, round(isc, 1), why))
        seen.add(aid)
        if len(picks) >= limit:
            break
    seed_rows = []
    for aid, title, up, cm, day, summary in sorted(rows, key=lambda r: -r[2]):
        if aid in seen:
            continue
        seed_rows.append((aid, title, up, cm, day, summary or "", 0.0, []))
        seen.add(aid)
        if len(seed_rows) >= seeds:
            break
    return picks, seed_rows


def search(query, limit=15):
    con, cur = _cur(autocommit=False)
    or_q = " | ".join(query.split())
    cur.execute(
        """SELECT name, url, likes, date, ts_rank_cd(tsv, to_tsquery('english', %s)) AS rank
           FROM research_index WHERE src='hf-daily'
             AND (tsv @@ to_tsquery('english', %s) OR name ILIKE %s)
           ORDER BY rank DESC, likes DESC LIMIT %s""",
        (or_q, or_q, f"%{query}%", limit),
    )
    rows = cur.fetchall()
    con.close()
    return rows


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "refresh"
    if cmd == "refresh":
        refresh(int(sys.argv[2]) if len(sys.argv) > 2 else 3)
    elif cmd == "top":
        for aid, title, up, cm, day in top(sys.argv[2] if len(sys.argv) > 2 else "day"):
            print(f"{up:5d}↑ {cm:3d}c {day} {title[:90]}  https://arxiv.org/abs/{aid}")
    elif cmd == "foryou":
        picks, seeds = interest_top(sys.argv[2] if len(sys.argv) > 2 else "week")
        print("── FOR YOU ──")
        for aid, title, up, cm, day, _s, isc, why in picks:
            print(f"{up:5d}↑ int={isc:6.1f} {title[:70]}  [{','.join(why)}]")
        print("── EXPLORE (trending seeds) ──")
        for aid, title, up, cm, day, _s, isc, why in seeds:
            print(f"{up:5d}↑ {title[:70]}")
    elif cmd == "figure":
        print(figure(sys.argv[2]) or "no figure found")
    elif cmd == "search":
        for name, url, likes, date, rank in search(" ".join(sys.argv[2:])):
            print(f"{likes:5d}↑ {date} {name[:90]}  {url}")
    else:
        print(__doc__)
