#!/usr/bin/env python3
"""honest-method-scout — a REUSABLE, config-driven agent that mines a Kaggle
competition for the HONEST (no-leak) leading method, so we never repeat the
manual discussion+kernel dig by hand.

Why it exists (user rule 2026-07-24: "make agent do same thing for future time"):
the ROGII honest-method research (discussion ingest -> filter leak vs honest
kernels -> extract target-author comments -> pull honest notebooks -> emit a
markdown findings report) was valuable but manual. This agent parametrizes it.

It REUSES existing tooling, it does not reinvent it:
  * kaggle CLI (`kaggle kernels list/pull`) via the user's ~/.kaggle/kaggle.json
  * nvidia-kaggle skill scripts (discussion_ingest / discussion_query /
    discussion_read) + their local discussions.db (sqlite).

Everything competition-specific is a CONFIG knob (no hardcoded ROGII logic):
  --leak-regex   : title regex marking LEAK / LB-tuned forks to EXCLUDE
                   (default catches "7.159"-style score-in-title + "leak").
  --authors      : comma list of usernames whose comments to extract verbatim
                   (e.g. known honest GMs: cdeotte).
  --terms        : comma list of technique terms to grep across all comments
                   (dtw, particle filter, cross-correlation, cv, oracle, ...).
  --top-kernels  : how many top-voted kernels to list / classify honest-vs-leak.

Output: a markdown report to <comp>/research/honest_method_scout_report.md
(and, with --pull, the honest notebooks into <comp>/research/honest_notebooks/).

CLI (standalone, no fleet runtime needed):
  python honest_method_scout.py <comp-slug> [--pull] [--authors cdeotte]
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sqlite3, sys
from pathlib import Path

DEFAULT_LEAK_REGEX = r"\b\d\.\d{2,3}\b|leak|as-?is|public.?rebuild|safe.?twin"
DEFAULT_TERMS = ["dtw", "dynamic time", "cross-corr", "particle filter",
                 "matched filter", "typewell", "oracle", "cv", "field-grouped",
                 "azimuth", "neighbor", "contact"]

def _kaggle_env() -> dict:
    """CLI wants ~/.kaggle/kaggle.json; the nvidia scripts want KAGGLE_API_TOKEN.
    Set the token from the json key so BOTH work; DON'T poison the CLI (unset it
    when we shell out to `kaggle`)."""
    env = dict(os.environ)
    jp = Path.home() / ".kaggle" / "kaggle.json"
    if jp.exists():
        j = json.loads(jp.read_text())
        env["KAGGLE_USERNAME"] = j.get("username", "")
        env["KAGGLE_API_TOKEN"] = j.get("key", "")
    return env

def _find_nvidia_scripts() -> Path | None:
    base = Path.home() / ".claude/plugins/cache/nvidia-kaggle"
    hits = list(base.glob("**/skills/nvidia-kaggle-skill/scripts/discussion_ingest.py"))
    return hits[0].parent if hits else None

def _run(cmd, env, cwd=None, timeout=1800):
    return subprocess.run(cmd, env=env, cwd=cwd, capture_output=True,
                          text=True, timeout=timeout)

def list_kernels(slug, env, n):
    e = dict(env); e.pop("KAGGLE_API_TOKEN", None)  # CLI uses kaggle.json, not token
    r = _run(["kaggle", "kernels", "list", "--competition", slug,
              "--sort-by", "voteCount", "--page-size", str(n), "--csv"], e, timeout=120)
    rows = []
    for line in r.stdout.splitlines()[1:]:
        parts = line.rsplit(",", 4)
        if len(parts) == 5:
            rows.append({"ref": parts[0], "title": parts[1], "votes": parts[4]})
    return rows

def classify_kernels(rows, leak_re):
    rx = re.compile(leak_re, re.I)
    honest, leak = [], []
    for k in rows:
        (leak if rx.search(k["title"]) or rx.search(k["ref"]) else honest).append(k)
    return honest, leak

def ingest_discussions(slug, scripts, env, pages):
    if not scripts:
        return "nvidia-kaggle scripts not found; skipped discussion ingest."
    r = _run([sys.executable, "discussion_ingest.py", slug, "--max-pages",
              str(pages), "--sort-by", "votes"], env, cwd=scripts, timeout=3600)
    return r.stdout.strip().splitlines()[-1] if r.stdout else r.stderr[-400:]

def _db_path(scripts):
    if not scripts:
        return None
    # scripts = .../<hash>/skills/nvidia-kaggle-skill/scripts ; db = .../<hash>/data/discussions.db
    for anc in [scripts, *scripts.parents]:
        cand = anc / "data" / "discussions.db"
        if cand.exists():
            return cand
    return None

def author_comments(db, slug, authors):
    out = {}
    if not db:
        return out
    con = sqlite3.connect(db)
    for a in authors:
        rows = con.execute(
            "SELECT discussion_id, substr(body_markdown,1,900) FROM discussion_comments "
            "WHERE competition_id=? AND (lower(author) LIKE ? OR lower(author_username) LIKE ?) "
            "ORDER BY discussion_id",
            (slug, f"%{a.lower()}%", f"%{a.lower()}%")).fetchall()
        out[a] = rows
    con.close()
    return out

def term_hits(db, slug, terms):
    out = {}
    if not db:
        return out
    con = sqlite3.connect(db)
    for t in terms:
        rows = con.execute(
            "SELECT discussion_id, author, substr(replace(body_markdown,char(10),' '),1,300) "
            "FROM discussion_comments WHERE competition_id=? AND lower(body_markdown) LIKE ? LIMIT 6",
            (slug, f"%{t.lower()}%")).fetchall()
        if rows:
            out[t] = rows
    con.close()
    return out

def pull_honest(honest, dest, env, limit):
    e = dict(env); e.pop("KAGGLE_API_TOKEN", None)
    dest.mkdir(parents=True, exist_ok=True)
    pulled = []
    for k in honest[:limit]:
        d = dest / k["ref"].replace("/", "_")
        r = _run(["kaggle", "kernels", "pull", k["ref"], "-p", str(d)], e, timeout=120)
        if r.returncode == 0:
            pulled.append(k["ref"])
    return pulled

def build_report(slug, honest, leak, ac, th, ingest_line):
    L = [f"# Honest-method scout report — {slug}\n",
         f"_Auto-generated by honest_method_scout.py. Discussion ingest: {ingest_line}_\n",
         "## Honest top kernels (leak/LB-tuned forks excluded)"]
    for k in honest:
        L.append(f"- `{k['ref']}` — {k['title']} ({k['votes']} votes)")
    L.append("\n## Excluded as leak / LB-tuned forks")
    for k in leak:
        L.append(f"- ~~`{k['ref']}`~~ — {k['title']}")
    for a, rows in ac.items():
        L.append(f"\n## Comments by `{a}`")
        for did, body in rows:
            L.append(f"- [disc {did}] {body.strip()}")
    L.append("\n## Technique-term hits across comments")
    for t, rows in th.items():
        L.append(f"\n### `{t}`")
        for did, au, body in rows:
            L.append(f"- [{did}/{au}] {body.strip()}")
    return "\n".join(L) + "\n"

def scout(slug, comp_root=None, leak_regex=DEFAULT_LEAK_REGEX,
          authors=("cdeotte",), terms=None, top_kernels=30, pages=8,
          pull=False, pull_limit=8):
    terms = list(terms) if terms else DEFAULT_TERMS
    env = _kaggle_env()
    scripts = _find_nvidia_scripts()
    comp_root = Path(comp_root) if comp_root else Path.cwd()
    research = comp_root / "research"; research.mkdir(exist_ok=True)

    rows = list_kernels(slug, env, top_kernels)
    honest, leak = classify_kernels(rows, leak_regex)
    ingest_line = ingest_discussions(slug, scripts, env, pages)
    db = _db_path(scripts)
    ac = author_comments(db, slug, list(authors))
    th = term_hits(db, slug, terms)
    if pull:
        pull_honest(honest, research / "honest_notebooks", env, pull_limit)

    report = build_report(slug, honest, leak, ac, th, ingest_line)
    out = research / "honest_method_scout_report.md"
    out.write_text(report)
    return out

def main():
    ap = argparse.ArgumentParser(description="Mine a Kaggle comp for the honest (no-leak) leading method.")
    ap.add_argument("slug")
    ap.add_argument("--comp-root", default=None, help="competition dir (default: cwd)")
    ap.add_argument("--leak-regex", default=DEFAULT_LEAK_REGEX)
    ap.add_argument("--authors", default="cdeotte")
    ap.add_argument("--terms", default=",".join(DEFAULT_TERMS))
    ap.add_argument("--top-kernels", type=int, default=30)
    ap.add_argument("--pages", type=int, default=8)
    ap.add_argument("--pull", action="store_true", help="also pull honest notebooks")
    args = ap.parse_args()
    out = scout(args.slug, comp_root=args.comp_root, leak_regex=args.leak_regex,
                authors=[a.strip() for a in args.authors.split(",") if a.strip()],
                terms=[t.strip() for t in args.terms.split(",") if t.strip()],
                top_kernels=args.top_kernels, pages=args.pages, pull=args.pull)
    print(f"wrote {out}")

if __name__ == "__main__":
    main()
