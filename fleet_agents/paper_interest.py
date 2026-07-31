"""paper-interest — build a weighted keyword INTEREST PROFILE from the user's own signals
(GitHub starred repos + active Kaggle competitions) and score papers against it, so the :7777
paper feed surfaces what THIS user cares about, not just what is globally trending.

Signals & weights (deterministic, no LLM):
  - Kaggle competition domain tokens (from ~/kaggle/2026/<slug>/ dir names)   ×5  — strongest
  - GitHub star repo `topics`                                                 ×3
  - GitHub star repo `language`                                              ×2
  - GitHub star repo description tokens (stopword-filtered, len>2)           ×1

Stored in PG `kaggle_papers` table `paper_interest_profile (kw TEXT PK, weight REAL, src TEXT)`.
`interest_score(text)` returns the summed profile weight of the profile keywords present in the
text (title+summary), used to re-rank the trending papers. Refreshable like the other indices.

Usage:
    python paper_interest.py build            # rebuild profile from stars + comps
    python paper_interest.py show [n]         # top-n profile keywords
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import ensure_db, _connect  # noqa: E402

SLUG = "papers"
COMP_ROOT = Path("/home/seshu/kaggle/2026")
_SKIP_DIRS = {"external", "scratchpad", "logs", "unsloth_compiled_cache", "kaggle_ai"}

PROFILE_SQL = """
CREATE TABLE IF NOT EXISTS paper_interest_profile (
    kw TEXT PRIMARY KEY, weight REAL DEFAULT 0, src TEXT, ts TIMESTAMPTZ DEFAULT now()
);
"""

_STOP = set("""the a an of to and or for with in on at by from as is are be this that these those
it its into via using use used based our we you your they their can will may not but if then than
new using toward towards over under between across per any all more most via approach method model
models data paper papers code framework library toolkit preprint arxiv official implementation repo""".split())

# generic mega-terms that match nearly every ML paper — carrying them makes the score undiscriminating
_GENERIC = set("""python jupyter jupyter-notebook notebook ai artificial-intelligence learning large
language deep-learning machine-learning typescript javascript go rust cpp c++ shell html css
awesome list tool tools app application software project""".split())

_TOKEN = re.compile(r"[a-z][a-z0-9+#-]{2,}")


def _tokens(text: str):
    return [t for t in _TOKEN.findall((text or "").lower()) if t not in _STOP]


def _comp_terms():
    """Kaggle competition domain tokens from live comp directories."""
    terms = Counter()
    if not COMP_ROOT.exists():
        return terms
    for d in COMP_ROOT.iterdir():
        if not d.is_dir() or d.name in _SKIP_DIRS or d.name.startswith("."):
            continue
        for t in _tokens(d.name.replace("-", " ")):
            terms[t] += 1
    return terms


def _star_signals():
    """(topics, languages, desc-tokens) counters from `gh api user/starred`."""
    topics, langs, desc = Counter(), Counter(), Counter()
    try:
        out = subprocess.run(
            ["gh", "api", "user/starred?per_page=100", "--paginate",
             "--jq", ".[] | {full_name, description, language, topics}"],
            capture_output=True, text=True, timeout=180, check=True,
        ).stdout
    except Exception as e:  # noqa: BLE001
        print(f"  ! gh starred fetch failed: {e}")
        return topics, langs, desc
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        for t in (r.get("topics") or []):
            topics[t.lower()] += 1
        lang = (r.get("language") or "").lower().strip()
        if lang:
            langs[lang] += 1
        for tok in _tokens(r.get("description") or ""):
            desc[tok] += 1
    return topics, langs, desc


def build():
    comps = _comp_terms()
    topics, langs, desc = _star_signals()
    # sqrt-dampen raw counts so a few mega-starred topics don't dominate the summed score
    from math import sqrt
    weighted = Counter()
    src = {}
    for tier, cnt in ((5, comps), (3, topics), (2, langs), (1, desc)):
        label = {5: "comp", 3: "star-topic", 2: "star-lang", 1: "star-desc"}[tier]
        for kw, c in cnt.items():
            if kw in _GENERIC:
                continue
            weighted[kw] += tier * sqrt(c)
            src.setdefault(kw, label)

    # keep only meaningful signal, cap to a workable profile size
    rows = [(kw, float(w), src[kw]) for kw, w in weighted.items() if w >= 2.5]
    rows.sort(key=lambda r: -r[1])
    rows = rows[:800]

    con = _connect(ensure_db(SLUG)); con.autocommit = True
    cur = con.cursor(); cur.execute(PROFILE_SQL)
    cur.execute("TRUNCATE paper_interest_profile")
    cur.executemany(
        "INSERT INTO paper_interest_profile (kw, weight, src) VALUES (%s,%s,%s)", rows,
    )
    con.close()
    print(f"paper-interest build: {len(rows)} keywords "
          f"(comps={len(comps)} topics={len(topics)} langs={len(langs)} desc={len(desc)})")
    return len(rows)


_PROFILE_CACHE = {}


def load_profile():
    if _PROFILE_CACHE:
        return _PROFILE_CACHE
    con = _connect(ensure_db(SLUG)); con.autocommit = True
    cur = con.cursor(); cur.execute(PROFILE_SQL)
    cur.execute("SELECT kw, weight FROM paper_interest_profile")
    for kw, w in cur.fetchall():
        _PROFILE_CACHE[kw] = float(w)
    con.close()
    return _PROFILE_CACHE


def interest_score(text: str):
    """Summed profile weight of matched keywords, plus the matched keywords (for 'why')."""
    prof = load_profile()
    if not prof:
        return 0.0, []
    toks = set(_tokens(text))
    hits = [(t, prof[t]) for t in toks if t in prof]
    hits.sort(key=lambda x: -x[1])
    return sum(w for _, w in hits), [t for t, _ in hits[:6]]


def show(n=40):
    con = _connect(ensure_db(SLUG)); con.autocommit = True
    cur = con.cursor(); cur.execute(PROFILE_SQL)
    cur.execute("SELECT kw, weight, src FROM paper_interest_profile ORDER BY weight DESC LIMIT %s", (n,))
    for kw, w, src in cur.fetchall():
        print(f"  {w:7.1f}  {kw:28s} [{src}]")
    con.close()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "show":
        show(int(sys.argv[2]) if len(sys.argv) > 2 else 40)
    else:
        print(__doc__)
