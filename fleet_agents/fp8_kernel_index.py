"""fp8-kernel-index — a standing, REFRESHABLE search index of FP8 GEMM/Linear kernels on the HF Hub.

Cross-competition (FP8 kernels aren't rogii-specific), reuses the existing per-"competition" Postgres
pattern in db.py (one DB, `kaggle_fp8_kernels`, same `research_index` table + FTS/trgm as every other
comp's research index) so it's queryable the same way: BM25/FTS over name/tags/summary, and inspectable
at :7777 via the standard PG tooling. `refresh()` re-queries the HF Hub so the index tracks new kernel
releases (torchao, kernels-community, Triton fp8 gemms, TransformerEngine, etc.) over time.

Usage:
    python fp8_kernel_index.py refresh      # pull latest from HF Hub, upsert into the index
    python fp8_kernel_index.py search "sm120 blackwell fp8 gemm"
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from db import ensure_db, _connect  # noqa: E402

SLUG = "fp8-kernels"

# search terms that find FP8 GEMM/quant kernel repos on the Hub (orgs + free-text)
QUERIES = [
    "fp8 gemm kernel", "fp8 linear kernel", "float8 triton kernel", "fp8 quantization kernel",
    "blackwell fp8", "sm120 fp8", "torchao float8", "transformer engine fp8",
    "deepgemm fp8", "marlin fp8", "int4 int8 fp8 kernel",
]
ORGS = ["kernels-community", "pytorch", "deepseek-ai", "NVIDIA"]


def _score(tags, downloads, likes):
    fp8_bonus = 5.0 if any("fp8" in t.lower() or "float8" in t.lower() for t in tags) else 0.0
    return fp8_bonus + (downloads or 0) ** 0.3 + (likes or 0) ** 0.5


def refresh(limit_per_query=15):
    from huggingface_hub import HfApi
    api = HfApi()
    db = ensure_db(SLUG)
    con = _connect(db); con.autocommit = True
    cur = con.cursor()
    seen, n = set(), 0
    for q in QUERIES:
        try:
            models = list(api.list_models(search=q, sort="downloads", limit=limit_per_query))
        except Exception as e:  # noqa: BLE001
            print(f"  ! query '{q}' failed: {e}")
            continue
        for m in models:
            if m.id in seen:
                continue
            seen.add(m.id)
            tags = list(m.tags or [])
            url = f"https://huggingface.co/{m.id}"
            summary = f"HF kernel repo; tags={tags[:8]}"
            score = _score(tags, getattr(m, "downloads", 0), getattr(m, "likes", 0))
            cur.execute(
                """INSERT INTO research_index (name, src, url, tags, downloads, likes, summary, query, score)
                   VALUES (%s,'hf',%s,%s,%s,%s,%s,%s,%s)
                   ON CONFLICT (url) DO UPDATE SET
                     downloads=EXCLUDED.downloads, likes=EXCLUDED.likes, tags=EXCLUDED.tags,
                     summary=EXCLUDED.summary, score=EXCLUDED.score, ts=now()""",
                (m.id, url, __import__("json").dumps(tags), getattr(m, "downloads", 0) or 0,
                 getattr(m, "likes", 0) or 0, summary, q, score),
            )
            n += 1
    con.close()
    print(f"fp8-kernel-index refresh: upserted {n} repos across {len(QUERIES)} queries -> db={db}")
    return n


def search(query, top=15):
    db = ensure_db(SLUG)
    con = _connect(db)
    cur = con.cursor()
    # OR-of-terms (not AND): a user query like "blackwell fp8 sm120" should surface rows matching
    # ANY term, ranked by how many/well they match — an AND-only tsquery returns nothing on 3+ rare terms.
    or_query = " | ".join(query.split())
    cur.execute(
        """SELECT name, url, tags, downloads, likes, score,
                  ts_rank_cd(tsv, to_tsquery('english', %s)) AS rank
           FROM research_index
           WHERE tsv @@ to_tsquery('english', %s) OR name ILIKE %s
           ORDER BY rank DESC, score DESC LIMIT %s""",
        (or_query, or_query, f"%{query}%", top),
    )
    rows = cur.fetchall()
    con.close()
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "refresh":
        refresh()
    elif sys.argv[1] == "search":
        for name, url, tags, dl, lk, score, rank in search(" ".join(sys.argv[2:]) or "fp8 gemm"):
            print(f"{score:6.2f} {name:45s} dl={dl:<8} likes={lk:<5} {url}")
    else:
        print(__doc__)
