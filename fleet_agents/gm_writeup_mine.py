"""gm-writeup-mine — the reusable pipeline that GROUNDS the fleet in real Kaggle top solutions. Given a set
of finished competition slugs, it fetches the top-N solution writeups (1st..Nth place) as markdown via the
official nvidia-kaggle bearer API (`fetch_leaderboard_writeups` → `fetch_writeup`, KGAT token) and saves them
under docs/gm_writeups/<slug>/, then reports what it got. The distillation of those writeups into techniques/
metrics/math is done by the research/extraction agents (research-search, trick-extractor) or extraction
sub-agents — this agent owns the reliable FETCH + INDEX so the catalog (docs/gm_techniques_grounded.md)
keeps growing as new comps finish. This is how the fleet stays current with 2025/2026 SOTA without a stale
hardcoded list.

Pure-Python fetch with an injectable script runner (`_run_script`) so the save/parse logic is data-wise
tested OFFLINE (stubbed network), while live runs use the real bearer scripts.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
from .base import BaseAgent, COMP

_SCRIPTS = ("/home/seshu/.claude/plugins/cache/nvidia-kaggle/nvidia-kaggle/c9336418905e/"
            "skills/nvidia-kaggle-skill/scripts")
OUT = COMP / "docs" / "gm_writeups"


def _py():
    p = COMP / "research" / "cellmot_venv" / "bin" / "python"
    return str(p) if p.exists() else sys.executable


def _run_script(script, args, timeout=90):
    """Run a nvidia-kaggle bearer script; returns stdout ('' on failure). Injectable for offline tests.
    timeout: per-call wall-clock cap in seconds (a hung/slow API call returns '' rather than blocking)."""
    env = dict(os.environ); env.setdefault("PROJECT_ROOT", str(COMP))
    try:
        r = subprocess.run([_py(), os.path.join(_SCRIPTS, script), *args],
                           capture_output=True, text=True, timeout=int(timeout), env=env)
        return r.stdout if r.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def _slugify(u):
    return re.sub(r"[^a-z0-9]+", "-", (u or "").lower()).strip("-")[-60:]


def writeup_urls(slug):
    """Top solution writeup URLs for a finished comp: [{'rank','team','writeup_url'}]. [] if none/unfinished."""
    raw = _run_script("fetch_leaderboard_writeups.py", [slug])
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return []


def mine(slugs, top_n=5, out_dir=None):
    """Fetch top-N writeups for each slug → out_dir/<slug>/rank<k>_*.md. Returns {slug: {'n':count}}."""
    out_dir = out_dir or OUT
    top_n = max(0, int(top_n))
    summary = {}
    for slug in (slugs or []):
        d = os.path.join(str(out_dir), slug); os.makedirs(d, exist_ok=True)
        wus = writeup_urls(slug) or []
        got = 0
        for w in wus[:top_n]:
            url = w.get("writeup_url")
            if not url:
                continue
            fn = os.path.join(d, f"rank{w.get('rank', '?')}_{_slugify(url)}.md")
            if os.path.exists(fn) and os.path.getsize(fn) > 200:
                got += 1; continue
            md = _run_script("fetch_writeup.py", [url])
            if md and len(md) > 200:
                open(fn, "w").write(md); got += 1
        summary[slug] = {"n": got}
    return summary


class GmWriteupMine(BaseAgent):
    name = "gm-writeup-mine"
    thread = "R"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        slugs = spec.get("slugs") or ([spec["slug"]] if spec.get("slug") else [])
        if not slugs:
            return self.escalate(worker, "researcher", "gm-writeup-mine needs spec.slugs (finished comp slugs).")
        summary = mine(slugs, top_n=int(spec.get("top_n", 5)), out_dir=spec.get("out_dir"))
        tot = sum(v["n"] for v in summary.values())
        got = [s for s, v in summary.items() if v["n"] > 0]
        msg = (f"gm-writeup-mine: fetched {tot} top-solution writeups across {len(got)} comps → docs/gm_writeups/. "
               f"Feed to trick-extractor/research-search to grow docs/gm_techniques_grounded.md.")
        self.log(msg, kind="finding", recommendation="distill with trick-extractor → recipe-adopt behind CompConfig")
        return self.done({"summary": summary, "total_writeups": tot, "comps": got}, msg)


_AGENT = GmWriteupMine()


def run(q, worker):
    return _AGENT.run(q, worker)
