"""kaggle_cli — ONE tested wrapper around the Kaggle CLI, shared by every agent that touches Kaggle.

Why this exists: 11 agents each shelled the CLI with their own private helper. That duplication is not
cosmetic — it hid a real bug. `comp-onboard` called
`kaggle competitions pages <slug> --content --page-name Evaluation`, which ALWAYS fails on CLI 2.2.4 (the
`pages` command now requires a subcommand, so the slug was parsed as the command), swallowed the error, and
silently fingerprinted every competition from filenames alone. Ten other agents could not catch it because
they had their own copies. Fix the call here once and every caller is fixed.

This is deliberately NOT one mega "kaggle agent": the agents do different jobs, carry their own data-wise
tests, and the guarded registry isolates a failure to a single capability. Shared PLUMBING, separate agents.

Everything returns text/lists and never raises on a CLI failure — a competition we cannot reach is a
RESULT, not a crash — except where a caller explicitly asks for strict mode.
"""
from __future__ import annotations

import os
import subprocess

BIN = os.environ.get("KAGGLE_BIN", "/home/seshu/miniconda3/envs/llm/bin/kaggle")


def run(args, timeout=60, strict=False):
    """Run the CLI. Returns stdout ('' on failure) — or raises when strict."""
    try:
        r = subprocess.run([BIN, *args], capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        if strict:
            raise
        return ""
    if r.returncode != 0 and strict:
        raise RuntimeError(f"kaggle {' '.join(args)} failed: {(r.stderr or r.stdout)[:200]}")
    return r.stdout or ""


def set_competition(slug):
    """Set the CLI's default competition. REQUIRED before `pages`, which takes no slug argument."""
    return run(["config", "set", "-n", "competition", "-v", slug])


def page_names(slug):
    """Every official page name for a competition (Evaluation, rules, Timeline, Prizes, ...)."""
    set_competition(slug)
    out = run(["competitions", "pages", "list", "--format", "csv"])
    names = []
    for ln in out.splitlines():
        n = ln.split(",")[0].strip().strip('"')
        if not n or n.lower() == "name" or n.startswith("Using competition"):
            continue
        names.append(n)
    return names


def page(slug, name):
    """The CONTENT of one official page. Correct 2.2.4 syntax: default-competition + `pages list`."""
    set_competition(slug)
    return run(["competitions", "pages", "list", "--content", "--page-name", name])


def all_pages(slug, save_dir=None):
    """{page_name: text} for EVERY page. Reading only Evaluation+Overview loses the rules, the submission
    limits and the timeline — all of which change how a competition must be played."""
    out = {}
    for n in page_names(slug):
        txt = page(slug, n)
        if txt and len(txt) > 40:
            out[n] = txt
    if save_dir:
        try:
            os.makedirs(save_dir, exist_ok=True)
            for n, txt in out.items():
                with open(os.path.join(save_dir, f"{n.replace('/', '_')}.txt"), "w") as fh:
                    fh.write(txt)
        except OSError:
            pass
    return out


def files(slug):
    """Competition file manifest (names only)."""
    raw = run(["competitions", "files", slug, "--csv"])
    return [ln.split(",")[0].strip().strip('"') for ln in raw.splitlines()[1:] if ln.strip()]


def _csv_rows(raw, top=None, expect=()):
    """Parse CLI csv output into dicts.

    Two measured traps: the CLI prints a `Next Page Token = ...` preamble BEFORE the header (so line 0 is
    not the header), and fields can contain commas — team names like "Smith, J" would be shredded by a naive
    split. Find the header by looking for the expected column, and parse with the csv module.
    """
    import csv as _csv
    import io
    lines = [l for l in raw.splitlines() if l.strip()]
    start = 0
    for i, ln in enumerate(lines):
        if (not expect and "," in ln) or any(e in ln for e in expect):
            start = i
            break
    else:
        return []
    rdr = _csv.DictReader(io.StringIO("\n".join(lines[start:])))
    rows = [{(k or "").strip(): (v or "").strip() for k, v in r.items()} for r in rdr]
    return rows[: int(top)] if top else rows


def leaderboard(slug, top=None):
    """Leaderboard rows as dicts. `-v` paginates, so always request csv (a standing lesson)."""
    raw = run(["competitions", "leaderboard", slug, "-s", "--csv"])
    return _csv_rows(raw, top, expect=("teamId", "teamName"))


def submissions(slug, top=None):
    """Our own submissions (ref, date, publicScore, privateScore)."""
    raw = run(["competitions", "submissions", slug, "--format", "csv"])
    return _csv_rows(raw, top, expect=("ref", "fileName", "publicScore"))


def kernels(slug, top=20, sort="voteCount"):
    """Top public notebooks for a competition."""
    raw = run(["kernels", "list", "--competition", slug, "--sort-by", sort,
               "--page-size", str(int(top)), "--csv"])
    return _csv_rows(raw, expect=("ref", "title"))


def kernel_pull(ref, dest):
    """Pull one notebook + its metadata into `dest`."""
    os.makedirs(dest, exist_ok=True)
    return run(["kernels", "pull", ref, "-p", dest, "-m"], timeout=180)
