"""Kaggle-scout agent — pull top public NOTEBOOKS (+ leaderboard) via the Kaggle CLI, so we don't miss.

Deterministic: shells out to the Kaggle CLI (kernels list by votes, leaderboard), writes
docs/kaggle_scout.md, and reports the top refs to the chat. The researcher/leader then read/pull the
promising ones. Matches the rule: Kaggle research = CLI/kernels only; the best public LB is the floor.
Set KAGGLE_COMP_SLUG to the real competition slug if the default is wrong.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
KAGGLE = os.environ.get("KAGGLE_BIN", "/home/seshu/miniconda3/envs/llm/bin/kaggle")
DEFAULT_SLUG = os.environ.get("KAGGLE_COMP_SLUG", "biohub-cell-tracking-during-development")


def _run(args, timeout=45):
    """timeout (s): cap the Kaggle CLI call so a hung network can't stall the scout."""
    try:
        r = subprocess.run([KAGGLE, *args], capture_output=True, text=True,
                           timeout=max(1, int(timeout)), env={**os.environ})
        return r.stdout if r.returncode == 0 else f"ERR: {r.stderr.strip()[-160:]}"
    except Exception as exc:  # noqa: BLE001
        return f"ERR: {exc}"


def scout(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}    # tolerate a missing/None spec
    slug = spec.get("slug", DEFAULT_SLUG)
    # OPTIONAL: page_size (kernels pulled), top_n (refs surfaced), timeout (per-CLI-call seconds).
    try:
        page_size = max(1, int(spec.get("page_size", 15)))
    except Exception:  # noqa: BLE001
        page_size = 15
    try:
        top_n = max(1, int(spec.get("top_n", 5)))
    except Exception:  # noqa: BLE001
        top_n = 5
    to = spec.get("timeout", 45)
    kernels = _run(["kernels", "list", "--competition", slug, "--sort-by", "voteCount",
                    "--page-size", str(page_size), "--csv"], timeout=to)
    lb = _run(["competitions", "leaderboard", slug, "--show", "--csv"], timeout=to)
    if kernels.startswith("ERR"):
        return ("escalated", {"slug": slug, "err": kernels}, "researcher",
                f"[{worker}] KAGGLE-SCOUT: CLI failed for slug '{slug}' ({kernels[:90]}). "
                f"Set KAGGLE_COMP_SLUG to the real competition slug and I'll pull top notebooks + LB.")
    top = []
    for ln in kernels.splitlines()[1:1 + top_n]:
        ref = ln.split(",")[0].strip().strip('"')
        if ref and "/" in ref:
            top.append(ref)
    out = COMP / "docs" / "kaggle_scout.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Kaggle scout — top public notebooks ({slug})\n\n"
                   f"## Top notebooks (by votes)\n```\n{kernels}\n```\n\n"
                   f"## Leaderboard (top)\n```\n{lb[:900]}\n```\n")
    return ("done", {"slug": slug, "top": top, "file": str(out)}, "all",
            f"[{worker}] KAGGLE-SCOUT: pulled top public notebooks + LB for {slug} → docs/kaggle_scout.md. "
            f"Top: {top}. Researcher: read/pull the promising ones (`kaggle kernels pull <ref>`); best public LB is the floor.")
