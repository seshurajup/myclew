"""Note-parser agent — ingest STRUCTURED research notes into the journal (no LLM).

The researcher writes each experiment as a machine-parseable block (docs/EXPERIMENT_TEMPLATE.md):
a '### EXP' header + 'key: value' lines. This agent scans docs/research_notes/*.md, parses every
block, and records it to the ledger — so Python gets EXACTLY what it needs from the research,
deterministically. This is what lets a structured research write-up drive the pipeline without Claude.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

from . import ledger

COMP = Path(__file__).resolve().parent.parent
NOTES_GLOB = str(COMP / "docs" / "research_notes" / "*.md")
FIELDS = ("stage", "parent", "change", "config", "script", "cv", "lb", "trn_set", "observation", "kept")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v  # keep strings like 'bad' / 'nan' / 'overfit'


def _parse_block(text: str) -> dict:
    d = {}
    for ln in text.splitlines():
        m = re.match(r"\s*([a-z_]+)\s*:\s*(.*)", ln)
        if m and m.group(1) in FIELDS:
            val = m.group(2).split("#")[0].strip()  # strip inline comments
            if val:
                d[m.group(1)] = val
    return d


def parse_notes(notes_glob=None, max_files=2000) -> list[dict]:
    """notes_glob: override the default docs/research_notes/*.md pattern. max_files: cap the scan (robustness).
    Any unreadable/badly-formed note file is skipped, never fatal."""
    exps = []
    try:
        files = glob.glob(notes_glob or NOTES_GLOB)[:max(1, int(max_files))]
    except Exception:  # noqa: BLE001
        files = []
    for f in files:
        try:
            txt = Path(f).read_text(errors="replace")
        except Exception:  # noqa: BLE001
            continue
        try:
            blocks = re.split(r"^#{2,3}\s*EXP.*$", txt, flags=re.M)[1:]
        except Exception:  # noqa: BLE001
            continue
        for b in blocks:
            d = _parse_block(b)
            if d.get("change") or d.get("config"):
                exps.append(d)
    return exps


def sync(q, worker):
    """Fleet handler — parse structured research notes → journal rows (dedup).
    OPTIONAL spec: notes_glob (override the note file pattern)."""
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    exps = parse_notes(spec.get("notes_glob"))
    for d in exps:
        try:
            cv = _num(d["cv"]) if d.get("cv") and d["cv"] != "pending" else None
            lb = _num(d["lb"]) if d.get("lb") and d["lb"] != "pending" else None
            stage = int(d["stage"]) if str(d.get("stage", "")).isdigit() else None
            kept = str(d.get("kept", "")).lower() in ("true", "yes", "1")
            script = d.get("script") or (f"bash start_train.sh {d['config']}" if d.get("config") else "")
            ledger.record(change=d.get("change") or d.get("config", ""), script=script,
                          cv=cv, lb=lb, train_set=d.get("trn_set", "loeo"), parent=d.get("parent") or None,
                          stage=stage, kept=kept, observation=d.get("observation", ""))
        except Exception:  # noqa: BLE001 — one malformed note must not abort the whole ingest
            continue
    return ("done", {"parsed": len(exps)}, "all",
            f"[{worker}] NOTES-SYNC: parsed {len(exps)} structured research note(s) → journal (dedup). "
            f"Research written in docs/EXPERIMENT_TEMPLATE.md format is ingested by Python directly — no LLM.")
