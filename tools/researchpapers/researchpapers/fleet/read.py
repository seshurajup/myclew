"""Read messages — Python agents READ the shared thread / an inbox with the structured template.

Every message posted via post.post_thread carries {from, to, kind, content, data, routine, ...}. This
module lets a deterministic agent react to OTHER agents' messages (a leader directive, another agent's
result) by parsing the structured `data`/`kind`, not the free text. So agents both send AND read.
"""
from __future__ import annotations

import glob
import json
import os
import re

from .post import runtime_dir

_TEMPLATE_KV = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]{1,20}):\s*(.+?)\s*$")


def parse_template(content: str) -> dict:
    """FIXED MESSAGE TEMPLATE — leading `KEY: value` lines → structured data (see runtime_cli). Lets
    a Python agent read a Claude leader's directive deterministically even if `data` wasn't set."""
    data: dict = {}
    for line in (content or "").splitlines():
        if not line.strip():
            break
        m = _TEMPLATE_KV.match(line)
        if not m:
            break
        data[m.group(1).lower()] = m.group(2).strip()
    return data


def _with_data(e: dict) -> dict:
    """Ensure a message carries `data`: fall back to parsing the template header from its content."""
    if not e.get("data"):
        e = {**e, "data": parse_template(e.get("content") or "")}
    return e


def read_thread(kinds=None, sender=None, to=None, contains=None, limit=50):
    """Return the last `limit` structured messages, filtered. kinds/sender/to filter by field;
    contains = substring in content. Newest-last (thread order)."""
    thread = runtime_dir() / "thread.jsonl"
    out = []
    if thread.exists():
        for ln in thread.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                e = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if kinds and e.get("kind") not in kinds:
                continue
            if sender and e.get("from") != sender:
                continue
            if to and e.get("to") not in (to, "all"):
                continue
            if contains and contains.lower() not in (e.get("content") or "").lower():
                continue
            out.append(_with_data(e))
    return out[-limit:]


def read_inbox(agent: str):
    """Return the structured messages sitting in an agent's inbox (directed messages to it)."""
    inbox = runtime_dir() / "inbox" / agent
    out = []
    if inbox.is_dir():
        for f in sorted(glob.glob(str(inbox / "*.json")), key=os.path.getmtime):
            try:
                out.append(json.load(open(f)))
            except Exception:  # noqa: BLE001
                pass
    return out


def latest(kind: str):
    """The most recent structured message of a given kind (or None) — e.g. read.latest('cv-build')."""
    msgs = read_thread(kinds=[kind], limit=1)
    return msgs[-1] if msgs else None
