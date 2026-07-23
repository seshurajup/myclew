"""context-offload — write a LARGE tool/worker output to disk and return a COMPACT summary + path.

The one context-management primitive the fleet lacked. The leader/researcher board thread and agent
inboxes are the working context; dumping a 5k-line score table, a full log, or a big JSON blob into them
poisons the thread (and every later read pays for it). This agent is the deterministic offload valve:

    offload(text, label) -> writes output/run_artifacts/<comp>/<ts>_<label>.<ext>
                            returns {path, bytes, lines, preview, summary}

The caller then posts the SHORT summary + the path to the board — not the full dump. Any agent (or the
Claude leader) reads the detail back on demand with read_slice(path, offset, limit).

This is exactly the pattern deepagents' harness uses (SummarizationMiddleware offloads evicted history to
`/conversation_history/{thread}.md`; FilesystemMiddleware replaces an oversized ToolMessage with a stub
pointing at `/large_tool_results/{id}` plus a head+tail preview) — ported to OUR stack (real filesystem +
fleet agents, no LangChain). It is also the neurogolf-9th-place working pattern: a ~200-line working memo
in context, full detail on disk.

Reusable across every competition: RP_COMP routes the artifact dir to the active comp, same as the ledger.
Deterministic, offline, no GPU, no network.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
PROJECTS_BASE = COMP.parent                                   # /home/seshu/kaggle/2026 (all comp dirs)

# ---- TWO-TIER thresholds (deepagents summarization.py: a LOWER in-place truncation trigger below the
# full-offload trigger, so an oversized tool-call arg / file-write / grep dump gets clipped at emit time
# BEFORE a full offload is ever needed). Characters, not tokens (deterministic, dependency-free). ----
TRUNCATE_LIMIT = 4_000        # LOWER tier: clip an oversized single arg/output in place (keep head+tail)
OFFLOAD_LIMIT = 20_000        # UPPER tier: content this big should be offload()'d to disk with a breadcrumb


def _active_comp() -> str:
    """The competition this offload targets: env RP_COMP else this comp (mirrors ledger._active_comp)."""
    return (os.environ.get("RP_COMP") or "").strip() or COMP.name


def artifacts_dir(slug: str | None = None) -> Path:
    """output/run_artifacts/ for the active comp (biohub-local; PROJECTS_BASE/<slug>/output for others)."""
    slug = slug or _active_comp()
    base = COMP if slug == COMP.name else (PROJECTS_BASE / slug)
    return base / "output" / "run_artifacts"


def _slug(label: str) -> str:
    """Filesystem-safe basename fragment from a free-text label."""
    keep = "".join(c if (c.isalnum() or c in "-_") else "_" for c in (label or "artifact").strip())
    return (keep.strip("_") or "artifact")[:60]


def _preview(text: str, head: int = 12, tail: int = 8, width: int = 500) -> str:
    """Head+tail preview with a truncation marker in the middle (deepagents _create_content_preview)."""
    lines = text.splitlines()
    if len(lines) <= head + tail:
        return "\n".join(ln[:width] for ln in lines)
    h = "\n".join(ln[:width] for ln in lines[:head])
    t = "\n".join(ln[:width] for ln in lines[-tail:])
    return f"{h}\n... [{len(lines) - head - tail} lines truncated — read the file for the middle] ...\n{t}"


def offload(text, label: str = "output", summary: str = "", *, ext: str = "md",
            head: int = 12, tail: int = 8, out_dir=None, slug: str | None = None) -> dict:
    """Write `text` to a run-artifact file; return a compact {path, bytes, lines, preview, summary, stub}.

    text     : the large content (str, or any object → JSON/str-coerced).
    label    : short name → becomes the artifact basename (<ts>_<label>.<ext>).
    summary  : optional one-line caller summary carried in the return (what the thread should show).
    ext      : file extension (md/txt/json/log). JSON objects auto-serialise when ext='json'.
    head/tail: preview line counts.
    out_dir  : override the artifacts directory (else output/run_artifacts/<comp>/).
    Returns a dict; `stub` is the ready-to-post message (summary + path + preview), never the full dump.
    """
    if not isinstance(text, str):
        try:
            text = json.dumps(text, indent=2, default=str) if ext == "json" else str(text)
        except Exception:  # noqa: BLE001
            text = str(text)
    d = Path(out_dir) if out_dir else artifacts_dir(slug)
    d.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = d / f"{ts}_{_slug(label)}.{ext.lstrip('.')}"
    # avoid clobber if two offloads land in the same second
    i = 1
    while path.exists():
        path = d / f"{ts}_{_slug(label)}_{i}.{ext.lstrip('.')}"
        i += 1
    path.write_text(text)
    nbytes = len(text.encode("utf-8", "replace"))
    nlines = text.count("\n") + 1
    prev = _preview(text, head=head, tail=tail)
    stub = (f"{summary or label} — full output offloaded to disk (not inlined).\n"
            f"path: {path}  ({nlines} lines, {nbytes} bytes)\n"
            f"read back: python -m researchpapers.fleet_dispatch context-offload read "
            f"'{{\"path\":\"{path}\",\"offset\":0,\"limit\":100}}'\n"
            f"--- preview (head/tail) ---\n{prev}")
    return {"path": str(path), "bytes": nbytes, "lines": nlines,
            "preview": prev, "summary": summary or label, "stub": stub}


def read_slice(path, offset: int = 0, limit: int = 200, width: int = 2000) -> dict:
    """Read lines [offset : offset+limit) of an offloaded artifact (deepagents read_file offset/limit)."""
    p = Path(path)
    if not p.exists():
        return {"error": f"no such artifact: {path}", "lines": 0}
    all_lines = p.read_text(errors="replace").splitlines()
    offset = max(0, int(offset)); limit = max(1, int(limit))
    chunk = all_lines[offset:offset + limit]
    body = "\n".join(ln[:width] for ln in chunk)
    more = offset + limit < len(all_lines)
    return {"path": str(p), "total_lines": len(all_lines), "offset": offset, "limit": limit,
            "returned": len(chunk), "more": more, "text": body}


def _clip_str(s: str, limit: int, path: str | None = None) -> str:
    """Clip one oversized string in place: keep head+tail, insert an elided-char breadcrumb marker."""
    if len(s) <= limit:
        return s
    head = max(1, (limit * 6) // 10)          # ~60% head, ~40% tail — recent+opening both survive
    tail = max(1, limit - head)
    elided = len(s) - head - tail
    marker = f"...[{elided} chars elided, full at {path or '<path>'} if offloaded]..."
    return s[:head] + marker + s[-tail:]


def truncate_args(text_or_dict, limit: int = TRUNCATE_LIMIT, *, path: str | None = None):
    """LOWER-tier in-place truncation of an oversized tool-call arg / file-write / grep-style output.

    This is deepagents' `_should_truncate_args`/`_truncate_tool_call` ported: a SEPARATE, LOWER threshold
    (`TRUNCATE_LIMIT`) than full offload (`OFFLOAD_LIMIT`). Any single arg/output over `limit` chars is
    clipped to head+tail with a `...[N chars elided, full at <path> if offloaded]...` marker, so one big
    write_file/grep/log dump never balloons the thread BEFORE a full offload() is even considered.

    text_or_dict :
        str  → returns the clipped string (strings are immutable, so nothing is mutated).
        dict → each oversized STRING value is clipped IN PLACE (typical tool-call `args`); returns same dict.
        list → each oversized string element clipped IN PLACE; returns same list.
        (other types are returned untouched.)
    limit : the LOWER threshold in characters (default TRUNCATE_LIMIT). Pass OFFLOAD_LIMIT to only clip the
        truly huge; the two-tier default keeps them distinct.
    path : optional artifact path to name in the marker (if the full content was/will be offloaded).
    """
    if isinstance(text_or_dict, str):
        return _clip_str(text_or_dict, limit, path)
    if isinstance(text_or_dict, dict):
        for k, v in text_or_dict.items():
            if isinstance(v, str):
                text_or_dict[k] = _clip_str(v, limit, path)
        return text_or_dict
    if isinstance(text_or_dict, list):
        for i, v in enumerate(text_or_dict):
            if isinstance(v, str):
                text_or_dict[i] = _clip_str(v, limit, path)
        return text_or_dict
    return text_or_dict


def run(q, worker):
    """Fleet handler. spec routes to three verbs:

      offload (default): {text|path, label, summary, ext, head, tail} → writes artifact, returns compact stub.
      read            : {path, offset, limit}                         → returns a slice of an artifact.
      truncate        : {text, limit}                                 → LOWER-tier in-place clip (head+tail),
                        no disk write — the emit-time valve BELOW full offload.

    A `mode`/`verb` == 'read' (or a spec with `path` and no `text`) selects read; else offload. With an
    empty spec (smoke), it offloads the question text — a valid, harmless no-op that still exercises the path.
    """
    self = BaseAgent(); self.name = "context-offload"
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    question = (q.get("question") if isinstance(q, dict) else str(q)) or ""
    verb = str(spec.get("mode") or spec.get("verb") or "").lower()

    if verb == "truncate":
        src = spec.get("text", question)
        clipped = truncate_args(src, int(spec.get("limit", TRUNCATE_LIMIT)), path=spec.get("path"))
        before = len(src) if isinstance(src, str) else len(str(src))
        after = len(clipped) if isinstance(clipped, str) else len(str(clipped))
        res = {"before_chars": before, "after_chars": after, "text": clipped,
               "shrunk": after < before, "limit": int(spec.get("limit", TRUNCATE_LIMIT))}
        return self.done(res, f"[{worker}] truncated {before}→{after} chars (limit {res['limit']})", to="all")

    is_read = verb == "read" or ("path" in spec and "text" not in spec and "file" not in spec and verb != "offload")
    if is_read:
        res = read_slice(spec.get("path", ""), spec.get("offset", 0), spec.get("limit", 200))
        if res.get("error"):
            return ("error", res, "all", f"[{worker}] context-offload read: {res['error']}")
        msg = (f"[{worker}] read {res['returned']} lines "
               f"(offset {res['offset']}/{res['total_lines']}, more={res['more']}) of {res['path']}")
        return self.done(res, msg, to="all")

    # offload
    text = spec.get("text")
    if text is None and (spec.get("path") or spec.get("file")):
        src = Path(spec.get("path") or spec.get("file"))
        text = src.read_text(errors="replace") if src.exists() else ""
    if text is None or text == "":
        text = question                                       # smoke / no-payload → offload the question
    res = offload(text, label=spec.get("label", "output"), summary=spec.get("summary", ""),
                  ext=spec.get("ext", "md"), head=int(spec.get("head", 12)), tail=int(spec.get("tail", 8)),
                  out_dir=spec.get("out_dir"))
    self.log(summary=f"offloaded {res['lines']}-line output → {res['path']}",
             detail=res["summary"], kind="finding")
    return self.done(res, f"[{worker}] {res['stub']}", to="all")
