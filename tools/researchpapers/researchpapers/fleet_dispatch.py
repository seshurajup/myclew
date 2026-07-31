"""fleet_dispatch — the leader/researcher's handle to the Python worker fleet.

The Claude agents (leader/researcher) coordinate on the board; the 69 deterministic Python agents do
the heavy lifting. This is the bridge: enqueue a question with a `kind`, and the matching worker claims
and runs it. Without this the leader could only delegate to the other Claude agent — it could not reach
the fleet at all.

    python -m researchpapers.fleet_dispatch <kind> "<question text>" ['<spec JSON>']
    python -m researchpapers.fleet_dispatch --list          # print the agent catalog (kind — purpose)

Validates `kind` against the LIVE registry (fleet_agents.HANDLERS) so a typo can't silently no-op.

FINAL-ANSWER-ONLY CONTRACT: a dispatched worker's CALLER sees ONLY the worker's final message — not its
intermediate work, tool results, or status. Put the COMPLETE, self-contained answer in that final message
and share ABSOLUTE PATHS for anything large; do not narrate intermediate steps as the deliverable.
"""
from __future__ import annotations
import inspect
import json
import os
import sys

# a sensible default board thread per agent family (workers read every thread; this is just grouping)
_THREAD = {"S": "S", "B": "B"}


def _load_agents():
    root = os.environ.get("FLEET_COMPETITION_ROOT")
    if root and root not in sys.path:
        sys.path.insert(0, root)
    name = os.environ.get("FLEET_COMPETITION", "fleet_agents")
    import importlib
    return importlib.import_module(name)


def _purpose(fn) -> str:
    mod = inspect.getmodule(fn)
    doc = (mod.__doc__ or "").strip() if mod else ""
    if not doc or doc.startswith("base —"):
        doc = (getattr(fn, "__doc__", None) or "").strip()
    first = doc.replace("\n", " ").split(". ")[0].strip() if doc else "(no docstring)"
    return " ".join(first.split())[:150]


def list_agents() -> int:
    F = _load_agents()
    raw = getattr(F, "_RAW_HANDLERS", F.HANDLERS)
    for k in sorted(raw):
        print(f"{k:26} — {_purpose(raw[k])}")
    print(f"\n{len(raw)} agents. Dispatch: python -m researchpapers.fleet_dispatch <kind> \"<q>\" '<spec-json>'")
    return 0


def dispatch(kind: str, question: str, spec: dict | None = None, thread: str = "B") -> int:
    F = _load_agents()
    raw = getattr(F, "_RAW_HANDLERS", F.HANDLERS)
    if kind not in raw:
        near = [k for k in raw if kind.replace("_", "-") in k or k in kind]
        print(f"ERROR: unknown agent kind '{kind}'. Did you mean: {near[:5] or sorted(raw)[:8]} … "
              f"(run --list for the full catalog)", file=sys.stderr)
        return 2
    from researchpapers.fleet import board
    board.add(thread, kind, question, spec or {})       # _conn() auto-creates the schema
    print(f"dispatched → kind={kind} thread={thread} :: {question[:80]}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--list":
        return list_agents()
    kind = argv[0]
    question = argv[1] if len(argv) > 1 else f"run {kind}"
    spec = None
    if len(argv) > 2 and argv[2].strip():
        try:
            spec = json.loads(argv[2])
        except json.JSONDecodeError as e:
            print(f"ERROR: spec is not valid JSON: {e}", file=sys.stderr)
            return 2
    return dispatch(kind, question, spec)


if __name__ == "__main__":
    raise SystemExit(main())
