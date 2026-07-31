"""comp_env — OPTIONAL per-competition DATA-root override for the researchpapers runtime.

The runtime board (``researchpapers.app:app``) normally serves the ONE competition it lives in,
because it resolves its data (fleet.db, runtime dir, ledger/journal, config, docs) via paths
relative to this package. Setting the ``RP_COMP_ROOT`` environment variable relocates that DATA
root to another competition dir, so a SECOND board instance (own port, own pidfile) can serve a
different competition's data using the SAME shared code.

NOTE: the UNIFIED board switches competition per-request via the ``?comp=<slug>`` URL param + the
``rp_comp`` cookie (see app.py ``_active_comp`` / fleet_agents.db), reading each competition's data
from Postgres ``kaggle_<slug>``. This env override remains for backward compatibility.

Contract:
  * ``RP_COMP_ROOT`` UNSET  → every helper returns ``None``; callers keep their existing default,
                              so behaviour is byte-identical to today (the live biohub instance).
  * ``RP_COMP_ROOT`` SET    → helpers return paths under ``<RP_COMP_ROOT>`` / its ``.research-mvp-data``.

CODE (fleet_agents package, identities, AGENTS.md) is NOT relocated — only DATA. Every helper is
best-effort and NEVER raises.
"""
from __future__ import annotations

import os
from pathlib import Path


def _raw() -> str:
    try:
        return (os.environ.get("RP_COMP_ROOT") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def comp_root() -> Path | None:
    """``<RP_COMP_ROOT>`` — the competition root (holds docs/, config/, experiments/, input/), or None."""
    v = _raw()
    if not v:
        return None
    try:
        return Path(v).expanduser().resolve()
    except Exception:  # noqa: BLE001
        return None


def data_root() -> Path | None:
    """``<RP_COMP_ROOT>/.research-mvp-data`` (control-plane: runtime, fleet.db, projects.json), or None."""
    root = comp_root()
    return (root / ".research-mvp-data") if root is not None else None


def runtime_root() -> Path | None:
    """``<RP_COMP_ROOT>/.research-mvp-data/runtime`` (thread, inboxes, agent state), or None."""
    root = data_root()
    return (root / "runtime") if root is not None else None


def fleet_db() -> Path | None:
    """``<RP_COMP_ROOT>/.research-mvp-data/fleet/fleet.db`` (deterministic fleet question board), or None."""
    root = data_root()
    return (root / "fleet" / "fleet.db") if root is not None else None
