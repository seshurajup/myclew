"""board_seed — make a competition's hub board FULLY populated & honest in one call. Idempotent, best-effort.

Does, for a slug:
  1. ensure the per-comp Postgres exists (kaggle_<slug>) + schema.
  2. mirror existing activity (experiment_decisions) into `fleet_board` so the Python-agents panel
     (/api/runtime/fleet) shows every deterministic fleet-agent + its latest action (not blank).
  3. write docs/INSIGHTS.md from the journal + findings if missing, so the /insights tab isn't a placeholder.

Called from competition_setup.py (every new competition) and runnable ad-hoc:
    python -c "import fleet_agents.board_seed as b; b.seed('rogii-wellbore-geology-prediction')"
"""
from __future__ import annotations
import datetime as _dt
from pathlib import Path


def _db():
    try:
        from fleet_agents import db as _d
        return _d
    except Exception:  # noqa: BLE001
        import importlib.util as _u
        p = Path(__file__).with_name("db.py")
        s = _u.spec_from_file_location("_fleet_db_seed", str(p))
        m = _u.module_from_spec(s); s.loader.exec_module(m); return m


# The reusable fleet-agent ROSTER a competition draws on — shown in the Python-agents panel so the board
# reflects the whole team we'll use, not only the ones that have posted yet. (agent, role/what-it-does).
DEFAULT_ROSTER = [
    ("research", "mine discussions/kernels for the honest method"),
    ("cv-builder", "build & validate the CV that mirrors the hidden test"),
    ("adversarial", "confirm the CV axis matches the test distribution"),
    ("engine", "the core model — alignment + trajectory"),
    ("affine-cal", "GR gain/offset calibration vs typewell"),
    ("ncc-matcher", "multi-scale windowed GR-signature correlation"),
    ("pf-tracker", "particle-filter stratigraphic trajectory"),
    ("dtw-align", "multiscale/stochastic DTW alignment candidate"),
    ("selector", "per-well guarded variant selection / fallback"),
    ("tab-fe", "feature assembly for the meta-stack"),
    ("tab-train", "GBM meta-stack (XGB/CatBoost GPU)"),
    ("blend", "blend the method OOFs"),
    ("blend-optimize", "optimal method+seed blend weights"),
    ("uncertainty", "conformal per-well intervals (Working-Note + fallback)"),
    ("hardware-tune", "pick GPU precision/batch for T4/5090"),
    ("notebook", "package the 2×T4 8h inference notebook"),
    ("kaggle-submit", "submit + calibrate CV↔LB"),
    ("math-master", "govern every knob mathematically"),
    ("xai", "explain feature/agent contributions"),
]


def refresh_fleet_board(slug: str) -> int:
    """Seed fleet_board with the FULL agent roster for this comp (DEFAULT_ROSTER) merged with real activity
    (experiment_decisions). Agents that have posted get their real status; the rest are 'open' (on the team,
    available for this comp) so the Python-agents panel shows the whole team, not just who has run."""
    db = _db()
    try:
        rows = db.all_decisions(slug)
    except Exception:  # noqa: BLE001
        rows = []
    import zlib
    latest: dict[str, dict] = {}
    for r in (rows or []):  # ts-ascending → last write per agent wins
        a = (r.get("agent") or "").strip()
        if a:
            latest[a] = r
    now = _dt.datetime.now()

    def _age_min(ts):
        try:
            t = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
            if t.tzinfo:
                t = t.replace(tzinfo=None)
            return (now - t).total_seconds() / 60.0
        except Exception:  # noqa: BLE001
            return 1e9
    board = []
    nowiso = now.isoformat(timespec="seconds")
    seen = set()
    # 1) agents with real activity → their live status
    for a, r in latest.items():
        ts = r.get("ts") or ""
        seen.add(a)
        board.append(dict(id=zlib.crc32(a.encode()) & 0x7FFFFFFF, thread="session", kind=a,
                          question=(r.get("summary") or "")[:200], spec={"kind": r.get("kind")},
                          status=("running" if _age_min(ts) <= 15 else "done"),
                          claimed_by=a, result=(r.get("detail") or "")[:300], created=ts, updated=ts))
    # 2) rest of the roster → 'open' (on the team for this comp, not yet run)
    for a, role in DEFAULT_ROSTER:
        if a in seen:
            continue
        board.append(dict(id=zlib.crc32(a.encode()) & 0x7FFFFFFF, thread="roster", kind=a,
                          question=role, spec={"kind": "roster"}, status="open",
                          claimed_by=a, result="", created=nowiso, updated=nowiso))
    try:
        return db.sync_board(slug, board)
    except Exception:  # noqa: BLE001
        return 0


def write_insights(slug: str, comp_dir: Path | None = None) -> bool:
    """Generate docs/INSIGHTS.md from the journal (best experiments) + decisions (findings) if missing."""
    comp_dir = comp_dir or (Path("/home/seshu/kaggle/2026") / slug)
    out = comp_dir / "docs" / "INSIGHTS.md"
    db = _db()
    try:
        j = db.all_journal(slug); d = db.all_decisions(slug)
    except Exception:  # noqa: BLE001
        j, d = [], []
    if not j and not d:
        return False
    js = sorted([r for r in j if isinstance(r.get("cv"), (int, float))], key=lambda r: r["cv"])[:8]
    lines = [f"# Insights — {slug}", "",
             f"_Auto-generated from the experiment journal + findings ({len(j)} experiments, {len(d)} findings)._", ""]
    if js:
        lines += ["## Best experiments (by CV)", "", "| Experiment | CV | Description |", "|---|---|---|"]
        lines += [f"| {r.get('exp','')} | {r.get('cv')} | {(r.get('desc') or r.get('description') or '')[:80]} |" for r in js]
        lines += [""]
    if d:
        lines += ["## Key findings & decisions", ""]
        for r in reversed(d):  # newest first
            s = (r.get("summary") or "").strip()
            rec = (r.get("recommendation") or "").strip()
            if s:
                lines.append(f"- **[{r.get('agent','')}]** {s}" + (f" → _{rec}_" if rec else ""))
        lines += [""]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    return True


def seed(slug: str, comp_dir: Path | None = None) -> dict:
    db = _db()
    try:
        db.ensure_db(slug)
    except Exception:  # noqa: BLE001
        return {"ok": False, "reason": "postgres unavailable"}
    n = refresh_fleet_board(slug)
    ins = write_insights(slug, comp_dir)
    return {"ok": True, "fleet_board_rows": n, "insights_written": ins}


if __name__ == "__main__":
    import sys
    print(seed(sys.argv[1] if len(sys.argv) > 1 else "rogii-wellbore-geology-prediction"))
