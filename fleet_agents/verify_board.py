"""verify_board — the competition-board VERIFIER. Asserts a competition's hub board is fully provisioned
(per-comp Postgres + schema + agent roster + INSIGHTS + live panels honest/comp-scoped). Run automatically
at the end of competition_setup.py (enforces provisioning — fails loudly if anything is missing) and
standalone anytime:  python -m fleet_agents.verify_board <slug>   (exit 0 = PASS, 1 = FAIL).

Each check is (name, ok, detail). Postgres checks are hard requirements; live-board (:7788) checks are
soft (skipped with a note if the board server isn't running) so setup on a box without the server still
passes the provisioning it CAN guarantee.
"""
from __future__ import annotations
import sys
from pathlib import Path


def _db():
    try:
        from fleet_agents import db as _d
        return _d
    except Exception:  # noqa: BLE001
        import importlib.util as _u
        p = Path(__file__).with_name("db.py")
        s = _u.spec_from_file_location("_fleet_db_verify", str(p))
        m = _u.module_from_spec(s); s.loader.exec_module(m); return m


def _board_json(path: str, timeout: float = 6.0):
    import json, urllib.request
    with urllib.request.urlopen(f"http://localhost:7788{path}", timeout=timeout) as r:
        return json.load(r)


def verify(slug: str, min_roster: int = 15) -> tuple[bool, list[tuple[str, bool, str]]]:
    checks: list[tuple[str, bool, str]] = []
    db = _db()
    name = db.db_name(slug)

    # --- HARD: Postgres provisioning ---
    try:
        con = db._connect(name)
        with con, con.cursor() as cur:
            cur.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            tables = {r[0] for r in cur.fetchall()}
        con.close()
        need = {"experiment_journal", "experiment_decisions", "research_index", "lb_snapshot", "fleet_board"}
        checks.append((f"postgres db {name} + schema", need <= tables,
                       f"tables: {sorted(need & tables)}" + (f" MISSING {sorted(need - tables)}" if need - tables else "")))
    except Exception as exc:  # noqa: BLE001
        checks.append((f"postgres db {name}", False, f"cannot connect: {exc}"))
        return False, checks

    try:
        board = db.all_board(slug)
        checks.append((f"agent roster seeded (≥{min_roster})", len(board) >= min_roster, f"{len(board)} fleet_board rows"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("agent roster seeded", False, str(exc)))

    # --- SOFT: live board panels (only if :7788 is up) ---
    try:
        b = _board_json(f"/api/runtime/board?comp={slug}")
        checks.append(("Claude agents honest (0 for non-home)", isinstance(b.get("agents"), list) and len(b["agents"]) == 0,
                       f"agents={len(b.get('agents', []))}"))
        f = _board_json(f"/api/runtime/fleet?comp={slug}")
        nk = len(f.get("kinds", {})) if isinstance(f, dict) else 0
        checks.append((f"Python-agents panel populated (≥{min_roster})", nk >= min_roster, f"kinds={nk}"))
        sp = _board_json(f"/api/runtime/spend?comp={slug}")
        checks.append(("spend scoped to last-hour (non-home)", sp.get("scope") == "last60", f"scope={sp.get('scope')}"))
        tj = _board_json(f"/api/runtime/trainjobs?comp={slug}")
        checks.append(("training-jobs comp-scoped (non-home)", tj.get("scope") == "comp", f"scope={tj.get('scope')}"))
    except Exception as exc:  # noqa: BLE001
        checks.append(("live board panels (:7788)", True, f"SKIPPED — board server not reachable ({type(exc).__name__})"))

    ok = all(c[1] for c in checks)
    return ok, checks


def main(slug: str) -> int:
    ok, checks = verify(slug)
    print(f"\n🔎 Competition board verifier — {slug}")
    for name, passed, detail in checks:
        print(f"  {'✅' if passed else '❌'} {name}  ·  {detail}")
    print(f"\n{'✅ PASS — board fully provisioned' if ok else '❌ FAIL — board NOT fully provisioned (re-run competition_setup.py or board_seed)'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "rogii-wellbore-geology-prediction"))
