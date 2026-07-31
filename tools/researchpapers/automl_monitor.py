"""automl_monitor — long-run health/progress monitor for the autonomous fleet.

Every INTERVAL it appends ONE JSON snapshot of the signals that prove real work is happening over
hours (not just that agents are wired): board completions, real golden-12 verify runs, combos scored,
the best VERIFIED golden-CV (≤1.0), and journal growth. It also flags a STALL (no completions across
several snapshots). Read logs/automl_monitor.jsonl to confirm a 1h / 6h run actually progressed.
"""
import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

COMP = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
RP = COMP / "tools" / "researchpapers"
os.environ.setdefault("RESEARCH_MVP_RUNTIME_DIR", str(RP / ".research-mvp-data" / "runtime"))
sys.path.insert(0, str(RP))
from researchpapers.fleet import board  # noqa: E402

OUT = RP / "logs" / "automl_monitor.jsonl"
COMBO = COMP / "config" / "_auto" / "combo_search_state.json"
CACHE = COMP / "config" / "_auto" / "verified_cv_cache.json"
LEDGER = COMP / "docs" / "experiment_ledger.jsonl"
INTERVAL = int(os.environ.get("MONITOR_INTERVAL", "300"))


def _load(p):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return {}


def _best_cv():
    best = None
    for v in (_load(CACHE) or {}).values():
        cv = v.get("cv")
        if isinstance(cv, (int, float)) and cv <= 1.0:
            best = cv if best is None else max(best, cv)
    return best


def _measured_golden():
    """count of golden-group rows with a numeric measured CV ≤ 1.0 (real, not pending/artifact)."""
    n = 0
    if LEDGER.exists():
        for l in LEDGER.read_text().splitlines():
            try:
                r = json.loads(l)
            except Exception:  # noqa: BLE001
                continue
            if r.get("trn_set") in ("golden12", "public", "golden4"):
                try:
                    if float(r.get("cv")) <= 1.0:
                        n += 1
                except (TypeError, ValueError):
                    pass
    return n


def snapshot():
    c = sqlite3.connect(board.DB, timeout=5)
    def cnt(k, s):
        return c.execute("SELECT count(*) FROM questions WHERE kind=? AND status=?", (k, s)).fetchone()[0]
    stats = dict(board.stats())
    snap = {
        "ts": (datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M IST"),
        "done_total": stats.get("done", 0),
        "claimed": stats.get("claimed", 0),
        "verify_done": cnt("verify-cv", "done"),
        "combo_done": cnt("combo-search", "done"),
        "combos_scored": len((_load(COMBO).get("evaluated") or {})),
        "verified_sigs": len(_load(CACHE) or {}),
        "best_golden_cv": _best_cv(),
        "measured_golden_rows": _measured_golden(),
        "ledger_rows": len(LEDGER.read_text().splitlines()) if LEDGER.exists() else 0,
    }
    c.close()
    return snap


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    hist = []
    print(f"[monitor] every {INTERVAL}s → {OUT}", flush=True)
    while True:
        s = snapshot()
        # stall detection: done_total flat across the last 3 snapshots AND workers idle
        hist.append(s["done_total"])
        s["stalled"] = len(hist) >= 4 and len(set(hist[-4:])) == 1 and s["claimed"] == 0
        with open(OUT, "a") as f:
            f.write(json.dumps(s) + "\n")
        print(f"[monitor] {s['ts']} done={s['done_total']} verify_done={s['verify_done']} "
              f"combos={s['combos_scored']} best_cv={s['best_golden_cv']} measured={s['measured_golden_rows']}"
              f"{' ⚠STALL' if s['stalled'] else ''}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
