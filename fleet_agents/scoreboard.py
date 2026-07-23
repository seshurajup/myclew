"""scoreboard — ONE live message on /runtime that is a MARKDOWN TABLE of the golden-CV leaderboard,
updated in place as scores land (no new message each time). Shows the search improving in real time:
every measured recipe, ranked by golden-CV, with the gap to the public bar. Pure Python, no Claude.
"""
from __future__ import annotations
import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
CACHE = COMP / "config" / "_auto" / "verified_cv_cache.json"
LEDGER = COMP / "docs" / "experiment_ledger.jsonl"
STATE = COMP / "config" / "_auto" / "scoreboard_state.json"
BAR = 0.885


def _load(p, d):
    try:
        return json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        return d


def _measured():
    """{recipe_label: cv} for every REAL measured golden-CV ≤ 1.0 (verify cache + golden ledger rows)."""
    out = {}
    for sig, v in (_load(CACHE, {}) or {}).items():
        cv = v.get("cv")
        if isinstance(cv, (int, float)) and cv <= 1.0:
            label = sig.replace("BIOHUB_", "").replace("OUTPUT_", "").replace("_UM", "").lower()
            out[label] = max(out.get(label, 0), cv)
    if LEDGER.exists():
        for ln in LEDGER.read_text().splitlines():
            try:
                r = json.loads(ln)
            except Exception:  # noqa: BLE001
                continue
            if r.get("trn_set") in ("golden12", "public", "golden4"):
                try:
                    cv = float(r.get("cv"))
                except (TypeError, ValueError):
                    continue
                if cv <= 1.0:
                    label = (r.get("change") or r.get("exp") or "?").replace("combo:", "").replace("BIOHUB_", "")[:34]
                    out[label] = max(out.get(label, 0), cv)
    return out


def _table(measured, top_n=8, bar=BAR):
    """top_n: how many recipes to list (default 8). bar: gap reference (default the public bar 0.885)."""
    try:
        top_n = max(1, int(top_n))
    except Exception:  # noqa: BLE001
        top_n = 8
    rows = sorted(measured.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    best = rows[0][1] if rows else None
    head = (f"### 🏁 golden-CV leaderboard — best `{best:.4f}` · bar `{bar}` · gap `{best - bar:+.4f}` · "
            f"{len(measured)} recipes" if best else "### 🏁 golden-CV leaderboard — no scores yet")
    lines = [head, "| # | recipe | CV | Δbar |", "|:-|:--|--:|--:|"]
    for i, (label, cv) in enumerate(rows, 1):
        lines.append(f"| {i}{' 🏆' if i == 1 else ''} | `{label}` | **{cv:.4f}** | {cv - bar:+.4f} |")
    return "\n".join(lines)


def report(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}   # OPTIONAL: top_n (rows), bar (gap reference)
    measured = _measured()
    if not measured:
        return ("done", {"measured": 0, "_posted": True}, "all",
                f"[{worker}] scoreboard: no measured golden-CV scores yet.")
    md = _table(measured, top_n=spec.get("top_n", 8), bar=spec.get("bar", BAR))

    from researchpapers.fleet import post
    st = _load(STATE, {})
    eid = st.get("eid")
    updated = post.update_thread(eid, md) if eid else False
    if not updated:                                    # first time (or message rotated out) → post fresh, visible
        eid = post.post_thread(worker, "all", md, routine=False)
        STATE.write_text(json.dumps({"eid": eid}))
    best = max(measured.values())
    return ("done", {"measured": len(measured), "best": best, "_posted": True}, "all",
            f"[{worker}] scoreboard: leaderboard updated in place — best golden-CV {best:.4f} ({len(measured)} recipes).")
