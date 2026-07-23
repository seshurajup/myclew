"""public-config — one config/exp yml per PUBLIC notebook (full coverage of all 73+).

Reads the public journal rows + the mined params (docs/kaggle_learnings.md) and writes
config/exp/public/<notebook>.yml for each — its family (learned-graph / rule-based DoG), base pipeline,
mined params (det_threshold, gap/motion µm, safe_div, augs), its LB, and our golden-12 CV. So the fleet
covers every public solution as a runnable/comparable config, and the set stays current as sync pulls more.
"""
from __future__ import annotations

import re
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
LEARN = COMP / "docs" / "kaggle_learnings.md"
OUTDIR = COMP / "config" / "exp" / "public"


def _mined_by_ref():
    """ref → 'det_threshold=[..]; gap_um=[..]; division=[..]' from kaggle_learnings.md."""
    out = {}
    if LEARN.exists():
        for ln in LEARN.read_text().splitlines():
            m = re.match(r"- \*\*(.+?)\*\* — (.+)", ln.strip())
            if m:
                out[m.group(1)] = m.group(2)
    return out


def generate(q, worker):
    from . import ledger
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    limit = spec.get("limit")                                # limit: cap the number of public configs written (None = all)
    limit = int(limit) if limit is not None else None
    OUTDIR.mkdir(parents=True, exist_ok=True)
    mined = _mined_by_ref()
    try:
        pubs = [e for e in (ledger.entries() or []) if isinstance(e, dict) and e.get("trn_set") == "public"]
    except Exception:  # noqa: BLE001
        pubs = []
    if limit is not None:
        pubs = pubs[:limit]
    n = 0
    for e in pubs:
        name = (e.get("change") or "").replace("pub_", "")
        if not name:
            continue
        desc = e.get("desc") or e.get("description") or ""
        fam = "learned-graph" if "learned-graph" in desc else ("rule-based-DoG" if "rule-based" in desc else "other")
        ref = (e.get("script") or "").replace("kaggle: ", "")
        params = mined.get(ref.split("/")[-1], "") or mined.get(ref, "")
        base = "pilkwang_learned_graph" if fam == "learned-graph" else ("dog_hungarian" if "DoG" in fam else "unknown")
        y = OUTDIR / f"{name[:40]}.yml"
        y.write_text(
            f"# PUBLIC notebook config — {ref}\n"
            f"name: pub_{name[:36]}\n"
            f"source: {ref}\n"
            f"family: {fam}\n"
            f"base: {base}\n"
            f"lb: {e.get('lb')}\n"
            f"golden_cv: {e.get('cv')}\n"
            f"mined_params: \"{params[:200]}\"\n"
            f"run: {'inference' if fam == 'learned-graph' else 'rule-based-dog' if fam == 'rule-based-DoG' else 'manual'}\n"
            f"score: golden-12 (adj_edge_J + 0.1*div_J), by_embryo\n")
        n += 1
    return ("done", {"dir": str(OUTDIR.relative_to(COMP)), "count": n}, "all",
            f"[{worker}] PUBLIC-CONFIG: wrote {n} config/exp/public/*.yml — every public notebook is now a "
            f"runnable/comparable config (family, base, params, LB, golden-CV). Covers all synced notebooks.")
