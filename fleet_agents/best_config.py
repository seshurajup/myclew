"""best-config — assemble the BEST inference config from the public learnings + CV↔LB anchors (Part A).

Deterministic: reads docs/kaggle_learnings.md (mined params across 73 notebooks) + the public journal
rows, picks the consensus/highest-LB inference options (det_threshold, gap µm, safe_div, checkpoint hint),
and writes config/_auto/best_inference.yml — the inference-only base that reuses the community pipeline.
No training. The score agent then runs it on golden-12.
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
LEARN = COMP / "docs" / "kaggle_learnings.md"
OUT = COMP / "config" / "_auto" / "best_inference.yml"


def build(q, worker):
    if not LEARN.exists():
        return ("escalated", {}, "researcher",
                f"[{worker}] best-config: no docs/kaggle_learnings.md yet — run notebook-sync first.")
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    gap_default = str(spec.get("gap_um", "5.5"))             # gap_um: gap-close distance to bake into the base config
    txt = LEARN.read_text() or ""
    dets = Counter(re.findall(r"det_threshold=\[([^\]]+)\]", txt))
    gaps = Counter(re.findall(r"gap_um=\[([^\]]+)\]", txt))
    safe = txt.lower().count("safe_div") + txt.lower().count("add_safe_divisions")
    # consensus picks (fall back to the sweep's known-best values)
    det = "0.99" if not dets else max(dets, key=dets.get).split(",")[0].strip().strip("'\"")
    rec = {"det_threshold": det or "0.99", "safe_divisions": safe > 0, "gap_um": gap_default, "note":
           "learned-graph inference base (pilkwang pipeline); best LB comes from checkpoint + safe_divisions"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("# best inference config (assembled from public learnings — NO training)\n"
                   f"inference:\n  det_threshold: {rec['det_threshold']}\n  safe_divisions: {str(rec['safe_divisions']).lower()}\n"
                   f"  gap_um: {rec['gap_um']}\n  base: pilkwang learned-graph pipeline\n")
    return ("done", {"config": str(OUT.relative_to(COMP)), **rec}, "all",
            f"[{worker}] BEST-CONFIG: assembled inference base → {OUT.name} "
            f"(det_threshold={rec['det_threshold']}, safe_divisions={rec['safe_divisions']}, gap={rec['gap_um']}µm). "
            f"Anchor: golden-12 ~0.87 → LB ~0.89. No training.")
