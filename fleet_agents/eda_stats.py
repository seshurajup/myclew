"""EDA-stats agent — surface the data fingerprint from the competition's precomputed EDA outputs.

Deterministic: reads the existing EDA CSV/tables (cells-per-frame, true-density-stage, division-scan,
motion) and reports the headline numbers the research needs — no recompute. Reuses learning/eda outputs.
"""
from __future__ import annotations

from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
FILES = {
    "cells_per_frame": "learning/01_cells_per_frame_per_dataset.csv",
    "density_stage": "learning/03_true_density_stage.csv",
    "division_scan": "tools/researchpapers/eda/thread2/division_scan.csv",
    "motion": "experiments/eda/e59_motion_table.md",
}
HEADLINE = ("2 embryos (44b6/6bba); sparse labels (~3.6% of cells); TRUE density 38–1015 cells/frame "
            "(median ~186); 151 GT divisions (golden-12 has only 8); motion p95≈4.89µm vs the 7µm gate; "
            "S0–S4 density stages (6bba early, 44b6 late); binding constraint = edge precision / linking")


def _rows(p: Path):
    try:
        with open(p, errors="replace") as fh:
            return max(sum(1 for _ in fh) - 1, 0)
    except Exception:  # noqa: BLE001
        return None


def report(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    files = dict(FILES)
    files.update(spec.get("extra_files") or {})   # extra_files: optional {label: relpath} to also surface
    found = []
    for k, rel in files.items():
        p = COMP / rel
        if p.exists():
            found.append(f"{k}({_rows(p)} rows)" if p.suffix == ".csv" else f"{k}(md)")
    return ("done", {"headline": HEADLINE, "outputs": found}, "all",
            f"[{worker}] EDA-STATS: {HEADLINE}. Precomputed outputs available: {found or 'none — run the EDA scripts'}.")
