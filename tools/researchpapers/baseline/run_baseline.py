#!/usr/bin/env python
"""baseline_v1 formal entrypoint (runtime convention).

    python baseline/run_baseline.py --config baseline/experiments_v1/<cfg>.yaml [--dry-run] [--fold 0]

Thin wrapper that dispatches to the observable launcher `python -m src.baseline.train`
(which owns the YAML->env/argv mapping, the STARTUP banner, and the tee'd per-run log).
Run under the competition venv python so pyyaml + the trainer are importable:
    research/cellmot_venv/bin/python.
"""
import argparse
import subprocess
import sys
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[1]  # tools/researchpapers (so `src.baseline` imports)


def main() -> None:
    ap = argparse.ArgumentParser(description="baseline_v1 formal runner -> src.baseline.train")
    ap.add_argument("--config", required=True, help="baseline/experiments_v1/<cfg>.yaml")
    ap.add_argument("--fold", type=int, default=0, help="split index (golden-12 = 0)")
    ap.add_argument("--dry-run", action="store_true", help="GPU-safe wiring validation; no training")
    args = ap.parse_args()

    cfg = str(Path(args.config).resolve())
    cmd = [sys.executable, "-m", "src.baseline.train", "--config", cfg, "--fold", str(args.fold)]
    if args.dry_run:
        cmd.append("--dry-run")
    print(f"[run_baseline] dispatch: {' '.join(cmd)}  (cwd={WORKDIR})", flush=True)
    sys.exit(subprocess.call(cmd, cwd=str(WORKDIR)))


if __name__ == "__main__":
    main()
