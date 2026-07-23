#!/usr/bin/env python
"""Track D entry — runs in the `kaggle_tabular` env (has TabFM installed, Python 3.12).

Invoked by experiment.py via `conda run -n kaggle_tabular python trackD_run.py --params <json>`.
Calls geology_trackD.trackD_oof, writes results/trackD_oof.csv + results/trackD_test.csv,
prints a final line: `TRACKD_CV <cv>` for the parent to parse.
"""
import argparse
import json
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLEET = HERE / "fleet_agents"
sys.path.insert(0, "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/tools/researchpapers")
pkg = types.ModuleType("fleet_agents"); pkg.__path__ = [str(FLEET.resolve())]; sys.modules["fleet_agents"] = pkg
from fleet_agents import geology_trackD as D   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="JSON dict of params")
    args = ap.parse_args()
    params = json.loads(args.params)
    res = HERE / "results"; res.mkdir(exist_ok=True)
    cv = D.trackD_oof(
        HERE / "config" / "_auto" / "train.csv", HERE / "config" / "_auto" / "test.csv",
        res / "trackD_oof.csv", res / "trackD_test.csv", params, log=print,
    )
    print(f"TRACKD_CV {cv:.6f}")


if __name__ == "__main__":
    main()
