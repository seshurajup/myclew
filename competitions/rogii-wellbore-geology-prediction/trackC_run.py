#!/usr/bin/env python
"""Track C entry — runs in the `kaggle_nlp` env (torch+kernels[HF FP8], cu128/sm_120).

Invoked by experiment.py via `conda run -n kaggle_nlp python trackC_run.py --params <json>`.
Calls geology_trackC.trackC_oof, writes results/trackC_oof.csv + results/trackC_test.csv,
prints a final line: `TRACKC_CV <cv> <precision>` for the parent to parse.
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
from fleet_agents import geology_trackC as C   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", required=True, help="JSON dict of params")
    args = ap.parse_args()
    params = json.loads(args.params)
    res = HERE / "results"; res.mkdir(exist_ok=True)
    cv, prec = C.trackC_oof(
        str(HERE / "input" / "train"), str(HERE / "input" / "test"),
        res / "trackC_oof.csv", res / "trackC_test.csv", params, log=print,
    )
    print(f"TRACKC_CV {cv:.6f} {prec}")


if __name__ == "__main__":
    main()
