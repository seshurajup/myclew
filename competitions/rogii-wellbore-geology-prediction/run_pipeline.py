#!/usr/bin/env python
"""Drive the fleet tabular pack for rogii-wellbore-geology-prediction.

comp-onboard (fingerprint) -> geology domain assemble -> tab-autobaseline (GBM, well-disjoint CV) -> submission.
Bypasses the heavy fleet_agents/__init__ (optional torch deps) via a stub package so only the tab modules load.
"""
import argparse
import importlib.util
import os
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FLEET = HERE / "fleet_agents"
RP = "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/tools/researchpapers"
sys.path.insert(0, RP)

# --- stub the package so submodules import without the mega-__init__ (avoids optional torch deps) ---
pkg = types.ModuleType("fleet_agents")
pkg.__path__ = [str(FLEET.resolve())]
sys.modules["fleet_agents"] = pkg

from fleet_agents import geology_pack as G          # noqa: E402
from fleet_agents import comp_config as CC          # noqa: E402
from fleet_agents import comp_onboard as ONB        # noqa: E402
from fleet_agents import tab_autobaseline as TA     # noqa: E402

AUTO = HERE / "config" / "_auto"
SUBS = HERE / "submissions"
AUTO.mkdir(parents=True, exist_ok=True)
SUBS.mkdir(parents=True, exist_ok=True)


def build_config():
    sample = pd.read_csv(HERE / "input" / "sample_submission.csv", nrows=1)
    cfg = ONB.infer_config(
        "rogii-wellbore-geology-prediction",
        files=["train.csv", "test.csv", "sample_submission.csv"],
        sample_header=list(sample.columns),
    )
    cfg.data = {
        "train": str(AUTO / "train.csv"),
        "test": str(AUTO / "test.csv"),
        "sample_sub": str(HERE / "input" / "sample_submission.csv"),
    }
    cfg.save(AUTO / "comp_config_rogii.json")
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="use only N wells (smoke test)")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    print(">> assembling flat tables (geology domain hook)...")
    G.geology_assemble(str(HERE / "input" / "train"), AUTO / "train.csv", training=True, limit=args.limit)
    G.geology_assemble(str(HERE / "input" / "test"), AUTO / "test.csv", training=False, limit=args.limit)
    tr = pd.read_csv(AUTO / "train.csv"); te = pd.read_csv(AUTO / "test.csv")
    print(f"   train rows={len(tr)} wells={tr.well.nunique()} | test rows={len(te)} wells={te.well.nunique()}")

    cfg = build_config()
    print(f">> onboard: modality={cfg.modality} task={cfg.task} metric={cfg.metric} "
          f"cv={cfg.cv_scheme} group={cfg.group_col} -> PACK '{cfg.pack()}'")
    assert cfg.pack() == "tab", f"expected tab pack, got {cfg.pack()}"

    # target 'tvt' is the residual drift (dtvt). const-continuation baseline = predict 0.
    base = np.sqrt((tr.tvt ** 2).mean())
    cand = np.sqrt(((tr.tvt - (tr.cand_tvt - tr.tvt_ps)) ** 2).mean())
    print(f">> baselines (residual target): const(dtvt=0) RMSE={base:.4f}  cand_tvt RMSE={cand:.4f}")

    print(">> tab-autobaseline (well-disjoint GroupKFold CV)...")
    res = TA.run_pipeline(cfg, out_path=str(SUBS / "submission.csv"), seed=42, fe=False,
                          n_folds=args.folds, gpu=(args.gpu or None), postprocess=True)
    print(">> per-backend CV RMSE:", {b: round(v, 4) for b, v in res["per_backend_cv"].items()})
    print(">> blend CV RMSE:", round(res["blend_cv"], 4), "| weights:", res["blend_weights"])

    # reconstruct ABSOLUTE tvt = predicted dtvt + tvt_ps (keyed by id), then validate
    sub = pd.read_csv(SUBS / "submission.csv")
    sub = sub.merge(te[["id", "tvt_ps"]], on="id", how="left")
    sub["tvt"] = sub["tvt"] + sub["tvt_ps"]
    sub = sub[["id", "tvt"]]
    sub.to_csv(SUBS / "submission.csv", index=False)
    samp = pd.read_csv(HERE / "input" / "sample_submission.csv")
    ok_cols = list(sub.columns) == list(samp.columns)
    ok_ids = set(sub.id) == set(samp.id) and len(sub) == len(samp)
    ok_nan = int(sub.tvt.isna().sum()) == 0
    print(f">> submission: rows={len(sub)} cols_ok={ok_cols} ids_ok={ok_ids} no_nan={ok_nan} -> {SUBS/'submission.csv'}")
    print(f">> VERDICT: blend CV {res['blend_cv']:.3f} vs const-baseline {base:.3f} "
          f"({'BEATS' if res['blend_cv'] < base else 'DOES NOT beat'} baseline)")


if __name__ == "__main__":
    main()
