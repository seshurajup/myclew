"""Minimal CPU dry-run for the CV-VALIDATION GATE on the embryo-disjoint density CV.

Validates that BOTH gate pipelines are correctly wired WITHOUT running any GPU inference:
  - splits_loeo_density.json loads and both folds resolve to real .zarr + .geff files
  - pilkwang: weights + sibling config.json resolve, model reconstructs on CPU, the config
    downsample is reported (this is where the (1,4,4) genuine-pilkwang vs (1,2,2) local-retrain
    distinction shows up), and the namespaced output dir is writable
  - canqiang: delegated to run_canqiang_loeodens.py --dry-run (loads DeepCenterUNet3D on CPU,
    resolves the fold's datasets, exits before the forward pass)

This proves config/path/data/dependency resolution only. The REAL predict is the trainer's
GPU lane. Exit code 0 = GREEN.

Usage:
  <venv>/python baseline/dryrun_gate_loeodens.py                 # both folds, default (1,4,4) pilkwang weights
  <venv>/python baseline/dryrun_gate_loeodens.py --folds 0,1
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
WORKDIR = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
SPLIT = ROOT / "learning/ensemble_work/finetune/splits_loeo_density.json"
PREDICT_SCRIPTS = ROOT / "research/official_repo/scripts"
PREDICTIONS = ROOT / "research/official_repo/predictions"
# Genuine pilkwang LB-0.890 detector = (1,4,4) resolution (pristine support pack).
# NOT research/official_repo/weights/unet_transformer/split_0 (that is a local (1,2,2) retrain).
PILK_WEIGHTS_DEFAULT = ROOT / "research/pilkwang_support_pack/weights/unet_transformer/split_0/edge_predictor_best.pth"
PILK_METHOD = "pilk_loeodens"  # namespaced so we never clobber existing predictions/seshu/unet_transformer/split_0


def check(label, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{(' — ' + detail) if detail else ''}", flush=True)
    return ok


def resolve_test(fold):
    folds = json.loads(SPLIT.read_text())
    return [s.replace(".zarr", "") for s in folds[fold]["test"]]


def datasets_present(test_ds):
    miss = []
    for ds in test_ds:
        if not (TRAIN / f"{ds}.geff").exists():
            miss.append(f"{ds}.geff")
        if not (TRAIN / f"{ds}.zarr").exists():
            miss.append(f"{ds}.zarr")
    return miss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0,1")
    ap.add_argument("--pilk-weights", default=str(PILK_WEIGHTS_DEFAULT))
    args = ap.parse_args()
    folds = [int(x) for x in args.folds.split(",") if x != ""]

    # dataspec resolves GT/images via CELLMOT_DATA_DIR -> point it at the real train dir on CPU
    os.environ["CELLMOT_DATA_DIR"] = str(TRAIN)
    os.environ.setdefault("USER", "seshu")
    all_ok = True

    print("=" * 70, flush=True)
    print(" CV-VALIDATION GATE — CPU DRY-RUN (no GPU, no real predict)", flush=True)
    print("=" * 70, flush=True)

    # --- common: split + datasets ---
    all_ok &= check("split file exists", SPLIT.exists(), str(SPLIT))
    for f in folds:
        test = resolve_test(f)
        emb = sorted({d.split("_")[0] for d in test})
        all_ok &= check(f"fold{f} test resolves ({len(test)} ds, embryos={emb})", True)
        miss = datasets_present(test)
        all_ok &= check(f"fold{f} all .zarr+.geff present", not miss,
                        "" if not miss else f"MISSING {miss[:4]}...")

    # --- pilkwang: weights + config + CPU model reconstruct ---
    print("\n-- pilkwang (unet_transformer) --", flush=True)
    w = Path(args.pilk_weights)
    cfgp = w.parent / "config.json"
    all_ok &= check("weights exist", w.exists(), str(w))
    all_ok &= check("sibling config.json exists", cfgp.exists(), str(cfgp))
    if cfgp.exists():
        ds_factor = json.loads(cfgp.read_text()).get("downsample")
        genuine = ds_factor == [1, 4, 4]
        check(f"config downsample = {ds_factor}", genuine,
              "GENUINE pilkwang (1,4,4)" if genuine else "WARNING: not (1,4,4) — is this really pilkwang's public model?")
    # pilkwang predict runs with cwd=research/official_repo + PYTHONPATH="src:scripts"
    # (tracking_cellmot lives in research/official_repo/src) — mirror that here.
    sys.path.insert(0, str(PREDICT_SCRIPTS))
    sys.path.insert(0, str(ROOT / "research/official_repo/src"))
    try:
        import predict_unet_transformer as put  # noqa: E402
        model, window, downs = put.load_model(w, torch.device("cpu"))
        n_params = sum(p.numel() for p in model.parameters())
        all_ok &= check("model reconstructs on CPU", True,
                        f"window={window} downsample={downs} params={n_params/1e6:.2f}M")
    except Exception as e:
        all_ok &= check("model reconstructs on CPU", False, repr(e))
    out_dir = PREDICTIONS / "seshu" / PILK_METHOD
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        all_ok &= check("namespaced output dir writable", True, str(out_dir))
    except Exception as e:
        all_ok &= check("namespaced output dir writable", False, repr(e))

    # --- canqiang: delegate to its --dry-run (CPU model load) ---
    print("\n-- canqiang (DeepCenterUNet3D) --", flush=True)
    for f in folds:
        cmd = [sys.executable, str(WORKDIR / "baseline" / "run_canqiang_loeodens.py"),
               "--fold", str(f), "--dry-run"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        green = "DRY-RUN GREEN" in r.stdout
        all_ok &= check(f"canqiang fold{f} dry-run", green,
                        "" if green else (r.stdout[-300:] + r.stderr[-300:]))

    print("\n" + "=" * 70, flush=True)
    print(f" DRY-RUN {'GREEN — wiring validated for both pipelines' if all_ok else 'RED — see FAIL lines above'}", flush=True)
    print("=" * 70, flush=True)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
