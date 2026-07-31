#!/usr/bin/env python
"""GPU-FREE dry-run for the v13 split_1 training package.

train_unet_transformer.py has no --dry-run and needs torch/GPU, so this validates everything
train_from_config.py would set up — WITHOUT importing torch or touching the GPU — then prints the
EXACT trainer + predict commands that WOULD run. Mirrors the repo's train.py --dry-run philosophy.

Checks: YAML parses + required keys; python/trainer/data_dir/splits/pythonpath/cache_dir paths resolve;
fold1 = train 44b6 (71) / test 6bba (128); every fold1 TRAIN .zarr+.geff present; augment block valid;
max_iters OMITTED (uncapped); output weights dir derivable. Exit 0 = GREEN.

Usage: research/cellmot_venv/bin/python tools/researchpapers/baseline/dryrun_split1_train.py \
           tools/researchpapers/baseline/experiments_v13/v13_split1_train.yml
"""
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


def main():
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        ROOT / "tools/researchpapers/baseline/experiments_v13/v13_split1_train.yml"
    errs, warns = [], []
    print(f"[DRY-RUN] validating split_1 training package: {cfg_path}")
    cfg = yaml.safe_load(open(cfg_path))
    paths, tr = cfg["paths"], cfg.get("train", {})

    # 1. path existence
    for key in ("python", "trainer", "data_dir", "splits"):
        rp = resolve(paths[key])
        if not rp.exists():
            errs.append(f"paths.{key} missing: {rp}")
    for pp in paths.get("pythonpath", []):
        if not resolve(pp).exists():
            warns.append(f"pythonpath entry missing: {resolve(pp)}")
    if paths.get("cache_dir") and not resolve(paths["cache_dir"]).exists():
        warns.append(f"cache_dir missing (trainer will build it): {resolve(paths['cache_dir'])}")

    # 2. split geometry: fold1 = train 44b6 / test 6bba
    split = int(tr.get("split", -1))
    if split != 1:
        errs.append(f"train.split must be 1 for split_1 (got {split})")
    folds = json.loads(resolve(paths["splits"]).read_text())
    fold = folds[1]
    tr_emb = Counter(d.split("_")[0] for d in fold["test"])   # note: fold['test'] is the HELD-OUT embryo
    train_emb = Counter(d.split("_")[0] for d in fold["train"])
    print(f"  fold1: train={dict(train_emb)} test={dict(tr_emb)}")
    if set(train_emb) != {"44b6"}:
        errs.append(f"fold1 train must be 44b6-only, got {dict(train_emb)}")
    if set(tr_emb) != {"6bba"}:
        errs.append(f"fold1 test must be 6bba-only, got {dict(tr_emb)}")

    # 3. every fold1 TRAIN dataset has .zarr + .geff
    data_dir = resolve(paths["data_dir"])
    missing = [ds for ds in fold["train"]
               if not (data_dir / f"{ds}.zarr").exists() or not (data_dir / f"{ds}.geff").exists()]
    if missing:
        errs.append(f"{len(missing)} fold1-train datasets missing .zarr/.geff, e.g. {missing[:3]}")
    else:
        print(f"  data OK: {len(fold['train'])} fold1-train (44b6) .zarr+.geff present")

    # 4. augment block valid + max_iters omitted (uncapped)
    aug = cfg.get("augment")
    if aug is None or not isinstance(aug, list) or not all("name" in a for a in aug):
        errs.append("augment block must be a non-empty list of {name,...} (regularizes cross-embryo)")
    else:
        print(f"  augment OK: {[a['name'] for a in aug]}")
    if tr.get("max_iters") is not None:
        errs.append(f"max_iters={tr['max_iters']} set — MUST be OMITTED (uncapped) to avoid the loeo_129ep overfit trap")
    else:
        print("  max_iters OMITTED -> uncapped full epochs (non-overfit recipe) OK")

    # 5. reconstruct the exact trainer CLI (what start_train.sh would launch)
    flag_map = {"method": "--method", "split": "--split", "downsample": "--downsample",
                "epochs": "--epochs", "lr": "--lr", "batch_size": "--batch-size",
                "num_workers": "--num-workers", "det_loss_weight": "--det-loss-weight",
                "det_neg_weight": "--det-neg-weight", "window_size": "--window-size",
                "pool_kernel_um": "--pool-kernel-um", "seed": "--seed"}
    cmd = [str(resolve(paths["python"])), str(resolve(paths["trainer"])),
           "--data-dir", str(resolve(paths["data_dir"])), "--splits", str(resolve(paths["splits"]))]
    for k, flag in flag_map.items():
        if tr.get(k) is not None:
            cmd += [flag, str(tr[k])]
    weights_out = f"research/official_repo/weights/{tr.get('method')}/split_{split}/edge_predictor_best.pth"

    print("\n  TRAIN cmd (GPU) that WOULD run:\n    " + " ".join(cmd))
    print(f"  -> weights: {weights_out}")
    print("  GPU quick-check (once GPU free): append  --epochs 1 --max-iters 1 --single-gpu")
    print(f"  PREDICT (after train): research/cellmot_venv/bin/python research/official_repo/scripts/"
          f"predict_unet_transformer.py --method {tr.get('method')} --split 1 "
          f"--splits {paths['splits']} --data-dir {paths['data_dir']} --weights {weights_out}")

    if warns:
        print("\n  WARNINGS:")
        for w in warns:
            print(f"    ! {w}")
    if errs:
        print("\n[DRY-RUN FAILED]")
        for e in errs:
            print(f"    ✗ {e}")
        sys.exit(1)
    print("\n[DRY-RUN GREEN] split_1 training package validated (GPU-free). Parked pending greenlight.")


if __name__ == "__main__":
    main()
