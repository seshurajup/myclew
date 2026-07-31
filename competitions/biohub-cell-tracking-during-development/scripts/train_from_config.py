#!/usr/bin/env python
"""Config-driven training launcher.

Reads a YAML experiment config (paths + mlflow + hyperparameters), sets up the
environment, and runs the detector trainer as a subprocess so its MLflow logging
(which reads MLFLOW_TRACKING_URI) is inherited. Everything — including WHERE the
python code and data live — comes from the YAML, so runs are fully reproducible.

Usage:  python scripts/train_from_config.py config/exp1.yml
"""
import sys, os, subprocess
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve(p):
    p = Path(p)
    return str(p if p.is_absolute() else (ROOT / p))


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: train_from_config.py <config.yml> [extra trainer flags]")
    cfg = yaml.safe_load(open(sys.argv[1]))
    extra = sys.argv[2:]
    paths = cfg["paths"]
    tr = cfg.get("train", {})
    ml = cfg.get("mlflow", {})

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(resolve(p) for p in paths.get("pythonpath", ["."]))
    if ml.get("tracking_uri"):
        env["MLFLOW_TRACKING_URI"] = ml["tracking_uri"]
    if ml.get("experiment"):
        env["MLFLOW_EXPERIMENT"] = ml["experiment"]
    env["MLFLOW_RUN_NAME"] = ml.get("run_name") or cfg.get("name", "run")
    if cfg.get("purpose"):
        env["MLFLOW_PURPOSE"] = " ".join(str(cfg["purpose"]).split())
    env["MLFLOW_LOG_ARTIFACTS"] = "1" if ml.get("log_artifacts") else "0"
    if paths.get("cache_dir"):
        env["CELLMOT_CACHE_DIR"] = resolve(paths["cache_dir"])
    if cfg.get("augment") is not None:           # YAML augment: block -> trainer builds the pipeline.
        import json as _json                     # `is not None` (not truthiness) so an EMPTY list [] is honored as
        env["CELLMOT_AUGMENT"] = _json.dumps(cfg["augment"])  # TRUE no-aug, not silently defaulted to brightness.
    if tr.get("early_stop_patience") is not None:  # config-driven early stopping
        env["CELLMOT_EARLY_STOP_PATIENCE"] = str(tr["early_stop_patience"])
    # config-driven training upgrades (LR schedule / EMA / weight decay / AMP)
    for _key, _env in [("lr_schedule", "CELLMOT_LR_SCHEDULE"), ("lr_warmup", "CELLMOT_LR_WARMUP"),
                       ("ema_decay", "CELLMOT_EMA_DECAY"), ("weight_decay", "CELLMOT_WEIGHT_DECAY"),
                       ("amp", "CELLMOT_AMP"),
                       ("det_fg_ignore", "CELLMOT_DET_FG_IGNORE"),  # sparse-annot masked det-loss (percentile)
                       ("edge_temporal", "CELLMOT_EDGE_TEMPORAL"),  # Plan-B opt-in windowed temporal linker
                       ("edge_temporal_blocks", "CELLMOT_EDGE_TEMPORAL_BLOCKS")]:
        if tr.get(_key) is not None:
            env[_env] = str(tr[_key])

    plimit = (cfg.get("gpu") or {}).get("power_limit_w")
    if plimit:
        subprocess.run(["sudo", "nvidia-smi", "-pl", str(plimit)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    flag_map = {  # yaml key -> CLI flag
        "method": "--method", "split": "--split", "downsample": "--downsample",
        "epochs": "--epochs", "lr": "--lr", "batch_size": "--batch-size",
        "num_workers": "--num-workers", "det_loss_weight": "--det-loss-weight",
        "det_neg_weight": "--det-neg-weight", "window_size": "--window-size",
        "pool_kernel_um": "--pool-kernel-um", "max_iters": "--max-iters",
        "unet_out_channels": "--unet-out-channels", "unet_layers": "--unet-layers",
        "seed": "--seed",
    }
    cmd = [resolve(paths["python"]), resolve(paths["trainer"]),
           "--data-dir", resolve(paths["data_dir"]), "--splits", resolve(paths["splits"])]
    for k, flag in flag_map.items():
        if k in tr and tr[k] is not None:
            cmd += [flag, str(tr[k])]
    cmd += extra

    print("=" * 70)
    print(f" EXPERIMENT: {cfg.get('name','?')}")
    print(f" trainer   : {resolve(paths['trainer'])}")
    print(f" data      : {resolve(paths['data_dir'])}")
    print(f" mlflow    : {ml.get('tracking_uri')} / {ml.get('experiment')}")
    print(f" params    : {tr}")
    print("=" * 70, flush=True)
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
