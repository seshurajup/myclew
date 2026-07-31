#!/usr/bin/env python
"""baseline_v1 observable training launcher.

Wraps the official UNet+transformer detector trainer
(`research/official_repo/scripts/train_unet_transformer.py`) which already supports:
  - a fast pre-downsampled fp16 `.npy` cache via CELLMOT_CACHE_DIR (scripts/build_cache.py),
  - config-driven augmentations via CELLMOT_AUGMENT (JSON),
  - built-in MLflow logging via MLFLOW_* env.

This launcher OWNS the YAML->env/argv mapping (ported from the repo's proven
scripts/train_from_config.py) and adds the runtime-required observability:
a clear STARTUP banner before the training loop, and a tee'd per-run log under
output/baseline_v1/<id>/train.log.

DRY-RUN is GPU-SAFE by design (the shared GPU may be busy): it validates config
schema + all paths + the augment spec WITHOUT importing torch, then runs the
trainer's argparse/import chain with CUDA_VISIBLE_DEVICES="" (`--help`, zero GPU
memory). It never launches the training loop. The real 1-iter GPU check
(`--epochs 1 --max-iters 1 --single-gpu`) is printed for use once the GPU frees.

Usage:
  python src/baseline/train.py --config baseline/experiments_v1/v1_2_hr_1x2x2.yaml --dry-run --fold 0
  python src/baseline/train.py --config baseline/experiments_v1/v1_2_hr_1x2x2.yaml --fold 0
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Reference code + data live under the PARENT competition root; our artifacts live in this workdir.
PARENT_REPO = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
WORKDIR = Path(__file__).resolve().parents[2]  # tools/researchpapers
OUTPUT_ROOT = WORKDIR / "output" / "baseline_v1"

# Mirror of research/official_repo/scripts/augmentations.py REGISTRY (kept torch-free so
# the dry-run can validate aug specs without importing torch during a busy-GPU window).
KNOWN_AUGS = {
    "brightness", "contrast", "gamma", "noise", "bias_field",
    "blur", "flip", "rot90_xy", "scale", "cutout", "crop",
}

# YAML train: key -> trainer CLI flag (same mapping as scripts/train_from_config.py).
FLAG_MAP = {
    "method": "--method", "split": "--split", "downsample": "--downsample",
    "epochs": "--epochs", "lr": "--lr", "batch_size": "--batch-size",
    "num_workers": "--num-workers", "det_loss_weight": "--det-loss-weight",
    "det_neg_weight": "--det-neg-weight", "window_size": "--window-size",
    "pool_kernel_um": "--pool-kernel-um", "max_iters": "--max-iters",
    "unet_out_channels": "--unet-out-channels", "unet_layers": "--unet-layers",
}


def _resolve(p) -> str:
    """Resolve a config path against the parent competition repo (matches train_from_config)."""
    p = Path(p)
    return str(p if p.is_absolute() else (PARENT_REPO / p))


def load_config(cfg_path: Path) -> dict:
    import yaml  # cellmot_venv provides pyyaml
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def build_env(cfg: dict) -> dict:
    """YAML -> environment for the official trainer (ported from train_from_config.py)."""
    paths = cfg["paths"]
    ml = cfg.get("mlflow", {})
    tr = cfg.get("train", {})
    env = dict(os.environ)
    # Each config MUST start its OWN MLflow run: params (method/downsample/batch_size) are
    # immutable per run, so a pinned MLFLOW_RUN_ID inherited from the job/submission env makes
    # the 2nd config collide ("Changing param values is not allowed") and crash. Drop it; the
    # per-config MLFLOW_RUN_NAME below keys a fresh run. (Group runs via tags/params, not one run.)
    env.pop("MLFLOW_RUN_ID", None)
    env["PYTHONPATH"] = os.pathsep.join(_resolve(p) for p in paths.get("pythonpath", ["."]))
    if ml.get("tracking_uri"):
        env["MLFLOW_TRACKING_URI"] = ml["tracking_uri"]
    if ml.get("experiment"):
        env["MLFLOW_EXPERIMENT"] = ml["experiment"]
    env["MLFLOW_RUN_NAME"] = ml.get("run_name") or cfg.get("name", "run")
    # always log system metrics (GPU/CPU/mem utilization over time) for every run
    env["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
    env["MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL"] = ml.get("system_metrics_interval", "10")
    if cfg.get("purpose"):
        env["MLFLOW_PURPOSE"] = " ".join(str(cfg["purpose"]).split())
    env["MLFLOW_LOG_ARTIFACTS"] = "1" if ml.get("log_artifacts") else "0"
    if paths.get("cache_dir"):
        env["CELLMOT_CACHE_DIR"] = _resolve(paths["cache_dir"])
    if cfg.get("augment"):
        env["CELLMOT_AUGMENT"] = json.dumps(cfg["augment"])
    if tr.get("early_stop_patience") is not None:
        env["CELLMOT_EARLY_STOP_PATIENCE"] = str(tr["early_stop_patience"])
    for _key, _env in [("lr_schedule", "CELLMOT_LR_SCHEDULE"), ("lr_warmup", "CELLMOT_LR_WARMUP"),
                       ("ema_decay", "CELLMOT_EMA_DECAY"), ("weight_decay", "CELLMOT_WEIGHT_DECAY"),
                       ("amp", "CELLMOT_AMP")]:
        if tr.get(_key) is not None:
            env[_env] = str(tr[_key])
    # i-epoch early-discard rungs (successive halving). Accept "3:0.85,8:0.90" or [[3,0.85],...].
    pr = tr.get("prune_rungs")
    if pr:
        env["BIOHUB_PRUNE_RUNGS"] = pr if isinstance(pr, str) else ",".join(f"{int(e)}:{b}" for e, b in pr)
    return env


def build_cmd(cfg: dict, fold: int, extra=None) -> list:
    """YAML -> trainer argv. `fold` overrides train.split (golden-12 = split 0)."""
    paths = cfg["paths"]
    tr = dict(cfg.get("train", {}))
    tr["split"] = fold  # runtime --fold wins over config split
    # Route through the FD-safe entry shim: it sets torch sharing_strategy='file_system' BEFORE the
    # trainer builds DataLoaders, preventing 'Too many open files' FD exhaustion on long runs.
    entry = str(Path(__file__).resolve().parent / "_trainer_entry.py")
    cmd = [_resolve(paths["python"]), entry, _resolve(paths["trainer"]),
           "--data-dir", _resolve(paths["data_dir"]),
           "--splits", _resolve(paths["splits"]),
           "--single-gpu"]  # one RTX 5090 -> disable DataParallel
    for key, flag in FLAG_MAP.items():
        if tr.get(key) is not None:
            cmd += [flag, str(tr[key])]
    if extra:
        cmd += list(extra)
    return cmd


def validate(cfg: dict, fold: int) -> list:
    """Torch-free wiring validation. Returns a list of human-readable check lines; raises on hard errors."""
    lines = []
    paths = cfg["paths"]
    for key in ("python", "trainer", "data_dir", "splits"):
        rp = Path(_resolve(paths[key]))
        ok = rp.exists()
        lines.append(f"  [{'OK' if ok else 'MISSING'}] {key}: {rp}")
        if not ok:
            raise FileNotFoundError(f"config path '{key}' does not resolve: {rp}")
    # cache is optional (trainer falls back to strided zarr) but we want it present for speed
    cache = paths.get("cache_dir")
    if cache:
        cp = Path(_resolve(cache))
        n = len(list(cp.glob("*.npy"))) if cp.exists() else 0
        lines.append(f"  [{'OK' if n else 'EMPTY'}] cache_dir: {cp} ({n} npy files)")
    # splits must contain the requested fold
    splits = json.load(open(_resolve(paths["splits"])))
    nfolds = len(splits) if isinstance(splits, list) else 0
    if not (isinstance(splits, list) and 0 <= fold < nfolds):
        raise ValueError(f"fold {fold} out of range for splits with {nfolds} folds")
    f = splits[fold]
    lines.append(f"  [OK] splits fold {fold}: train={len(f.get('train', []))} test={len(f.get('test', []))}")
    # augment spec names must be known (mirrors trainer REGISTRY; torch-free)
    for item in cfg.get("augment", []) or []:
        name = item.get("name")
        if name not in KNOWN_AUGS:
            raise KeyError(f"unknown augmentation '{name}'; known: {sorted(KNOWN_AUGS)}")
    aug_names = [i["name"] for i in (cfg.get("augment") or [])] or ["DEFAULT (brightness)"]
    lines.append(f"  [OK] augment: {aug_names}")
    return lines


def _import_check(cfg: dict) -> tuple:
    """Run the trainer's import+argparse chain with NO GPU (CUDA_VISIBLE_DEVICES=''), via --help.
    Confirms torch/dataspec/augmentations import and the CLI parses, without touching GPU memory."""
    env = build_env(cfg)
    env["CUDA_VISIBLE_DEVICES"] = ""  # force CPU: zero GPU contention with any running job
    cmd = [_resolve(cfg["paths"]["python"]), _resolve(cfg["paths"]["trainer"]), "--help"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
    ok = r.returncode == 0 and "--downsample" in (r.stdout + r.stderr)
    return ok, (r.stdout + r.stderr)


def run(cfg_path: Path, fold: int, dry_run: bool) -> int:
    cfg = load_config(cfg_path)
    name = cfg.get("name", cfg_path.stem)
    tr = cfg.get("train", {})
    run_dir = OUTPUT_ROOT / name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78, flush=True)
    print(f" baseline_v1 TRAIN LAUNCH  |  {'DRY-RUN' if dry_run else 'REAL'}", flush=True)
    print(f"   experiment : {name}", flush=True)
    print(f"   config     : {cfg_path}", flush=True)
    print(f"   downsample : {tr.get('downsample')}   batch={tr.get('batch_size')}  "
          f"lr={tr.get('lr')}  epochs={tr.get('epochs')}  window={tr.get('window_size')}", flush=True)
    print(f"   cache_dir  : {_resolve(cfg['paths'].get('cache_dir', '(none -> strided zarr)'))}", flush=True)
    print(f"   splits/fold: {_resolve(cfg['paths']['splits'])}  fold={fold}", flush=True)
    print(f"   mlflow     : {cfg.get('mlflow', {}).get('tracking_uri')} / "
          f"{cfg.get('mlflow', {}).get('experiment')}", flush=True)
    print(f"   output log : {run_dir / 'train.log'}", flush=True)
    print("=" * 78, flush=True)

    print("[validate] wiring (torch-free):", flush=True)
    for line in validate(cfg, fold):
        print(line, flush=True)

    real_cmd = build_cmd(cfg, fold)
    if dry_run:
        print("\n[import-check] trainer argparse+imports with CUDA_VISIBLE_DEVICES='' (no GPU) ...", flush=True)
        ok, out = _import_check(cfg)
        print(f"[import-check] {'PASS' if ok else 'FAIL'} (trainer --help exit ok, --downsample present)", flush=True)
        if not ok:
            print(out[-2000:], flush=True)
            return 2
        print("\n[dry-run] resolved REAL command (NOT executed):", flush=True)
        print("  " + " ".join(real_cmd), flush=True)
        gpu_check = build_cmd(cfg, fold, extra=["--epochs", "1", "--max-iters", "1"])
        print("\n[dry-run] deferred 1-iter GPU smoke-check (run once GPU frees):", flush=True)
        print("  " + " ".join(gpu_check), flush=True)
        print("\n[dry-run] OK — wiring validated, no training launched, no GPU used.", flush=True)
        return 0

    # REAL run: stream trainer output to console + tee to the per-run log.
    env = build_env(cfg)
    # track WHICH config/yaml drove this run (logged to MLflow by the trainer)
    env["BIOHUB_CONFIG_FILE"] = str(cfg_path)
    env["BIOHUB_CONFIG_JSON"] = json.dumps(cfg)
    log_path = run_dir / "train.log"
    print(f"\n[train] launching real training -> {log_path}\n  {' '.join(real_cmd)}\n", flush=True)
    with open(log_path, "w") as logf:
        proc = subprocess.Popen(real_cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        proc.wait()
    print(f"\n[train] finished rc={proc.returncode}  log={log_path}", flush=True)
    return proc.returncode


def main() -> None:
    ap = argparse.ArgumentParser(description="baseline_v1 observable training launcher")
    ap.add_argument("--config", required=True, help="path to baseline/experiments_v1/<config>.yaml")
    ap.add_argument("--fold", type=int, default=0, help="split index (golden-12 = 0)")
    ap.add_argument("--dry-run", action="store_true", help="GPU-safe wiring validation; no training")
    args = ap.parse_args()
    sys.exit(run(Path(args.config).resolve(), args.fold, args.dry_run))


if __name__ == "__main__":
    main()
