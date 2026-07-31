#!/usr/bin/env python
"""Successive-halving / early-discard driver — gated on MINI-OFFICIAL, not acc*recall.

CRITICAL METHODOLOGY FIX (2026-07-05): the trainer's per-epoch best_score = acc*recall is a PROVEN
poor surrogate for the competition official metric — across baseline_v1 it was nearly FLAT
(0.944-0.956) while the golden-12 official adjJ spanned 0.61-0.82. Ranking/pruning a bracket on it is
BLIND (prunes on noise). So this driver ranks each rung on the OFFICIAL metric computed on a
golden-12-DISTRIBUTION-MATCHED, leak-free mini-VAL set (`splits_screen_matched.json`): train the rung
-> predict on the rung's val embryos -> pilk_post -> src.metric official -> rank on that mini-official.
Because the mini-val mirrors golden-12 (group + density, see docs/screen_miniset_v2.md), mini-official
tracks golden-12 cheaply (12 embryos). golden-12 stays the FINAL judge on the survivor.

Multi-fidelity rules (2024-2026 lit): mini->full DATA fidelity + few->many EPOCHS per rung, keep top
half, CONSERVATIVE bar = worst survivor's mini-official (never kill a winner on a noisy dip).

Usage:
  python baseline/successive_halving.py --bracket baseline/brackets/screen_v3.yml --fold 0 --dry-run
  python baseline/successive_halving.py --bracket baseline/brackets/screen_v3.yml --fold 0   # needs GPU
"""
import argparse
import math
import re
import subprocess
import time
from pathlib import Path

import yaml

WORKDIR = Path(__file__).resolve().parents[1]           # tools/researchpapers
PARENT_REPO = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
PY = str(PARENT_REPO / "research" / "cellmot_venv" / "bin" / "python")
TRAIN = WORKDIR / "src" / "baseline" / "train.py"
SCORE = WORKDIR / "baseline" / "score_v1.py"
PREDICT = PARENT_REPO / "research/official_repo/scripts/predict_unet_transformer.py"
WEIGHTS = PARENT_REPO / "research/official_repo/weights"
OFF_SRC = PARENT_REPO / "research/official_repo"
DATA_DIR = PARENT_REPO / "input/biohub-cell-tracking-during-development/train"
OUTPUT_ROOT = WORKDIR / "output" / "baseline_v1"


def _abs(p):
    p = Path(p)
    return p if p.is_absolute() else (PARENT_REPO / p)


def mini_official(method: str, split: str, fold: int) -> float | None:
    """Predict the rung checkpoint on the (mini or golden-12) VAL embryos, then score the FULL
    official metric on them. Returns the golden-12-faithful mini-official adjJ (None on failure)."""
    ckpt = WEIGHTS / method / f"split_{fold}" / "edge_predictor_best.pth"
    if not ckpt.exists():
        print(f"    [mini-official] no checkpoint {ckpt} — cannot score", flush=True)
        return None
    # 1) predict on the split's val embryos (fast: matched mini-val is ~12)
    env = {"PYTHONPATH": f"{OFF_SRC}/src:{OFF_SRC}/scripts:."}
    pred = subprocess.run(
        [PY, str(PREDICT), "--data-dir", str(DATA_DIR), "--splits", str(_abs(split)),
         "--split", str(fold), "--weights", str(ckpt), "--method", method,
         "--det-threshold", "0.99", "--use-ilp"],
        cwd=str(OFF_SRC), env={**__import__("os").environ, **env}, capture_output=True, text=True)
    if pred.returncode != 0:
        print(f"    [mini-official] predict failed: {pred.stdout[-400:]}{pred.stderr[-400:]}", flush=True)
        return None
    # 2) official score on those val embryos (--split-file targets the mini-val, not golden-12)
    sc = subprocess.run(
        [PY, str(SCORE), "--method", method, "--split-file", str(split), "--fold", str(fold),
         "--run-name", f"{method}_minival", "--no-mlflow"],
        cwd=str(WORKDIR), capture_output=True, text=True)
    m = re.findall(r"MINI_OFFICIAL_SCORE=([0-9.]+)", sc.stdout)
    if not m:
        print(f"    [mini-official] score failed: {sc.stdout[-400:]}{sc.stderr[-400:]}", flush=True)
        return None
    return float(m[-1])


def run_config(base_cfg: Path, name: str, epochs: int, fold: int, dry: bool, splits: str):
    """Materialise a per-round config (data + epochs, namespaced checkpoint, NO acc*recall in-run
    prune) and train it; then score MINI-OFFICIAL on the round's val embryos."""
    cfg = yaml.safe_load(open(base_cfg if base_cfg.is_absolute() else (WORKDIR / base_cfg)))
    cfg.setdefault("train", {})["epochs"] = epochs
    cfg["train"]["method"] = name          # namespace checkpoint per rung (weights/<name>/)
    cfg["train"].pop("prune_rungs", None)   # DROP acc*recall in-run prune (unfaithful); gate on mini-official
    if splits:                              # per-round DATA fidelity: mini leak-free -> full golden-12
        cfg.setdefault("paths", {})["splits"] = splits
    cfg["name"] = name
    cfg.setdefault("mlflow", {})["run_name"] = name
    out = OUTPUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "config.yml"
    yaml.safe_dump(cfg, open(tmp, "w"))
    cmd = [PY, str(TRAIN), "--config", str(tmp), "--fold", str(fold)] + (["--dry-run"] if dry else [])
    print(f"    $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(WORKDIR))
    if dry:
        return None
    return mini_official(name, splits, fold)  # rank on golden-12-faithful mini-official


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bracket", required=True, help="YAML: configs[] + rungs[{epochs,keep,splits}]")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--budget-hours", type=float, default=0.0, help="0 = no cap")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(open(args.bracket))
    survivors = [Path(c) for c in spec["configs"]]
    rungs = spec["rungs"]
    t_start = time.monotonic()

    print("=" * 78)
    print(f" SUCCESSIVE HALVING (gate=MINI-OFFICIAL)  |  {'DRY-RUN' if args.dry_run else 'REAL'}  |  fold={args.fold}")
    print(f"   configs : {[c.stem for c in survivors]}")
    print(f"   rungs   : {rungs}")
    print(f"   budget  : {args.budget_hours or 'uncapped'} h")
    print("=" * 78)

    bar = None
    for ri, rung in enumerate(rungs):
        epochs, keep = int(rung["epochs"]), float(rung["keep"])
        splits = rung.get("splits")
        data_tag = Path(splits).stem if splits else "config-default"
        elapsed = (time.monotonic() - t_start) / 3600
        if args.budget_hours and elapsed >= args.budget_hours:
            print(f"[budget] {elapsed:.2f}h >= {args.budget_hours}h — stopping before round {ri}.")
            break
        print(f"\n### ROUND {ri}: {len(survivors)} config(s) @ {epochs} epochs on '{data_tag}' "
              f"(conservative bar={bar if bar is None else round(bar,4)}, keep top {keep})")
        scored = []
        for cfg in survivors:
            name = f"sh_r{ri}_e{epochs}_{cfg.stem}"
            sc = run_config(cfg, name, epochs, args.fold, args.dry_run, splits=splits)
            scored.append((sc, cfg))
            if sc is not None:
                print(f"    -> {cfg.stem}: mini_official={sc:.4f}", flush=True)

        if args.dry_run:
            n_keep = max(1, math.ceil(keep * len(survivors)))
            print(f"[dry] round {ri}: would train {len(survivors)} @ {epochs}ep, mini-official-score, keep top {n_keep}.")
            continue

        scored = [(s, c) for s, c in scored if s is not None]
        if not scored:
            print(f"[warn] round {ri}: no configs produced a mini-official score; stopping.")
            break
        scored.sort(key=lambda x: -x[0])
        n_keep = max(1, math.ceil(keep * len(scored)))
        # CONSERVATIVE rule: keep top-half, but never drop a config at/above the previous worst survivor.
        survivors = [c for s, c in scored[:n_keep]]
        if bar is not None:
            survivors += [c for s, c in scored[n_keep:] if s >= bar and c not in survivors]
        bar = scored[n_keep - 1][0]  # worst survivor's mini-official = next round's conservative floor
        print(f"  round {ri} result (keep {len(survivors)}, next bar={bar:.4f}):")
        for s, c in scored:
            print(f"    {s:.4f}  {c.stem:34s} {'KEEP' if c in survivors else 'drop'}")

    print("\n" + "=" * 78)
    print(f" SURVIVOR(S): {[c.stem for c in survivors]}")
    print(f" wall-clock: {(time.monotonic()-t_start)/3600:.2f}h")
    print(" NOTE: survivor already scored on golden-12-faithful mini-official; confirm the FULL "
          "golden-12 official (splits_ft.json) before trusting/submitting.")
    print("=" * 78)


if __name__ == "__main__":
    main()
