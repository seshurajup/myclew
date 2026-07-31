"""Run canqiang ZebraCellTrace (LB 0.866, DeepCenterUNet3D) on the EMBRYO-DISJOINT
density CV (splits_loeo_density.json), score with the official metric, one fold at a time.

This is the CV-VALIDATION GATE variant of the parent's run_canqiang.py (which is HARD-CODED
to the 12 golden-12 datasets and cuda-only). canqiang uses a single fold-independent
best.pt applied per-dataset, so we simply run it on the fold's held-out (test) embryo set.

The pipeline is: canqiang full-frame detect+track (its OWN post-proc, NOT pilkwang's) ->
official_counts -> official_score (adj_edge_jaccard + 0.1*div_jaccard). Same official metric
as the pilkwang path, so the two pipelines are directly comparable for the gate.

Usage:
  # dry-run (CPU-safe; loads model on CPU, resolves datasets, exits before the forward pass)
  <venv>/python baseline/run_canqiang_loeodens.py --fold 0 --dry-run
  # real predict+score (GPU)
  <venv>/python baseline/run_canqiang_loeodens.py --fold 0
  <venv>/python baseline/run_canqiang_loeodens.py --fold 1

Startup log prints the fold, split file, dataset count, checkpoint and output dir BEFORE
any heavy work so humans/agents can confirm the job truly started.
"""
import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
WORKDIR = Path(__file__).resolve().parents[1]  # tools/researchpapers
sys.path.insert(0, str(ROOT))
from src import golden_cv as gcv          # noqa: E402
from src.config import Config as SrcCfg   # noqa: E402
from src import metric, io                # noqa: E402

SRC = SrcCfg()
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
OUT = ROOT / "learning/ensemble_work"
CKPT = ROOT / "learning/public_pull/data/canqiang_ckpt/best.pt"
CQ = ROOT / "learning/public_pull/canqiang_zebracelltrace-full-frame-submit/zebracelltrace-full-frame-submit.py"
DEFAULT_SPLIT = ROOT / "learning/ensemble_work/finetune/splits_loeo_density.json"


def load_cq():
    """Import canqiang's submission script as a module (defines DeepCenterUNet3D etc.)."""
    spec = importlib.util.spec_from_file_location("cq", CQ)
    cq = importlib.util.module_from_spec(spec)
    sys.modules["cq"] = cq
    spec.loader.exec_module(cq)
    return cq


def resolve_test_datasets(split_file: Path, fold: int) -> list[str]:
    folds = json.loads(Path(split_file).read_text())
    return [s.replace(".zarr", "") for s in folds[fold]["test"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="canqiang on the embryo-disjoint density CV")
    ap.add_argument("--fold", type=int, required=True, help="density-CV fold (0=test 44b6, 1=test 6bba)")
    ap.add_argument("--split-file", default=str(DEFAULT_SPLIT))
    ap.add_argument("--peak-threshold", type=float, default=0.20)
    ap.add_argument("--min-peak-distance", type=int, default=1)
    ap.add_argument("--dry-run", action="store_true",
                    help="validate wiring on CPU (model load + dataset resolution) then exit "
                         "BEFORE the forward pass; no GPU, no real inference")
    ap.add_argument("--device", default=None, help="override device (default: cuda if available)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--exp-id", default=None, help="experiment-journal id, logged as MLflow tag exp_id")
    ap.add_argument("--score-json", default=None)
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    split_file = Path(args.split_file)
    test_ds = resolve_test_datasets(split_file, args.fold)
    eval_split = split_file.stem
    run_name = args.run_name or f"canqiang_loeodens_f{args.fold}"
    out_nodes = OUT / "canqiang_loeodens_nodes"
    out_tracks = OUT / "canqiang_loeodens_tracks"

    # ---- startup log (before any heavy work) ----
    print("=" * 66, flush=True)
    print(f" canqiang density-CV  |  fold={args.fold}  split={split_file.name}", flush=True)
    print(f" test datasets (n={len(test_ds)}): {test_ds}", flush=True)
    print(f" ckpt={CKPT}", flush=True)
    print(f" out_nodes={out_nodes}", flush=True)
    print(f" dry_run={args.dry_run}", flush=True)
    print("=" * 66, flush=True)

    # ---- resolve device ----
    if args.device:
        device = torch.device(args.device)
    elif args.dry_run:
        device = torch.device("cpu")  # dry-run never touches the GPU
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] device={device}  torch.cuda.is_available={torch.cuda.is_available()}", flush=True)

    # ---- validate inputs exist (both dry-run and real) ----
    missing = []
    for ds in test_ds:
        g = TRAIN / f"{ds}.geff"
        z = TRAIN / f"{ds}.zarr"
        if not g.exists():
            missing.append(str(g))
        if not z.exists():
            missing.append(str(z))
    if not CKPT.exists():
        missing.append(str(CKPT))
    if not CQ.exists():
        missing.append(str(CQ))
    if missing:
        print("[FAIL] missing required inputs:", flush=True)
        for m in missing:
            print(f"   - {m}", flush=True)
        sys.exit(2)
    print(f"[ok] all {len(test_ds)} datasets (.geff+.zarr) + ckpt + cq script resolve", flush=True)

    # ---- load model (CPU in dry-run) ----
    cq = load_cq()
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = cq.FullFrameTrainingConfig(**ckpt["config"])
    model = cq.DeepCenterUNet3D(base_channels=cfg.base_channels)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    baseline_cfg = cq.public_reference_baseline_config()
    print(f"[ok] model reconstructed on {device}  epoch={ckpt.get('epoch')} "
          f"best={ckpt.get('best_score')}", flush=True)

    if args.dry_run:
        print("[DRY-RUN GREEN] wiring validated (split/datasets/ckpt/model). "
              "Exiting before forward pass — no GPU inference performed.", flush=True)
        return

    # ---- real predict + score ----
    out_nodes.mkdir(parents=True, exist_ok=True)
    out_tracks.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds in test_ds:
        t0 = time.time()
        srows, stats = cq.process_dataset_full_frame(
            TRAIN, ds, model, cfg, baseline_cfg, device,
            peak_threshold=args.peak_threshold, min_peak_distance=args.min_peak_distance)
        sub = pd.DataFrame(srows)
        nd = sub[sub.row_type == "node"][["node_id", "t", "z", "y", "x"]].reset_index(drop=True)
        ed = sub[sub.row_type == "edge"][["source_id", "target_id"]].reset_index(drop=True)
        nd.to_csv(out_nodes / f"{ds}.csv", index=False)
        ed.to_csv(out_tracks / f"{ds}_edges.csv", index=False)
        gn, ge = io.read_geff(TRAIN / f"{ds}.geff")
        tt = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
        r = metric.official_counts(gn, ge, nd, ed, SRC.SCALE, SRC.MATCH_GATE_UM, t_true=tt)
        r["dataset"] = ds
        r["embryo"] = ds.split("_")[0]
        r["node_recall"] = float("nan")  # canqiang path does not compute the node-match proxy here
        r["count_ratio"] = r["t_pred"] / max(r["t_true"], 1.0)
        rows.append(r)
        print(f"  {ds:16s} adjJ={r['adj_jaccard']:.3f}  nodes={r['t_pred']} "
              f"estN={r['t_true']:.0f} (x{r['count_ratio']:.2f})  div_tp={r['div_tp']}  "
              f"{time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    off = metric.official_score(df.to_dict("records"))
    micro = float((df.w * df.adj_jaccard).sum() / df.w.sum())
    per = df.groupby("embryo").apply(
        lambda x: (x.w * x.adj_jaccard).sum() / x.w.sum(), include_groups=False)
    official = float(off["score"])
    print(f"\nMINI_OFFICIAL_SCORE={official:.6f}", flush=True)
    print("=" * 66, flush=True)
    print(f" canqiang density-CV fold{args.fold}  official_score(adj_edge+0.1div) = {official:.4f}", flush=True)
    print(f"   micro adjJ = {micro:.4f}   per-embryo {dict(per.round(4))}", flush=True)
    print("=" * 66, flush=True)

    # ---- sidecar (same schema as baseline/score_v1.py -> journal auto-fill) ----
    sidecar = {
        "run_name": run_name, "exp_id": args.exp_id, "method": "canqiang",
        "fidelity": "mini", "eval_split": eval_split, "fold": args.fold, "n_embryos": int(len(df)),
        "official_score": official, "micro_adjJ": micro,
        "mean_count_ratio": float(df.count_ratio.mean()), "div_tp_total": int(df.div_tp.sum()),
        "adjJ_44b6": float(per.get("44b6", float("nan"))),
        "adjJ_6bba": float(per.get("6bba", float("nan"))),
        "pipeline": "canqiang_full_frame", "status": "DONE",
    }
    sc_path = Path(args.score_json) if args.score_json else (WORKDIR / "output" / "scores" / f"{run_name}.json")
    sc_path.parent.mkdir(parents=True, exist_ok=True)
    sc_path.write_text(json.dumps(sidecar, indent=2))
    df.to_csv(OUT / f"canqiang_loeodens_f{args.fold}_scores.csv", index=False)
    print(f"[sidecar] wrote {sc_path}", flush=True)

    # ---- MLflow (config_file + ALWAYS-ON system metrics, per recipe) ----
    if not args.no_mlflow:
        os.environ.pop("MLFLOW_RUN_ID", None)
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        os.environ.setdefault("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "10")
        try:
            import mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("kaggle-biohub-cell-tracking")
            with mlflow.start_run(run_name=run_name, log_system_metrics=True):
                mlflow.set_tags({"exp_id": args.exp_id or "", "fidelity": "mini",
                                 "eval_split": eval_split, "pipeline": "canqiang_full_frame",
                                 "config_path": str(CKPT)})
                mlflow.log_param("config_file", "canqiang_best.pt")
                mlflow.log_param("fold", args.fold)
                mlflow.log_param("peak_threshold", args.peak_threshold)
                mlflow.log_metric("official_score", official)
                mlflow.log_metric("micro_adjJ", micro)
                mlflow.log_metric("div_tp_total", int(df.div_tp.sum()))
                mlflow.log_artifact(str(sc_path))
            print("[mlflow] logged canqiang density-CV run", flush=True)
        except Exception as e:  # never let MLflow failure lose the score
            print(f"[mlflow] skipped ({e})", flush=True)


if __name__ == "__main__":
    main()
