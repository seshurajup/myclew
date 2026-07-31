#!/usr/bin/env python
"""baseline_v1 golden-12 OFFICIAL scorer (researcher-owned measurement).

The trainer only reports the detector's acc*recall. The competition OFFICIAL metric
(adj edge-Jaccard + 0.1*div-Jaccard, 7um node match) needs the FULL pipeline:
  predicted geff -> pilk_post.filter_output_graph (motion-relink/gap/safe-div/smooth)
                 -> src.metric.official_counts/official_score over the golden-12 fold.

This scores a directory of predicted .geff files (produced by predict_unet_transformer.py
with a trained checkpoint) and reports, per IMPROVE_PLAYBOOK Rule 1:
  * golden-12 OFFICIAL score + micro adjJ + golden_cv()  (CV-judgeable part)
  * detector node-RECALL proxy + predicted-count vs true estN (density)  -> detection gains
    are density-CHANGING => golden-12 partly blind => flag NEEDS-LB (human submits).

Reuses the EXACT primitives that reproduced pilkwang 0.8708 (pilk_post + src.metric +
src.golden_cv + learning/ensemble_work/score_pilkwang.py helpers) so our numbers are
comparable to the recorded baseline.

Usage:
  # validate the scorer reproduces 0.8708 on pilkwang's existing golden-12 geffs (CPU, no GPU):
  python baseline/score_v1.py --validate [--limit 2] [--no-mlflow]
  # score a trained run's predictions:
  python baseline/score_v1.py --method baseline_v1_v1_2_hr_baseaug --run-name baseline_v1_v1_2_hr_baseaug
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PARENT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
WORKDIR = Path(__file__).resolve().parents[1]  # tools/researchpapers
BASELINE_DIR = WORKDIR / "baseline"  # holds experiments_v1/, experiments_v2/, ...
ENSEMBLE = PARENT / "learning/ensemble_work"
TRAIN = PARENT / "input/biohub-cell-tracking-during-development/train"
# predict_unet_transformer.py writes to PREDICTIONS_PATH/USERNAME/<method>/split_<fold>
PRED_ROOT = PARENT / "research/official_repo/predictions"
PILK_GEFFS = PARENT / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
BASELINE_REF = 0.8708  # recorded golden-12 FULL post-proc micro adjJ

# make parent-repo src + ensemble helpers importable
sys.path.insert(0, str(PARENT))
sys.path.insert(0, str(ENSEMBLE))
from src import io, golden_cv as gcv          # noqa: E402
from src.config import Config                 # noqa: E402
from src.metric import official_counts, official_score, _match_nodes  # noqa: E402
import pilk_post as P                         # noqa: E402
from score_pilkwang import geff_to_dicts, dicts_to_dfs, GOLDEN12  # noqa: E402  (proven helpers)

SRC = Config()


def score_dir(geff_dir: Path, limit: int | None = None, target: set | None = None) -> pd.DataFrame:
    """Score geffs in `geff_dir` with the FULL post-proc + official metric.
    `target` = the embryo-stem set to score (default golden-12). Pass a matched mini-VAL set to get
    a golden-12-faithful 'mini-official' score for cheap per-rung screening."""
    tgt = target if target is not None else GOLDEN12
    geffs = sorted(g for g in geff_dir.glob("*.geff")
                   if g.name.replace(".zarr.geff", "").replace(".geff", "") in tgt)
    if limit:
        geffs = geffs[:limit]
    if not geffs:
        raise FileNotFoundError(f"no target geffs in {geff_dir} (target set size={len(tgt)})")
    print(f"scoring {len(geffs)} geffs from {geff_dir}", flush=True)
    rows = []
    for g in geffs:
        ds = g.name.replace(".zarr.geff", "").replace(".geff", "")
        nbi, raw_edges = geff_to_dicts(g)
        # full pilkwang post-proc chain (verbatim), then official counts
        nbi2, edges2, _ = P.filter_output_graph(dict(nbi), list(raw_edges), dataset=ds)
        pn2, pe2 = dicts_to_dfs(nbi2, edges2)
        gn, ge = io.read_geff(TRAIN / f"{ds}.geff")
        tt = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
        r = official_counts(gn, ge, pn2, pe2, SRC.SCALE, SRC.MATCH_GATE_UM, t_true=tt)
        # detector recall proxy (density-blind on sparse GT) + density (count vs estN)
        g2p = _match_nodes(gn, pn2, SRC.SCALE, SRC.MATCH_GATE_UM)
        r["node_recall"] = len(g2p) / max(len(gn), 1)
        r["count_ratio"] = r["t_pred"] / max(r["t_true"], 1.0)   # >1 => denser than true estN
        r["embryo"] = ds.split("_")[0]
        r["dataset"] = ds
        rows.append(r)
        print(f"  {ds:16s} adjJ={r['adj_jaccard']:.3f}  recall={r['node_recall']:.3f}  "
              f"nodes={r['t_pred']} estN={r['t_true']:.0f} (x{r['count_ratio']:.2f})  div_tp={r['div_tp']}",
              flush=True)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> dict:
    off = official_score(df.to_dict("records"))            # adj_edge_jaccard + 0.1*div_jaccard
    micro = float((df.w * df.adj_jaccard).sum() / df.w.sum())
    cv = gcv.golden_cv(df)["golden_cv"]
    per = df.groupby("embryo").apply(
        lambda x: (x.w * x.adj_jaccard).sum() / x.w.sum(), include_groups=False)
    return {
        "official_score": float(off["score"]),
        "adj_edge_jaccard": float(off["adj_edge_jaccard"]),
        "division_jaccard": float(off["division_jaccard"]) if not np.isnan(off["division_jaccard"]) else 0.0,
        "micro_adjJ": micro,
        "golden_cv": float(cv),
        "mean_node_recall": float(df.node_recall.mean()),
        "mean_count_ratio": float(df.count_ratio.mean()),
        "div_tp_total": int(df.div_tp.sum()),
        "adjJ_44b6": float(per.get("44b6", float("nan"))),
        "adjJ_6bba": float(per.get("6bba", float("nan"))),
        "n": int(len(df)),
    }


def _resolve_config_file(method: str | None, explicit: str | None) -> Path | None:
    """Find the experiment yaml that drove the scored checkpoint (for MLflow config tracking).
    Explicit path wins; else map method 'baseline_v<N>_<stem>' -> baseline/experiments_v<N>/<stem>.yaml
    (searches every experiments_v*/ so v1, v2, ... all resolve)."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    if method:
        # strip a 'baseline_v<N>_' prefix if present
        import re
        m = re.match(r"baseline_v\d+_(.+)", method)
        stem = m.group(1) if m else method
        for exp_dir in sorted(BASELINE_DIR.glob("experiments_v*")):
            cand = exp_dir / f"{stem}.yaml"
            if cand.exists():
                return cand
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="baseline_v1 golden-12 official scorer")
    ap.add_argument("--method", help="trained method name -> predictions/seshu/<method>/split_<fold>")
    ap.add_argument("--geff-dir", help="explicit geff dir (overrides --method)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--config-file", default=None,
                    help="experiment yaml that produced the scored checkpoint (MLflow config tracking)")
    ap.add_argument("--validate", action="store_true",
                    help="score pilkwang's existing golden-12 geffs; assert micro adjJ ~= 0.8708")
    ap.add_argument("--limit", type=int, default=None, help="score only first N embryos (fast smoke)")
    ap.add_argument("--split-file", default=None,
                    help="score the TEST embryos of this split[fold] instead of golden-12 "
                         "(e.g. splits_screen_matched.json for a golden-12-faithful MINI-OFFICIAL score)")
    ap.add_argument("--run-name", default=None, help="MLflow run name (default: <method>_score)")
    ap.add_argument("--exp-id", default=None, help="experiment journal id (e.g. EXP-001) — logged as "
                    "MLflow tag exp_id and written into the score.json sidecar for auto-fill")
    ap.add_argument("--score-json", default=None,
                    help="path for the results sidecar (default output/scores/<run_name>.json)")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()

    # target embryo set: golden-12 by default, or a split's test fold (matched mini-VAL)
    target = None
    if args.split_file:
        sp = json.load(open(args.split_file if Path(args.split_file).is_absolute()
                            else PARENT / args.split_file))
        target = {s.replace(".zarr", "") for s in sp[args.fold]["test"]}
        print(f"[target] mini-VAL from {Path(args.split_file).name} fold{args.fold}: {len(target)} embryos", flush=True)

    if args.validate:
        geff_dir = PILK_GEFFS
        run_name = args.run_name or "pilkwang_baseline_score_validate"
    elif args.geff_dir:
        geff_dir = Path(args.geff_dir)
        run_name = args.run_name or (Path(args.geff_dir).name + "_score")
    elif args.method:
        geff_dir = PRED_ROOT / "seshu" / args.method / f"split_{args.fold}"
        run_name = args.run_name or (args.method + "_score")
    else:
        ap.error("provide --method or --geff-dir (or --validate)")

    df = score_dir(geff_dir, limit=args.limit, target=target)
    s = summarize(df)
    # machine-parseable line for the successive-halving driver to gate on (mini-official)
    print(f"MINI_OFFICIAL_SCORE={s['official_score']:.6f}", flush=True)

    print("\n" + "=" * 66)
    print(f" GOLDEN-12 OFFICIAL SCORE  (n={s['n']}{' PARTIAL' if args.limit else ''})")
    print("=" * 66)
    print(f"  official_score (adj_edge + 0.1*div) = {s['official_score']:.4f}")
    print(f"  micro adjJ (edge)                   = {s['micro_adjJ']:.4f}   [vs baseline {BASELINE_REF:.4f}]")
    print(f"  golden_cv()                         = {s['golden_cv']:.4f}")
    print(f"  division_jaccard / div_tp           = {s['division_jaccard']:.4f} / {s['div_tp_total']}")
    print(f"  per-embryo  44b6={s['adjJ_44b6']:.4f}  6bba={s['adjJ_6bba']:.4f}")
    print(f"  [recall proxy] mean node_recall     = {s['mean_node_recall']:.4f}   (density-blind on sparse GT)")
    print(f"  [density]     mean count_ratio      = {s['mean_count_ratio']:.2f}x true estN  "
          f"(>1 => denser => golden-CV over-credits; NEEDS-LB)")
    print("=" * 66, flush=True)

    # ---- results sidecar (score.json) — the auto-fill SOURCE for the experiment journal ----
    fidelity, eval_split = "golden12", "golden12"
    if args.split_file:
        eval_split = Path(args.split_file).stem
        fidelity = "golden12" if "splits_ft" in eval_split else "mini"
    sidecar = {
        "run_name": run_name, "exp_id": args.exp_id, "method": args.method,
        "fidelity": fidelity, "eval_split": eval_split, "n_embryos": s["n"],
        "official_score": s["official_score"], "micro_adjJ": s["micro_adjJ"],
        "golden_cv": s["golden_cv"], "mean_node_recall": s["mean_node_recall"],
        "mean_count_ratio": s["mean_count_ratio"], "div_tp_total": s["div_tp_total"],
        "adjJ_44b6": s["adjJ_44b6"], "adjJ_6bba": s["adjJ_6bba"],
        "geff_dir": str(geff_dir), "status": "DONE",
    }
    sc_path = Path(args.score_json) if args.score_json else (WORKDIR / "output" / "scores" / f"{run_name}.json")
    sc_path.parent.mkdir(parents=True, exist_ok=True)
    sc_path.write_text(json.dumps(sidecar, indent=2))
    print(f"[sidecar] wrote {sc_path}", flush=True)

    if not args.no_mlflow:
        # STANDING RULE (human directive): every MLflow run owner replicates (1) system metrics
        # ALWAYS ON and (2) config tracking (config_file param + config_path tag + config artifact).
        import os
        os.environ.pop("MLFLOW_RUN_ID", None)  # own fresh run per score (avoid inherited-run param collision)
        os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"
        os.environ.setdefault("MLFLOW_SYSTEM_METRICS_SAMPLING_INTERVAL", "10")
        cfg_file = _resolve_config_file(args.method, args.config_file)
        try:
            import json as _json

            import mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("kaggle-biohub-cell-tracking")
            with mlflow.start_run(run_name=run_name, log_system_metrics=True):  # (1) system metrics ON
                mlflow.set_tag("phase", "score")
                mlflow.set_tag("fidelity", fidelity)          # mini | golden12
                mlflow.set_tag("eval_split", eval_split)
                if args.exp_id:
                    mlflow.set_tag("exp_id", args.exp_id)     # journal keying (trainer writeback)
                mlflow.log_metrics({k: v for k, v in s.items() if isinstance(v, (int, float))})
                mlflow.log_param("geff_dir", str(geff_dir))
                # (2) config tracking — same mechanism as src/baseline/train.py
                if cfg_file is not None:
                    mlflow.log_param("config_file", cfg_file.name)
                    mlflow.set_tag("config_path", str(cfg_file))
                    import yaml as _yaml
                    _cfg = _yaml.safe_load(open(cfg_file))
                    mlflow.log_dict(_cfg, "config.yaml.json")
                    mlflow.log_artifact(str(cfg_file))
                else:
                    mlflow.log_param("config_file", args.method or Path(str(args.geff_dir)).name)
            print(f"[mlflow] logged score run '{run_name}' (system-metrics ON, "
                  f"config={cfg_file.name if cfg_file else 'n/a'})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[mlflow] skipped ({e})", flush=True)

    if args.validate and not args.limit:
        ok = abs(s["micro_adjJ"] - BASELINE_REF) < 0.005
        print(f"\n[validate] micro adjJ {s['micro_adjJ']:.4f} vs {BASELINE_REF:.4f} -> "
              f"{'PASS (scorer reproduces baseline)' if ok else 'FAIL (scorer mismatch!)'}", flush=True)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
