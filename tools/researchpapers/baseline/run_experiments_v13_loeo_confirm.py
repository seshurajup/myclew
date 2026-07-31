#!/usr/bin/env python
"""baseline_v13 LOEO CONFIRMATION runner — promote min_track_len to the PRIMARY 2-fold LOEO.

Confirms the cheap golden-12 win (min_track_len10 = 0.8818, +0.0083) on the PRIMARY 2-fold
embryo-disjoint LOEO (fold0 test=44b6, fold1 test=6bba; `learning/ensemble_work/finetune/fleet_loeo_mini.json`).

TWO PHASES:
  Phase 1 (GPU, TRAINER): generate raw pilk-model predictions for ALL fold test datasets ->
    <preds-dir>/<ds>.zarr.geff (pilk-native format, with `solution` col). fold0/44b6 already
    exist under output/stage3/loeo_predictions_full (as .geff, non-native — REGENERATE native).
    fold1/6bba (128 datasets) NOT yet generated. Gen script: baseline/generate_loeo_geffs_v13.sh.
    ** BLOCKER FLAGGED: only `unet_transformer/split_0` pilk weights exist — no split_1. For a
    strictly embryo-disjoint fold1 the pilk model must not have trained on 6bba; confirm split_0's
    train embryos or train split_1 before trusting fold1. (See design note.) **
  Phase 2 (CPU, THIS runner): pilk_post(gap=GAP) + min_track_len_prune(MTL) per dataset ->
    official full metric, per-fold weighted + 2-fold LOEO mean.

Usage (queue-ready; locked config passed at submit time):
  research/cellmot_venv/bin/python tools/researchpapers/baseline/run_experiments_v13_loeo_confirm.py \
      --preds-dir <raw .zarr.geff dir> --mtl 10 --gap 6.0 [--mtl-robust 8]
  # GPU-free dry-run (validates schema/paths/post-proc on the golden-12 raw cache, no GPU):
  ... run_experiments_v13_loeo_confirm.py --dry-run
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "learning/ensemble_work"))
sys.path.insert(0, str(ROOT / "tools/researchpapers/baseline/postproc"))

# CRITICAL: pilk_post binds GAP_CLOSE_UM from the env at IMPORT time (pilk_post.py:47), so the --gap
# env MUST be set BEFORE `import pilk_post`. Setting os.environ after import is a silent no-op.
if "--gap" in sys.argv:
    os.environ["BIOHUB_GAP_CLOSE_UM"] = sys.argv[sys.argv.index("--gap") + 1]

from src import io, metric
from src.config import Config
import pilk_post
import _geff_glue
from score_pilkwang import geff_to_dicts

S = Config()
PY = str(ROOT / "research/cellmot_venv/bin/python")
MR = ROOT / "tools/researchpapers/baseline/postproc"
GT = ROOT / "input/biohub-cell-tracking-during-development/train"
SPLIT = ROOT / "learning/ensemble_work/finetune/fleet_loeo_mini.json"
GOLDEN12_RAW = ROOT / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
OUT = ROOT / "tools/researchpapers/output/baseline_v13/loeo_confirm"


def postproc_and_score(datasets, preds_dir: Path, work: Path, mtl: int, gap: float) -> dict:
    """pilk_post(gap) -> min_track_len_prune(mtl) -> official metric over `datasets`. Weighted micro."""
    anchor = work / "anchor"
    anchor.mkdir(parents=True, exist_ok=True)
    # gap-close is applied INSIDE pilk_post, bound at import from BIOHUB_GAP_CLOSE_UM (set before import above).
    assert abs(pilk_post.GAP_CLOSE_UM - gap) < 1e-9, (
        f"gap mismatch: pilk_post bound {pilk_post.GAP_CLOSE_UM} but requested {gap} — "
        f"set BIOHUB_GAP_CLOSE_UM before importing pilk_post")
    present = []
    for ds in datasets:
        raw = preds_dir / f"{ds}.zarr.geff"
        if not raw.is_dir():
            raw = preds_dir / f"{ds}.geff"
        if not raw.is_dir():
            continue
        nbi, redges = geff_to_dicts(raw)
        a_nbi, a_edges, _ = pilk_post.filter_output_graph(nbi, redges, ds)
        _geff_glue.write_geff(anchor / f"{ds}.geff", a_nbi, a_edges)
        present.append(ds)
    pruned = work / f"mtl{mtl}"
    subprocess.run([PY, str(MR / "min_track_len_prune.py"), "--in-dir", str(anchor),
                    "--out-dir", str(pruned), "--min-track-len", str(mtl)], check=True,
                   capture_output=True, text=True)
    counts = []
    for ds in present:
        gn, ge = io.read_geff(GT / f"{ds}.geff")
        estN = io.geff_estimated_nodes(GT / f"{ds}.geff")
        pn, pe = io.read_geff(pruned / f"{ds}.geff")
        counts.append(metric.official_counts(gn, ge, pn[["node_id", "t", "z", "y", "x"]],
                      pe[["source_id", "target_id"]], S.SCALE, S.MATCH_GATE_UM, t_true=estN))
    agg = metric.official_score(counts)
    return {"score": agg["score"], "adj_edge_jaccard": agg["adj_edge_jaccard"], "n": len(present)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", type=Path, help="raw pilk .zarr.geff dir for all fold datasets (GPU-generated)")
    ap.add_argument("--mtl", type=int, default=10, help="locked min_track_len (primary)")
    ap.add_argument("--mtl-robust", type=int, default=8, help="robustness min_track_len")
    ap.add_argument("--gap", type=float, default=6.0, help="locked gap_close_um")
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1],
                    help="fold indices to score (fold0=44b6, fold1=6bba). Use --folds 0 for the interim.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    all_folds = json.loads(SPLIT.read_text())
    folds = [(i, all_folds[i]) for i in args.folds]
    interim = args.folds != [0, 1]
    OUT.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # Validate the full Phase-2 wiring on the golden-12 raw cache (available, no GPU).
        g12 = [p.name.replace(".zarr.geff", "") for p in sorted(GOLDEN12_RAW.glob("*.zarr.geff"))]
        print(f"[DRY-RUN] Phase-2 wiring on golden-12 raw cache ({len(g12)} ds), mtl={args.mtl} gap={args.gap}")
        r = postproc_and_score(g12, GOLDEN12_RAW, OUT / "dryrun", args.mtl, args.gap)
        print(f"    golden-12 official={r['score']:.4f} adj_edge={r['adj_edge_jaccard']:.4f} n={r['n']}")
        print("[DRY-RUN OK] pilk_post(gap) + min_track_len + official scorer wired. "
              "Phase-1 GPU gen (fold1/6bba) is the trainer's step.")
        (OUT / "dryrun_result.json").write_text(json.dumps({"golden12": r, "mtl": args.mtl, "gap": args.gap}, indent=2))
        return

    if not args.preds_dir or not args.preds_dir.is_dir():
        sys.exit("ERROR: --preds-dir with GPU-generated raw preds required (or use --dry-run). "
                 "Generate via baseline/generate_loeo_geffs_v13.sh first.")

    scope = "FOLD0-ONLY INTERIM" if interim else "2-fold LOEO"
    print(f"=== baseline_v13 LOEO confirm ({scope}) mtl={args.mtl} gap={args.gap} folds={args.folds} ===")
    results = {"mtl": args.mtl, "gap": args.gap, "folds_run": args.folds, "interim": interim, "results": {}}
    for mtl in sorted({args.mtl, args.mtl_robust}):
        fold_scores = []
        for i, fold in folds:
            r = postproc_and_score(fold["test"], args.preds_dir, OUT / f"mtl{mtl}_fold{i}", mtl, args.gap)
            results["results"].setdefault(f"mtl{mtl}", {})[f"fold{i}"] = r
            fold_scores.append(r["score"])
            print(f"mtl={mtl} fold{i}: official={r['score']:.4f} adj_edge={r['adj_edge_jaccard']:.4f} n={r['n']}")
        if not interim and fold_scores:
            mean = sum(fold_scores) / len(fold_scores)
            results["results"][f"mtl{mtl}"]["loeo_mean"] = mean
            print(f"mtl={mtl} 2-fold LOEO mean = {mean:.4f}\n")
    (OUT / "loeo_confirm_results.json").write_text(json.dumps(results, indent=2))
    print(f"Results -> {OUT / 'loeo_confirm_results.json'}")


if __name__ == "__main__":
    main()
