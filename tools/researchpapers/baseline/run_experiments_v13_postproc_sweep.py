#!/usr/bin/env python
"""baseline_v13 CHEAP post-proc sweep on pilkwang CACHED golden-12 predictions (GPU-free).

Stage 3b pivot (EdgeThresholdGapRecovery SHELVED for O(n*m) hang). Screens cheap,
CPU-only post-proc levers on the golden-12 LB-faithful CV gate, scored with the
OFFICIAL metric (src.metric.official_*), delta vs the 0.8735 anchor
(= pilk full postproc + min_track_len4, our bankable best).

Levers (all stack ON the pilk full-postproc anchor, which is the 0.8708-in-memory
chain reproduced via pilk_post.filter_output_graph + write_geff round-trip ~0.8691):

  EXP-B  min_track_len sweep {4,5,6,8,10,12,14}   (min4 = anchor 0.8735; competitor
         intel drkongvis/boristown swept min6..min14 -> find golden-12 optimum)
  EXP-A' edge-precision prune (consensus_prune, beicicc lb884 preset) stacked on the
         best min_track_len -> the CACHED analog of raising det_threshold (both relieve
         the over-prediction penalty min(1, estN/predN); TRUE det_thresh needs GPU
         re-detection -- no per-node prob in the cached geff -- see design note).

NOT run here (no cheap eval surface on the cached golden-12; require GPU inference):
  EXP-A true det_threshold sweep  -> baseline/postproc/det_grid_sweep.sh (needs weights)
  EXP-C fork-based division prune -> golden-12 is div-BLIND (8 GT divs) AND the div-rich
         minisplit datasets are NOT in the pilk cache; defer to a GPU div-rich job.

Usage:
  research/cellmot_venv/bin/python tools/researchpapers/baseline/run_experiments_v13_postproc_sweep.py \
      [--min-track-lens 4 5 6 8 10 12 14] [--dry-run]

--dry-run: validate wiring on ONE dataset (44b6_0113de3b) + min_track_len4 only, no full sweep.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "learning/ensemble_work"))
sys.path.insert(0, str(ROOT / "tools/researchpapers/baseline/postproc"))

from src import io, metric
from src.config import Config
import pilk_post
import _geff_glue
from score_pilkwang import geff_to_dicts

S = Config()
PY = str(ROOT / "research/cellmot_venv/bin/python")
MR = ROOT / "tools/researchpapers/baseline/postproc"
PILK = ROOT / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
GT = ROOT / "input/biohub-cell-tracking-during-development/train"
OUT = ROOT / "tools/researchpapers/output/baseline_v13"
ANCHOR_ANCHOR_SCORE = 0.8735  # pilk full postproc + min_track_len4 (bankable best)

GOLDEN12 = {
    "44b6": ["44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc",
             "44b6_0db75fae", "44b6_12dfb391", "44b6_144b256d"],
    "6bba": ["6bba_05b6850b", "6bba_05db0fb1", "6bba_062c8d37",
             "6bba_07477033", "6bba_07e24132", "6bba_085bf656"],
}
ALL12 = GOLDEN12["44b6"] + GOLDEN12["6bba"]


def score_ds(geff_dir: Path, ds: str, suffix: str) -> dict:
    p = geff_dir / f"{ds}{suffix}"
    gn, ge = io.read_geff(GT / f"{ds}.geff")
    estN = io.geff_estimated_nodes(GT / f"{ds}.geff")
    pn, pe = io.read_geff(p)
    r = metric.official_counts(gn, ge, pn[["node_id", "t", "z", "y", "x"]],
                               pe[["source_id", "target_id"]], S.SCALE, S.MATCH_GATE_UM, t_true=estN)
    r["dataset"] = ds
    return r


def score_dir(geff_dir: Path, suffix: str, datasets=None) -> dict:
    """Micro official_score over the given datasets + by-embryo adj_edge."""
    datasets = datasets or ALL12
    counts = [score_ds(geff_dir, ds, suffix) for ds in datasets if (geff_dir / f"{ds}{suffix}").is_dir()]
    agg = metric.official_score(counts)
    out = {"score": agg["score"], "adj_edge_jaccard": agg["adj_edge_jaccard"], "n": len(counts)}
    for emb, dss in GOLDEN12.items():
        ec = [c for c in counts if c["dataset"] in dss]
        if ec:
            out[f"adj_{emb}"] = metric.official_score(ec)["adj_edge_jaccard"]
    return out


def build_anchor(anchor_dir: Path, datasets=None) -> int:
    """pilk full-postproc chain -> .geff round-trip (the ~0.8691 anchor)."""
    datasets = datasets or ALL12
    anchor_dir.mkdir(parents=True, exist_ok=True)
    relink = 0
    for ds in datasets:
        nbi, redges = geff_to_dicts(PILK / f"{ds}.zarr.geff")
        a_nbi, a_edges, stats = pilk_post.filter_output_graph(nbi, redges, ds)
        relink += stats.get("motion_relink_edges", 0)
        _geff_glue.write_geff(anchor_dir / f"{ds}.geff", a_nbi, a_edges)
    return relink


def run_module(script: str, in_dir: Path, out_dir: Path, extra=None):
    cmd = [PY, str(MR / script), "--in-dir", str(in_dir), "--out-dir", str(out_dir)] + (extra or [])
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{script} failed:\n{r.stderr[-2000:]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-track-lens", type=int, nargs="+", default=[4, 5, 6, 8, 10, 12, 14])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    datasets = ALL12
    lens = args.min_track_lens
    if args.dry_run:
        datasets = ["44b6_0113de3b"]
        lens = [4]
        print("[DRY-RUN] 1 dataset (44b6_0113de3b), min_track_len4 only, wiring check.\n")

    OUT.mkdir(parents=True, exist_ok=True)
    anchor_dir = OUT / "anchor"
    print(f"[1] Building pilk full-postproc anchor ({len(datasets)} ds)...")
    relink = build_anchor(anchor_dir, datasets)
    anchor = score_dir(anchor_dir, ".geff", datasets)
    print(f"    anchor (pilk_post, ~0.8691 on full-12): official={anchor['score']:.4f} "
          f"adj_edge={anchor['adj_edge_jaccard']:.4f} [relink={relink} edges]\n")

    results = [{"variant": "anchor_pilk_post", **anchor}]

    # EXP-B: min_track_len sweep
    print(f"[2] EXP-B min_track_len sweep {lens} (delta vs 0.8735 anchor):")
    best_mtl = (None, -1, None)
    for L in lens:
        d = OUT / f"mtl{L}"
        run_module("min_track_len_prune.py", anchor_dir, d, ["--min-track-len", str(L)])
        r = score_dir(d, ".geff", datasets)
        delta = r["score"] - ANCHOR_ANCHOR_SCORE
        flag = "  <-- PROMOTE" if delta > 0.001 else ""
        print(f"    min_track_len={L:<3} official={r['score']:.4f} ({delta:+.4f}) "
              f"adj_edge={r['adj_edge_jaccard']:.4f}{flag}")
        results.append({"variant": f"mtl{L}", **r, "delta_vs_anchor": delta})
        if r["score"] > best_mtl[1]:
            best_mtl = (L, r["score"], d)

    # EXP-A' edge-precision prune stacked on best min_track_len
    if not args.dry_run and best_mtl[2] is not None:
        print(f"\n[3] EXP-A' consensus_prune (cached det_thresh analog) on best mtl{best_mtl[0]}:")
        d = OUT / f"mtl{best_mtl[0]}_consensus"
        run_module("consensus_prune.py", best_mtl[2], d)
        r = score_dir(d, ".geff", datasets)
        delta = r["score"] - ANCHOR_ANCHOR_SCORE
        flag = "  <-- PROMOTE" if delta > 0.001 else ""
        print(f"    mtl{best_mtl[0]}+consensus_prune official={r['score']:.4f} ({delta:+.4f}) "
              f"adj_edge={r['adj_edge_jaccard']:.4f}{flag}")
        results.append({"variant": f"mtl{best_mtl[0]}_consensus", **r, "delta_vs_anchor": delta})

    out_json = OUT / ("dryrun_results.json" if args.dry_run else "v13_postproc_sweep_results.json")
    with open(out_json, "w") as f:
        json.dump({"anchor_score": ANCHOR_ANCHOR_SCORE, "eval_set": "golden_12",
                   "datasets": datasets, "results": results}, f, indent=2)
    print(f"\nResults -> {out_json}")
    if args.dry_run:
        print("[DRY-RUN OK] wiring validated: anchor build + min_track_len + official scorer.")


if __name__ == "__main__":
    main()
