#!/usr/bin/env python
"""baseline_v14 — PER-DENSITY min_track_len prototype (golden-12 PoC).

The v13 per-embryo diagnostic showed OPPOSITE mtl optima, but true density does NOT track embryo
label (6bba spans S0..S4, 44b6 spans S2..S4). So bin by a MECHANISTIC per-dataset density proxy, not
embryo. Mechanism: high true-density => many spurious short tracks => wants aggressive pruning (mtl14);
low-density => gentle (mtl10).

Density proxy: `learning/03_true_density_stage.csv` column `stage` (S0..S4, precomputed over 199 datasets
from estN_per_frame). 2-bin mapping (NOT fit to golden-12 scores):
    stage in {S0,S1,S2} -> mtl 10   (low/mid density)
    stage in {S3,S4}     -> mtl 14   (high density)
gap fixed at 5.5. Compare 2-bin vs GLOBAL mtl10 on golden-12 (micro + per-embryo). Round-trip scale
(all rows share the write_geff base, so deltas are clean; in-memory anchor is ~+0.0014 higher).

Usage: research/cellmot_venv/bin/python tools/researchpapers/baseline/run_experiments_v14_perdensity_mtl.py
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "learning/ensemble_work"))
sys.path.insert(0, str(ROOT / "tools/researchpapers/baseline/postproc"))
os.environ["BIOHUB_GAP_CLOSE_UM"] = "5.5"  # BEFORE importing pilk_post (binds gap at import)

from src import io, metric
from src.config import Config
import pilk_post
import _geff_glue
from score_pilkwang import geff_to_dicts

S = Config()
assert abs(pilk_post.GAP_CLOSE_UM - 5.5) < 1e-9, "gap not bound to 5.5"
PY = str(ROOT / "research/cellmot_venv/bin/python")
MR = ROOT / "tools/researchpapers/baseline/postproc"
PILK = ROOT / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
GT = ROOT / "input/biohub-cell-tracking-during-development/train"
DENSITY_CSV = ROOT / "learning/03_true_density_stage.csv"
OUT = ROOT / "tools/researchpapers/output/baseline_v14"

GOLDEN12 = {
    "44b6": ["44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae", "44b6_12dfb391", "44b6_144b256d"],
    "6bba": ["6bba_05b6850b", "6bba_05db0fb1", "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"],
}
ALL12 = GOLDEN12["44b6"] + GOLDEN12["6bba"]
LOW_STAGES = {"S0", "S1", "S2"}   # -> mtl 10
HIGH_STAGES = {"S3", "S4"}        # -> mtl 14
MTL_LOW, MTL_HIGH = 10, 14


def load_density_bins():
    """dataset -> mtl, via the mechanistic stage->mtl map (199-row proxy, NOT golden-fit)."""
    m = {}
    for r in csv.DictReader(open(DENSITY_CSV)):
        st = r["stage"]
        m[r["dataset"]] = MTL_LOW if st in LOW_STAGES else MTL_HIGH
    return m


def build_anchor(anchor: Path):
    anchor.mkdir(parents=True, exist_ok=True)
    for ds in ALL12:
        if (anchor / f"{ds}.geff").is_dir():
            continue
        nbi, redges = geff_to_dicts(PILK / f"{ds}.zarr.geff")
        a_nbi, a_edges, _ = pilk_post.filter_output_graph(nbi, redges, ds)
        _geff_glue.write_geff(anchor / f"{ds}.geff", a_nbi, a_edges)


def prune(in_dir: Path, out_dir: Path, mtl: int, only=None):
    """min_track_len_prune on `in_dir`; if `only` given, first stage just those datasets."""
    src_dir = in_dir
    if only is not None:
        src_dir = out_dir.parent / f"_stage_{out_dir.name}"
        src_dir.mkdir(parents=True, exist_ok=True)
        for ds in only:
            g = in_dir / f"{ds}.geff"
            if g.is_dir():
                dst = src_dir / f"{ds}.geff"
                if not dst.exists():
                    dst.symlink_to(g)
    subprocess.run([PY, str(MR / "min_track_len_prune.py"), "--in-dir", str(src_dir),
                    "--out-dir", str(out_dir), "--min-track-len", str(mtl)], check=True, capture_output=True, text=True)


def score(dirs_by_ds: dict) -> dict:
    """dirs_by_ds: dataset -> geff dir holding <ds>.geff. Micro official + per-embryo adj_edge."""
    counts = []
    for ds in ALL12:
        p = dirs_by_ds[ds] / f"{ds}.geff"
        gn, ge = io.read_geff(GT / f"{ds}.geff")
        estN = io.geff_estimated_nodes(GT / f"{ds}.geff")
        pn, pe = io.read_geff(p)
        c = metric.official_counts(gn, ge, pn[["node_id", "t", "z", "y", "x"]],
                                   pe[["source_id", "target_id"]], S.SCALE, S.MATCH_GATE_UM, t_true=estN)
        c["dataset"] = ds
        counts.append(c)
    agg = metric.official_score(counts)
    out = {"score": agg["score"], "adj_edge_jaccard": agg["adj_edge_jaccard"]}
    for emb, dss in GOLDEN12.items():
        out[f"adj_{emb}"] = metric.official_score([c for c in counts if c["dataset"] in dss])["adj_edge_jaccard"]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bins = load_density_bins()
    low = [d for d in ALL12 if bins[d] == MTL_LOW]
    high = [d for d in ALL12 if bins[d] == MTL_HIGH]
    stage_of = {r["dataset"]: r["stage"] for r in csv.DictReader(open(DENSITY_CSV))}
    print(f"Density 2-bin (stage {sorted(LOW_STAGES)}->mtl{MTL_LOW}, {sorted(HIGH_STAGES)}->mtl{MTL_HIGH}), gap=5.5")
    print(f"  LOW/mtl{MTL_LOW}  ({len(low)}): {[f'{d}({stage_of[d]})' for d in low]}")
    print(f"  HIGH/mtl{MTL_HIGH} ({len(high)}): {[f'{d}({stage_of[d]})' for d in high]}")

    anchor = OUT / "anchor"
    build_anchor(anchor)

    # global baselines
    prune(anchor, OUT / "global_mtl10", 10)
    prune(anchor, OUT / "global_mtl14", 14)
    g10 = score({d: OUT / "global_mtl10" for d in ALL12})
    g14 = score({d: OUT / "global_mtl14" for d in ALL12})

    # per-density 2-bin: low datasets from mtl10 dir, high datasets from mtl14 dir
    twobin = score({**{d: OUT / "global_mtl10" for d in low}, **{d: OUT / "global_mtl14" for d in high}})

    print(f"\n{'config':<16}{'official':>10}{'adj_44b6':>10}{'adj_6bba':>10}{'Δ vs g_mtl10':>14}")
    for name, r in [("global_mtl10", g10), ("global_mtl14", g14), ("perdensity_2bin", twobin)]:
        d = r["score"] - g10["score"]
        flag = "  <-- BEAT +0.001" if d > 0.001 else ""
        print(f"{name:<16}{r['score']:>10.4f}{r['adj_44b6']:>10.4f}{r['adj_6bba']:>10.4f}{d:>+14.4f}{flag}")

    res = {"gap": 5.5, "mapping": {"low_stages": sorted(LOW_STAGES), "high_stages": sorted(HIGH_STAGES),
           "mtl_low": MTL_LOW, "mtl_high": MTL_HIGH}, "low_datasets": low, "high_datasets": high,
           "global_mtl10": g10, "global_mtl14": g14, "perdensity_2bin": twobin,
           "delta_2bin_vs_global10": twobin["score"] - g10["score"]}
    (OUT / "v14_perdensity_results.json").write_text(json.dumps(res, indent=2))
    print(f"\nResults -> {OUT / 'v14_perdensity_results.json'}")


if __name__ == "__main__":
    main()
