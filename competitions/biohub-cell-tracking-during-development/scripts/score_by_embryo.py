"""Per-embryo (44b6 / 6bba) canonical golden-12 scorer.

Thin wrapper that reuses scripts/score_golden12_official.py internals so the metric,
cached-pilk source, and env-driven pilk_post are byte-identical to the canonical gate.
Emits per-embryo + combined adj_edge_jaccard so improvements can be judged BOTH embryos
(44b6 = the detection lever, 6bba = saturated), per the report-per-embryo-2cv rule.

Usage (same BIOHUB_* env as score_golden12_official.py):
  research/cellmot_venv/bin/python scripts/score_by_embryo.py [--raw] [--tag STR] [--pred-dir DIR]
Prints one JSON line: {tag, combined:{...}, "44b6":{...}, "6bba":{...}, per_ds:{ds:adjJ}}
"""
import argparse, json, sys, tempfile
from pathlib import Path

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "sg12", str(Path(__file__).resolve().parent / "score_golden12_official.py"))
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def score_ds(ds, tmp, raw, pred_dir):
    gt_ds = S.open_dataset(S.TRAIN / ds, require_tracks=True, load_image=False, device="cpu")
    pred_path = S._pred_geff_for(ds, tmp, raw, pred_dir)
    pred_res = S.td.graph.IndexedRXGraph.from_geff(pred_path)
    pred_graph = pred_res[0] if isinstance(pred_res, tuple) else pred_res
    er = S.compute_metric(pred_graph, gt_ds.tracks, scale=S.SCALE, max_distance=S.MATCH_GATE)
    if pred_graph.num_edges() > 0 and pred_graph.num_nodes() > 0:
        recall = S.node_recall(pred_graph, gt_ds.tracks)
    else:
        recall = 0.0
    return S.per_sample_metrics(er, S._estimated_n(S.TRAIN / f"{ds}.geff"), recall)


def summ(rows):
    if not rows:
        return None
    s = S.summarise(rows)
    return {k: (round(float(s[k]), 4) if s[k] == s[k] else None)
            for k in ("score", "adj_edge_jaccard", "edge_jaccard", "node_recall", "division_jaccard")} | {"n": s["n"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--pred-dir", type=str, default=None)
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir) if args.pred_dir else None
    datasets = sorted(S.GOLDEN12)
    if pred_dir is not None:
        datasets = [d for d in datasets if S._find_in_dir(pred_dir, d) is not None]
    by_emb = {"44b6": [], "6bba": []}
    per_ds = {}
    all_rows = []
    with tempfile.TemporaryDirectory() as td_dir:
        tmp = Path(td_dir)
        for i, ds in enumerate(datasets, 1):
            print(f"PROGRESS {i}/{len(datasets)} {ds}", file=sys.stderr, flush=True)
            row = score_ds(ds, tmp, args.raw, pred_dir)
            all_rows.append(row)
            emb = ds.split("_")[0]
            by_emb[emb].append(row)
            per_ds[ds] = round(float(row["adj_edge_jaccard"]), 4)
    out = {"tag": args.tag, "raw": args.raw,
           "combined": summ(all_rows), "44b6": summ(by_emb["44b6"]), "6bba": summ(by_emb["6bba"]),
           "per_ds": per_ds}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
