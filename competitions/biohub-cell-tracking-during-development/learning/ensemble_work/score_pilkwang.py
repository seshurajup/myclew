"""Score pilkwang's predicted geffs (raw ILP vs full post-processed) with OUR golden CV,
and save per-dataset node detections for later fusion with canqiang.

  cellmot_venv/bin/python score_pilkwang.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from src import io, metric, golden_cv as gcv
from src.config import Config
import pilk_post as P

SRC = Config()
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
GEFF_DIR = ROOT / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
NODES_OUT = Path(__file__).parent / "pilkwang_nodes"
NODES_OUT.mkdir(exist_ok=True)
GOLDEN12 = {"44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"}


def geff_to_dicts(path):
    """Load a predicted geff -> nodes_by_id dict + raw_edges list (notebook format)."""
    graph = P.graph_from_geff(path)
    nodes_by_id = {}
    for row in graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nodes_by_id[nid] = {"node_id": nid, "t": int(row["t"]),
                            "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}
    raw_edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        ep = row.get("edge_prob") if hasattr(row, "get") else None
        raw_edges.append({"source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                          "edge_prob": None if ep is None else float(ep)})
    return nodes_by_id, raw_edges


def dicts_to_dfs(nodes_by_id, edges):
    pn = pd.DataFrame([{"node_id": n["node_id"], "t": n["t"], "z": n["z"], "y": n["y"], "x": n["x"]}
                       for n in nodes_by_id.values()])
    pe = pd.DataFrame([{"source_id": int(e["source_id"]), "target_id": int(e["target_id"])} for e in edges]) \
        if edges else pd.DataFrame(columns=["source_id", "target_id"])
    return pn, pe


def score_one(ds, pn, pe):
    gn, ge = io.read_geff(TRAIN / f"{ds}.geff")
    tt = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
    r = metric.official_counts(gn, ge, pn, pe, SRC.SCALE, SRC.MATCH_GATE_UM, t_true=tt)
    r["embryo"] = ds.split("_")[0]
    r["dataset"] = ds
    return r


def main():
    geffs = sorted(GEFF_DIR.glob("*.geff"))
    geffs = [g for g in geffs if g.name.replace(".zarr.geff", "").replace(".geff", "") in GOLDEN12]
    print(f"found {len(geffs)} golden-12 predicted geffs")
    rows_raw, rows_post = [], []
    for g in geffs:
        ds = g.name.replace(".zarr.geff", "").replace(".geff", "")
        nbi, raw_edges = geff_to_dicts(g)
        # RAW (detection + ILP only)
        pn, pe = dicts_to_dfs(nbi, raw_edges)
        rows_raw.append(score_one(ds, pn, pe))
        # POST-PROCESSED (full pilkwang chain)
        nbi2, edges2, _ = P.filter_output_graph(dict(nbi), list(raw_edges), dataset=ds)
        pn2, pe2 = dicts_to_dfs(nbi2, edges2)
        rows_post.append(score_one(ds, pn2, pe2))
        # save node detections (original-res voxels) for fusion
        pn2.to_csv(NODES_OUT / f"{ds}.csv", index=False)
        print(f"  {ds:16s} raw adjJ={rows_raw[-1]['adj_jaccard']:.3f}  "
              f"post adjJ={rows_post[-1]['adj_jaccard']:.3f}  nodes={len(pn2)}")

    for name, rows in [("RAW det+ILP", rows_raw), ("FULL post-proc", rows_post)]:
        df = pd.DataFrame(rows)
        cv = gcv.golden_cv(df)["golden_cv"]
        # simple micro over the 12 (comparable subset number)
        micro = (df.w * df.adj_jaccard).sum() / df.w.sum()
        emb = df.groupby("embryo").apply(lambda x: (x.w * x.adj_jaccard).sum() / x.w.sum(),
                                         include_groups=False)
        print(f"\n=== {name} ===")
        print(f"  micro adjJ over golden-12 = {micro:.4f}   (est LB ~= +0.11 -> {micro+0.11:.3f})")
        print(f"  golden_cv() = {cv:.4f}")
        print(f"  per-embryo: 44b6={emb.get('44b6', float('nan')):.4f}  6bba={emb.get('6bba', float('nan')):.4f}")
        df.to_csv(Path(__file__).parent / f"pilkwang_{name.split()[0].lower()}_scores.csv", index=False)


if __name__ == "__main__":
    main()
