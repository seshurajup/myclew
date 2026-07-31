import sys, pandas as pd, polars as pl, tracksdata as td
from pathlib import Path
K = td.DEFAULT_ATTR_KEYS
OUTD = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/research/official_repo/predictions/seshu/pilk_full890/split_0")
OUTD.mkdir(parents=True, exist_ok=True)
df = pd.read_csv("submission.csv")
for ds, sub in df.groupby("dataset"):
    nodes = sub[sub.row_type == "node"]; edges = sub[sub.row_type == "edge"]
    g = td.graph.IndexedRXGraph()
    for k in (K.T, K.Z, K.Y, K.X):
        try: g.add_node_attr_key(k, pl.Float64, default_value=0.0)
        except Exception: pass
    idmap = {}
    for r in nodes.itertuples(index=False):
        idmap[int(r.node_id)] = g.add_node({K.T: int(r.t), K.Z: float(r.z), K.Y: float(r.y), K.X: float(r.x)}, index=int(r.node_id))
    for r in edges.itertuples(index=False):
        s, t = int(r.source_id), int(r.target_id)
        if s in idmap and t in idmap: g.add_edge(idmap[s], idmap[t], {})
    g.to_geff(str(OUTD / f"{ds}.geff"))
    print("  wrote", ds, "nodes", len(nodes), "edges", len(edges))
print("done ->", OUTD)
