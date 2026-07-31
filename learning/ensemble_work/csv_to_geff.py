"""Convert node+edge CSVs (node_id,t,z,y,x / source_id,target_id) to a geff
in the official predictions/{user}/{method}/split_0/ layout, so the OFFICIAL
metric (scripts/evaluate.py) can score it exactly like pilkwang."""
import sys, argparse
from pathlib import Path
import pandas as pd
import polars as pl
import tracksdata as td

K = td.DEFAULT_ATTR_KEYS


def build_geff(nodes_csv, edges_csv, out_geff):
    nd = pd.read_csv(nodes_csv)
    ed = pd.read_csv(edges_csv) if Path(edges_csv).exists() else pd.DataFrame(columns=["source_id", "target_id"])
    g = td.graph.IndexedRXGraph()
    existing = set(g.node_attr_keys())
    for key in (K.T, K.Z, K.Y, K.X):
        if key not in existing:
            g.add_node_attr_key(key, pl.Float64, default_value=0.0)
    idmap = {}
    for r in nd.itertuples(index=False):
        nid = g.add_node({K.T: int(r.t), K.Z: float(r.z), K.Y: float(r.y), K.X: float(r.x)},
                         index=int(r.node_id))
        idmap[int(r.node_id)] = nid
    for r in ed.itertuples(index=False):
        s, t = int(r.source_id), int(r.target_id)
        if s in idmap and t in idmap:
            g.add_edge(idmap[s], idmap[t], {})
    g.to_geff(str(out_geff))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes-dir", required=True)
    ap.add_argument("--edges-dir", required=True)
    ap.add_argument("--edges-suffix", default="_edges.csv")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    outd = Path(a.out_dir); outd.mkdir(parents=True, exist_ok=True)
    stems = sorted(p.stem for p in Path(a.nodes_dir).glob("*.csv"))
    for stem in stems:
        ne = Path(a.nodes_dir) / f"{stem}.csv"
        ee = Path(a.edges_dir) / f"{stem}{a.edges_suffix}"
        try:
            build_geff(ne, ee, outd / f"{stem}.geff")
            print(f"  ok {stem}")
        except Exception as e:
            print(f"  FAIL {stem}: {e}")
    print(f"wrote geffs -> {outd}")
