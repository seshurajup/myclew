"""Process pilkwang RAW geffs -> full postproc with pilk_post (params from BIOHUB_* env)
-> write geffs to official predictions/seshu/<method>/split_0. Fast sweep: set
BIOHUB_GAP_REFINE_SYNTHETIC=0 to skip slow frame reads.

  BIOHUB_OUTPUT_SAFE_DIVISIONS=0 rule_variant.py nodiv
"""
import sys, os
from pathlib import Path
import polars as pl
import tracksdata as td

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(Path(__file__).parent))
import pilk_post as P

K = td.DEFAULT_ATTR_KEYS
RAW = ROOT / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
GOLDEN12 = {"44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"}


def geff_to_dicts(path):
    graph = P.graph_from_geff(path)
    nbi = {}
    for row in graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nbi[nid] = {"node_id": nid, "t": int(row["t"]), "z": float(row["z"]),
                    "y": float(row["y"]), "x": float(row["x"])}
    edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        ep = row.get("edge_prob") if hasattr(row, "get") else None
        edges.append({"source_id": int(row["source_id"]), "target_id": int(row["target_id"]),
                      "edge_prob": None if ep is None else float(ep)})
    return nbi, edges


def write_geff(nbi, edges, out):
    g = td.graph.IndexedRXGraph()
    for key in (K.T, K.Z, K.Y, K.X):
        if key not in set(g.node_attr_keys()):
            g.add_node_attr_key(key, pl.Float64, default_value=0.0)
    idmap = {}
    for nid, n in nbi.items():
        idmap[nid] = g.add_node({K.T: int(n["t"]), K.Z: float(n["z"]),
                                 K.Y: float(n["y"]), K.X: float(n["x"])}, index=int(nid))
    for e in edges:
        s, t = int(e["source_id"]), int(e["target_id"])
        if s in idmap and t in idmap:
            g.add_edge(idmap[s], idmap[t], {})
    g.to_geff(str(out))


def main():
    method = sys.argv[1]
    outm = ROOT / f"research/official_repo/predictions/seshu/{method}/split_0"
    outm.mkdir(parents=True, exist_ok=True)
    for raw in sorted(RAW.glob("*.geff")):
        s = raw.name.replace(".zarr.geff", "").replace(".geff", "")
        if s not in GOLDEN12:
            continue
        nbi, edges = geff_to_dicts(raw)
        nbi2, edges2, _ = P.filter_output_graph(dict(nbi), list(edges), dataset=s)
        write_geff(nbi2, edges2, outm / f"{s}.geff")
    print(f"wrote -> {outm}")


if __name__ == "__main__":
    main()
