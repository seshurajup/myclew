"""Part B — learned division recovery from the cached candidate pool.
For each parent with 1 child after pilkwang post-proc (safe-div OFF), restore the
2nd child that the edge-predictor scored highly (pool) AND that continues.
Writes geffs to research/official_repo/predictions/seshu/<method>/split_0/ for official scoring.

  recover_div.py <method> <prob_thr> <max_add_frac> [reassign=0]
"""
import os, sys
from pathlib import Path
from collections import defaultdict
import numpy as np, polars as pl
import tracksdata as td

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning/ensemble_work"))
os.environ["BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "0"   # turn OFF geometric safe-div
os.environ["BIOHUB_GAP_REFINE_SYNTHETIC"] = "0"    # fast
import pilk_post as P

K = td.DEFAULT_ATTR_KEYS
CACHE = ROOT / "learning/ensemble_work/pool_cache"
GOLDEN12 = ["44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_085bf656", "6bba_07e24132"]


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


def recover(method, prob_thr, max_add_frac, reassign=False):
    outd = ROOT / f"research/official_repo/predictions/seshu/{method}/split_0"
    if outd.exists():
        import shutil; shutil.rmtree(outd)
    outd.mkdir(parents=True, exist_ok=True)
    tot_add = 0
    for name in GOLDEN12:
        d = np.load(CACHE / f"{name}.npz")
        coords, pool, base_edges = d["coords"], d["pool"], d["base_edges"]
        nbi = {i: {"node_id": i, "t": int(coords[i, 0]), "z": float(coords[i, 1]),
                   "y": float(coords[i, 2]), "x": float(coords[i, 3])} for i in range(len(coords))}
        prob_lu = {(int(s), int(t)): float(p) for s, t, p, dist in pool}   # learned edge prob
        raw_edges = [{"source_id": int(s), "target_id": int(t),
                      "edge_prob": prob_lu.get((int(s), int(t)))}
                     for s, t in base_edges]
        # pilkwang post-proc, safe-div OFF
        nbi2, edges2, _ = P.filter_output_graph(dict(nbi), list(raw_edges), dataset=name)
        # existing topology after post-proc
        out_by = defaultdict(list); in_deg = defaultdict(int)
        for e in edges2:
            out_by[int(e["source_id"])].append(int(e["target_id"]))
            in_deg[int(e["target_id"])] += 1
        # candidate pool: per source -> [(prob, tgt)] sorted desc
        cand = defaultdict(list)
        for s, t, p, dist in pool:
            cand[int(s)].append((float(p), int(t)))
        for s in cand:
            cand[s].sort(reverse=True)
        cont = {n for n in out_by}                    # nodes that continue (have a successor)
        cap = int(max_add_frac * len(edges2))
        added = 0
        for parent, kids in list(out_by.items()):
            if added >= cap:
                break
            if len(kids) != 1:
                continue                              # only single-child parents
            c1 = kids[0]
            for prob, c2 in cand.get(parent, []):
                if prob < prob_thr:
                    break
                if c2 == c1 or c2 not in nbi2:
                    continue
                if nbi2[c2]["t"] != nbi2[parent]["t"] + 1:
                    continue
                if c2 not in cont:                    # 2nd daughter must continue -> real division
                    continue
                if in_deg.get(c2, 0) >= 1:
                    if not reassign:
                        continue
                    # reassign: drop c2's existing incoming edge, then make it P's 2nd child
                    edges2 = [e for e in edges2
                              if not (int(e["target_id"]) == c2)]
                edges2.append({"source_id": parent, "target_id": c2})
                in_deg[c2] = 1; added += 1
                break
        tot_add += added
        write_geff(nbi2, edges2, outd / f"{name}.geff")
    print(f"{method}: prob_thr={prob_thr} max_frac={max_add_frac} total_div_added={tot_add}", flush=True)


if __name__ == "__main__":
    ra = len(sys.argv) > 4 and sys.argv[4] == "1"
    recover(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), reassign=ra)
