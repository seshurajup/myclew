"""Geometric division-recovery rule on base pred geffs (NO GT used).
For each node P with exactly 1 outgoing child C1, add a 2nd child edge P->C2 when:
  - C2 at t+1, parent dist |P-C2| <= PARENT_UM
  - sister dist |C1-C2| <= SISTER_UM
  - C2 continues (has a successor)  [makes it a real fork -> TP not FP]
  - C2 currently has <= MAXPAR parents (default 0 = parentless only)
Writes new geffs to predictions/seshu/<METHOD>/split_0 and prints.
Env: PARENT_UM, SISTER_UM, REQUIRE_PARENTLESS(1/0), METHOD
"""
import os, sys
from pathlib import Path
import numpy as np, polars as pl
import tracksdata as td

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning/ensemble_work"))
import pilk_post as P

BASE = ROOT / f"research/official_repo/predictions/seshu/{os.environ.get('BASE_METHOD','nodiv')}/split_0"
VOX = np.array([1.625, 0.40625, 0.40625])
K = td.DEFAULT_ATTR_KEYS
G12 = ["44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
       "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
       "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"]

PARENT_UM = float(os.environ.get("PARENT_UM", "5.0"))
SISTER_UM = float(os.environ.get("SISTER_UM", "7.0"))
REQ_PARENTLESS = os.environ.get("REQUIRE_PARENTLESS", "1") == "1"
METHOD = os.environ.get("METHOD", "divrec")


def load(path):
    g = P.graph_from_geff(str(path))
    nodes = {int(r[K.NODE_ID]): (int(r["t"]), np.array([r["z"], r["y"], r["x"]], float))
             for r in g.node_attrs().iter_rows(named=True)}
    edges = [(int(r[K.EDGE_SOURCE]), int(r[K.EDGE_TARGET])) for r in g.edge_attrs().iter_rows(named=True)]
    return nodes, edges


def write(nodes, edges, out):
    g = td.graph.IndexedRXGraph()
    for key in (K.T, K.Z, K.Y, K.X):
        if key not in set(g.node_attr_keys()):
            g.add_node_attr_key(key, pl.Float64, default_value=0.0)
    idmap = {}
    for nid, (t, p) in nodes.items():
        idmap[nid] = g.add_node({K.T: int(t), K.Z: float(p[0]), K.Y: float(p[1]), K.X: float(p[2])}, index=int(nid))
    for s, t in edges:
        if s in idmap and t in idmap:
            g.add_edge(idmap[s], idmap[t], {})
    g.to_geff(str(out))


def recover(nodes, edges):
    succ, preds = {}, {}
    for s, t in edges:
        succ.setdefault(s, []).append(t)
        preds.setdefault(t, []).append(s)
    by_t = {}
    for nid, (t, p) in nodes.items():
        by_t.setdefault(t, []).append(nid)
    added = 0
    remove = set()
    new_pairs = []
    used_c2 = set()
    for pnode, kids in list(succ.items()):
        if len(kids) != 1:
            continue
        if pnode not in preds:            # divider must have a parent (one-node stage)
            continue
        c1 = kids[0]
        tp, pp = nodes[pnode]
        tc1, pc1 = nodes[c1]
        best = None
        for c2 in by_t.get(tp + 1, []):
            if c2 == c1 or c2 in used_c2:
                continue
            np_c2 = len(preds.get(c2, []))
            if REQ_PARENTLESS and np_c2 > 0:
                continue
            if np_c2 > 1:                 # too tangled to reassign safely
                continue
            if len(succ.get(c2, [])) == 0:      # must continue
                continue
            tc2, pc2 = nodes[c2]
            dp = np.linalg.norm((pp - pc2) * VOX)
            ds = np.linalg.norm((pc1 - pc2) * VOX)
            if dp <= PARENT_UM and ds <= SISTER_UM:
                sc = dp + 0.15 * ds
                if best is None or sc < best[0]:
                    best = (sc, c2)
        if best is not None:
            c2 = best[1]
            for pp_old in preds.get(c2, []):       # reassign: drop c2's old parent edge
                remove.add((pp_old, c2))
            new_pairs.append((pnode, c2))
            used_c2.add(c2); added += 1
    new_edges = [e for e in edges if e not in remove] + new_pairs
    return new_edges, added


def main():
    outd = ROOT / f"research/official_repo/predictions/seshu/{METHOD}/split_0"
    import shutil
    if outd.exists():
        shutil.rmtree(outd)
    outd.mkdir(parents=True, exist_ok=True)
    tot = 0
    for ds in G12:
        nodes, edges = load(BASE / f"{ds}.geff")
        ne, added = recover(nodes, edges)
        write(nodes, ne, outd / f"{ds}.geff")
        tot += added
    print(f"{METHOD}: PARENT_UM={PARENT_UM} SISTER_UM={SISTER_UM} parentless={REQ_PARENTLESS} added_edges={tot}")


if __name__ == "__main__":
    main()
