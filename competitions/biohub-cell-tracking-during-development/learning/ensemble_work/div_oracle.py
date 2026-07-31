"""ORACLE ceiling check: use GT to add EXACTLY the correct 2nd-child edge for each
linking-fixable GT division, then score. Confirms whether the TP mechanism responds
(if oracle can't get TP, no geometric rule can). GT used only to pick the edge."""
import sys
from pathlib import Path
import numpy as np, polars as pl
import tracksdata as td

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning/ensemble_work"))
import pilk_post as P

TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
BASE = ROOT / "research/official_repo/predictions/seshu/base_fast/split_0"
VOX = np.array([1.625, 0.40625, 0.40625]); GATE = 7.0
K = td.DEFAULT_ATTR_KEYS
G12 = ["44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
       "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
       "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"]


def load(path):
    g = P.graph_from_geff(str(path))
    nodes = {int(r[K.NODE_ID]): (int(r["t"]), np.array([r["z"], r["y"], r["x"]], float))
             for r in g.node_attrs().iter_rows(named=True)}
    succ = {}
    edges = []
    for r in g.edge_attrs().iter_rows(named=True):
        s, t = int(r[K.EDGE_SOURCE]), int(r[K.EDGE_TARGET]); succ.setdefault(s, []).append(t); edges.append((s, t))
    return nodes, edges, succ


def write(nodes, edges, out):
    g = td.graph.IndexedRXGraph()
    for key in (K.T, K.Z, K.Y, K.X):
        if key not in set(g.node_attr_keys()):
            g.add_node_attr_key(key, pl.Float64, default_value=0.0)
    idm = {}
    for nid, (t, p) in nodes.items():
        idm[nid] = g.add_node({K.T: int(t), K.Z: float(p[0]), K.Y: float(p[1]), K.X: float(p[2])}, index=int(nid))
    for s, t in edges:
        if s in idm and t in idm:
            g.add_edge(idm[s], idm[t], {})
    g.to_geff(str(out))


def nearest(nodes, t, pos):
    best, bd = None, 1e9
    for nid, (tt, p) in nodes.items():
        if tt == t:
            d = np.linalg.norm((p - pos) * VOX)
            if d < bd:
                bd, best = d, nid
    return best, bd


outd = ROOT / "research/official_repo/predictions/seshu/divrec_oracle/split_0"
outd.mkdir(parents=True, exist_ok=True)
tot_added = 0
for ds in G12:
    gn, ge, gsucc = load(TRAIN / f"{ds}.geff")
    pn, pe, psucc = load(BASE / f"{ds}.geff")
    dividers = [n for n, ch in gsucc.items() if len(ch) >= 2]
    add = []
    for dv in dividers:
        tdv, pdv = gn[dv]
        pdv_id, dd = nearest(pn, tdv, pdv)
        if dd > GATE:
            continue
        dau_pred = []
        for c in gsucc[dv][:2]:
            tc, pc = gn[c]; pid, dc = nearest(pn, tc, pc); dau_pred.append(pid if dc <= GATE else None)
        if all(d is not None for d in dau_pred) and dau_pred[0] != dau_pred[1]:
            kids = set(psucc.get(pdv_id, []))
            unl = [d for d in dau_pred if d not in kids]
            if len(unl) == 1 and sum(1 for d in dau_pred if d in kids) == 1:
                add.append((pdv_id, unl[0]))
    write(pn, pe + add, outd / f"{ds}.geff")
    tot_added += len(add)
print(f"oracle added {tot_added} edges (expected ~4)")
