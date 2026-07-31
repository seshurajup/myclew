"""Diagnose the 8 GT divisions in golden-12: are they DETECTION-blocked or LINKING-fixable?
For each GT divider: is there a base-pred node near the divider? near each of the 2 daughters?
and is the pred-divider linked to the pred-daughters?"""
import sys
from pathlib import Path
import numpy as np
import tracksdata as td

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning/ensemble_work"))
import pilk_post as P

TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
BASE = ROOT / "research/official_repo/predictions/seshu/base_fast/split_0"
VOX = np.array([1.625, 0.40625, 0.40625])
GATE = 7.0
K = td.DEFAULT_ATTR_KEYS
G12 = ["44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
       "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
       "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"]


def load(path):
    g = P.graph_from_geff(str(path))
    na = g.node_attrs()
    nodes = {int(r[K.NODE_ID]): (int(r["t"]), np.array([r["z"], r["y"], r["x"]], float))
             for r in na.iter_rows(named=True)}
    succ, pred = {}, {}
    for r in g.edge_attrs().iter_rows(named=True):
        s, t = int(r[K.EDGE_SOURCE]), int(r[K.EDGE_TARGET])
        succ.setdefault(s, []).append(t); pred.setdefault(t, []).append(s)
    return nodes, succ, pred


def nearest(nodes, t, pos):
    best, bd = None, 1e9
    for nid, (tt, p) in nodes.items():
        if tt != t:
            continue
        d = np.linalg.norm((p - pos) * VOX)
        if d < bd:
            bd, best = d, nid
    return best, bd


tot = dict(n=0, div_hit=0, both_daughters=0, linked_both=0, linked_one=0, linked_none=0, fixable=0)
for ds in G12:
    gn, gsucc, gpred = load(TRAIN / f"{ds}.geff")
    pn, psucc, ppred = load(BASE / f"{ds}.geff")
    dividers = [nid for nid, ch in gsucc.items() if len(ch) >= 2]
    for dv in dividers:
        tot["n"] += 1
        tdv, pdv = gn[dv]
        pdv_id, dd = nearest(pn, tdv, pdv)
        if dd > GATE:
            continue  # divider not detected in pred
        tot["div_hit"] += 1
        # 2 daughters
        daus = gsucc[dv][:2]
        dau_pred = []
        for c in daus:
            tc, pc = gn[c]
            pid, dc = nearest(pn, tc, pc)
            dau_pred.append(pid if dc <= GATE else None)
        if all(d is not None for d in dau_pred) and dau_pred[0] != dau_pred[1]:
            tot["both_daughters"] += 1
            kids = set(psucc.get(pdv_id, []))
            hit = sum(1 for d in dau_pred if d in kids)
            if hit == 2:
                tot["linked_both"] += 1
            elif hit == 1:
                tot["linked_one"] += 1
                # fixable if the unlinked daughter has no parent OR reassignable, and continues
                other = [d for d in dau_pred if d not in kids][0]
                continues = len(psucc.get(other, [])) > 0
                if continues:
                    tot["fixable"] += 1
            else:
                tot["linked_none"] += 1

print("GT divisions in golden-12:", tot["n"])
print(f"  divider detected in pred (<=7um)     : {tot['div_hit']}")
print(f"  BOTH daughters present as pred nodes  : {tot['both_daughters']}")
print(f"    divider linked to BOTH daughters    : {tot['linked_both']}  (already a pred fork)")
print(f"    divider linked to ONE daughter      : {tot['linked_one']}")
print(f"    divider linked to NEITHER           : {tot['linked_none']}")
print(f"  LINKING-FIXABLE (one linked, other continues, add 2nd edge): {tot['fixable']}")
