"""At the 4 linking-fixable GT dividers, does the GEOMETRIC pick == the real daughter?
If a closer decoy wins, pure geometry can't isolate divisions at this density."""
import sys
from pathlib import Path
import numpy as np
import tracksdata as td
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning/ensemble_work"))
import pilk_post as P
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
BASE = ROOT / "research/official_repo/predictions/seshu/nodiv/split_0"
VOX = np.array([1.625, 0.40625, 0.40625]); GATE = 7.0
K = td.DEFAULT_ATTR_KEYS
G12 = ["44b6_0113de3b","44b6_0b24845f","44b6_0c582fdc","44b6_0db75fae","44b6_12dfb391","44b6_144b256d",
       "6bba_05b6850b","6bba_05db0fb1","6bba_062c8d37","6bba_07477033","6bba_07e24132","6bba_085bf656"]

def load(path):
    g=P.graph_from_geff(str(path))
    nodes={int(r[K.NODE_ID]):(int(r["t"]),np.array([r["z"],r["y"],r["x"]],float)) for r in g.node_attrs().iter_rows(named=True)}
    succ,preds={},{}
    for r in g.edge_attrs().iter_rows(named=True):
        s,t=int(r[K.EDGE_SOURCE]),int(r[K.EDGE_TARGET]); succ.setdefault(s,[]).append(t); preds.setdefault(t,[]).append(s)
    return nodes,succ,preds
def nearest(nodes,t,pos,exclude=()):
    best,bd=None,1e9
    for nid,(tt,p) in nodes.items():
        if tt==t and nid not in exclude:
            d=np.linalg.norm((p-pos)*VOX)
            if d<bd: bd,best=d,nid
    return best,bd

for ds in G12:
    gn,gsucc,_=load(TRAIN/f"{ds}.geff")
    pn,psucc,ppreds=load(BASE/f"{ds}.geff")
    for dv in [n for n,ch in gsucc.items() if len(ch)>=2]:
        tdv,pdv=gn[dv]; pdv_id,dd=nearest(pn,tdv,pdv)
        if dd>GATE: continue
        dau=[]
        for c in gsucc[dv][:2]:
            tc,pc=gn[c]; pid,dc=nearest(pn,tc,pc); dau.append((pid if dc<=GATE else None))
        if not(all(d is not None for d in dau) and dau[0]!=dau[1]): continue
        kids=set(psucc.get(pdv_id,[]))
        linked=[d for d in dau if d in kids]; unl=[d for d in dau if d not in kids]
        if len(linked)!=1 or len(unl)!=1: continue
        real_c2=unl[0]; c1=linked[0]
        tp,pp=pn[pdv_id]
        # geometric nearest continuing node at t+1 (excluding c1)
        cont=[n for n in pn if pn[n][0]==tp+1 and len(psucc.get(n,[]))>0 and n!=c1]
        cont_sorted=sorted(cont,key=lambda n:np.linalg.norm((pn[n][1]-pp)*VOX))[:5]
        real_rank=[i for i,n in enumerate(cont_sorted) if n==real_c2]
        dp_real=np.linalg.norm((pn[real_c2][1]-pp)*VOX)
        dp_near=np.linalg.norm((pn[cont_sorted[0]][1]-pp)*VOX) if cont_sorted else -1
        print(f"{ds[:12]} div: real_daughter dist={dp_real:.1f}um  nearest_continuing={dp_near:.1f}um  "
              f"real_rank_among_continuing={'#'+str(real_rank[0]+1) if real_rank else '>5'}  real_c2_parented={len(ppreds.get(real_c2,[]))>0}")
