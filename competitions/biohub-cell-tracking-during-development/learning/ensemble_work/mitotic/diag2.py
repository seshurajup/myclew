"""Test sister-proximity + combined discriminators for the real 2nd daughter."""
import sys
from pathlib import Path
import numpy as np, zarr, tracksdata as td
ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT / "learning/ensemble_work"))
import pilk_post as P
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
BASE = ROOT / "research/official_repo/predictions/seshu/nodiv/split_0"
VOX = np.array([1.625,0.40625,0.40625]); GATE=7.0; K=td.DEFAULT_ATTR_KEYS
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
res=[]
for ds in G12:
    gn,gsucc,_=load(TRAIN/f"{ds}.geff"); pn,psucc,ppreds=load(BASE/f"{ds}.geff")
    for dv in [n for n,ch in gsucc.items() if len(ch)>=2]:
        tdv,pdv=gn[dv]; pdv_id,dd=nearest(pn,tdv,pdv)
        if dd>GATE: continue
        dau=[]
        for c in gsucc[dv][:2]:
            tc,pc=gn[c]; pid,dc=nearest(pn,tc,pc); dau.append(pid if dc<=GATE else None)
        if not(all(d is not None for d in dau) and dau[0]!=dau[1]): continue
        kids=set(psucc.get(pdv_id,[])); linked=[d for d in dau if d in kids]; unl=[d for d in dau if d not in kids]
        if len(linked)!=1 or len(unl)!=1: continue
        real_c2,c1=unl[0],linked[0]; tp,pp=pn[pdv_id]
        cont=[n for n in pn if pn[n][0]==tp+1 and len(psucc.get(n,[]))>0 and n!=c1 and np.linalg.norm((pn[n][1]-pp)*VOX)<=14.0]
        # rank real_c2 by distance to c1 (sister proximity), and by dist to parent
        by_sis=sorted(cont,key=lambda n:np.linalg.norm((pn[n][1]-pn[c1][1])*VOX))
        by_par=sorted(cont,key=lambda n:np.linalg.norm((pn[n][1]-pp)*VOX))
        sis_rank=by_sis.index(real_c2)+1 if real_c2 in by_sis else 99
        par_rank=by_par.index(real_c2)+1 if real_c2 in by_par else 99
        d_sis=np.linalg.norm((pn[real_c2][1]-pn[c1][1])*VOX)
        res.append((ds[:12],len(cont),sis_rank,par_rank,d_sis))
print(f"{'dataset':13}{'nCand':>6}{'sisterRank':>11}{'parentRank':>11}{'sisterDist':>11}")
sw=0
for r in res:
    print(f"{r[0]:13}{r[1]:6}{r[2]:11}{r[3]:11}{r[4]:11.1f}")
    sw+= (r[2]==1)
print(f"\nreal daughter is CLOSEST-to-sister (rank#1) in {sw}/{len(res)} — if high, sister-proximity is the rule.")
