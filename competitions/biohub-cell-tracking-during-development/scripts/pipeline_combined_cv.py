"""Combined golden-12 CV = inference base (pilkwang) + div-model divisions. Runs under cellmot_venv.
Prints JSON: base score (div_J=0) vs combined (adj_edge + 0.1*div_J after adding high-precision sisters)."""
import sys, os, math, json, pickle
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(COMP, "src")); sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "learning", "ensemble_work"))
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers", "baseline"))
from src import io
from src.metric import official_counts, official_score
from score_pilkwang import geff_to_dicts, dicts_to_dfs, GOLDEN12
import pilk_post as P
import glob, pandas as pd
import numpy as np
SCALE = np.array([1.625, 0.40625, 0.40625]); MATCH_GATE = 7.0
VOX = SCALE
TRAIN = os.path.join(COMP, "input", "biohub-cell-tracking-during-development", "train")
PILK = os.path.join(COMP, "research", "pilkwang_support_pack", "repo", "predictions", "seshu", "unet_transformer", "split_0")
clf = pickle.load(open(os.path.join(COMP, "models", "div_clf.pkl"), "rb"))
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
FEATS = ["parent_daughter_um","sister_sister_um","existing_child_um","cand_parent_um","local_density","frame_gap"]
def _um(a,b): return math.sqrt(sum((VOX[i]*(a[i]-b[i]))**2 for i in range(3)))
def rows_for(add_div):
    out=[]
    for g in sorted(glob.glob(os.path.join(PILK, "*.geff"))):
        ds=os.path.basename(g).replace(".zarr.geff","").replace(".geff","")
        if ds not in GOLDEN12: continue
        nbi, raw = geff_to_dicts(g)
        nbi2, edges2, _ = P.filter_output_graph(dict(nbi), list(raw), dataset=ds)
        if add_div:
            posn={n:(nbi2[n]["z"],nbi2[n]["y"],nbi2[n]["x"],nbi2[n]["t"]) for n in nbi2}
            kids={}
            for e in edges2: kids.setdefault(e["source_id"],[]).append(e["target_id"])
            by_t={}
            for n,(z,y,x,t) in posn.items(): by_t.setdefault(t,[]).append(n)
            for m,ch in list(kids.items()):
                if m not in posn or len(ch)!=1: continue
                d1=ch[0]
                if d1 not in posn: continue
                mt=posn[m][3]
                cands=[c for c in by_t.get(mt+1,[]) if c in posn and c!=d1 and _um(posn[c],posn[m])<12]
                for c in cands:
                    f=[[_um(posn[d1],posn[m]),_um(posn[d1],posn[c]),_um(posn[d1],posn[m]),_um(posn[c],posn[m]),len(cands),1]]
                    try: p=clf.predict_proba(pd.DataFrame(f,columns=FEATS))[0][1]
                    except Exception: p=0.0
                    if p>THR:
                        edges2.append({"source_id":m,"target_id":c}); break
        pn2, pe2 = dicts_to_dfs(nbi2, edges2)
        gn, ge = io.read_geff(os.path.join(TRAIN, ds+".geff"))
        tt = io.geff_estimated_nodes(os.path.join(TRAIN, ds+".geff"))
        out.append(official_counts(gn, ge, pn2, pe2, SCALE, MATCH_GATE, t_true=tt))
    return out
base=official_score(rows_for(False)); comb=official_score(rows_for(True))
print(json.dumps({"base_score":round(base["score"],4),"base_adjE":round(base["adj_edge_jaccard"],4),
  "combined_score":round(comb["score"],4),"combined_adjE":round(comb["adj_edge_jaccard"],4),
  "combined_divJ":round(comb["division_jaccard"],4) if comb["division_jaccard"]==comb["division_jaccard"] else 0.0,
  "delta":round(comb["score"]-base["score"],4),"thr":THR}))
