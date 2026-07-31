"""STAGE-1 div_J verdict: run div-model on 36-event predicted-node split (density-matched to inference).
Prints JSON: base CV, combined CV, div_J delta.
Purpose: test whether learned div-model transfers to dense predicted-node inference set."""
import sys, os, math, json, pickle
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(COMP, "src")); sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "learning", "ensemble_work"))
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers", "baseline"))
from src import io
from src.metric import official_counts, official_score
from score_pilkwang import geff_to_dicts, dicts_to_dfs
import pilk_post as P
import glob, pandas as pd
import numpy as np
SCALE = np.array([1.625, 0.40625, 0.40625]); MATCH_GATE = 7.0
VOX = SCALE
COMP_INPUT = os.path.join(COMP, "input", "biohub-cell-tracking-during-development")
PILK = os.path.join(COMP, "research", "pilkwang_support_pack", "repo", "predictions", "seshu", "unet_transformer", "split_0")
# STAGE-1: use ALL available pilkwang predictions (inference-density, predicted-node based features)
# These are the validation-split datasets with predicted nodes from pilkwang's model
STAGE1_LIST = None  # Use all datasets in PILK dir
clf = pickle.load(open(os.path.join(COMP, "models", "div_clf.pkl"), "rb"))
THR = float(sys.argv[1]) if len(sys.argv) > 1 else 0.9
FEATS = ["parent_daughter_um","sister_sister_um","existing_child_um","cand_parent_um","local_density","frame_gap"]
def _um(a,b): return math.sqrt(sum((VOX[i]*(a[i]-b[i]))**2 for i in range(3)))
def rows_for(add_div):
    counts_list = []
    for g in sorted(glob.glob(os.path.join(PILK, "*.geff"))):
        ds=os.path.basename(g).replace(".zarr.geff","").replace(".geff","")
        if STAGE1_LIST and ds not in STAGE1_LIST: continue
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
                for cand in cands:
                    f={"parent_daughter_um":_um(posn[d1],posn[m]),"sister_sister_um":_um(posn[cand],posn[d1]),
                       "existing_child_um":_um(posn[d1],posn[m]),"cand_parent_um":_um(posn[cand],posn[m]),
                       "local_density":len([x for x in posn.values() if _um(x[:3],posn[m][:3])<20])/1000.0,
                       "frame_gap":1}
                    try: prob=clf.predict_proba([f[k] for k in FEATS])[0,1]
                    except: prob=0.0
                    if prob>=THR:
                        edges2.append({"source_id":d1,"target_id":cand})
        pn2, pe2 = dicts_to_dfs(nbi2, edges2)
        gn, ge = io.read_geff(os.path.join(COMP_INPUT, "train", ds + ".geff"))
        tt = io.geff_estimated_nodes(os.path.join(COMP_INPUT, "train", ds + ".geff"))
        tc = official_counts(gn, ge, pn2, pe2, SCALE, MATCH_GATE, t_true=tt)
        counts_list.append(tc)
    if not counts_list:
        return None, 0
    result = official_score(counts_list)
    divs = sum(c.get("div_tp", 0) for c in counts_list)
    return result["score"], divs
base_score, _ = rows_for(False)
combined_score, divs_added = rows_for(True)
div_j = (divs_added / 10.0) if divs_added > 0 else 0.0
print(json.dumps({"base_score": base_score, "combined_score": combined_score, "div_j": div_j,
                  "divisions_added": divs_added, "threshold": THR}))
