"""Fit the sister/division classifier — runs under cellmot_venv (has geff/zarr deps + src). Prints JSON."""
import sys, os, glob, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import io
import pandas as pd
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(COMP, "input", "biohub-cell-tracking-during-development", "train")
VOX = (1.625, 0.40625, 0.40625)
def _um(a, b): return math.sqrt(sum((VOX[i]*(a[i]-b[i]))**2 for i in range(3)))
rows = []
for g in sorted(glob.glob(os.path.join(TRAIN, "*.geff"))):
    try: nodes, edges = io.read_geff(g)
    except Exception: continue
    pos = {r.node_id: (r.z, r.y, r.x, r.t) for r in nodes.itertuples()}
    kids = {}
    for e in edges.itertuples(): kids.setdefault(e.source_id, []).append(e.target_id)
    by_t = {}
    for nid, (z, y, x, t) in pos.items(): by_t.setdefault(t, []).append(nid)
    for m, ch in kids.items():
        if m not in pos or not ch: continue
        d1 = ch[0]
        if d1 not in pos: continue
        mt = pos[m][3]; real = set(ch)
        cands = [n for n in by_t.get(mt+1, []) if n in pos and n != d1 and _um(pos[n], pos[m]) < 12]
        for c in cands:
            rows.append({"parent_daughter_um": _um(pos[d1], pos[m]), "sister_sister_um": _um(pos[d1], pos[c]),
                         "existing_child_um": _um(pos[d1], pos[m]), "cand_parent_um": _um(pos[c], pos[m]),
                         "local_density": len(cands), "frame_gap": 1,
                         "label": 1 if (len(real) >= 2 and c in real) else 0})
df = pd.DataFrame(rows)
npos, nneg = int((df.label==1).sum()), int((df.label==0).sum())
if npos < 10:
    print(json.dumps({"ok": False, "pos": npos, "neg": nneg})); sys.exit(0)
feats = ["parent_daughter_um","sister_sister_um","existing_child_um","cand_parent_um","local_density","frame_gap"]
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
X, y = df[feats], df.label
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
try:
    import xgboost as xgb
    clf = xgb.XGBClassifier(tree_method="hist", n_estimators=200, max_depth=4,
                            scale_pos_weight=max(1, nneg/max(1,npos))); kind="xgboost"
except Exception:
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=500, class_weight="balanced"); kind="logistic"
clf.fit(Xtr, ytr)
p = clf.predict_proba(Xte)[:,1]
os.makedirs(os.path.join(COMP,"models"), exist_ok=True)
import pickle; pickle.dump(clf, open(os.path.join(COMP,"models","div_clf.pkl"),"wb"))
print(json.dumps({"ok": True, "kind": kind, "pos": npos, "neg": nneg,
    "auc": round(float(roc_auc_score(yte, p)),3), "precision": round(float(precision_score(yte, p>0.8, zero_division=0)),3),
    "recall": round(float(recall_score(yte, p>0.8, zero_division=0)),3)}))
