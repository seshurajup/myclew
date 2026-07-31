"""XAI on the trained division model — SEE what it keys on before trusting it (permutation importance +
integrated gradients on the 5 features), and measure the train→apply distribution shift that makes it flood FP."""
import torch, numpy as np, pandas as pd
from torch import nn
from sklearn.metrics import average_precision_score
from scipy.spatial import cKDTree
FEAT = ["density_t", "density_t+1", "count_change", "nn_dist", "z"]

c = torch.load("results/gnn_link/gnn_link.pt", map_location="cpu", weights_only=False)
def mlp(nin, h, nl, out):
    L, d = [], nin
    for _ in range(nl):
        L += [nn.Linear(d, h), nn.GELU()]; d = h
    L += [nn.Linear(d, out)]; return nn.Sequential(*L)
net = mlp(5, c["hidden"], c["n_layers"], 1); net.load_state_dict(c["div"]); net.eval()
mu, sd = c["mu"], c["sd"]

df = pd.read_parquet("results/flow_gt/flow_node_gt_clean.parquet",
                     columns=["embryo", "t", "z", "y", "x", "is_division"])
def feats(sub, radius=6.0, frames=30):
    ts = sorted(sub["t"].unique()); pick = ts[::max(1, len(ts)//frames)][:frames]; X, Y = [], []
    for t in pick:
        a = sub[sub.t == t]; b = sub[sub.t == t+1]
        if len(a) < 8 or len(b) < 8: continue
        pa = a[["z","y","x"]].to_numpy(); pb = b[["z","y","x"]].to_numpy()
        ta = cKDTree(pa); tb = cKDTree(pb)
        for i in range(len(a)):
            nA = len(ta.query_ball_point(pa[i], radius))-1; nB = len(tb.query_ball_point(pa[i], radius))
            nn = float(ta.query(pa[i], k=2)[0][1]) if len(a) > 1 else 0.0
            X.append([nA, nB, nB-nA, nn, float(pa[i][0])]); Y.append(int(a.is_division.iloc[i]))
    return np.array(X, dtype="float32"), np.array(Y, dtype="float32")

X, Y = feats(df[df.embryo == "ZSNS005"])
Xn = (X - mu) / sd
with torch.no_grad(): p = torch.sigmoid(net(torch.tensor(Xn))).numpy().ravel()
base_ap = average_precision_score(Y, p)
print(f"=== PERMUTATION IMPORTANCE (division model, ZSNS005 test AP {base_ap:.3f}) ===")
rng = np.random.RandomState(0)
for i, f in enumerate(FEAT):
    Xp = Xn.copy(); Xp[:, i] = rng.permutation(Xp[:, i])
    with torch.no_grad(): pp = torch.sigmoid(net(torch.tensor(Xp))).numpy().ravel()
    ap = average_precision_score(Y, pp)
    flag = "  <== RELIES ON THIS" if base_ap - ap > 0.02 else ""
    print(f"  shuffle {f:13s}: AP {ap:.3f}  drop {base_ap-ap:+.3f}{flag}")

# integrated gradients (attribution of each feature for the positive class, averaged over true divisions)
pos = Xn[Y == 1]
if len(pos):
    baseline = np.zeros((1, 5), dtype="float32")
    steps = 32
    ig = np.zeros(5)
    for a in pos[:200]:
        acc = np.zeros(5)
        for s in range(1, steps+1):
            xin = torch.tensor(baseline + (a - baseline) * s/steps, requires_grad=True)
            out = net(xin).sum(); out.backward()
            acc += xin.grad.numpy().ravel()
        ig += (a - baseline).ravel() * acc / steps
    ig /= len(pos[:200])
    print("\n=== INTEGRATED GRADIENTS (avg attribution on TRUE divisions) ===")
    for f, v in sorted(zip(FEAT, ig), key=lambda kv: -abs(kv[1])):
        print(f"  {f:13s}: {v:+.3f}")

print("\n=== WHY IT FLOODS FP: the division signal it learned ===")
print(f"  true-div count_change mean={X[Y==1,2].mean():+.2f}  vs  non-div mean={X[Y==0,2].mean():+.2f}")
print(f"  true-div density_t   mean={X[Y==1,0].mean():.1f}   vs  non-div mean={X[Y==0,0].mean():.1f}")
print(f"  → if it keys on density/count_change, ANY dense region on golden-12 triggers a false division.")
