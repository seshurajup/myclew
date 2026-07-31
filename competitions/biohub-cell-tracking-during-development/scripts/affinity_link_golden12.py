"""affinity-link — apply the trained division model to golden-12 and RE-SCORE (prove the div_J lever).

Loads the gnn-link division head, computes the same features on each golden-12 dataset's POST-PROC nodes
(self-normalised per embryo so they match the training scale), and at high-confidence nodes ADDS a
division edge (a 2nd outgoing edge to the nearest unclaimed next-frame node). Re-scores with the official
metric and prints base vs augmented, so we see honestly whether learned divisions beat 0.8803.

argv[1] = subset size (0 = full golden-12). BIOHUB_LEARNED_DIV_THRESH / _TOPK env tune injection.
"""
import sys, os, json
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(COMP, "src")); sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "learning", "ensemble_work"))
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers", "baseline"))
import numpy as np
import torch
from torch import nn
from scipy.spatial import cKDTree
from src import io
from src.metric import official_counts, official_score
from score_pilkwang import geff_to_dicts, dicts_to_dfs, GOLDEN12
import pilk_post as P
import glob

SCALE = np.array([1.625, 0.40625, 0.40625]); MATCH_GATE = 7.0
TRAIN = os.path.join(COMP, "input", "biohub-cell-tracking-during-development", "train")
PILK = os.path.join(COMP, "research", "pilkwang_support_pack", "repo", "predictions", "seshu", "unet_transformer", "split_0")
CKPT = os.path.join(COMP, "results", "gnn_link", "gnn_link.pt")
THRESH = float(os.environ.get("BIOHUB_LEARNED_DIV_THRESH", "0.9"))
RADIUS = float(os.environ.get("BIOHUB_LEARNED_DIV_RADIUS", "6.0"))
_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def _mlp(nin, hidden, nl, out):
    layers, d = [], nin
    for _ in range(nl):
        layers += [nn.Linear(d, hidden), nn.GELU()]; d = hidden
    layers += [nn.Linear(d, out)]
    return nn.Sequential(*layers)


def _load_div():
    c = torch.load(CKPT, map_location="cpu", weights_only=False)
    net = _mlp(6, c["hidden"], c["n_layers"], 1); net.load_state_dict(c["div"]); net.eval()
    return net, c["mu"], c["sd"]


def _features(pn, radius):
    """transfer-robust sister-geometry: [d1,d2,ratio,sister_dist,symmetry,nn] — MUST match gnn_link_train.
    Self-normalised per embryo (median spacing → training scale) so features are comparable."""
    pn = pn.sort_values("t").reset_index(drop=True)
    ts = sorted(pn["t"].unique())
    feats, idx = [], []
    p0 = pn[pn["t"] == ts[0]][["z", "y", "x"]].to_numpy()
    scale = 1.0
    if len(p0) > 5:
        d, _ = cKDTree(p0).query(p0[: min(300, len(p0))], k=2)
        med = np.median(d[:, 1])
        scale = 1.3 / max(med, 1e-6)                        # normalise median spacing to training ref ~1.3
    for t in ts:
        a = pn[pn["t"] == t]; b = pn[pn["t"] == t + 1]
        if len(a) < 3:
            continue
        pa = a[["z", "y", "x"]].to_numpy() * scale
        pb = b[["z", "y", "x"]].to_numpy() * scale if len(b) else np.zeros((0, 3))
        ta = cKDTree(pa); tb = cKDTree(pb) if len(pb) >= 2 else None
        nids = a["node_id"].to_numpy() if "node_id" in a.columns else a.index.to_numpy()
        if tb is not None:
            dd, ii = tb.query(pa, k=2)                       # BATCH: all nodes at once
            d1 = dd[:, 0]; d2 = dd[:, 1]
            sister = np.linalg.norm(pb[ii[:, 0]] - pb[ii[:, 1]], axis=1)
            ratio = d2 / np.maximum(d1, 1e-3); symm = np.abs(d1 - d2) / np.maximum(d1 + d2, 1e-3)
        else:
            d1 = d2 = sister = np.full(len(a), 10.0); ratio = np.ones(len(a)); symm = np.zeros(len(a))
        nnq = ta.query(pa, k=min(2, len(a)))[0]
        nnv = nnq[:, -1] if nnq.ndim > 1 else np.zeros(len(a))
        for i in range(len(a)):
            feats.append([float(d1[i]), float(d2[i]), float(ratio[i]), float(sister[i]), float(symm[i]), float(nnv[i])])
            idx.append(int(nids[i]))
    return np.array(feats, dtype="float32"), idx


def main():
    net, mu, sd = _load_div()
    ds_all = sorted(GOLDEN12)
    if _LIMIT > 0:
        ds_all = ds_all[:_LIMIT]
    base_rows, aug_rows = [], []
    total_added = 0
    for gi, ds in enumerate(ds_all, 1):
        g = os.path.join(PILK, ds + ".geff")
        if not os.path.exists(g):
            g2 = glob.glob(os.path.join(PILK, ds + "*.geff"))
            if not g2:
                continue
            g = g2[0]
        print(f"PROGRESS {gi}/{len(ds_all)} {ds}", file=sys.stderr, flush=True)
        nbi, raw = geff_to_dicts(g)
        nbi2, edges2, _ = P.filter_output_graph(dict(nbi), list(raw), dataset=ds)
        pn2, pe2 = dicts_to_dfs(nbi2, edges2)
        gn, ge = io.read_geff(os.path.join(TRAIN, ds + ".geff"))
        tt = io.geff_estimated_nodes(os.path.join(TRAIN, ds + ".geff"))
        base_rows.append(official_counts(gn, ge, pn2, pe2, SCALE, MATCH_GATE, t_true=tt))

        # apply division model
        X, idx = _features(pn2, RADIUS)
        aug_edges = list(map(tuple, pe2[["source_id", "target_id"]].to_numpy())) if len(pe2) else []
        if len(X):
            with torch.no_grad():
                prob = torch.sigmoid(net(torch.tensor((X - mu) / sd))).numpy().ravel()
            # fast lookups (no iterrows): node_id -> (t,z,y,x) and per-frame node arrays
            nid_arr = pn2["node_id"].to_numpy(); tpos = pn2["t"].to_numpy()
            zyx = pn2[["z", "y", "x"]].to_numpy()
            pos = {int(nid_arr[i]): (int(tpos[i]), zyx[i]) for i in range(len(pn2))}
            frame_nodes = {}
            for i in range(len(pn2)):
                frame_nodes.setdefault(int(tpos[i]), []).append(i)
            claimed = {tgt for _, tgt in aug_edges}
            edge_set = set(aug_edges)
            order = np.argsort(-prob)
            for k in order:
                if prob[k] < THRESH:
                    break
                nid = idx[k]
                if nid not in pos:
                    continue
                t, p = pos[nid]
                rows_next = frame_nodes.get(t + 1, [])
                if not rows_next:
                    continue
                cp = zyx[rows_next] * SCALE
                d = np.linalg.norm(cp - p * SCALE, axis=1)
                for j in np.argsort(d):
                    if d[j] > 10.0:
                        break
                    cid = int(nid_arr[rows_next[j]])
                    if cid not in claimed and (nid, cid) not in edge_set:
                        aug_edges.append((nid, cid)); edge_set.add((nid, cid))
                        claimed.add(cid); total_added += 1; break
        import pandas as pd
        pe_aug = pd.DataFrame(aug_edges, columns=["source_id", "target_id"]) if aug_edges else pe2
        aug_rows.append(official_counts(gn, ge, pn2, pe_aug, SCALE, MATCH_GATE, t_true=tt))

    sb = official_score(base_rows); sa = official_score(aug_rows)
    print(json.dumps({"base": round(sb["score"], 4), "base_adjE": round(sb["adj_edge_jaccard"], 4),
                      "augmented": round(sa["score"], 4), "aug_adjE": round(sa["adj_edge_jaccard"], 4),
                      "delta": round(sa["score"] - sb["score"], 4), "divisions_added": total_added,
                      "thresh": THRESH, "n": len(base_rows)}))


if __name__ == "__main__":
    main()
