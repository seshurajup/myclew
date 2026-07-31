"""Score the BASE pilkwang post-processing on golden-12 CV — params come from BIOHUB_* env vars
(read by pilk_post at import). NO div-model, NO training needed. Prints JSON {score, adjE, n}.

Used by the fleet combo-search agent: it sets BIOHUB_DET_THRESHOLD / BIOHUB_GAP_CLOSE_UM /
BIOHUB_OUTPUT_MIN_TRACK_LEN / BIOHUB_OUTPUT_SAFE_DIVISIONS / BIOHUB_SAFE_DIV_MAX_UM, runs this as a
fresh subprocess per combination, and reads the golden-12 score back. Fast: post-proc + score only."""
import sys, os, json
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(COMP, "src")); sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "learning", "ensemble_work"))
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers", "baseline"))
from src import io
from src.metric import official_counts, official_score
from score_pilkwang import geff_to_dicts, dicts_to_dfs, GOLDEN12
import pilk_post as P
import glob
import numpy as np
SCALE = np.array([1.625, 0.40625, 0.40625]); MATCH_GATE = 7.0
TRAIN = os.path.join(COMP, "input", "biohub-cell-tracking-during-development", "train")
PILK = os.path.join(COMP, "research", "pilkwang_support_pack", "repo", "predictions", "seshu", "unet_transformer", "split_0")


# optional fast screen: argv[1] = max datasets (embryo-balanced subset); 0/absent = full golden-12
_LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def _subset():
    ds_all = sorted(GOLDEN12)
    if _LIMIT <= 0 or _LIMIT >= len(ds_all):
        return set(ds_all)
    by_emb = {}
    for d in ds_all:
        by_emb.setdefault(d.split("_")[0], []).append(d)
    picked, i = [], 0
    while len(picked) < _LIMIT:                       # round-robin across embryos → balanced screen
        for emb in sorted(by_emb):
            if i < len(by_emb[emb]) and len(picked) < _LIMIT:
                picked.append(by_emb[emb][i])
        i += 1
        if i > max(len(v) for v in by_emb.values()):
            break
    return set(picked)


SUBSET = _subset()


def score():
    out = []
    todo = [g for g in sorted(glob.glob(os.path.join(PILK, "*.geff")))
            if os.path.basename(g).replace(".zarr.geff", "").replace(".geff", "") in SUBSET]
    for i, g in enumerate(todo, 1):
        ds = os.path.basename(g).replace(".zarr.geff", "").replace(".geff", "")
        print(f"PROGRESS {i}/{len(todo)} {ds}", file=sys.stderr, flush=True)   # live progress for the agent
        nbi, raw = geff_to_dicts(g)
        nbi2, edges2, _ = P.filter_output_graph(dict(nbi), list(raw), dataset=ds)  # env-driven params
        pn2, pe2 = dicts_to_dfs(nbi2, edges2)
        gn, ge = io.read_geff(os.path.join(TRAIN, ds + ".geff"))
        tt = io.geff_estimated_nodes(os.path.join(TRAIN, ds + ".geff"))
        out.append(official_counts(gn, ge, pn2, pe2, SCALE, MATCH_GATE, t_true=tt))
    return out


rows = score()
s = official_score(rows)
print(json.dumps({"score": round(s["score"], 4), "adjE": round(s["adj_edge_jaccard"], 4), "n": len(rows)}))
