"""DIAGNOSTIC: at the linking-fixable GT dividers, does the raw-image mitotic
signature (parent brightness + daughter intensity/symmetry) separate the REAL
2nd daughter from the geometric decoys? If not, image rescue can't work."""
import sys
from pathlib import Path
import numpy as np
import zarr
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
    g = P.graph_from_geff(str(path))
    nodes = {int(r[K.NODE_ID]): (int(r["t"]), np.array([r["z"], r["y"], r["x"]], float))
             for r in g.node_attrs().iter_rows(named=True)}
    succ, preds = {}, {}
    for r in g.edge_attrs().iter_rows(named=True):
        s, t = int(r[K.EDGE_SOURCE]), int(r[K.EDGE_TARGET])
        succ.setdefault(s, []).append(t); preds.setdefault(t, []).append(s)
    return nodes, succ, preds


def nearest(nodes, t, pos, exclude=()):
    best, bd = None, 1e9
    for nid, (tt, p) in nodes.items():
        if tt == t and nid not in exclude:
            d = np.linalg.norm((p - pos) * VOX)
            if d < bd: bd, best = d, nid
    return best, bd


def blob_feat(vol, zyx, rz=1, ry=4, rx=4):
    """Local intensity feature at a coordinate: peak & mean in a small window,
    and a compactness/brightness ratio vs the window's low baseline."""
    z, y, x = [int(round(v)) for v in zyx]
    Z, Y, X = vol.shape
    zl, zh = max(0, z-rz), min(Z, z+rz+1)
    yl, yh = max(0, y-ry), min(Y, y+ry+1)
    xl, xh = max(0, x-rx), min(X, x+rx+1)
    patch = vol[zl:zh, yl:yh, xl:xh].astype(np.float32)
    if patch.size == 0:
        return dict(peak=0., mean=0., bright=0.)
    base = np.percentile(patch, 20)
    peak = patch.max()
    mean = patch.mean()
    bright = (peak - base)                 # contrast above local floor
    return dict(peak=float(peak), mean=float(mean), bright=float(bright))


def main():
    rows = []
    for ds in G12:
        gn, gsucc, _ = load(TRAIN / f"{ds}.geff")
        pn, psucc, ppreds = load(BASE / f"{ds}.geff")
        zroot = zarr.open(str(TRAIN / f"{ds}.zarr/0"), mode="r")
        for dv in [n for n, ch in gsucc.items() if len(ch) >= 2]:
            tdv, pdv = gn[dv]; pdv_id, dd = nearest(pn, tdv, pdv)
            if dd > GATE: continue
            dau = []
            for c in gsucc[dv][:2]:
                tc, pc = gn[c]; pid, dc = nearest(pn, tc, pc); dau.append(pid if dc <= GATE else None)
            if not (all(d is not None for d in dau) and dau[0] != dau[1]): continue
            kids = set(psucc.get(pdv_id, []))
            linked = [d for d in dau if d in kids]; unl = [d for d in dau if d not in kids]
            if len(linked) != 1 or len(unl) != 1: continue
            real_c2, c1 = unl[0], linked[0]
            tp, pp = pn[pdv_id]
            volp = np.asarray(zroot[tp]); voln = np.asarray(zroot[tp+1])
            # candidate 2nd daughters = continuing nodes at t+1 within 14um, excluding c1
            cont = [n for n in pn if pn[n][0] == tp+1 and len(psucc.get(n, [])) > 0 and n != c1
                    and np.linalg.norm((pn[n][1]-pp)*VOX) <= 14.0]
            fp = blob_feat(volp, pp)                       # parent (mitotic?) in frame t
            fc1 = blob_feat(voln, pn[c1][1])               # daughter1
            def cand_score(n):
                fc = blob_feat(voln, pn[n][1])
                d_p = np.linalg.norm((pn[n][1]-pp)*VOX)
                d_c1 = np.linalg.norm((pn[n][1]-pn[c1][1])*VOX)
                sym_dist = abs(d_p - np.linalg.norm((pn[c1][1]-pp)*VOX))    # symmetric split
                sym_int = abs(fc["bright"]-fc1["bright"])/(fc1["bright"]+1e-6)  # similar brightness
                return dict(n=n, d_p=d_p, d_c1=d_c1, sym_dist=sym_dist, sym_int=sym_int, bright=fc["bright"])
            cands = [cand_score(n) for n in cont]
            real = next((c for c in cands if c["n"] == real_c2), None)
            if real is None: continue
            # rank real among cands by: symmetry (low sym_dist + low sym_int) — lower=better
            for c in cands:
                c["symscore"] = c["sym_dist"] + 8.0*c["sym_int"]   # combined symmetry cost
            ranked = sorted(cands, key=lambda c: c["symscore"])
            real_rank = [i for i, c in enumerate(ranked) if c["n"] == real_c2][0]
            geo_rank = [i for i, c in enumerate(sorted(cands, key=lambda c: c["d_p"])) if c["n"] == real_c2][0]
            rows.append(dict(ds=ds[:12], parent_bright=fp["bright"], n_cand=len(cands),
                             real_symscore=round(real["symscore"],1),
                             best_decoy_symscore=round(min([c["symscore"] for c in cands if c["n"]!=real_c2], default=9e9),1),
                             real_symrank=f"#{real_rank+1}", geo_rank=f"#{geo_rank+1}"))
    print(f"{'dataset':13} {'parentBright':>12} {'nCand':>6} {'realSym':>8} {'bestDecoySym':>13} {'symRank':>8} {'geoRank':>8}")
    wins = 0
    for r in rows:
        better = r["real_symscore"] < r["best_decoy_symscore"]
        wins += better
        print(f"{r['ds']:13} {r['parent_bright']:12.0f} {r['n_cand']:6} {r['real_symscore']:8.1f} "
              f"{r['best_decoy_symscore']:13.1f} {r['real_symrank']:>8} {r['geo_rank']:>8}  {'<-- SYM WINS' if better else ''}")
    print(f"\nSYMMETRY separates real daughter from decoys in {wins}/{len(rows)} fixable dividers "
          f"(geometry alone: real is rarely rank #1).")


if __name__ == "__main__":
    main()
