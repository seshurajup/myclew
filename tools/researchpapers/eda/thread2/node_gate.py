"""THREAD 2 candidate B — estN-CALIBRATED NODE GATING on pilkwang's FIXED detections (CPU only).

Question: L2 says over-prediction owns the low-scoring tail (07e24132 @ 2.4x estN). Can we recover it
by dropping SPURIOUS excess nodes down toward estN, purely post-proc (no detector retrain)?

Nodes have no continuous confidence (only ILP `solution` bool), so we rank each node by its EDGE
SUPPORT = max incident edge_prob (a node held only by weak/no edges is the spurious-detection
candidate). Two interventions on the fixed solution graph:
  Variant G_estN : drop lowest-support nodes until t_pred ~= estN (calibrated to the count target).
  Variant G_thr  : drop nodes whose max incident edge_prob < thr (sweep), keep their surviving edges.
Dropping a node also drops its incident edges. Re-score OFFICIAL on golden-12, weighted.

research/cellmot_venv/bin/python eda/thread2/node_gate.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
ENSEMBLE = ROOT / "learning/ensemble_work"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ENSEMBLE))
from src import io, metric  # noqa: E402
from src.config import Config  # noqa: E402
import pilk_post as P  # noqa: E402

SRC = Config()
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
GEFF_DIR = ROOT / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
OUT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/tools/researchpapers/eda/thread2")
GOLDEN12 = {"44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"}
THRESHOLDS = [0.0, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8]


def load_solution(path):
    g = P.graph_from_geff(path)
    nd = g.node_attrs().to_pandas(); ed = g.edge_attrs().to_pandas()
    if "solution" in nd.columns: nd = nd[nd["solution"] == True]  # noqa: E712
    if "solution" in ed.columns: ed = ed[ed["solution"] == True]  # noqa: E712
    nodes = nd[["node_id", "t", "z", "y", "x"]].astype(
        {"node_id": int, "t": int, "z": float, "y": float, "x": float})
    edges = ed[["source_id", "target_id", "edge_prob"]].astype(
        {"source_id": int, "target_id": int, "edge_prob": float})
    return nodes, edges


def node_support(nodes, edges):
    """max incident edge_prob per node; 0 if isolated."""
    sup = {int(n): 0.0 for n in nodes.node_id}
    for s, t, p in edges[["source_id", "target_id", "edge_prob"]].to_numpy():
        s, t = int(s), int(t)
        if s in sup and p > sup[s]: sup[s] = p
        if t in sup and p > sup[t]: sup[t] = p
    return pd.Series(sup)


def rescore(nodes, edges, gn, ge, estN):
    kept_ids = set(nodes.node_id)
    e = edges[edges.source_id.isin(kept_ids) & edges.target_id.isin(kept_ids)]
    c = metric.official_counts(gn, ge, nodes, e[["source_id", "target_id"]],
                               SRC.SCALE, SRC.MATCH_GATE_UM, t_true=estN)
    return c


def main():
    geffs = sorted(g for g in GEFF_DIR.glob("*.geff")
                   if g.name.replace(".zarr.geff", "").replace(".geff", "") in GOLDEN12)
    base = {}
    for g in geffs:
        ds = g.name.replace(".zarr.geff", "").replace(".geff", "")
        gn, ge = io.read_geff(TRAIN / f"{ds}.geff")
        estN = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
        nodes, edges = load_solution(g)
        sup = node_support(nodes, edges)
        base[ds] = (gn, ge, estN, nodes, edges, sup)

    def agg(per):
        pdf = pd.DataFrame(per); W = pdf.w.sum()
        cp = 1 - 0.1 * (pdf.t_pred - pdf.t_true) / pdf.t_true
        return dict(adj=(pdf.w * pdf.adj_jaccard).sum() / W,
                    edge_r=pdf.edge_tp.sum() / (pdf.edge_tp.sum() + pdf.edge_fn.sum()),
                    edge_p=pdf.edge_tp.sum() / max(1, pdf.edge_tp.sum() + pdf.edge_fp.sum()),
                    cpen=(pdf.w * cp).sum() / W, tpred=int(pdf.t_pred.sum()))

    rows = []
    # baseline
    per0 = [dict(**rescore(nd, ed, gn, ge, eN), dataset=ds)
            for ds, (gn, ge, eN, nd, ed, sup) in base.items()]
    b = agg(per0)
    rows.append(dict(variant="baseline", param=0.0, **b))
    print(f"baseline: adjJ={b['adj']:.4f} edgeR={b['edge_r']:.3f} edgeP={b['edge_p']:.3f} "
          f"cpen={b['cpen']:.3f} Tpred={b['tpred']}", flush=True)

    # G_estN: drop lowest-support nodes until t_pred <= estN
    per = []
    for ds, (gn, ge, eN, nd, ed, sup) in base.items():
        target = int(round(eN)) if eN else len(nd)
        if len(nd) > target:
            keep = sup.sort_values(ascending=False).head(target).index
            nd2 = nd[nd.node_id.isin(set(int(i) for i in keep))]
        else:
            nd2 = nd
        per.append(dict(**rescore(nd2, ed, gn, ge, eN), dataset=ds))
    g = agg(per)
    rows.append(dict(variant="G_estN", param=1.0, **g))
    print(f"G_estN (calibrate to estN): adjJ={g['adj']:.4f} (Δ={g['adj']-b['adj']:+.4f}) "
          f"edgeR={g['edge_r']:.3f} edgeP={g['edge_p']:.3f} cpen={g['cpen']:.3f} Tpred={g['tpred']}", flush=True)

    # G_thr: drop nodes with support < thr
    for thr in THRESHOLDS:
        per = []
        for ds, (gn, ge, eN, nd, ed, sup) in base.items():
            keep = set(int(i) for i in sup[sup >= thr].index)
            nd2 = nd[nd.node_id.isin(keep)]
            per.append(dict(**rescore(nd2, ed, gn, ge, eN), dataset=ds))
        gg = agg(per)
        rows.append(dict(variant="G_thr", param=thr, **gg))
        print(f"G_thr>={thr:.2f}: adjJ={gg['adj']:.4f} (Δ={gg['adj']-b['adj']:+.4f}) "
              f"edgeR={gg['edge_r']:.3f} cpen={gg['cpen']:.3f} Tpred={gg['tpred']}", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "node_gate.csv", index=False)
    best = res.loc[res.adj.idxmax()]
    print(f"\nBEST: {best.variant} param={best.param} adjJ={best.adj:.4f} "
          f"(Δ vs baseline {best.adj-b['adj']:+.4f})", flush=True)
    print("[done] -> eda/thread2/node_gate.csv", flush=True)


if __name__ == "__main__":
    main()
