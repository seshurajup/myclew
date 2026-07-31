"""THREAD 1 — METRIC DECOMPOSITION (pure Python, NO GPU).

Decompose pilkwang's golden-12 score into failure buckets to find the BINDING CONSTRAINT.

Metric (src/metric.py, faithful): score = adj_edge_jaccard + 0.1*div_jaccard
  * node match = 1-to-1 Hungarian <= 7um / timepoint (scaled um)
  * EDGE TP needs BOTH endpoints matched to GT endpoints of a GT edge => R_edge ~ R_node^2 * Q_link
  * adj = J*(1 - 0.1*(T_pred - T_true)/T_true), T_true = estimated_number_of_nodes (over-pred penalty)
  * div_jaccard = MICRO over rare out-deg>=2 nodes, weight 0.1 (high variance)

Per embryo we compute: node_recall, node_precision, edge_recall, edge_precision,
implied link-quality Q_link = edge_recall / node_recall^2, div tp/fp/fn + realized 0.1*div_j,
and the count-ratio penalty magnitude (1 - 0.1*(Tpred-Ttrue)/Ttrue). Then correlate each bucket
vs data properties: density (cells/frame), group 44b6 (dense/late) vs 6bba (sparse/early), T_true.

We score the FULL post-processed pilkwang chain (the real ~0.87 pipeline) and, for reference,
the RAW detector+ILP output. Outputs: per-embryo CSV + charts under eda/thread1/, and the
markdown table/verdict is written by the caller into docs/research/.

Run:  research/cellmot_venv/bin/python eda/thread1/decompose.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
ENSEMBLE = ROOT / "learning/ensemble_work"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ENSEMBLE))  # pilk_post lives here

from src import io, metric, golden_cv as gcv  # noqa: E402
from src.config import Config  # noqa: E402
import pilk_post as P  # noqa: E402

SRC = Config()
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
GEFF_DIR = ROOT / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
OUT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/tools/researchpapers/eda/thread1")
OUT.mkdir(parents=True, exist_ok=True)
GOLDEN12 = {"44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"}


def geff_to_dfs_raw(path):
    """pred geff -> (nodes_by_id, raw_edges list, pred_nodes_df, pred_edges_df)."""
    graph = P.graph_from_geff(path)
    nbi = {}
    for row in graph.node_attrs().iter_rows(named=True):
        nid = int(row["node_id"])
        nbi[nid] = {"node_id": nid, "t": int(row["t"]),
                    "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}
    raw_edges = []
    for row in graph.edge_attrs().iter_rows(named=True):
        raw_edges.append({"source_id": int(row["source_id"]), "target_id": int(row["target_id"])})
    return nbi, raw_edges


def dicts_to_dfs(nbi, edges):
    pn = pd.DataFrame([{"node_id": n["node_id"], "t": n["t"], "z": n["z"], "y": n["y"], "x": n["x"]}
                      for n in nbi.values()])
    pe = pd.DataFrame([{"source_id": int(e["source_id"]), "target_id": int(e["target_id"])} for e in edges]) \
        if edges else pd.DataFrame(columns=["source_id", "target_id"])
    return pn, pe


def decompose_one(ds, pn, pe, gn, ge, estN):
    """Full per-embryo decomposition into buckets."""
    scale = SRC.SCALE
    gate = SRC.MATCH_GATE_UM
    # official counts (edge/div/adj) — authoritative
    c = metric.official_counts(gn, ge, pn, pe, scale, gate, t_true=estN)
    # node matching (recall/precision) — the SQUARED input to edge recall
    g2p = metric._match_nodes(gn, pn, scale, gate)
    n_match = len(g2p)
    n_gt, n_pred = len(gn), len(pn)
    node_r = n_match / n_gt if n_gt else float("nan")
    node_p = n_match / n_pred if n_pred else float("nan")
    # edge recall/precision from official counts
    etp, efp, efn = c["edge_tp"], c["edge_fp"], c["edge_fn"]
    edge_r = etp / (etp + efn) if (etp + efn) else float("nan")
    edge_p = etp / (etp + efp) if (etp + efp) else float("nan")
    # implied link quality: how much of the node-recall^2 ceiling the linker realizes
    node_r2 = node_r * node_r if np.isfinite(node_r) else float("nan")
    q_link = edge_r / node_r2 if node_r2 and np.isfinite(node_r2) and node_r2 > 0 else float("nan")
    # divisions
    dtp, dfp, dfn = c["div_tp"], c["div_fp"], c["div_fn"]
    div_j = dtp / (dtp + dfp + dfn) if (dtp + dfp + dfn) else float("nan")
    # count-ratio penalty: factor multiplying jaccard (>1 possible if under-pred; clamp in metric)
    t_pred, t_true = c["t_pred"], c["t_true"]
    count_pen = 1 - 0.1 * (t_pred - t_true) / t_true if t_true else float("nan")
    # frames / density
    n_frames = int(gn["t"].nunique())
    density = n_gt / n_frames if n_frames else float("nan")  # GT cells per frame
    n_gt_div = dtp + dfn  # GT divisions present
    return dict(
        dataset=ds, embryo=ds.split("_")[0],
        n_frames=n_frames, n_gt=n_gt, n_pred=n_pred, estN=t_true, density=density,
        count_ratio=(t_pred / t_true if t_true else float("nan")),
        node_r=node_r, node_p=node_p, node_r2=node_r2,
        edge_r=edge_r, edge_p=edge_p, q_link=q_link,
        edge_tp=etp, edge_fp=efp, edge_fn=efn,
        jaccard=c["jaccard"], count_pen=count_pen, adj_jaccard=c["adj_jaccard"], w=c["w"],
        n_gt_div=n_gt_div, div_tp=dtp, div_fp=dfp, div_fn=dfn, div_j=div_j,
    )


def run_variant(variant):
    """variant in {'raw','post'}; returns per-embryo DataFrame."""
    geffs = sorted(g for g in GEFF_DIR.glob("*.geff")
                   if g.name.replace(".zarr.geff", "").replace(".geff", "") in GOLDEN12)
    rows = []
    for g in geffs:
        ds = g.name.replace(".zarr.geff", "").replace(".geff", "")
        gn, ge = io.read_geff(TRAIN / f"{ds}.geff")
        estN = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
        nbi, raw_edges = geff_to_dfs_raw(g)
        if variant == "raw":
            pn, pe = dicts_to_dfs(nbi, raw_edges)
        else:
            nbi2, edges2, _ = P.filter_output_graph(dict(nbi), list(raw_edges), dataset=ds)
            pn, pe = dicts_to_dfs(nbi2, edges2)
        rows.append(decompose_one(ds, pn, pe, gn, ge, estN))
        r = rows[-1]
        print(f"  [{variant}] {ds:16s} adjJ={r['adj_jaccard']:.3f} nodeR={r['node_r']:.3f} "
              f"edgeR={r['edge_r']:.3f} Qlink={r['q_link']:.3f} cpen={r['count_pen']:.3f} "
              f"divJ={r['div_j'] if np.isfinite(r['div_j']) else float('nan'):.3f}", flush=True)
    return pd.DataFrame(rows)


def summarize(df, name):
    W = df.w.sum()
    adj_edge = (df.w * df.adj_jaccard).sum() / W
    dtp, dfp, dfn = df.div_tp.sum(), df.div_fp.sum(), df.div_fn.sum()
    div_j = dtp / (dtp + dfp + dfn) if (dtp + dfp + dfn) else float("nan")
    score = adj_edge + 0.1 * div_j
    # weighted-mean buckets
    def wm(col):
        return (df.w * df[col]).sum() / W
    out = dict(
        variant=name, adj_edge_jaccard=adj_edge, division_jaccard=div_j, score=score,
        w_node_r=wm("node_r"), w_edge_r=wm("edge_r"), w_q_link=wm("q_link"),
        w_jaccard=wm("jaccard"), w_count_pen=wm("count_pen"),
        div_tp=dtp, div_fp=dfp, div_fn=dfn,
    )
    return out


def main():
    print("=== THREAD 1 metric decomposition ===", flush=True)
    results = {}
    summaries = []
    for variant in ("post", "raw"):
        print(f"\n--- variant: {variant} ---", flush=True)
        df = run_variant(variant)
        df.to_csv(OUT / f"decomp_{variant}.csv", index=False)
        results[variant] = df
        s = summarize(df, variant)
        summaries.append(s)
        print(f"  SUMMARY[{variant}]: score={s['score']:.4f} adjEdgeJ={s['adj_edge_jaccard']:.4f} "
              f"divJ={s['division_jaccard']:.4f} | w.nodeR={s['w_node_r']:.3f} w.edgeR={s['w_edge_r']:.3f} "
              f"w.Qlink={s['w_q_link']:.3f} w.cpen={s['w_count_pen']:.3f}", flush=True)
    pd.DataFrame(summaries).to_csv(OUT / "decomp_summary.csv", index=False)

    # correlations (POST = real pipeline): each bucket vs data properties
    df = results["post"]
    print("\n=== CORRELATIONS (post) bucket vs density / estN ===", flush=True)
    props = ["density", "estN", "n_gt"]
    buckets = ["node_r", "edge_r", "q_link", "count_pen", "adj_jaccard", "div_j"]
    cor_rows = []
    for b in buckets:
        row = {"bucket": b}
        sub = df[np.isfinite(df[b])]
        for pr in props:
            if len(sub) >= 3:
                row[f"corr_{pr}"] = float(np.corrcoef(sub[pr], sub[b])[0, 1])
            else:
                row[f"corr_{pr}"] = float("nan")
        cor_rows.append(row)
    cor = pd.DataFrame(cor_rows)
    cor.to_csv(OUT / "decomp_correlations.csv", index=False)
    print(cor.to_string(index=False), flush=True)

    # group means 44b6 (dense/late) vs 6bba (sparse/early)
    grp = df.groupby("embryo")[["density", "estN", "node_r", "edge_r", "q_link",
                                "count_pen", "adj_jaccard", "div_j", "div_tp", "div_fn"]].mean()
    grp.to_csv(OUT / "decomp_group_means.csv")
    print("\n=== GROUP MEANS (post) ===", flush=True)
    print(grp.to_string(), flush=True)

    # charts
    try:
        make_charts(df, results["post"])
    except Exception as e:
        print(f"[charts] skipped: {e}", flush=True)
    print("\n[done] wrote decomp_*.csv + charts under", OUT, flush=True)


def make_charts(df, post):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = df.embryo.map({"44b6": "#c0392b", "6bba": "#2980b9"})

    # 1) node_recall^2 vs edge_recall — is edge recall bounded by the squared node recall?
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(df.node_r2, df.edge_r, c=colors, s=60, edgecolor="k", linewidth=0.5)
    lim = [0, 1]
    ax.plot(lim, lim, "k--", alpha=0.4, label="edge_r = node_r$^2$ (Q_link=1)")
    ax.set_xlabel("node_recall$^2$ (detection ceiling on edges)")
    ax.set_ylabel("edge_recall (realized)")
    ax.set_title("Edge recall vs node-recall$^2$ ceiling\nred=44b6 dense, blue=6bba sparse")
    ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "fig1_edgeR_vs_nodeR2.png", dpi=110); plt.close(fig)

    # 2) buckets vs density
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, b, ttl in zip(axes, ["node_r", "q_link", "count_pen"],
                          ["node recall", "link quality Q_link", "count penalty"]):
        ax.scatter(df.density, df[b], c=colors, s=55, edgecolor="k", linewidth=0.5)
        ax.set_xlabel("GT density (cells/frame)"); ax.set_ylabel(b); ax.set_title(ttl)
    fig.suptitle("Failure buckets vs density (red=44b6 dense/late, blue=6bba sparse/early)")
    fig.tight_layout(); fig.savefig(OUT / "fig2_buckets_vs_density.png", dpi=110); plt.close(fig)

    # 3) waterfall: adj_edge_jaccard ceiling losses (weighted means)
    W = df.w.sum()
    wm = lambda c: (df.w * df[c]).sum() / W
    node_r = wm("node_r"); node_r2 = node_r ** 2; edge_r = wm("edge_r")
    edge_p = wm("edge_p"); jac = wm("jaccard"); cpen = wm("count_pen"); adj = wm("adj_jaccard")
    stages = ["perfect\n(1.0)", "after node\nrecall$^2$", "after link\nquality", "after edge\nprecision(J)",
              "after count\npenalty (adj)"]
    vals = [1.0, node_r2, edge_r, jac, adj]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(len(vals)), vals, color=["#7f8c8d", "#e67e22", "#e74c3c", "#9b59b6", "#27ae60"])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(stages))); ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylim(0, 1.05); ax.set_ylabel("weighted-mean edge jaccard")
    ax.set_title("Where the adj-edge-jaccard ceiling is lost (golden-12, post)")
    fig.tight_layout(); fig.savefig(OUT / "fig3_waterfall.png", dpi=110); plt.close(fig)
    print("[charts] wrote fig1/fig2/fig3", flush=True)


if __name__ == "__main__":
    main()
