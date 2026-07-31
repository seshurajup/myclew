"""THREAD 2 candidate A (+B) — EDGE-CONFIDENCE THRESHOLDING on pilkwang's FIXED detections (CPU only).

Question (leader): how much of the +0.060 edge-precision headroom (kill FP edges) is reachable by
edge_prob thresholding ALONE — pure post-proc, no learned linker, no retraining?

Substrate = pilkwang's RAW ILP output (native `edge_prob` on edges, `solution` bool). Node recall is
SATURATED (0.993, L4) so we hold detections fixed and only touch edges/orphan-nodes:
  Variant A (linking precision): keep edges with edge_prob >= thr; keep ALL nodes.
  Variant B (A + over-prediction): also drop nodes left with NO incident kept edge (spurious singletons).
Re-score with the OFFICIAL metric (src.metric.official_counts) on golden-12, weighted.

research/cellmot_venv/bin/python eda/thread2/edgeconf_sweep.py
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
OUT.mkdir(parents=True, exist_ok=True)
GOLDEN12 = {"44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"}
THRESHOLDS = [0.0, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def load_solution(path):
    """pilkwang RAW ILP prediction: nodes+edges with solution==true, edges carry edge_prob."""
    g = P.graph_from_geff(path)
    nd = g.node_attrs().to_pandas()
    ed = g.edge_attrs().to_pandas()
    if "solution" in nd.columns:
        nd = nd[nd["solution"] == True]  # noqa: E712
    if "solution" in ed.columns:
        ed = ed[ed["solution"] == True]  # noqa: E712
    nodes = nd[["node_id", "t", "z", "y", "x"]].astype(
        {"node_id": int, "t": int, "z": float, "y": float, "x": float})
    edges = ed[["source_id", "target_id", "edge_prob"]].astype(
        {"source_id": int, "target_id": int, "edge_prob": float})
    return nodes, edges


def score(nodes, edges, gn, ge, estN):
    pe = edges[["source_id", "target_id"]]
    c = metric.official_counts(gn, ge, nodes, pe, SRC.SCALE, SRC.MATCH_GATE_UM, t_true=estN)
    return c


def main():
    geffs = sorted(g for g in GEFF_DIR.glob("*.geff")
                   if g.name.replace(".zarr.geff", "").replace(".geff", "") in GOLDEN12)
    # preload GT + estN + base detections
    base = {}
    for g in geffs:
        ds = g.name.replace(".zarr.geff", "").replace(".geff", "")
        gn, ge = io.read_geff(TRAIN / f"{ds}.geff")
        estN = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
        nodes, edges = load_solution(g)
        base[ds] = (gn, ge, estN, nodes, edges)
        print(f"  loaded {ds}: {len(nodes)} nodes, {len(edges)} sol-edges "
              f"(edge_prob {edges.edge_prob.min():.2f}-{edges.edge_prob.max():.2f})", flush=True)

    rows = []
    for variant in ("A_edge_only", "B_edge+orphan"):
        for thr in THRESHOLDS:
            per = []
            for ds, (gn, ge, estN, nodes, edges) in base.items():
                kept = edges[edges.edge_prob >= thr]
                nds = nodes
                if variant.startswith("B") and len(nds):
                    used = set(kept.source_id) | set(kept.target_id)
                    # drop nodes with NO incident kept edge (spurious singletons)
                    nds = nds[nds.node_id.isin(used)] if len(kept) else nds.iloc[0:0]
                c = score(nds, kept, gn, ge, estN)
                c["dataset"] = ds; c["embryo"] = ds.split("_")[0]
                per.append(c)
            pdf = pd.DataFrame(per)
            W = pdf.w.sum()
            adj = (pdf.w * pdf.adj_jaccard).sum() / W
            edge_p = (pdf.edge_tp.sum() / (pdf.edge_tp.sum() + pdf.edge_fp.sum())) \
                if (pdf.edge_tp.sum() + pdf.edge_fp.sum()) else float("nan")
            edge_r = pdf.edge_tp.sum() / (pdf.edge_tp.sum() + pdf.edge_fn.sum())
            cpen = (pdf.w * pdf.count_pen if "count_pen" in pdf else 0)
            # count_pen not in official_counts return; recompute weighted from t_pred/t_true
            cp = 1 - 0.1 * (pdf.t_pred - pdf.t_true) / pdf.t_true
            wcpen = (pdf.w * cp).sum() / W
            rows.append(dict(variant=variant, thr=thr, adj_edge_jaccard=adj,
                             edge_precision=edge_p, edge_recall=edge_r, w_count_pen=wcpen,
                             tot_edge_fp=int(pdf.edge_fp.sum()), tot_edge_tp=int(pdf.edge_tp.sum()),
                             tot_pred_nodes=int(pdf.t_pred.sum())))
            print(f"  [{variant}] thr={thr:.2f}  adjJ={adj:.4f}  edgeP={edge_p:.3f} "
                  f"edgeR={edge_r:.3f}  cpen={wcpen:.3f}  FP={int(pdf.edge_fp.sum())}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "edgeconf_sweep.csv", index=False)

    print("\n=== SUMMARY ===", flush=True)
    for variant in ("A_edge_only", "B_edge+orphan"):
        sub = res[res.variant == variant]
        b = sub[sub.thr == 0.0].iloc[0]
        best = sub.loc[sub.adj_edge_jaccard.idxmax()]
        print(f"{variant}: baseline(thr0) adjJ={b.adj_edge_jaccard:.4f} -> "
              f"best adjJ={best.adj_edge_jaccard:.4f} @thr={best.thr:.2f}  "
              f"(Δ={best.adj_edge_jaccard-b.adj_edge_jaccard:+.4f}; "
              f"FP {int(b.tot_edge_fp)}->{int(best.tot_edge_fp)}, "
              f"edgeP {b.edge_precision:.3f}->{best.edge_precision:.3f})", flush=True)
    print(f"\n[ref] post-proc production adjJ=0.8708; raw-ILP theoretical edge-precision ceiling "
          f"(kill ALL FP edges) is the L1 +0.060 target.", flush=True)
    try:
        make_chart(res)
    except Exception as e:
        print(f"[chart] skipped: {e}", flush=True)


def make_chart(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for variant, c in [("A_edge_only", "#e74c3c"), ("B_edge+orphan", "#27ae60")]:
        s = res[res.variant == variant].sort_values("thr")
        ax.plot(s.thr, s.adj_edge_jaccard, "-o", color=c, label=variant)
    ax.axhline(0.8708, ls="--", color="gray", alpha=0.7, label="post-proc production 0.8708")
    ax.set_xlabel("edge_prob threshold"); ax.set_ylabel("weighted adj_edge_jaccard (golden-12)")
    ax.set_title("Edge-confidence thresholding on fixed detections (CPU)\nA=edges only, B=+drop orphan nodes")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT / "fig_edgeconf_sweep.png", dpi=110); plt.close(fig)
    print("[chart] wrote fig_edgeconf_sweep.png", flush=True)


if __name__ == "__main__":
    main()
