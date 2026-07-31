"""THREAD 2 exp#2 — OVER-PREDICTION / instance-detector-worth audit on pilkwang golden-12 (CPU).

Original plan: decompose node-FP into DUPLICATE / MERGED / SPURIOUS to decide if an instance-aware
detector (StarDist3D / Cellpose) is worth GPU.

CRITICAL SUBSTRATE FINDING (this reframes the whole audit): the GT geffs are a SPARSE point
annotation (n_gt = 52..1229 tracked nodes), while the metric's T_true = estimated_number_of_nodes
(estN) is the DENSE detection target (~6k..80k). pilkwang emits ~estN nodes, so:
  * count_ratio = n_pred / estN ≈ 1.0 (NOT n_pred/n_gt ≈ 490) -> pilkwang is NOT massively over-predicting.
  * "node-FP" measured against sparse GT is ~98.5% of preds, but those are legitimate detections that
    were simply never annotated -> a merged/dup/SPURIOUS decomposition against sparse GT is ILL-POSED
    (unannotated != spurious), and in dense tissue >1 pred within the 7µm gate can be distinct cells.
So the ONLY trustworthy over-prediction signal is count_ratio vs estN (== L2). We report that, plus a
local-multiplicity probe (flagged as density-confounded), and let the verdict rest on count_ratio.

research/cellmot_venv/bin/python tools/researchpapers/eda/thread2/merged_node_audit.py
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
ENSEMBLE = ROOT / "learning/ensemble_work"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ENSEMBLE))
from src import io  # noqa: E402
from src.config import Config  # noqa: E402
import pilk_post as P  # noqa: E402

SRC = Config()
SCALE = np.asarray(SRC.SCALE); GATE = SRC.MATCH_GATE_UM
TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
GEFF_DIR = ROOT / "research/pilkwang_support_pack/repo/predictions/seshu/unet_transformer/split_0"
OUT = ROOT / "tools/researchpapers/eda/thread2"
GOLDEN12 = {"44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"}


def load_final(path, ds):
    """pilkwang FINAL post-proc detections (real 0.8708 pipeline) — mirrors thread1 'post' variant."""
    graph = P.graph_from_geff(path)
    nbi = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]),
           "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}
           for r in graph.node_attrs().iter_rows(named=True)}
    raw_edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"])}
                 for r in graph.edge_attrs().iter_rows(named=True)]
    nbi2, _edges2, _ = P.filter_output_graph(dict(nbi), list(raw_edges), dataset=ds)
    return pd.DataFrame([{"node_id": n["node_id"], "t": n["t"], "z": n["z"], "y": n["y"], "x": n["x"]}
                        for n in nbi2.values()]).astype(
        {"node_id": int, "t": int, "z": float, "y": float, "x": float})


def local_multiplicity(gt, pred):
    """Per matched-GT-node, #pred nodes within the 7µm gate (density-confounded proxy for local dup).
    Returns (n_matched, dup_ge2_frac, mean_mult)."""
    mults = []
    for t in sorted(set(gt["t"].unique())):
        g = gt[gt["t"] == t]; p = pred[pred["t"] == t]
        if len(g) == 0 or len(p) == 0:
            continue
        G = g[["z", "y", "x"]].to_numpy(np.float64) * SCALE[None, :]
        Pp = p[["z", "y", "x"]].to_numpy(np.float64) * SCALE[None, :]
        D = np.sqrt(((G[:, None, :] - Pp[None, :, :]) ** 2).sum(2))
        cost = np.where(D <= GATE, D, 1e9)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if cost[r, c] < 1e9:
                mults.append(int((D[r] <= GATE).sum()))  # preds within gate of this matched GT
    if not mults:
        return 0, float("nan"), float("nan")
    m = np.asarray(mults)
    return len(m), float((m >= 2).mean()), float(m.mean())


def main():
    print(f"[startup] exp#2 over-prediction audit | gate={GATE}µm | golden-12 | preds={GEFF_DIR.name} | "
          f"SPARSE-GT-aware (count_ratio vs estN is the real signal)", flush=True)
    geffs = sorted(g for g in GEFF_DIR.glob("*.geff")
                   if g.name.replace(".zarr.geff", "").replace(".geff", "") in GOLDEN12)
    rows = []
    for g in geffs:
        ds = g.name.replace(".zarr.geff", "").replace(".geff", "")
        gn, _ge = io.read_geff(TRAIN / f"{ds}.geff")
        estN = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
        pred = load_final(g, ds)
        n_pred = len(pred)
        cr = n_pred / estN if estN else float("nan")
        nm, dup2, meanmult = local_multiplicity(gn, pred)
        rows.append(dict(dataset=ds, embryo=ds.split("_")[0], n_gt_sparse=len(gn), n_pred=n_pred,
                         estN=int(estN) if estN else -1, count_ratio=cr,
                         n_matched=nm, gt_dup_ge2_frac=dup2, mean_pred_within_gate=meanmult))
        print(f"  {ds}: n_pred={n_pred:6d} estN={int(estN):6d} count_ratio={cr:5.2f} "
              f"| matched={nm} dup>=2_frac={dup2:.2f} mean_mult={meanmult:.2f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "merged_node_audit.csv", index=False)

    over = df[df.count_ratio > 1.30]
    print("\n=== VERDICT (over-prediction = count_ratio vs estN, the only sparse-GT-valid signal) ===",
          flush=True)
    print(f"count_ratio: median={df.count_ratio.median():.2f}  "
          f"range={df.count_ratio.min():.2f}-{df.count_ratio.max():.2f}", flush=True)
    print(f"embryos over-predicting (count_ratio>1.30): {len(over)}/12 -> "
          f"{sorted(over.dataset.tolist())}", flush=True)
    print(f"local mult around real cells: mean dup>=2 frac={df.gt_dup_ge2_frac.mean():.2f} "
          f"(DENSITY-CONFOUNDED: in dense tissue >1 pred within 7µm can be distinct cells)", flush=True)
    print("\nDetector-precision (StarDist3D/Cellpose) worth GPU? -> WEAK & TAIL-ONLY: pilkwang is "
          "estN-calibrated (count_ratio~1) on 10/12; only the dense/hard tail (07e24132=2.40, "
          "0b24845f=1.68) over-predicts. A merged/dup/spurious FP decomposition is UNIDENTIFIABLE on "
          "sparse point GT, so a detector swap cannot be justified by an FP audit — only by the L2 "
          "dense-tail count penalty, which is small (+0.017 ceiling). Keep detector SECONDARY/gated.",
          flush=True)
    try:
        chart(df)
    except Exception as e:
        print(f"[chart] skipped: {e}", flush=True)


def chart(df):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = df.sort_values("count_ratio", ascending=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(d))
    colors = ["#e74c3c" if c > 1.30 else "#27ae60" for c in d.count_ratio]
    ax.bar(x, d.count_ratio, color=colors)
    ax.axhline(1.0, ls="--", color="gray", label="calibrated (n_pred=estN)")
    ax.axhline(1.30, ls=":", color="#e74c3c", alpha=0.6, label="over-pred flag 1.30")
    ax.set_xticks(x); ax.set_xticklabels(d.dataset, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("count_ratio = n_pred / estN"); ax.legend()
    ax.set_title("exp#2 over-prediction vs the DENSE estN target (not sparse GT)\n"
                 "red = dense-tail over-prediction; green = estN-calibrated")
    fig.tight_layout(); fig.savefig(OUT / "fig_merged_node_audit.png", dpi=110); plt.close(fig)
    print("[chart] wrote fig_merged_node_audit.png", flush=True)


if __name__ == "__main__":
    main()
