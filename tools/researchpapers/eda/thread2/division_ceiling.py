"""THREAD 2 exp#3 — DIVISION-JACCARD CEILING re-measure (CPU only).

L3 said divisions are the largest RAW upside but variance-dominated on golden-12 (only 8 GT divs).
Leader asked to re-measure the div_j ceiling on a division-rich split. HARD CONSTRAINT: pilkwang
predictions exist ONLY for the 12 golden-12 embryos, so we CANNOT score pilkwang preds on the
div-rich split. Instead:

(a) golden-12 WITH pilkwang FINAL post-proc preds (available): recompute div_tp/fp/fn via the
    official metric, confirm L3 (expect div_tp=0), and measure the FP-fork RATE = div_fp / pred_nodes.
(b) div-rich split (36 GT divs, GT-only, from division_rich_minisplit.json): a GT-STRUCTURAL ceiling
    curve div_j(r) = r·D / (r·D + FP_est + (1-r)·D), FP_est = fp_rate · (proxy pred-node count).
    Proxy pred-node count = GT node count of the split (from division_scan.csv), stated as an
    assumption since no preds exist there. Report perfect vs realistic div_j and the +0.1·div_j score
    delta, plus the variance advantage (36 vs 8 divisions).

research/cellmot_venv/bin/python tools/researchpapers/eda/thread2/division_ceiling.py
"""
import sys, json
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
OUT = ROOT / "tools/researchpapers/eda/thread2"
GOLDEN12 = {"44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"}


def load_final(path, ds):
    """FINAL post-proc pilkwang nodes+edges (real 0.8708 pipeline)."""
    graph = P.graph_from_geff(path)
    nbi = {int(r["node_id"]): {"node_id": int(r["node_id"]), "t": int(r["t"]),
           "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}
           for r in graph.node_attrs().iter_rows(named=True)}
    raw_edges = [{"source_id": int(r["source_id"]), "target_id": int(r["target_id"])}
                 for r in graph.edge_attrs().iter_rows(named=True)]
    nbi2, edges2, _ = P.filter_output_graph(dict(nbi), list(raw_edges), dataset=ds)
    pn = pd.DataFrame([{"node_id": n["node_id"], "t": n["t"], "z": n["z"], "y": n["y"], "x": n["x"]}
                      for n in nbi2.values()])
    pe = pd.DataFrame([{"source_id": int(e["source_id"]), "target_id": int(e["target_id"])}
                       for e in edges2]) if edges2 else pd.DataFrame(columns=["source_id", "target_id"])
    return pn, pe


def main():
    print(f"[startup] exp#3 division-ceiling | golden-12 preds + 36-div GT split | "
          f"gate={SRC.MATCH_GATE_UM}µm", flush=True)

    # ---------- (a) golden-12 with pilkwang FINAL preds ----------
    print("\n--- (a) golden-12 division performance (pilkwang FINAL post-proc) ---", flush=True)
    geffs = sorted(g for g in GEFF_DIR.glob("*.geff")
                   if g.name.replace(".zarr.geff", "").replace(".geff", "") in GOLDEN12)
    rows = []
    for g in geffs:
        ds = g.name.replace(".zarr.geff", "").replace(".geff", "")
        gn, ge = io.read_geff(TRAIN / f"{ds}.geff")
        estN = io.geff_estimated_nodes(TRAIN / f"{ds}.geff")
        pn, pe = load_final(g, ds)
        c = metric.official_counts(gn, ge, pn, pe, SRC.SCALE, SRC.MATCH_GATE_UM, t_true=estN)
        n_gt_div = c["div_tp"] + c["div_fn"]
        rows.append(dict(dataset=ds, n_pred=c["t_pred"], n_gt_div=n_gt_div,
                         div_tp=c["div_tp"], div_fp=c["div_fp"], div_fn=c["div_fn"]))
        print(f"  {ds}: pred_nodes={c['t_pred']:6d} gt_div={n_gt_div} "
              f"div_tp={c['div_tp']} div_fp={c['div_fp']} div_fn={c['div_fn']}", flush=True)
    a = pd.DataFrame(rows)
    tot_pred = int(a.n_pred.sum()); tot_gt_div = int(a.n_gt_div.sum())
    tot_dtp = int(a.div_tp.sum()); tot_dfp = int(a.div_fp.sum()); tot_dfn = int(a.div_fn.sum())
    g12_divj = tot_dtp / (tot_dtp + tot_dfp + tot_dfn) if (tot_dtp + tot_dfp + tot_dfn) else 0.0
    fp_rate = tot_dfp / tot_pred if tot_pred else 0.0
    print(f"\n  golden-12 TOTAL: pred_nodes={tot_pred} gt_div={tot_gt_div} "
          f"div_tp={tot_dtp} div_fp={tot_dfp} div_fn={tot_dfn}", flush=True)
    print(f"  golden-12 div_jaccard = {g12_divj:.4f}  (contributes {0.1*g12_divj:+.4f} to score)", flush=True)
    print(f"  FP-fork RATE = div_fp/pred_nodes = {tot_dfp}/{tot_pred} = {fp_rate:.3e} "
          f"forks per pred node", flush=True)

    # ---------- (b) div-rich split GT-structural ceiling ----------
    print("\n--- (b) div-rich 36-div split — GT-structural div_j ceiling (NO preds available) ---",
          flush=True)
    split = json.load(open(OUT / "division_rich_minisplit.json"))["division_rich_minisplit"]
    members = [d for v in split.values() for d in v]
    scan = pd.read_csv(OUT / "division_scan.csv")
    sub = scan[scan.dataset.isin(members)]
    D = int(sub.n_div.sum())                     # 36 GT divisions
    # FP forks scale with the DENSE pred-node count (~estN), NOT the sparse GT annotation. Using the
    # sparse GT count would understate FP_est ~40× and inflate the ceiling. Dense proxy = Σ estN.
    estN_sum = int(sum(io.geff_estimated_nodes(TRAIN / f"{d}.geff") or 0 for d in members))
    sparse_gt = int(sub.n_nodes.sum())
    FP_est = fp_rate * estN_sum
    print(f"  split members={len(members)}  GT divisions D={D}", flush=True)
    print(f"  dense proxy Σ estN={estN_sum} -> FP_est={FP_est:.1f} forks (PRIMARY)  |  "
          f"sparse GT Σ={sparse_gt} -> {fp_rate*sparse_gt:.1f} forks (understated, not used)", flush=True)

    def div_j(r, fp):
        tp = r * D; fn = (1 - r) * D
        return tp / (tp + fp + fn) if (tp + fp + fn) else 0.0

    ceil_rows = []
    for r in [1.0, 0.75, 0.5, 0.25]:
        for label, fp in [("perfect(FP=0)", 0.0), ("realistic(FP=est)", FP_est)]:
            j = div_j(r, fp)
            ceil_rows.append(dict(recall=r, fp_scenario=label, FP=round(fp, 1),
                                  div_j=round(j, 4), score_delta=round(0.1 * j, 4)))
    cdf = pd.DataFrame(ceil_rows)
    print(cdf.to_string(index=False), flush=True)

    # variance note: each division event = 1/D of the micro metric
    print(f"\n  VARIANCE: golden-12 has 8 divs (each = {1/8:.3f} of div_j); the 36-div split has "
          f"each = {1/D:.3f} -> {D/8:.1f}× lower per-event variance.", flush=True)
    perfect = div_j(1.0, 0.0); realistic = div_j(1.0, FP_est)
    print(f"\n  CEILING: perfect head (recall=1,FP=0) div_j=1.000 -> +0.1000 score.  "
          f"realistic (recall=1, FP={FP_est:.0f}) div_j={realistic:.3f} -> {0.1*realistic:+.4f} score.",
          flush=True)

    a.to_csv(OUT / "division_ceiling_golden12.csv", index=False)
    cdf.to_csv(OUT / "division_ceiling.csv", index=False)
    print("\n[out] division_ceiling.csv, division_ceiling_golden12.csv", flush=True)
    return dict(g12_divj=g12_divj, fp_rate=fp_rate, D=D, FP_est=FP_est,
                realistic_divj=realistic, realistic_delta=0.1 * realistic)


if __name__ == "__main__":
    main()
