"""THREAD-2 exp#1b comparison — 3-way coord-scale de-confound on golden-12.

Reads exp1_trackastra_greedy_{voxel,iso_z,um}_percell.csv and reports, per config: weighted
adj_edge_jaccard, the DENSE-TAIL weighted adjJ (top-5 embryos by #detections — the ones that collapsed in
exp#1), and per-embryo adjJ vs #detections so we can see if the anisotropy fix rescues the dense tail.
Verdict rule (leader): exp#4 fine-tune is only worth it if a de-confound pulls the DENSE embryos toward
anchor (proving learned linking has real headroom on our data)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ANCHOR = 0.8708
CONFIGS = ["voxel", "iso_z", "um"]


def wmean(df, col="adj_jaccard"):
    return float((df.w * df[col]).sum() / df.w.sum()) if df.w.sum() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)

    frames = {}
    for cs in CONFIGS:
        p = out / f"exp1_trackastra_greedy_{cs}_percell.csv"
        if p.exists():
            frames[cs] = pd.read_csv(p)
    if not frames:
        print("no exp#1b per-cell CSVs found in", out)
        return

    # dense tail = top-5 by #detections (same set across configs)
    ref = next(iter(frames.values()))
    dense_ids = ref.sort_values("n_pilk_nodes", ascending=False).head(5).dataset.tolist()

    print(f"\n=== exp#1b coord-scale sweep (golden-12, anchor {ANCHOR:.4f}) ===")
    print(f"{'config':7} {'W-adjJ':>8} {'Δanchor':>8} {'dense-adjJ':>11} {'div_fp':>7}")
    rows = []
    for cs, df in frames.items():
        w = wmean(df)
        dense = df[df.dataset.isin(dense_ids)]
        dw = wmean(dense)
        dfp = int(df.div_fp.sum())
        rows.append(dict(config=cs, w_adjJ=w, delta=w - ANCHOR, dense_adjJ=dw, div_fp=dfp))
        print(f"{cs:7} {w:8.4f} {w-ANCHOR:+8.4f} {dw:11.4f} {dfp:7d}")

    print(f"\ndense-tail embryos (top-5 by #detections): {dense_ids}")
    print(f"\n=== per-embryo adjJ by config (sorted by #detections desc) ===")
    piv = ref[["dataset", "n_pilk_nodes"]].copy()
    for cs, df in frames.items():
        piv = piv.merge(df[["dataset", "adj_jaccard"]].rename(columns={"adj_jaccard": cs}), on="dataset")
    piv = piv.sort_values("n_pilk_nodes", ascending=False)
    print(piv.to_string(index=False))
    piv.to_csv(out / "exp1b_compare.csv", index=False)

    best = max(rows, key=lambda r: r["dense_adjJ"])
    print(f"\nBEST dense-tail config: {best['config']} (dense-adjJ {best['dense_adjJ']:.4f}); "
          f"baseline voxel dense-adjJ {[r for r in rows if r['config']=='voxel'][0]['dense_adjJ']:.4f}"
          if any(r["config"] == "voxel" for r in rows) else "")
    try:
        make_chart(piv, out)
    except Exception as e:
        print(f"[chart] skipped: {e}")


def make_chart(piv, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"voxel": "#888", "iso_z": "#27ae60", "um": "#e74c3c"}
    for cs in CONFIGS:
        if cs in piv.columns:
            ax.plot(piv.n_pilk_nodes, piv[cs], "-o", color=colors.get(cs), label=cs)
    ax.axhline(ANCHOR, ls="--", color="k", alpha=0.6, label=f"pilkwang anchor {ANCHOR}")
    ax.set_xscale("log")
    ax.set_xlabel("#fixed detections (embryo density proxy, log)")
    ax.set_ylabel("adj_edge_jaccard")
    ax.set_title("exp#1b: does anisotropy correction rescue the DENSE tail?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "exp1b_compare.png", dpi=110)
    print(f"[chart] wrote {out}/exp1b_compare.png")


if __name__ == "__main__":
    main()
