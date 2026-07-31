"""THREAD 2 precursor (candidate D) — scan ALL 199 train embryos' GT geffs for division events
(out-degree >= 2 nodes) and per-embryo density, to (a) re-measure the realistic div_j ceiling on a
DIVISION-RICH embryo-disjoint mini-split and (b) show golden-12 is division-poor.

Pure Python, NO GPU.  research/cellmot_venv/bin/python eda/thread2/division_scan.py
"""
import sys, json
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
sys.path.insert(0, str(ROOT))
from src import io  # noqa: E402

TRAIN = ROOT / "input/biohub-cell-tracking-during-development/train"
OUT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/tools/researchpapers/eda/thread2")
OUT.mkdir(parents=True, exist_ok=True)
GOLDEN12 = {"44b6_0113de3b", "44b6_0b24845f", "44b6_0c582fdc", "44b6_0db75fae",
            "44b6_12dfb391", "44b6_144b256d", "6bba_05b6850b", "6bba_05db0fb1",
            "6bba_062c8d37", "6bba_07477033", "6bba_07e24132", "6bba_085bf656"}


def main():
    geffs = sorted(TRAIN.glob("*.geff"))
    print(f"scanning {len(geffs)} train geffs for divisions + density", flush=True)
    rows = []
    for i, g in enumerate(geffs):
        ds = g.name.replace(".geff", "")
        try:
            gn, ge = io.read_geff(g)
        except Exception as e:
            print(f"  SKIP {ds}: {e}", flush=True); continue
        n_nodes = len(gn)
        n_frames = int(gn["t"].nunique()) if n_nodes else 0
        density = n_nodes / n_frames if n_frames else float("nan")  # cells/frame
        # divisions = GT nodes with out-degree >= 2
        od = Counter(int(s) for s in ge["source_id"]) if len(ge) else Counter()
        n_div = sum(1 for c in od.values() if c >= 2)
        rows.append(dict(dataset=ds, embryo=ds.split("_")[0], n_nodes=n_nodes,
                         n_frames=n_frames, density=density, n_edges=len(ge),
                         n_div=n_div, div_rate=n_div / n_nodes if n_nodes else float("nan"),
                         golden12=ds in GOLDEN12))
        if (i + 1) % 25 == 0:
            print(f"  ...{i+1}/{len(geffs)}", flush=True)
    df = pd.DataFrame(rows).sort_values("n_div", ascending=False)
    df.to_csv(OUT / "division_scan.csv", index=False)

    tot_div = df.n_div.sum()
    g12 = df[df.golden12]
    print(f"\n=== TOTALS ===", flush=True)
    print(f"embryos: {len(df)}  total GT divisions: {tot_div}  "
          f"embryos with >=1 div: {(df.n_div>0).sum()}  with 0 div: {(df.n_div==0).sum()}", flush=True)
    print(f"golden-12 divisions: {g12.n_div.sum()}  ({g12.n_div.sum()/tot_div*100:.1f}% of all) "
          f"-> golden-12 is division-{'POOR' if g12.n_div.sum()/tot_div < len(g12)/len(df) else 'rich'}", flush=True)
    print(f"density range (cells/frame): {df.density.min():.1f} - {df.density.max():.1f}", flush=True)
    print(f"\n=== per embryo group ===", flush=True)
    print(df.groupby("embryo").agg(n=("dataset","size"), tot_div=("n_div","sum"),
          mean_div=("n_div","mean"), mean_dens=("density","mean")).to_string(), flush=True)
    print(f"\n=== TOP-12 division-rich embryos (candidate mini-split pool) ===", flush=True)
    print(df.head(12)[["dataset","embryo","n_nodes","density","n_div","div_rate"]].to_string(index=False), flush=True)

    # build a DIVISION-RICH embryo-disjoint mini-split proposal: top division-rich, balanced by group,
    # embryo-disjoint means fold split by embryo prefix (44b6 vs 6bba) — already the natural CV axis.
    rich = df[df.n_div > 0].sort_values("n_div", ascending=False)
    prop = {}
    for emb in ["44b6", "6bba"]:
        prop[emb] = rich[rich.embryo == emb].head(6)["dataset"].tolist()
    json.dump({"division_rich_minisplit": prop,
               "total_div_in_split": int(df[df.dataset.isin(sum(prop.values(), []))].n_div.sum())},
              open(OUT / "division_rich_minisplit.json", "w"), indent=2)
    print(f"\nproposed division-rich mini-split (6+6, embryo-disjoint) -> "
          f"{OUT/'division_rich_minisplit.json'}", flush=True)
    print(f"  divisions in proposed split: "
          f"{df[df.dataset.isin(sum(prop.values(), []))].n_div.sum()} "
          f"(vs {g12.n_div.sum()} in golden-12)", flush=True)


if __name__ == "__main__":
    main()
