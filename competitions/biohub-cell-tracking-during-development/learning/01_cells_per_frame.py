"""
Learning step 01 — How many cells are in each frame?

Each training embryo is stored as a `.geff` tracking graph. Every NODE in that
graph is one cell detected at one timepoint. The node property `t` is the frame
index (0, 1, 2, ...). So the number of cells in frame `t` is simply how many
nodes have that value of `t`.

This script reads `t` for every embryo and answers, per dataset:
  - how many frames does it span (t range)
  - cells-per-frame: min / max / mean / median
And globally it prints a histogram: "cells-per-frame value -> how many frames".
"""

from pathlib import Path
import numpy as np
import pandas as pd
import zarr

DATA = Path(
    "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/"
    "input/biohub-cell-tracking-during-development/train"
)


def read_t(geff_dir: Path) -> np.ndarray:
    """Return the frame index `t` of every node (cell) in one embryo."""
    z = zarr.open(str(geff_dir / "nodes" / "props" / "t" / "values"), mode="r")
    return np.asarray(z[:])


def main() -> None:
    geffs = sorted(DATA.glob("*.geff"))
    print(f"Found {len(geffs)} embryos (.geff) in train/\n")

    rows = []
    all_counts = []  # every (embryo, frame) -> cell count, for the global histogram

    for g in geffs:
        t = read_t(g)
        # cells per frame = how many nodes share each t value
        vc = pd.Series(t).value_counts().sort_index()  # index=frame, value=#cells
        cpf = vc.values                                # cells-per-frame array
        stage = g.name[:4]                             # 44b6 or 6bba embryo group

        rows.append({
            "dataset": g.name.replace(".geff", ""),
            "group": stage,
            "n_frames": len(vc),
            "t_min": int(t.min()),
            "t_max": int(t.max()),
            "total_cells": int(len(t)),
            "cpf_min": int(cpf.min()),
            "cpf_max": int(cpf.max()),
            "cpf_mean": round(float(cpf.mean()), 1),
            "cpf_median": int(np.median(cpf)),
        })
        all_counts.extend(cpf.tolist())

    df = pd.DataFrame(rows).sort_values("cpf_max", ascending=False)

    # ---- per-dataset table ----
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 160)
    print("=" * 90)
    print("PER-DATASET cells-per-frame summary (sorted by densest frame)")
    print("=" * 90)
    print(df.to_string(index=False))

    out_csv = Path(__file__).parent / "01_cells_per_frame_per_dataset.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved -> {out_csv}")

    # ---- overall dataset-level summary ----
    print("\n" + "=" * 90)
    print("OVERALL (across all embryos)")
    print("=" * 90)
    print(f"embryos                : {len(df)}")
    print(f"  group 44b6           : {(df.group == '44b6').sum()}")
    print(f"  group 6bba           : {(df.group == '6bba').sum()}")
    print(f"frames per embryo      : {df.n_frames.min()} .. {df.n_frames.max()} "
          f"(median {int(df.n_frames.median())})")
    print(f"total cells per embryo : {df.total_cells.min()} .. {df.total_cells.max()} "
          f"(median {int(df.total_cells.median())})")
    print(f"cells in a single frame: {df.cpf_min.min()} .. {df.cpf_max.max()}")

    # ---- global histogram: cells-per-frame binned -> how many frames ----
    ac = np.array(all_counts)
    bins = [0, 10, 25, 50, 100, 200, 400, 800, 1600, 3200, np.inf]
    labels = ["1-10", "11-25", "26-50", "51-100", "101-200", "201-400",
              "401-800", "801-1600", "1601-3200", "3200+"]
    binned = pd.cut(ac, bins=bins, labels=labels, right=True)
    hist = binned.value_counts().reindex(labels)

    print("\n" + "=" * 90)
    print(f"GLOBAL histogram — how many frames fall in each cells-per-frame range")
    print(f"(total frames across all embryos = {len(ac)})")
    print("=" * 90)
    maxc = int(hist.max())
    for lab, cnt in hist.items():
        cnt = int(cnt)
        bar = "#" * int(50 * cnt / maxc) if maxc else ""
        print(f"  {lab:>10} cells/frame : {cnt:6d} frames  {bar}")


if __name__ == "__main__":
    main()
