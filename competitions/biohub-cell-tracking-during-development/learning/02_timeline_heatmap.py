"""
Learning step 02 — Follow TIME clearly for every dataset.

We lay each embryo out as one horizontal track:
    x-axis = frame index t (0 .. 99)   <- the timeline
    color  = number of cells in that frame (dark = few, bright = many)
    blank  = no data at that t  (shows the START timing / cropped window)

Rows are sorted low -> high by mean cells-per-frame, so the picture reads as a
gradient from sparse embryos (bottom) to dense embryos (top). A colored strip on
the left marks the two groups: 44b6 vs 6bba.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import zarr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

DATA = Path(
    "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/"
    "input/biohub-cell-tracking-during-development/train"
)
OUT = Path(__file__).parent


def read_t(geff_dir: Path) -> np.ndarray:
    z = zarr.open(str(geff_dir / "nodes" / "props" / "t" / "values"), mode="r")
    return np.asarray(z[:])


def main() -> None:
    geffs = sorted(DATA.glob("*.geff"))
    T_MAX = 100  # frames 0..99

    recs = []
    for g in geffs:
        t = read_t(g)
        vc = pd.Series(t).value_counts()          # frame -> #cells
        row = np.full(T_MAX, np.nan)
        row[vc.index.values] = vc.values          # place counts on the timeline
        recs.append({
            "name": g.name.replace(".geff", ""),
            "group": g.name[:4],
            "row": row,
            "mean_cpf": float(np.nanmean(row)),
            "t_min": int(t.min()),
            "t_max": int(t.max()),
        })

    df = pd.DataFrame(recs).sort_values("mean_cpf").reset_index(drop=True)
    mat = np.vstack(df["row"].values)             # [199 x 100]

    # ---- figure ----
    n = len(df)
    fig, (axg, ax) = plt.subplots(
        1, 2, figsize=(15, 26),
        gridspec_kw={"width_ratios": [1, 40], "wspace": 0.02},
    )

    # left strip: group color per row
    gcol = {"44b6": 0, "6bba": 1}
    gstrip = df["group"].map(gcol).values.reshape(-1, 1)
    axg.imshow(gstrip, aspect="auto", cmap=ListedColormap(["#E15759", "#4E79A7"]))
    axg.set_xticks([])
    axg.set_yticks([])
    axg.set_ylabel(f"{n} datasets  —  sorted sparse (bottom) → dense (top)",
                   fontsize=12)
    axg.set_title("group", fontsize=10)

    # main heatmap: cells per frame over time
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#f2f2f2")                  # NaN = no data = light grey
    im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest",
                   vmin=0, vmax=np.nanpercentile(mat, 99))
    ax.set_xlabel("frame index  t  (the timeline →)", fontsize=13)
    ax.set_yticks([])
    ax.set_title(
        "Cells per frame over time — each row = one embryo\n"
        "grey = no data at that t (shows start timing / cropped window)",
        fontsize=13,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("cells in frame", fontsize=11)

    # legend for the two groups
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(color="#E15759", label="44b6 (sparse group)"),
                 Patch(color="#4E79A7", label="6bba (dense group)")],
        loc="lower right", framealpha=0.9, fontsize=11,
    )

    out_png = OUT / "02_timeline_heatmap.png"
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print(f"Saved -> {out_png}")

    # quick text confirmation of start timing spread
    print(f"\nstart frame (t_min): {df.t_min.min()} .. {df.t_min.max()} "
          f"(median {int(df.t_min.median())})")
    print(f"  embryos starting at t=0 : {(df.t_min == 0).sum()} / {n}")
    print(f"  embryos starting later  : {(df.t_min > 0).sum()} / {n}")


if __name__ == "__main__":
    main()
