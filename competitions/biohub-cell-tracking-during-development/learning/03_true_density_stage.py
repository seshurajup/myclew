"""
Learning step 03 — The REAL cell density and developmental stage.

Step 01/02 counted annotated .geff nodes (1-33/frame) — but those are SPARSE
labels, not real cells. Each .geff carries the organisers' estimate of the true
count in `attributes.geff.extra.estimated_number_of_nodes` (estN).

Here we pull estN for all 199 embryos, compare it to the annotated count (to see
the labeling fraction), reduce to a per-frame density, and map that to
developmental stage (Kimmel 1995: gastrula ~thousands -> segmentation ~tens of
thousands of cells).
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import zarr

DATA = Path(
    "/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/"
    "input/biohub-cell-tracking-during-development/train"
)
OUT = Path(__file__).parent


def read_t(geff_dir: Path) -> np.ndarray:
    z = zarr.open(str(geff_dir / "nodes" / "props" / "t" / "values"), mode="r")
    return np.asarray(z[:])


def read_estN(geff_dir: Path) -> int:
    attrs = json.load(open(geff_dir / "zarr.json"))["attributes"]["geff"]
    return int(attrs.get("extra", {}).get("estimated_number_of_nodes", -1))


def main() -> None:
    rows = []
    for g in sorted(DATA.glob("*.geff")):
        t = read_t(g)
        n_frames = len(np.unique(t))
        rows.append({
            "dataset": g.name.replace(".geff", ""),
            "group": g.name[:4],
            "n_frames": n_frames,
            "annotated_nodes": int(len(t)),
            "estN": read_estN(g),
        })
    df = pd.DataFrame(rows)

    # is estN a per-frame or a total-across-frames number?
    corr = np.corrcoef(df.estN, df.n_frames)[0, 1]
    df["estN_per_frame"] = (df.estN / df.n_frames).round(0).astype(int)
    df["label_frac_pct"] = (100 * df.annotated_nodes / df.estN).round(1)

    print(f"estN vs n_frames correlation = {corr:.2f}  "
          f"({'scales with frames -> estN is a TOTAL' if corr > 0.4 else 'independent of frames -> estN is PER-FRAME'})")
    print(f"estN raw range        : {df.estN.min():,} .. {df.estN.max():,}")
    print(f"estN per-frame range  : {df.estN_per_frame.min():,} .. {df.estN_per_frame.max():,}")
    print(f"label fraction (annotated/estN): "
          f"{df.label_frac_pct.min():.1f}% .. {df.label_frac_pct.max():.1f}% "
          f"(median {df.label_frac_pct.median():.1f}%)")

    # ---- developmental stage from per-frame true density (5 log-bins ~ E56 S0..S4) ----
    x = np.log10(df.estN_per_frame.clip(lower=1))
    edges = np.linspace(x.min(), x.max() + 1e-9, 6)
    df["stage"] = pd.cut(x, bins=edges, labels=["S0", "S1", "S2", "S3", "S4"],
                         include_lowest=True)

    print("\n=== developmental stage (per-frame true density) ===")
    tab = df.groupby("stage", observed=True).agg(
        n=("dataset", "size"),
        cells_per_frame_min=("estN_per_frame", "min"),
        cells_per_frame_max=("estN_per_frame", "max"),
        n_44b6=("group", lambda s: (s == "44b6").sum()),
        n_6bba=("group", lambda s: (s == "6bba").sum()),
    )
    print(tab.to_string())

    # ---- the inversion: annotation density vs TRUE density, by group ----
    print("\n=== annotation vs TRUE density, by group ===")
    gg = df.groupby("group").agg(
        n=("dataset", "size"),
        annot_per_frame=("annotated_nodes", lambda s: round(s.sum() / df.loc[s.index, "n_frames"].sum(), 1)),
        true_per_frame_med=("estN_per_frame", "median"),
        label_frac_med=("label_frac_pct", "median"),
    )
    print(gg.to_string())

    out = OUT / "03_true_density_stage.csv"
    df.sort_values("estN_per_frame").to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
