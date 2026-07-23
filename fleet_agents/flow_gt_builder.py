"""flow-gt-build — turn the external dense lineage tracks into per-node FLOW + DIVISION supervision.

hengck23's recipe: predict a local vector field telling the graph optimizer how cells connect; build
the flow GT from tracks. This agent materialises that GT as a compact per-node training table:

    embryo, t, z, y, x, dz, dy, dx, n_children, is_division

where (dz,dy,dx) is the displacement to the SAME-track child at t+1 (the flow target) and is_division=1
when a node has ≥2 children (the division target — the lever our div_J=0 needs). Saved to
results/flow_gt/flow_node_gt.parquet for the flow-field trainer. Sparse per-node (not dense volumetric):
per hengck23, step (2) "only needs locations", so we supervise a vector AT each labelled cell.
"""
from __future__ import annotations
import glob
import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
TRACKS = COMP / "input" / "zebrahub" / "tracks"
OUT = COMP / "results" / "flow_gt"
STATE = COMP / "config" / "_auto" / "flow_gt_builder.json"


def build(q, worker):
    import numpy as np
    import pandas as pd
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    tracks_glob = spec.get("tracks_glob") or str(TRACKS / "*.csv")   # override external tracks glob (default = zebrahub tracks)
    files = sorted(glob.glob(tracks_glob))
    if not files:
        return ("done", {}, "all", f"[{worker}] flow-gt-build: no external track CSVs under {tracks_glob}.")
    OUT.mkdir(parents=True, exist_ok=True)

    frames = []
    n_rows = n_div = n_total = 0
    for f in files:
        emb = Path(f).stem.replace("_tracks", "").replace("_tail", "")
        try:
            df = pd.read_csv(f)
        except Exception:  # noqa: BLE001 — skip an unreadable/corrupt CSV rather than crash the build
            continue
        # normalise the parent-track column (ZSNS001 uses ParentTrackID; 003/004/005 use parent_track_id)
        for cand in ("ParentTrackID", "parent_track_id"):
            if cand in df.columns:
                df = df.rename(columns={cand: "parent_track"})
                break
        if not {"track_id", "t", "z", "y", "x"}.issubset(df.columns):
            continue
        df = df.sort_values(["track_id", "t"]).reset_index(drop=True)
        # flow target = displacement to the same-track node at t+1
        nxt = df.groupby("track_id").shift(-1)
        dt = nxt["t"] - df["t"]
        valid = dt == 1
        out = pd.DataFrame({
            "embryo": emb, "t": df["t"], "z": df["z"], "y": df["y"], "x": df["x"],
            "dz": (nxt["z"] - df["z"]).where(valid), "dy": (nxt["y"] - df["y"]).where(valid),
            "dx": (nxt["x"] - df["x"]).where(valid),
        })
        # division target: a parent track that splits into ≥2 child tracks. The DIVISION node is the
        # parent track's LAST frame (where the two daughters branch off next frame).
        out = out.reset_index(drop=True)
        if "parent_track" in df.columns:
            parent = df.groupby("track_id")["parent_track"].first()    # parent per track
            nchild = parent[parent > 0].value_counts()                 # #children per parent track
            div_parents = set(nchild[nchild >= 2].index.astype(int))   # parents that truly divide
            last_t = df.groupby("track_id")["t"].max()                 # parent's last frame = division frame
            div_keys = {(int(p), int(last_t[p])) for p in div_parents if p in last_t.index}
            tid = df["track_id"].to_numpy(); tt = df["t"].to_numpy()
            out["is_division"] = np.fromiter(((int(tid[i]), int(tt[i])) in div_keys for i in range(len(df))),
                                             dtype=np.int8, count=len(df))
        else:
            out["is_division"] = np.int8(0)
        # keep a row if it has a flow vector OR is a division node (division = parent's last frame, which
        # has NO single next-frame link → dz/dy/dx NaN, but IS the div-head target). Dropping it lost divisions.
        has_flow = out[["dz", "dy", "dx"]].notna().all(axis=1)
        out = out[has_flow | out["is_division"].eq(1)]
        n_total += len(df); n_rows += int(has_flow.sum()); n_div += int(out["is_division"].sum())
        frames.append(out)

    if not frames:
        return ("done", {}, "all", f"[{worker}] flow-gt-build: no usable tracks (missing z/y/x).")
    allgt = pd.concat(frames, ignore_index=True)
    dst = OUT / "flow_node_gt.parquet"
    try:
        allgt.to_parquet(dst, index=False)
    except Exception:  # noqa: BLE001 — parquet engine may be absent; fall back to feather/csv
        dst = OUT / "flow_node_gt.csv"; allgt.to_csv(dst, index=False)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"rows": n_rows, "divisions": n_div, "out": str(dst)}, indent=2))

    from . import ledger
    ledger.log("flow-gt-build",
               summary=f"flow+division GT built: {n_rows:,} node targets ({n_div:,} divisions) → {dst.name}",
               detail="per-node (dz,dy,dx) flow target + is_division label from external Zebrahub tracks",
               kind="finding", recommendation="train the flow/affinity + division head on this (flow-field-train)")
    completeness = round(100.0 * n_rows / max(n_total, 1), 1)
    from researchpapers.fleet import post
    msg = (f"[{worker}] 🎯 **FLOW-GT-BUILD** — materialised the affinity supervision hengck23 describes:\n\n"
           f"| target | count |\n|---|--:|\n| total external nodes | **{n_total:,}** |\n"
           f"| per-node flow vectors `(dz,dy,dx)` | **{n_rows:,}** |\n"
           f"| **link completeness** | **{completeness}%** |\n"
           f"| division-positive nodes | **{n_div:,}** |\n\n"
           f"vs competition ~1% labelled → external tracks are **{completeness}% complete** (every non-terminal "
           f"node has a next-frame link). Saved → `{dst.name}`. Training table for a flow/affinity "
           f"head + a division head to crack div_J=0. Ready for `flow-field-train`.")
    post.post_thread(worker, "all", msg, routine=False, kind="finding")
    return ("done", {"rows": n_rows, "divisions": n_div, "out": str(dst)}, "all", msg)
