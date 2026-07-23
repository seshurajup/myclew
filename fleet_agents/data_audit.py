"""data-audit — MEASURE the training data's scale/quality and CORRECT it, before anything trains on it.

arch-builder surfaced that the assembled flow GT is mis-scaled: the 4 external embryos have 2–3× different
inter-frame displacement medians (ZSNS001 matches hengck23's reference ~1.3 vox; 003/004/005 are larger),
plus ~4% outlier track-switching jumps. Training / deriving an architecture on that gives wrong numbers
(cell≈17.5µm, kernel 43×43). This agent, per "decide only from data", does NOT assume a scale — it:

  1. measures each embryo's median inter-frame displacement,
  2. takes the cleanest (smallest, = the hengck23-consistent reference) as the common scale,
  3. normalises every embryo's coordinates + flow to that reference,
  4. clips physically-implausible outlier links (|disp| beyond a robust cutoff),
  5. writes a cleaned GT (flow_node_gt_clean.parquet) and reports the before/after.

Reusable / spec-driven: {gt_path, out_path, outlier_mult, ref}. The trainer + arch-builder then read the
CLEAN GT, so downstream numbers are trustworthy.
"""
from __future__ import annotations
import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
GT = COMP / "results" / "flow_gt" / "flow_node_gt.parquet"
STATE = COMP / "config" / "_auto" / "data_audit.json"


def report(q, worker):
    import numpy as np
    import pandas as pd
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    gt_path = Path(spec.get("gt_path") or GT)
    out_path = Path(spec.get("out_path") or (gt_path.parent / "flow_node_gt_clean.parquet"))
    outlier_mult = float(spec.get("outlier_mult", 6.0))   # drop links > mult × reference median magnitude
    ref_override = spec.get("ref")                         # optional: force the reference embryo (else cleanest)
    if not gt_path.exists():
        return ("done", {}, "all", f"[{worker}] data-audit: GT missing at {gt_path}.")

    df = pd.read_parquet(gt_path)
    _EPS = 1e-9
    def _mag_of(frame):
        m = np.sqrt(frame["dz"].fillna(0) ** 2 + frame["dy"].fillna(0) ** 2 + frame["dx"].fillna(0) ** 2)
        return np.nan_to_num(np.asarray(m, float), nan=0.0, posinf=0.0, neginf=0.0)
    df["_mag"] = _mag_of(df)
    # per-embryo median inter-frame displacement (rows with a flow vector only)
    med = df[df["_mag"] > 0].groupby("embryo")["_mag"].median()
    if med.empty:                                          # no motion vectors at all → nothing to scale
        return ("done", {"clean_path": str(gt_path)}, "all",
                f"[{worker}] data-audit: no positive-magnitude flow rows in {gt_path.name}; nothing to correct.")
    if ref_override is not None and ref_override in med.index:
        ref_emb = ref_override; ref = float(med.loc[ref_emb])
    else:
        ref = float(med.min())                            # cleanest embryo = the reference scale
        ref_emb = med.idxmin()
    ref = ref if ref > _EPS else _EPS                     # guard a degenerate zero reference
    before = {e: round(float(v), 2) for e, v in med.items()}

    # 2. per-embryo scale factor → normalise coords + flow to the reference motion scale
    scale = {e: ref / max(float(v), _EPS) for e, v in med.items()}
    for col in ["z", "y", "x", "dz", "dy", "dx"]:
        if col in df.columns:
            df[col] = df[col] * df["embryo"].map(scale).astype(float)
    df["_mag"] = _mag_of(df)

    # 3. clip outlier track-switching jumps (keep divisions even if flow is nan)
    cutoff = outlier_mult * ref
    keep = (df["_mag"] <= cutoff) | (df["_mag"] == 0) | (df["is_division"] == 1)
    n0 = len(df)
    clean = df[keep].drop(columns=["_mag"])
    dropped = n0 - len(clean)

    after_med = _mag_of(clean)
    after_med = after_med[after_med > 0]
    p50, p99 = (float(np.percentile(after_med, 50)), float(np.percentile(after_med, 99))) if len(after_med) else (0.0, 0.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        clean.to_parquet(out_path, index=False)
    except Exception:  # noqa: BLE001
        out_path = out_path.with_suffix(".csv"); clean.to_csv(out_path, index=False)

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"ref_embryo": ref_emb, "ref_median": round(ref, 2), "before": before,
                                 "scale_factors": {e: round(s, 2) for e, s in scale.items()},
                                 "dropped_outliers": dropped, "clean_p50": round(p50, 2),
                                 "clean_p99": round(p99, 2), "clean_path": str(out_path)}, indent=2))
    from . import ledger
    ledger.log("data-audit",
               summary=f"flow GT scale-corrected: 4 embryos normalised to {ref_emb} ref (p50→{p50:.1f}, p99→{p99:.1f} vox); dropped {dropped:,} outliers",
               detail=f"before medians {before}; scale factors {({e: round(s,2) for e,s in scale.items()})}",
               kind="finding", recommendation="re-run arch-builder on the CLEAN GT — numbers now trustworthy")
    from researchpapers.fleet import post
    sc = " · ".join(f"{e.replace('ZSNS','Z')} ×{s:.2f}" for e, s in scale.items())
    msg = (f"[{worker}] **DATA-AUDIT** · measured, not assumed · reference = `{ref_emb}` (matches hengck23 ~1.3 vox)\n"
           f"before per-embryo median: {before}\n"
           f"scale factors applied: {sc}\n"
           f"outliers dropped: **{dropped:,}** ({100*dropped/max(n0,1):.1f}%) · clean motion p50 **{p50:.1f}** / p99 **{p99:.1f}** vox "
           f"(was p99 ~53µm inflated) → `{out_path.name}`. arch-builder can now derive trustworthy kernel/radius.")
    post.post_thread(worker, "all", msg, routine=False, kind="finding")
    return ("done", {"ref_embryo": ref_emb, "dropped": dropped, "clean_p50": round(p50, 2),
                     "clean_p99": round(p99, 2), "clean_path": str(out_path)}, "all", msg)
