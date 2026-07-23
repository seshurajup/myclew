"""trick-extractor — mine EVERY top-solution trick from the downloaded Kaggle notebooks, by area.

The user's ask: "take all best solutions from kaggle — metrics, pre-processing, post-processing, all
tricks." kaggle-scout pulls the LB, notebook-sync syncs daily, block-synth composes post-proc blocks;
none produces the full cross-solution TRICK MATRIX. This agent scans the notebook corpus for a curated
library of CV / tracking tricks grouped by pipeline stage (pre-proc · detection · linking · division ·
post-proc · metric · TTA · ensemble), counts how many top solutions use each, and flags which we've
already adopted — so nothing high-value is missed.

Reusable / spec-driven: {notebook_glob, tricks (extra patterns), ours_glob}. The trick library is data,
so the SAME agent works for any competition by passing a different `tricks` dict.
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
STATE = COMP / "config" / "_auto" / "trick_extractor.json"

# trick library: stage -> {trick_name: regex}. Curated from top CV/tracking solutions.
TRICKS = {
    "pre-proc": {
        "percentile-norm": r"percentile|np\.clip|clip\(|quantile",
        "CLAHE/hist-eq": r"clahe|equalize_hist|adapthist",
        "denoise/blur": r"gaussian_filter|denoise|median_filter|blur",
        "downsample-XY": r"XY_DS|downsample|zoom|rescale|resize",
        "z-normalize": r"z.?norm|standardi|mean.?std|normalize",
        "background-sub": r"background|rolling_ball|tophat|white_tophat",
    },
    "detection": {
        "DoG/blob": r"difference_of_gauss|blob_dog|blob_log|DoG",
        "multi-scale": r"multi.?scale|scale.?space|sigma.?list|scales",
        "peak-local-max": r"peak_local_max|local_max|argrelmax",
        "NMS": r"\bnms\b|non_max|physical_nms|suppress",
        "UNet/CNN": r"unet|u-net|conv3d|encoder|decoder|TemporalUNet",
        "cellpose/stardist": r"cellpose|stardist|cpsam",
        "watershed": r"watershed|segmentation",
    },
    "linking": {
        "hungarian/LSA": r"linear_sum_assignment|hungarian|munkres",
        "ILP/min-cost": r"\bilp\b|min.?cost|pulp|milp|integer.?program",
        "greedy-NN": r"greedy|nearest.?neighbou?r|kdtree|cKDTree",
        "motile/motiletrack": r"motile",
        "ultrack": r"ultrack",
        "trackastra": r"trackastra",
        "kalman/velocity": r"kalman|velocity|inertia|motion.?prior",
        "gap-closing": r"gap.?clos|gap.?close|GAP_CLOSE|close_gap",
    },
    "division": {
        "safe-divisions": r"safe_division|SAFE_DIV|add_safe",
        "detect-divisions": r"detect_division|DETECT_DIVISION|division.?detect",
        "parent/sister-dist": r"parent.?dist|sister.?dist|PARENT_DIST|SISTER",
        "division-weight": r"division_weight|DIV.*WEIGHT|div_weight",
        "validate-divisions": r"validate_division|prune.?division|daughter",
    },
    "post-proc": {
        "filter-short-tracks": r"filter_short|min_track_len|MIN_TRACK|short.?track",
        "gap2-recovery": r"gap2|GAP2|second.?pass|two.?pass",
        "linefit-smooth": r"linefit|LINEFIT|smooth|savgol|spline",
        "motion-relink": r"motion.?relink|MOTION_RELINK|relink",
        "prune-isolated": r"prune_isolated|PRUNE_ISOLATED|isolated",
        "edge-veto/repair": r"edge_veto|repair|EDGE_VETO|single_parent",
    },
    "metric": {
        "jaccard/edge-J": r"jaccard|edge_jaccard|adjusted",
        "match-gate-7um": r"7\.?0?.?um|match.?gate|MATCH_GATE|matching",
        "official-score": r"official_score|golden_cv|division_jaccard",
    },
    "tta": {
        "flip-TTA": r"flip.*(tta|augment|inference)|tta.*flip|hflip|vflip",
        "rotate-TTA": r"rot90.*(tta|inference)|tta.*rot|rotate.*augment",
        "multi-crop-TTA": r"tta|test.?time|multi.?crop",
    },
    "ensemble": {
        "avg/weighted": r"ensemble|weighted.?average|blend|mean.?pred",
        "vote/consensus": r"vote|consensus|majority|agreement",
        "WBF/NMS-merge": r"wbf|weighted_box|fusion|merge.?box",
        "fold/seed-avg": r"fold.?avg|seed.?avg|kfold|oof",
    },
}


def report(q, worker):
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    globs = spec.get("notebook_glob") or ["research/public_notebooks/**/*.ipynb", "research/kernels/**/*.ipynb"]
    extra = spec.get("tricks")
    tricks = {**TRICKS, **(extra if isinstance(extra, dict) else {})}
    # OPTIONAL max_files: cap the corpus scan so a huge notebook tree can't stall the agent (default 2000).
    try:
        max_files = int(spec.get("max_files", 2000))
    except Exception:  # noqa: BLE001
        max_files = 2000
    files = []
    for g in globs:
        try:
            files += glob.glob(str(COMP / g), recursive=True)
        except Exception:  # noqa: BLE001
            pass
    files = sorted(set(files))[:max_files]
    if not files:
        return ("done", {}, "all", f"[{worker}] trick-extractor: no notebooks under {globs}.")

    # what WE'VE adopted (our pilk_post + configs) — for the "adopted?" flag
    ours = ""
    for p in ["learning/ensemble_work/pilk_post.py", "src/link.py", "src/config.py"]:
        fp = COMP / p
        if fp.exists():
            ours += fp.read_text(errors="replace").lower()

    counts = {stage: {} for stage in tricks}
    for f in files:
        try:
            txt = Path(f).read_text(errors="replace").lower()
        except Exception:  # noqa: BLE001
            continue
        for stage, lib in tricks.items():
            for name, pat in lib.items():
                try:
                    hit = re.search(pat, txt, re.I)
                except Exception:  # noqa: BLE001 — malformed regex in a custom trick dict → skip, don't crash
                    continue
                if hit:
                    counts[stage][name] = counts[stage].get(name, 0) + 1

    n = len(files)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"notebooks": n, "counts": counts}, indent=2))

    # compact neat markdown: one tight table per stage, top tricks first, ✓ = we already use it
    lines = [f"**TRICK-EXTRACTOR** · {n} top solutions scanned · ✓=adopted, ✗=gap"]
    total_gaps = []
    for stage, lib in tricks.items():
        rows = sorted(counts[stage].items(), key=lambda kv: -kv[1])
        if not rows:
            continue
        cells = []
        for name, c in rows:
            pat = lib[name]
            try:
                adopted = bool(re.search(pat, ours, re.I)) if ours else False
            except Exception:  # noqa: BLE001
                adopted = False
            cells.append(f"{'✓' if adopted else '✗'} {name} `{c}`")
            if not adopted and c >= max(2, n // 8):
                total_gaps.append(f"{stage}/{name}")
        lines.append(f"`{stage}` — " + " · ".join(cells))
    gap_txt = (" · ".join(total_gaps[:6]) or "none — we cover the common tricks") if total_gaps else "none"
    lines.append(f"**High-value gaps (used widely, not adopted):** {gap_txt}")
    msg = f"[{worker}] " + "\n".join(lines)

    from . import ledger
    ledger.log("trick-extractor",
               summary=f"scanned {n} top solutions → trick matrix across 8 stages; gaps: {gap_txt[:120]}",
               detail=f"stages={list(tricks)}", kind="finding",
               recommendation="adopt the high-value gaps (widely-used tricks we don't have yet)")
    from researchpapers.fleet import post
    post.post_thread(worker, "all", msg, routine=False, kind="finding")
    return ("done", {"notebooks": n, "gaps": total_gaps[:10],
                     "stages": {s: len(c) for s, c in counts.items()}}, "all", msg)
