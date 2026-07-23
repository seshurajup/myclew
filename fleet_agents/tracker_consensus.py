"""tracker-consensus — hengck23's core recipe: run MANY trackers, keep the links they AGREE on.

The competition has <1% of links labelled. hengck23: "if i use 5 open source trackers, and using
consistency, i can have more short tracks." This agent implements exactly that IN-DOMAIN (on the
competition embryos' own detections), producing high-confidence pseudo-labelled links to train the
flow/GNN model on — complementing the external Zebrahub GT with real-embryo supervision.

How: load per-frame detections, run `src.link.track_dataset` under N diverse linker configs (nearest-
neighbour / velocity-prior / gap-closing / wider-gate / division-aware = the "trackers"), then keep every
edge that ≥K trackers agree on. Reports the consensus link yield and per-tracker agreement.

Reusable / spec-driven: {nodes_dir, embryo_filter, agreement_k, trackers:[{name, overrides}], max_embryos}.
"""
from __future__ import annotations
import glob
import json
from dataclasses import replace
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
NODES_DIR = COMP / "learning" / "ensemble_work" / "pilkwang_nodes"
OUT = COMP / "results" / "consensus_links"
STATE = COMP / "config" / "_auto" / "tracker_consensus.json"

# the "5 trackers" — diverse linker configs; each is a distinct short-track hypothesis
DEFAULT_TRACKERS = [
    {"name": "nn_tight",   "overrides": {"MAX_LINK_DIST_UM": 6.0, "USE_VELOCITY_PRIOR": False, "USE_GAP_CLOSING": False}},
    {"name": "nn_wide",    "overrides": {"MAX_LINK_DIST_UM": 10.0, "USE_VELOCITY_PRIOR": False, "USE_GAP_CLOSING": False}},
    {"name": "velocity",   "overrides": {"MAX_LINK_DIST_UM": 8.0, "USE_VELOCITY_PRIOR": True, "USE_GAP_CLOSING": False}},
    {"name": "gap_close",  "overrides": {"MAX_LINK_DIST_UM": 8.0, "USE_VELOCITY_PRIOR": False, "USE_GAP_CLOSING": True}},
    {"name": "division",   "overrides": {"MAX_LINK_DIST_UM": 8.0, "DETECT_DIVISIONS": True, "USE_GAP_CLOSING": False}},
]
DEFAULTS = {"nodes_dir": str(NODES_DIR), "embryo_filter": [], "agreement_k": 3,
            "trackers": DEFAULT_TRACKERS, "max_embryos": 4}


def _frames_from_nodes(csv, pd):
    try:
        df = pd.read_csv(csv)
    except Exception:  # noqa: BLE001 — unreadable node CSV → skip this embryo
        return None, None
    if not {"t", "z", "y", "x"}.issubset(df.columns):
        return None, None
    frames, ids = [], []
    for t in sorted(df["t"].unique()):
        sub = df[df["t"] == t]
        frames.append(sub[["z", "y", "x"]].to_numpy())
        ids.append(sub["node_id"].to_numpy() if "node_id" in sub.columns else sub.index.to_numpy())
    return frames, ids


def run(q, worker):
    import numpy as np
    import pandas as pd
    try:
        from src.link import track_dataset
        from src.config import Config
    except Exception as e:  # noqa: BLE001
        return ("escalated", {"error": str(e)}, "researcher",
                f"[{worker}] tracker-consensus: can't import src.link/src.config ({e}).")
    spec = (q.get("spec") or {}) if isinstance(q, dict) else {}
    cfg = {**DEFAULTS, **{k: spec[k] for k in DEFAULTS if k in spec}}
    trackers = cfg["trackers"]; K = int(cfg["agreement_k"])
    files = sorted(glob.glob(str(Path(cfg["nodes_dir"]) / "*.csv")))
    files = [f for f in files if "edges" not in f]
    if cfg["embryo_filter"]:
        files = [f for f in files if any(s in f for s in cfg["embryo_filter"])]
    files = files[: int(cfg["max_embryos"])]
    if not files:
        return ("done", {}, "all", f"[{worker}] tracker-consensus: no node CSVs under {cfg['nodes_dir']}.")
    OUT.mkdir(parents=True, exist_ok=True)

    tot_consensus = tot_union = 0
    per_tracker = {t["name"]: 0 for t in trackers}
    rows_out = []
    for f in files:
        emb = Path(f).stem
        frames, ids = _frames_from_nodes(f, pd)
        if frames is None or len(frames) < 2:
            continue
        # each tracker → set of directed edges keyed by (src_node_id, dst_node_id)
        edge_sets = []
        for t in trackers:
            try:
                c = replace(Config(), **{k: v for k, v in t["overrides"].items() if hasattr(Config(), k)})
            except Exception:  # noqa: BLE001
                c = Config()
            nodes, edges = track_dataset([fr.copy() for fr in frames], c)
            # track_dataset re-ids nodes internally; map its node ids back to (t, position) → our node_id
            id2pos = {n["node_id"]: (n["t"], round(n["z"], 2), round(n["y"], 2), round(n["x"], 2)) for n in nodes}
            es = set()
            for s, d in edges:
                if s in id2pos and d in id2pos:
                    es.add((id2pos[s], id2pos[d]))
            edge_sets.append(es)
            per_tracker[t["name"]] += len(es)
        # consensus = edges present in >= K trackers
        from collections import Counter
        cnt = Counter()
        for es in edge_sets:
            cnt.update(es)
        consensus = {e for e, n in cnt.items() if n >= K}
        union = set(cnt.keys())
        tot_consensus += len(consensus); tot_union += len(union)
        for (sp, dp) in consensus:
            rows_out.append({"embryo": emb, "t": sp[0], "z": sp[1], "y": sp[2], "x": sp[3],
                             "dz": dp[1] - sp[1], "dy": dp[2] - sp[2], "dx": dp[3] - sp[3],
                             "agreement": cnt[(sp, dp)]})

    dst = OUT / "consensus_links.parquet"
    if rows_out:
        df = pd.DataFrame(rows_out)
        try:
            df.to_parquet(dst, index=False)
        except Exception:  # noqa: BLE001
            dst = OUT / "consensus_links.csv"; df.to_csv(dst, index=False)
    frac = round(100.0 * tot_consensus / max(tot_union, 1), 1)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"consensus": tot_consensus, "union": tot_union, "frac": frac,
                                 "per_tracker": per_tracker, "k": K, "embryos": len(files)}, indent=2))
    from . import ledger
    ledger.log("tracker-consensus",
               summary=f"multi-tracker consensus: {tot_consensus:,} in-domain links (≥{K}/{len(trackers)} agree, {frac}% of union)",
               detail=f"per-tracker edges: {per_tracker}; from {len(files)} embryos → {dst.name}",
               kind="finding", recommendation="use as in-domain pseudo-labels alongside external GT for flow/GNN training")
    from researchpapers.fleet import post
    ptxt = " · ".join(f"{k} {v:,}" for k, v in per_tracker.items())
    msg = (f"[{worker}] 🤝 **TRACKER-CONSENSUS** — hengck23's '5 trackers + consistency', in-domain:\n\n"
           f"| metric | value |\n|---|--:|\n| trackers | {len(trackers)} |\n| agreement threshold | ≥{K} |\n"
           f"| **consensus links** | **{tot_consensus:,}** |\n| union (any tracker) | {tot_union:,} |\n"
           f"| consensus / union | {frac}% |\n\n"
           f"Per-tracker edges: {ptxt}. Saved → `{dst.relative_to(COMP)}`. High-confidence IN-DOMAIN "
           f"pseudo-labels (competition embryos) to train the flow/GNN model on, alongside the external GT.")
    post.post_thread(worker, "all", msg, routine=False, kind="finding")
    return ("done", {"consensus": tot_consensus, "union": tot_union, "frac": frac, "out": str(dst)}, "all", msg)
