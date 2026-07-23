"""temporal-audit — the DEEP temporal / data-quality analysis our per-node profiling was missing. It exists
because the 9-column profile (position, flow, depth, crowding) is BLIND to time-axis artifacts, so we missed
what the discussion flagged: FROZEN frames (vol(t)==vol(t+1)), GLOBAL time-jumps (the whole setup moves → every
cell shifts by the SAME vector at once), non-constant Δt, and natural LONG-JUMPER cells.

The core idea: for each consecutive frame pair, match cells (nearest-neighbour) and DECOMPOSE their displacement
into a GLOBAL component (robust median vector = the setup jump) + a PER-CELL residual (true biology). Then:
  • frozen        = global≈0 AND residual≈0 (and, if the raw volume is loaded, vol(t)≈vol(t+1))
  • global jump   = |global| ≫ typical residual  → a setup translation, CORRECTABLE by subtracting it
  • long-jumper   = a cell whose residual (post-global-correction) is in the far tail
This makes the artifacts MEASURABLE (new columns for matching) and separates a global setup-jump (fixable by
registration before linking) from real motion. GT stays authoritative; this is analysis + optional preprocessing.

A BaseAgent subclass with its own data-wise test. spec: {embryos, per_embryo_limit, check_volume, jump_k}.
"""
from __future__ import annotations
import glob
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent


def _match_disp(a, b, np, cKDTree, r=20.0):
    """Nearest-neighbour displacement vectors from frame a-points to frame b-points (within radius r)."""
    if not len(a) or not len(b):
        return np.zeros((0, 3))
    d, idx = cKDTree(b).query(a, k=1)
    ok = d < r
    return b[idx[ok]] - a[ok]


def audit_dataset(nodes, np, cKDTree, load_vol=None, ad_meta=None, jump_k=3.0):
    """Per-frame-pair global/per-cell motion decomposition + frozen/jump flags for ONE dataset."""
    ts = sorted(nodes["t"].unique())
    pos = {t: nodes.loc[nodes["t"] == t, ["z", "y", "x"]].to_numpy(float) for t in ts}
    globals_mag, percell_med, frozen, njump, jump_frames, longtail = [], [], 0, 0, [], []
    for i in range(len(ts) - 1):
        a, b = pos[ts[i]], pos[ts[i + 1]]
        disp = _match_disp(a, b, np, cKDTree)
        if not len(disp):
            continue
        g = np.median(disp, axis=0); gmag = float(np.linalg.norm(g))     # GLOBAL setup shift (robust)
        resid = np.linalg.norm(disp - g, axis=1)                          # PER-CELL motion after removing global
        rmed = float(np.median(resid))
        globals_mag.append(gmag); percell_med.append(rmed)
        longtail.append(float(np.percentile(resid, 99)) if len(resid) else 0.0)
        # frozen: nothing moved globally OR locally (and same count) — a duplicate/near-duplicate frame
        is_frozen = gmag < 0.5 and rmed < 0.5 and abs(len(a) - len(b)) <= 1
        if load_vol is not None and ad_meta is not None and i % max(1, (len(ts) // 6)) == 0:
            try:                                                         # definitive frozen test on the raw volume
                v0 = load_vol(*ad_meta, ts[i]); v1 = load_vol(*ad_meta, ts[i + 1])
                if np.array_equal(v0, v1): is_frozen = True
            except Exception:  # noqa: BLE001
                pass
        if is_frozen: frozen += 1
        if gmag > jump_k * max(rmed, 1e-6) and gmag > 3.0:               # GLOBAL jump: setup moved ≫ biology
            njump += 1; jump_frames.append((int(ts[i]), round(gmag, 1)))
    n = max(1, len(ts) - 1)
    return {"n_frames": len(ts), "frame_pairs": len(globals_mag),
            "frozen_frame_pct": round(100 * frozen / n, 1),
            "global_jump_frame_pct": round(100 * njump / n, 1),
            "global_shift_med": round(float(np.median(globals_mag)), 2) if globals_mag else 0.0,
            "global_shift_p95": round(float(np.percentile(globals_mag, 95)), 2) if globals_mag else 0.0,
            "global_shift_max": round(float(np.max(globals_mag)), 2) if globals_mag else 0.0,
            "per_cell_motion_med": round(float(np.median(percell_med)), 2) if percell_med else 0.0,
            "long_jumper_p99": round(float(np.median(longtail)), 2) if longtail else 0.0,
            "example_jumps": jump_frames[:5]}


def correct_global_shift(nodes, np, cKDTree, r=20.0):
    """PREPROCESSING (the 'nice addition'): estimate the GLOBAL setup shift per frame-pair (robust median of all
    matched-cell displacements) and subtract the ACCUMULATED drift from every later frame — normalising the setup
    motion away so the linker sees only true per-cell biology (genuine long-jumpers survive as residual). GT stays
    authoritative; this corrects the SIGNAL. Returns a copy of `nodes` with corrected z,y,x + a per-frame shift log."""
    nodes = nodes.sort_values("t").copy()
    ts = sorted(nodes["t"].unique())
    orig = nodes[["z", "y", "x"]].to_numpy(float).copy(); tarr = nodes["t"].to_numpy()
    med = {t: np.median(orig[tarr == t], axis=0) for t in ts}   # RAW matching-free cloud centre per frame
    cum = np.zeros(3); log = []
    for i in range(1, len(ts)):
        g = med[ts[i]] - med[ts[i - 1]]                          # global shift (robust to big jumps, no matching)
        cum = cum + g; log.append((int(ts[i]), round(float(np.linalg.norm(g)), 2)))
        m = tarr == ts[i]
        nodes.loc[nodes["t"] == ts[i], ["z", "y", "x"]] = orig[m] - cum   # subtract ACCUMULATED drift from originals
    return nodes, log


def label_integrity(gn, ge, np, cKDTree):
    """GT/label anomalies discovered FROM the data: duplicate points, out-of-bounds coords, division-degree
    anomalies (>2 children / multi-parent / orphan), overlapping (touching) nuclei, coordinate-scale sanity."""
    out = {}
    P = gn[["z", "y", "x"]].to_numpy(float)
    out["neg_or_nan_coords_pct"] = round(100 * float(np.mean(~np.isfinite(P).all(1) | (P < 0).any(1))), 2)
    # duplicate points within a frame (two labels on one cell)
    dup = 0; overlap = 0; tot = 0
    for t, fr in gn.groupby("t"):
        Q = fr[["z", "y", "x"]].to_numpy(float)
        if len(Q) < 2:
            continue
        d = cKDTree(Q); pairs = d.query_pairs(1.0); dup += len(pairs)
        near = d.query_pairs(3.0); overlap += len(near); tot += len(Q)
    out["dup_point_pct"] = round(100 * dup / max(tot, 1), 2)
    out["overlapping_nuclei_pct"] = round(100 * overlap / max(tot, 1), 2)
    # division-degree anomalies from edges
    if len(ge):
        from collections import Counter
        nid = gn["node_id"].to_numpy() if "node_id" in gn.columns else np.arange(len(gn))
        edges = ge[["source_id", "target_id"]].to_numpy()
        outdeg = Counter(int(s) for s, _ in edges); indeg = Counter(int(t) for _, t in edges)
        out["multi_child_gt3"] = int(sum(1 for v in outdeg.values() if v > 2))   # >2 children = anomaly
        out["multi_parent"] = int(sum(1 for v in indeg.values() if v > 1))       # >1 parent = anomaly
        out["n_divisions"] = int(sum(1 for v in outdeg.values() if v == 2))
    return out


def intensity_quality(load_vol, ad_meta, ts, np, sample=8):
    """Image-quality anomalies FROM the volumes: blank/near-empty frames, saturation, and intensity DRIFT over
    time (acquisition instability). Sampled to stay feasible."""
    if load_vol is None or ad_meta is None:
        return {}
    idx = list(range(0, len(ts), max(1, len(ts) // sample)))[:sample]
    means, blanks, sat = [], 0, []
    for i in idx:
        try:
            v = load_vol(*ad_meta, ts[i]).astype(np.float32)
        except Exception:  # noqa: BLE001
            continue
        mx = float(v.max()) or 1.0; vn = v / mx
        means.append(float(vn.mean()))
        if vn.mean() < 0.01: blanks += 1
        sat.append(float(np.mean(v >= 0.999 * v.max())))
    if not means:
        return {}
    return {"blank_frame_pct": round(100 * blanks / len(means), 1),
            "intensity_drift_cv": round(float(np.std(means) / (np.mean(means) + 1e-9)), 3),   # >0.3 = unstable exposure
            "saturation_pct": round(100 * float(np.mean(sat)), 2)}


class TemporalAudit(BaseAgent):
    name = "temporal-audit"
    thread = "A"
    kind = "finding"

    def run(self, q, worker):
        import numpy as np
        from scipy.spatial import cKDTree
        import sys as _s; _s.path.insert(0, str(COMP)); from src import io
        spec = self.spec(q)
        if spec.get("linker_errors"):                              # XAI: WHERE does the (GMC) linker miss GT edges?
            return self._linker_errors(spec, worker, np, cKDTree, io)
        if spec.get("correct_parquet"):                            # APPLY global-shift correction to a node parquet
            import pandas as pd
            src = COMP / spec["correct_parquet"] if not Path(spec["correct_parquet"]).is_absolute() else Path(spec["correct_parquet"])
            try:
                df = pd.read_parquet(src)
            except Exception as e:  # noqa: BLE001
                return self.done({}, f"[{worker}] temporal-audit: could not read {src} ({str(e)[:60]}).")
            if not len(df) or "embryo" not in df.columns:
                return self.done({}, f"[{worker}] temporal-audit: {src} empty or missing 'embryo' column.")
            out = []
            for emb, g in df.groupby("embryo"):                    # correct per crop/dataset (independent Δt drift)
                gc, _ = correct_global_shift(g, np, cKDTree); out.append(gc)
            cdf = pd.concat(out, ignore_index=True)
            dst = COMP / spec.get("out", str(src).replace(".parquet", "_gcorr.parquet"))
            cdf.to_parquet(dst, index=False)
            msg = f"[{worker}] temporal-audit: global-shift-corrected {df['embryo'].nunique()} crops → {dst.name}"
            return self.done({"out": str(dst), "crops": int(df["embryo"].nunique())}, msg)
        train = spec.get("train_dir") or str(COMP / "input" / "biohub-cell-tracking-during-development" / "train")
        embryos = spec.get("embryos") or ["44b6", "6bba"]
        lim = int(spec.get("per_embryo_limit", 8)); check_vol = bool(spec.get("check_volume", True))
        out = {}
        for e in embryos:
            per = []
            for gf in sorted(glob.glob(f"{train}/{e}_*.geff"))[:lim]:
                try:
                    gn, ge = io.read_geff(gf)
                except Exception:  # noqa: BLE001 — skip an unreadable geff
                    continue
                if not len(gn) or gn["t"].nunique() < 3:
                    continue
                ad_meta = None; lv = None
                if check_vol:
                    zf = gf.replace(".geff", ".zarr")
                    if Path(zf).exists():
                        try: ad_meta = io.read_array_meta(zf); lv = io.load_volume
                        except Exception: ad_meta = None  # noqa: BLE001
                row = audit_dataset(gn, np, cKDTree, lv, ad_meta, float(spec.get("jump_k", 3.0)))   # temporal
                row.update(label_integrity(gn, ge, np, cKDTree))                                     # label/GT
                row.update(intensity_quality(lv, ad_meta, sorted(gn["t"].unique()), np))             # image quality
                per.append(row)
            if per:                                                      # aggregate across this embryo's datasets
                keys = set().union(*[set(p) for p in per]) - {"example_jumps"}
                agg = {k: round(float(np.mean([p[k] for p in per if isinstance(p.get(k), (int, float))])), 2)
                       for k in keys if any(isinstance(p.get(k), (int, float)) for p in per)}
                agg["n_datasets"] = len(per); agg["example_jumps"] = sum((p.get("example_jumps", []) for p in per), [])[:6]
                out[e] = agg
        # AUTO-FLAG anomalies: thresholds that mark a metric as an ISSUE the data revealed (not hand-picked).
        THRESH = {"frozen_frame_pct": (">", 1), "global_jump_frame_pct": (">", 2), "dup_point_pct": (">", 0.5),
                  "overlapping_nuclei_pct": (">", 10), "multi_child_gt3": (">", 0), "multi_parent": (">", 0),
                  "neg_or_nan_coords_pct": (">", 0), "blank_frame_pct": (">", 0), "intensity_drift_cv": (">", 0.3),
                  "saturation_pct": (">", 1)}
        issues = []
        for e, v in out.items():
            for k, (op, thr) in THRESH.items():
                val = v.get(k)
                if isinstance(val, (int, float)) and val > thr:
                    issues.append(f"{e}: {k}={val} (>{thr})")
            # jump signature: global shift ≫ per-cell motion = a setup translation
            if v.get("global_shift_p95", 0) > 3 * max(v.get("per_cell_motion_med", 1), 0.1):
                issues.append(f"{e}: GLOBAL-JUMP signature (gshift-p95 {v.get('global_shift_p95')} ≫ per-cell {v.get('per_cell_motion_med')})")
        # WRITE the temporal profile so box-sample (inject_temporal="auto") reproduces these rates, and the
        # matching framework gains the corrected columns (per-cell motion REPLACES the confounded 'speed').
        if out:
            import json as _j
            prof = {"frozen_prob": round(float(np.median([v.get("frozen_frame_pct", 0) for v in out.values()])) / 100, 3),
                    "jump_prob": round(float(np.median([v.get("global_jump_frame_pct", 0) for v in out.values()])) / 100, 3),
                    "jump_scale": round(float(np.median([v.get("global_shift_p95", 7) for v in out.values()])), 2),
                    "per_cell_motion": round(float(np.median([v.get("per_cell_motion_med", 0) for v in out.values()])), 2),
                    "long_jumper_p99": round(float(np.median([v.get("long_jumper_p99", 0) for v in out.values()])), 2),
                    "per_embryo": out}
            (COMP / "config" / "_auto").mkdir(parents=True, exist_ok=True)
            (COMP / "config" / "_auto" / "temporal_pattern.json").write_text(_j.dumps(prof, indent=2))
        self.save_state({"per_embryo": out, "issues": issues})
        from . import ledger
        ledger.log("temporal-audit", summary=f"data-quality scan: {len(issues)} anomaly types found — "
                   + "; ".join(issues[:6]), detail=str(out), kind="finding",
                   recommendation="global jumps SEPARABLE (all cells same shift) → register/subtract before linking; "
                                  "reproduce frozen+jumps+long-jumpers in box-sample; frozen=dedup; DON'T override GT")
        rows = "\n".join(f"| {e} | {v.get('frozen_frame_pct')}% | {v.get('global_jump_frame_pct')}% | "
                         f"{v.get('global_shift_p95')} | {v.get('per_cell_motion_med')} | {v.get('long_jumper_p99')} | "
                         f"{v.get('dup_point_pct')}% | {v.get('overlapping_nuclei_pct')}% | {v.get('multi_child_gt3')} | "
                         f"{v.get('intensity_drift_cv')} |" for e, v in out.items())
        ilist = "\n".join(f"  ⚠️ {s}" for s in issues) or "  (none over threshold)"
        msg = (f"[{worker}] **DATA-QUALITY SCAN** · temporal + label + image-quality anomalies discovered FROM the data\n"
               f"| embryo | frozen% | jump% | gshift-p95 | percell-med | longjmp-p99 | dup% | overlap% | >2child | int-drift |\n"
               f"|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n{rows}\n\n**ISSUES DISCOVERED ({len(issues)}):**\n{ilist}\n"
               f"→ global-jump = SETUP moved (all cells same shift, correctable by registration); frozen = duplicate frames.")
        self.post(worker, "all", msg, routine=False, kind="finding")
        return self.done({"per_embryo": out, "issues": issues}, msg)


    def _linker_errors(self, spec, worker, np, cKDTree, io):
        """Run the GMC linker on GT DETECTIONS (isolates linking from detection) and characterise the GT edges it
        MISSES — by displacement, is-at-division, and local crowding — to find the next linking lever from OUR data."""
        import glob
        from src.link import track_dataset
        from src.config import Config
        from collections import Counter
        cfg = Config()
        train = spec.get("train_dir") or str(COMP / "input" / "biohub-cell-tracking-during-development" / "train")
        out = {}
        for e in spec.get("embryos", ["44b6", "6bba"]):
            miss_disp, hit_disp, miss_div, miss_crowd, n_gt, n_miss = [], [], 0, [], 0, 0
            for gf in sorted(glob.glob(f"{train}/{e}_*.geff"))[:int(spec.get("per_embryo_limit", 8))]:
                try:
                    gn, ge = io.read_geff(gf)
                except Exception:  # noqa: BLE001 — skip an unreadable geff
                    continue
                if not len(gn) or not len(ge) or gn["t"].nunique() < 3:
                    continue
                ts = sorted(gn["t"].unique())
                frames = [gn[gn["t"] == t][["z", "y", "x"]].to_numpy(float) for t in ts]
                nodes, edges = track_dataset(frames, cfg)                      # linker on GT positions
                # map linker (t, rounded pos) -> predicted successor pos; and GT (t,pos) successor pos
                pmap = {(int(n["t"]), round(n["z"]), round(n["y"]), round(n["x"])): int(n["node_id"]) for n in nodes}
                nid2pos = {int(n["node_id"]): (n["z"], n["y"], n["x"]) for n in nodes}
                pred = set();
                for s, d in edges:
                    pred.add((round(nid2pos[s][0]), round(nid2pos[s][1]), round(nid2pos[s][2]),
                              round(nid2pos[d][0]), round(nid2pos[d][1]), round(nid2pos[d][2])))
                nid2 = gn.set_index("node_id"); tmap = nid2["t"].to_dict()
                from collections import Counter as C2
                outdeg = C2(int(s) for s, _ in ge[["source_id", "target_id"]].to_numpy())
                for s, d in ge[["source_id", "target_id"]].to_numpy():
                    if int(s) not in tmap or int(d) not in tmap:
                        continue
                    ps = nid2.loc[int(s), ["z", "y", "x"]].to_numpy(float); pd_ = nid2.loc[int(d), ["z", "y", "x"]].to_numpy(float)
                    disp = float(np.linalg.norm(pd_ - ps)); n_gt += 1
                    key = (round(ps[0]), round(ps[1]), round(ps[2]), round(pd_[0]), round(pd_[1]), round(pd_[2]))
                    if key in pred:
                        hit_disp.append(disp)
                    else:
                        n_miss += 1; miss_disp.append(disp)
                        if outdeg[int(s)] >= 2: miss_div += 1                  # miss at a division
                        fr = gn[gn["t"] == tmap[int(s)]][["z", "y", "x"]].to_numpy(float)
                        if len(fr) >= 2:
                            dd = np.linalg.norm(fr - ps, axis=1); dd = dd[dd > 0]
                            miss_crowd.append(float(dd.min()) if len(dd) else 99)
            if n_gt:
                out[e] = {"gt_edges": n_gt, "missed": n_miss, "miss_rate_pct": round(100 * n_miss / n_gt, 1),
                          "miss_disp_med": round(float(np.median(miss_disp)), 2) if miss_disp else 0,
                          "hit_disp_med": round(float(np.median(hit_disp)), 2) if hit_disp else 0,
                          "miss_at_division_pct": round(100 * miss_div / max(n_miss, 1), 1),
                          "miss_local_nn_med": round(float(np.median(miss_crowd)), 2) if miss_crowd else 0}
        from . import ledger
        ledger.log("temporal-audit", summary="LINKER-XAI (GMC on, GT detections): " + "; ".join(
            f"{e}: miss {v['miss_rate_pct']}% (missed-disp {v['miss_disp_med']} vs hit {v['hit_disp_med']}, "
            f"{v['miss_at_division_pct']}% at divisions)" for e, v in out.items()), detail=str(out), kind="finding",
            recommendation="target the dominant miss cause: high missed-disp→better motion model; high at-division→division linking; low miss-nn→crowding")
        rows = "\n".join(f"| {e} | {v['miss_rate_pct']}% | {v['miss_disp_med']} | {v['hit_disp_med']} | {v['miss_at_division_pct']}% | {v['miss_local_nn_med']} |" for e, v in out.items())
        msg = (f"[{worker}] **LINKER-XAI** (GMC on, GT detections → which GT edges does the linker MISS?)\n"
               f"| embryo | miss% | missed-disp | hit-disp | %at-division | miss local-nn |\n|---|--:|--:|--:|--:|--:|\n{rows}\n"
               f"→ missed-disp ≫ hit-disp = motion-model gap; high %at-division = division-linking gap; low local-nn = crowding.")
        self.post(worker, "all", msg, routine=False, kind="finding")
        return self.done({"per_embryo": out}, msg)


_AGENT = TemporalAudit()


def run(q, worker):
    return _AGENT.run(q, worker)
