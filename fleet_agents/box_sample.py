"""box-sample — make the DENSE external embryos look like the SPARSE competition crops by sampling
sub-boxes with a matching CELL COUNT (the user's insight: crop a vertex box so external ≈ competition).

The competition golden-12 datasets are spatial CROPS with ~82–438 cells/frame; the external ZSNS embryos
are whole embryos with ~7k–11k cells/frame — a 20–140× density gap that kills transfer. This agent slides/
samples spatial boxes over each external frame, sized so each box holds ~`target_cells` (competition-like),
relabels each box as its own pseudo-dataset, and preserves per-node flow + division labels. The result is a
DENSITY-MATCHED external GT that a model can learn from and actually transfer to the competition.

A BaseAgent subclass with its own data-wise test. Reusable/spec-driven:
{gt_path, target_cells, boxes_per_frame, out_path, keep_min_cells}.
"""
from __future__ import annotations
import json
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent


def _inject_temporal_artifacts(s, np, rng, frozen_prob=0.0, jump_prob=0.0, jump_scale=7.0, long_jump_prob=0.0, long_jump_scale=4.0):
    """Reproduce the temporal artifacts temporal-audit found in the REAL data so the linker trains on the true
    Δt-irregular regime: FROZEN frames (cells don't move: copy prev-frame positions), GLOBAL setup-jumps (ALL
    cells shift by ONE random vector = the setup moving, accumulated as drift), and natural LONG-JUMPER cells
    (a few cells get extra displacement). Operates on a crop's nodes (t, z, y, x, track_id)."""
    if not (frozen_prob or jump_prob or long_jump_prob):
        return s
    s = s.sort_values("t").copy()
    ts = sorted(s["t"].unique()); cum = np.zeros(3)
    for i, t in enumerate(ts):
        m = (s["t"] == t).to_numpy()
        if i > 0 and rng.random() < frozen_prob:                     # FROZEN: copy previous frame's positions per track
            prev = s[s["t"] == ts[i - 1]]
            pmap = {r.track_id: (r.z, r.y, r.x) for r in prev.itertuples()}
            for j in np.where(m)[0]:
                tid = s.iloc[j]["track_id"]
                if tid in pmap:
                    s.iloc[j, [s.columns.get_loc(c) for c in ("z", "y", "x")]] = pmap[tid]
        if i > 0 and rng.random() < jump_prob:                       # GLOBAL setup jump → accumulate drift
            cum = cum + rng.randn(3) * jump_scale
        if cum.any():
            s.loc[m, ["z", "y", "x"]] = s.loc[m, ["z", "y", "x"]].to_numpy() + cum
        if long_jump_prob:                                           # a few natural long-jumper cells (per-cell tail)
            idx = np.where(m)[0]
            for j in idx:
                if rng.random() < long_jump_prob:
                    s.iloc[j, [s.columns.get_loc(c) for c in ("z", "y", "x")]] = \
                        s.iloc[j][["z", "y", "x"]].to_numpy() + rng.randn(3) * long_jump_scale
    return s


def _keep_complete_tracks(seg, n_tracks, np, rng, track_len_pool=None, full_span_frac=0.0, partial_cap=0.75):
    """Match the FULL competition track pattern: keep `n_tracks` COMPLETE (unbroken) but PARTIAL-length
    lineages, assign a `track_id` per lineage. Each track starts at a random frame and runs a length drawn
    from `track_len_pool` (the competition track-length distribution → matches track-len), following the flow
    field forward (child ≈ pos + (dz,dy,dx)) so frames are contiguous (broken=0). A precise fraction
    `full_span_frac` of tracks span the whole window (length = all frames) → directly sets full-span %."""
    ts = sorted(seg["t"].unique())
    if not ts:
        return seg.assign(track_id=-1)
    seg = seg.copy(); seg["track_id"] = -1
    has_div = "is_division" in seg.columns
    pool = list(track_len_pool) if track_len_pool else None
    keep, tid = [], 0
    for _ in range(int(n_tracks)):
        if pool is None or rng.random() < full_span_frac:      # NO length distribution → keep COMPLETE (full-span)
            L = len(ts)                                        # unbroken tracks (default + the full-span fraction)
        else:                                                  # partial: cap < 0.8·window so it's NEVER full-span
            L = int(min(int(partial_cap * len(ts)), max(2, rng.choice(pool))))  # cap<1 decouples; cap=1 lets full-span emerge
        start = int(rng.randint(0, max(1, len(ts) - L + 1)))
        chain_ts = ts[start:start + L]
        avail = seg[(seg["t"] == chain_ts[0]) & (seg["track_id"] == -1)]
        if not len(avail):
            continue
        i0 = avail.sample(1, random_state=int(rng.randint(10 ** 9))).index[0]
        seg.loc[i0, "track_id"] = tid; keep.append(i0)
        cur0 = seg.loc[i0, ["z", "y", "x", "dz", "dy", "dx"]].to_numpy(dtype=float)
        # DIVISION-AWARE follow: a BFS queue of active heads; at a division node keep BOTH nearest daughters
        # (the SISTER paths, each a new lineage id) so the parent→2-children split is preserved — the +0.1 term.
        heads = [(i0, cur0, 1, tid)]; tid += 1
        while heads:
            idx, cur, fp, my = heads.pop()
            if fp >= len(chain_ts):
                continue
            nxt = seg[(seg["t"] == chain_ts[fp]) & (seg["track_id"] == -1)]
            if not len(nxt):
                continue                                       # chain ends (still unbroken up to here)
            pos = nxt[["z", "y", "x"]].to_numpy(dtype=float); nidx = nxt.index.to_numpy()
            d = np.linalg.norm(pos - (cur[:3] + cur[3:6]), axis=1)
            is_div = has_div and int(seg.loc[idx, "is_division"] or 0) == 1 and len(nxt) >= 2
            picks = list(np.argsort(d)[:2]) if is_div else [int(np.argmin(d))]
            for k in picks:                                    # 2 sisters at a division, else 1 continuation
                jidx = int(nidx[k])
                child = tid if is_div else my                  # sisters get fresh lineage ids; normal step keeps id
                if is_div:
                    tid += 1
                seg.loc[jidx, "track_id"] = child; keep.append(jidx)
                cv = seg.loc[jidx, ["z", "y", "x", "dz", "dy", "dx"]].to_numpy(dtype=float)
                heads.append((jidx, cv, fp + 1, child))
    return seg.loc[sorted(set(keep))]


class BoxSample(BaseAgent):
    name = "box-sample"
    thread = "A"

    def run(self, q, worker):
        import numpy as np
        import pandas as pd
        spec = self.spec(q)
        gt_path = Path(spec.get("gt_path") or (COMP / "results" / "flow_gt" / "flow_node_gt_clean.parquet"))
        if not gt_path.is_absolute():                          # fleet workers run from tools/researchpapers —
            gt_path = COMP / gt_path                            # resolve relative paths against the competition root
        target = float(spec.get("target_cells", 250))          # competition-like cells per box (median ~150-438)
        bpf = int(spec.get("boxes_per_frame", spec.get("n", 3)))   # `n` — alias for boxes_per_frame (crops per frame)
        keep_min = float(spec.get("keep_min_cells", target * 0.4))
        out_path = Path(spec.get("out_path") or (gt_path.parent / "flow_node_gt_boxed.parquet"))
        if not out_path.is_absolute():
            out_path = COMP / out_path
        if not gt_path.exists():
            return self.done({}, f"[{worker}] box-sample: GT missing at {gt_path}.")

        try:
            df = pd.read_parquet(gt_path)
        except Exception as e:  # noqa: BLE001
            return self.done({}, f"[{worker}] box-sample: could not read GT {gt_path}: {str(e)[:80]}.")
        if df is None or not len(df) or "embryo" not in df.columns:
            return self.done({}, f"[{worker}] box-sample: empty/invalid GT at {gt_path}.")
        seed = int(spec.get("seed", 0))                        # RNG seed for reproducible box sampling (default 0 = legacy)
        rng = np.random.RandomState(seed)
        # LABEL-DROP AUGMENTATION (opt-in): the competition author labels only a FEW cells/frame (sparse);
        # external is fully labelled. When on, drop each box's labels to a competition-sampled count with
        # per-box randomness → matches the sparse labelling AND multiplies augmentation (different subset each box).
        match_label = bool(spec.get("match_label_sparsity"))
        # MATCH THE FULL competition pattern (tracks/crop, track-len, full-span, div, broken=0). Reuse the
        # pattern cached by the `ext-label-stats` agent (competition_pattern mode) — no ad-hoc analysis here.
        tracks_pool = spec.get("tracks_per_crop_pool"); tlen_pool = spec.get("track_len_pool")
        if match_label and (not tracks_pool or not tlen_pool):
            import json as _j
            cp = COMP / "config" / "_auto" / "competition_pattern.json"
            if cp.exists():
                try:
                    pat = _j.loads(cp.read_text())
                    tracks_pool = tracks_pool or [v["tracks_per_crop_med"] for v in pat.values() if isinstance(v, dict)]
                    tlen_pool = tlen_pool or [v["track_len_frames_med"] for v in pat.values() if isinstance(v, dict)]
                except Exception:  # noqa: BLE001
                    pass
        tracks_pool = tracks_pool or [3, 27]                    # competition 44b6/6bba tracks-per-crop
        tlen_pool = tlen_pool or [20, 47]                       # competition track length (frames)
        # RANGE MATCH (opt-in): the competition is TWO very different embryos (44b6 few-long-tracks, 6bba
        # many-short-tracks). Matching the median collapses to a fake middle — instead span MIN→MAX by drawing
        # a per-crop regime α∈[0,1] and setting ALL params COHERENTLY along the real 44b6↔6bba line (params stay
        # correlated: few tracks ⇒ long ⇒ full-span ⇒ shallow, like a real embryo). So boxed crops cover both.
        match_range = bool(spec.get("match_range"))
        endpts = None
        if match_range:
            import json as _j
            cp = COMP / "config" / "_auto" / "competition_pattern.json"
            if cp.exists():
                pat = [v for v in _j.loads(cp.read_text()).values() if isinstance(v, dict)]
                if len(pat) >= 2:
                    endpts = sorted(pat, key=lambda v: v.get("tracks_per_crop_med", 0))   # e0=few-tracks, e1=many
        # DISTRIBUTION MATCH (opt-in, the rigorous mode): resample box-sample's knobs from the competition's FULL
        # empirical per-column distributions (every per-crop track-count, every track-length, every crop frame-count,
        # every per-crop z-thickness, every division count) → the boxed distribution matches by construction, not
        # just its median/min/max. Because box-sample already draws via rng.choice(pool), feeding the empirical
        # arrays as pools reproduces the distribution; full-span emerges from track_len vs window (both resampled).
        match_dist = bool(spec.get("match_dist")); dpool = None
        if match_dist:
            import json as _j
            dp = COMP / "config" / "_auto" / "competition_dist.json"
            if dp.exists():
                dd = _j.loads(dp.read_text())
                dpool = {k: [x for e in dd.values() for x in e.get(k, [])]         # pool both embryos' arrays
                         for k in ("tracks_per_crop", "track_len", "total_frames", "divisions_per_crop", "z_std")}
                if dpool.get("tracks_per_crop"):
                    tracks_pool = dpool["tracks_per_crop"]        # resample the empirical distributions
                if dpool.get("track_len"):
                    tlen_pool = dpool["track_len"]
        out_frames = []
        for emb in df["embryo"].unique():
            e = df[df["embryo"] == emb]
            # size the box (y,x) so expected cells ≈ target: box_area/frame_area = target/cells_per_frame
            cpf = e.groupby("t").size().median()
            if cpf <= target:
                out_frames.append(e.assign(embryo=f"{emb}_full")); continue
            ymin, ymax = e["y"].min(), e["y"].max(); xmin, xmax = e["x"].min(), e["x"].max()
            zmin, zmax = e["z"].min(), e["z"].max()
            ry, rx = ymax - ymin, xmax - xmin
            zrange = float(zmax - zmin) or 1.0
            ycol = e["y"].to_numpy(); xcol = e["x"].to_numpy(); zcol = e["z"].to_numpy()
            # Z-SLAB: the competition crops are THIN in depth (z_std ~16); a full-depth column has z_std ~68.
            # Crop a z-slab of height `z_slab` (uniform slab z_std = h/sqrt(12)) → matches the competition depth.
            z_slab = spec.get("z_slab")                        # None = keep full depth (legacy)
            # DECOUPLE density from depth: a z-slab keeps only z_slab/zrange of the cells, so WIDEN the y/x box by
            # that factor → cells/frame (and nn_dist crowding) stay at `target` while z_slab sets depth independently.
            zcomp = (zrange / float(z_slab)) if z_slab else 1.0
            frac = float(np.sqrt((target / cpf) * zcomp))
            by, bx = min(ry, ry * frac), min(rx, rx * frac)    # clamp to the embryo extent
            # FIXED spatial boxes applied across ALL frames → a cell stays in its box over time, so its
            # COMPLETE path is preserved (the author labels full trajectories; a per-frame random box would
            # fragment them). Each box is a spatio-temporal crop, like a real competition crop.
            n_boxes = int(spec.get("n_boxes", bpf * 6))
            win = int(spec.get("target_frames", 56))         # temporal window ≈ competition track length
            zk = float(spec.get("z_slab_per_std", 4.16))     # empirical z_slab/z_std ratio (slab 48 → z_std 11.5)
            z_slab_pool = spec.get("z_slab_pool")             # per-crop depth range (INPUT match, dense labels ok)
            if z_slab_pool == "auto":                          # derive from competition per-crop z_std × zk (spans both embryos)
                import json as _j
                dp = COMP / "config" / "_auto" / "competition_dist.json"
                if dp.exists():
                    zsd = [x for e in _j.loads(dp.read_text()).values() for x in e.get("z_std", [])]
                    z_slab_pool = [max(8.0, v * zk) for v in zsd] or None
            for bi in range(n_boxes):
                a_crop = None; z_slab_c = z_slab; by_c, bx_c = by, bx; win_c = None
                if z_slab_pool:                                # vary depth per crop to SPAN both embryos (44b6 shallow ↔ 6bba deep)
                    z_slab_c = float(rng.choice(z_slab_pool))
                    zc = (zrange / z_slab_c) if z_slab_c else 1.0
                    fr = float(np.sqrt((target / cpf) * zc))
                    by_c, bx_c = min(ry, ry * fr), min(rx, rx * fr)
                elif dpool:                                    # DIST MODE: draw crop depth + window from the empirical arrays
                    if dpool.get("z_std"):
                        z_slab_c = max(8.0, float(rng.choice(dpool["z_std"])) * zk)
                    if dpool.get("total_frames"):
                        win_c = int(rng.choice(dpool["total_frames"]))
                    zc = (zrange / z_slab_c) if z_slab_c else 1.0
                    fr = float(np.sqrt((target / cpf) * zc))
                    by_c, bx_c = min(ry, ry * fr), min(rx, rx * fr)
                elif endpts:                                   # per-crop regime α → coherent point on 44b6↔6bba line
                    def _lp(k, a=None):
                        a = a_crop if a is None else a
                        return endpts[0].get(k, 0) * (1 - a) + endpts[1].get(k, 0) * a
                    a_crop = float(rng.random())
                    z_slab_c = max(8.0, _lp("z_std") * zk)     # depth for this regime
                    zc = (zrange / z_slab_c) if z_slab_c else 1.0
                    fr = float(np.sqrt((target / cpf) * zc))
                    by_c, bx_c = min(ry, ry * fr), min(rx, rx * fr)
                # DIVISION-TARGETED sampling (XAI: divisions are the linker's biggest miss + the +0.1 term): center a
                # fraction of boxes ON a real division event so the training data over-represents the weak case.
                divp = float(spec.get("div_oversample", 0.0))
                if divp > 0 and rng.random() < divp and "is_division" in e.columns:
                    dv = e[e["is_division"] == 1]
                    if len(dv):
                        c = dv.sample(1, random_state=int(rng.randint(10 ** 9))).iloc[0]
                        y0 = float(np.clip(c["y"] - by_c / 2, ymin, max(ymin, ymax - by_c)))
                        x0 = float(np.clip(c["x"] - bx_c / 2, xmin, max(xmin, xmax - bx_c)))
                    else:
                        y0 = rng.uniform(ymin, max(ymin, ymax - by_c)); x0 = rng.uniform(xmin, max(xmin, xmax - bx_c))
                else:
                    y0 = rng.uniform(ymin, max(ymin, ymax - by_c)); x0 = rng.uniform(xmin, max(xmin, xmax - bx_c))
                m = (ycol >= y0) & (ycol < y0 + by_c) & (xcol >= x0) & (xcol < x0 + bx_c)
                if z_slab_c:                                   # thin z-slab → match competition depth (z_std)
                    z0 = rng.uniform(zmin, max(zmin, zmax - float(z_slab_c)))
                    m = m & (zcol >= z0) & (zcol < z0 + float(z_slab_c))
                if m.sum() < keep_min:
                    continue
                box = e[m]                                    # same region, all frames = full paths preserved
                if box.groupby("t").size().median() < keep_min:
                    continue
                # VARIABLE window per box: the author's crops vary in frame-count, so a fixed-length track is
                # full-span in a SHORT crop and partial in a LONG one → matches full-span %. Draw win per box
                # from window_pool (default = [target_frames]); short entries create the full-span tail.
                if win_c is not None:                          # dist mode → window resampled from competition frame-counts
                    win = win_c
                else:
                    wpool = spec.get("window_pool") or [win]
                    win = int(rng.choice(wpool))
                t0, t1 = int(box["t"].min()), int(box["t"].max())
                stride = int(spec.get("time_stride", 1))          # subsample frames → cells move `stride`× further/frame
                for w in range(t0, t1 + 1, win):
                    seg = box[(box["t"] >= w) & (box["t"] < w + win)]
                    if stride > 1:                                # keep every `stride`-th frame, renumber t contiguous → speed↑
                        kept = sorted(seg["t"].unique())[::stride]
                        seg = seg[seg["t"].isin(kept)].copy()
                        seg["t"] = seg["t"].map({t: i for i, t in enumerate(kept)})
                    if seg["t"].nunique() >= min(10, win // 2) and seg.groupby("t").size().median() >= keep_min:
                        s = seg.copy()
                        if match_label:                       # keep n_tracks COMPLETE but PARTIAL-length lineages
                            if dpool is not None:             # DIST mode: resample counts+lengths → full-span EMERGES (cap=1)
                                n_tr = int(rng.choice(tracks_pool))   # empirical tracks/crop
                                s = _keep_complete_tracks(s, n_tr, np, rng, tlen_pool, 0.0, partial_cap=1.0)
                                # divisions: resample per-crop count → keep-prob = P(div>0) in the empirical dist
                                dv = dpool.get("divisions_per_crop") or [1]
                                kdp = 1.0 if int(rng.choice(dv)) > 0 else 0.0
                            elif a_crop is not None:          # RANGE mode: this crop's params interpolate 44b6↔6bba
                                n_tr = max(1, int(round(_lp("tracks_per_crop_med"))))
                                tlen_c = [max(2, int(round(_lp("track_len_frames_med"))))]
                                fsf = float(_lp("full_span_pct")) / 100.0
                                kdp = float(min(1.0, _lp("divisions_per_crop_med")))   # 0-div regime keeps none
                                s = _keep_complete_tracks(s, n_tr, np, rng, tlen_c, fsf)
                            else:
                                n_tr = int(rng.choice(tracks_pool))  # → matches tracks/crop + track-len + full-span + broken=0
                                fsf = float(spec.get("full_span_frac", 0.0))   # precise full-span % control
                                s = _keep_complete_tracks(s, n_tr, np, rng, tlen_pool, fsf)
                                kdp = float(spec.get("keep_division_prob", 1.0))   # thin divisions to match comp div/crop
                            if kdp < 1.0 and "track_id" in s.columns and "is_division" in s.columns:
                                # drop WHOLE division-containing tracks (not nodes) → div↓ WITHOUT breaking lineages
                                div_tracks = s.loc[s["is_division"] == 1, "track_id"].unique()
                                drop_t = [t for t in div_tracks if rng.random() > kdp]
                                if drop_t:
                                    s = s[~s["track_id"].isin(drop_t)]
                        if "track_id" in s.columns and spec.get("inject_temporal"):   # reproduce real Δt artifacts
                            ta = spec.get("inject_temporal")
                            if ta == "auto":                                          # read measured competition rates
                                import json as _j
                                tp = COMP / "config" / "_auto" / "temporal_pattern.json"
                                p = _j.loads(tp.read_text()) if tp.exists() else {}
                                ta = {"frozen_prob": p.get("frozen_prob", 0.05), "jump_prob": p.get("jump_prob", 0.15),
                                      "jump_scale": p.get("jump_scale", 7.0), "long_jump_scale": p.get("long_jumper_p99", 4.0)}
                            ta = {} if ta is True else dict(ta)
                            s = _inject_temporal_artifacts(
                                s, np, rng,
                                frozen_prob=float(ta.get("frozen_prob", 0.05)),        # ~ competition frozen rate (1-8%)
                                jump_prob=float(ta.get("jump_prob", 0.15)),            # ~ competition global-jump rate
                                jump_scale=float(ta.get("jump_scale", 7.0)),           # global-shift magnitude (gshift-p95 ~7)
                                long_jump_prob=float(ta.get("long_jump_prob", 0.01)),  # rare natural long-jumper cells
                                long_jump_scale=float(ta.get("long_jump_scale", 4.0)))
                        s["embryo"] = f"{emb}_box{bi}_w{w}"
                        out_frames.append(s)

        if not out_frames:
            return self.done({}, f"[{worker}] box-sample: produced no boxes (check target/keep_min).")
        boxed = pd.concat(out_frames, ignore_index=True)
        med_cells = float(np.nan_to_num(boxed.groupby(["embryo", "t"]).size().groupby("embryo").median().median()))
        n_boxes = boxed["embryo"].nunique()
        n_div = int(boxed["is_division"].sum()) if "is_division" in boxed.columns else 0   # guard missing div col
        try:
            boxed.to_parquet(out_path, index=False)
        except Exception:  # noqa: BLE001
            out_path = out_path.with_suffix(".csv"); boxed.to_csv(out_path, index=False)

        self.save_state({"boxes": n_boxes, "median_cells_per_box": round(med_cells, 1),
                         "target": target, "out": str(out_path), "divisions": n_div})
        self.log(summary=f"box-sample: {n_boxes} density-matched boxes (~{med_cells:.0f} cells, target {target:.0f}); {n_div} divisions kept",
                 detail=f"from {df['embryo'].nunique()} dense external embryos → competition-density crops",
                 recommendation="train the affinity/division model on the BOXED GT — now matches competition density → transferable")
        msg = (f"[{worker}] **BOX-SAMPLE** · density-match external to competition crops\n"
               f"| | cells/frame |\n|---|--:|\n| target (competition) | ~{target:.0f} |\n"
               f"| **boxed external (median)** | **~{med_cells:.0f}** |\n\n"
               f"Made **{n_boxes}** competition-density boxes from the dense embryos ({n_div} "
               f"divisions kept) → `{out_path.name}`. Train on THIS (not the raw 10k-cell frames) so it transfers.")
        self.post(worker, "all", msg, routine=False, kind="finding")
        return self.done({"boxes": n_boxes, "median_cells_per_box": round(med_cells, 1),
                          "target": target, "out": str(out_path)}, msg)


_AGENT = BoxSample()


def run(q, worker):
    return _AGENT.run(q, worker)
