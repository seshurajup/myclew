"""sample-match — understand HOW the author sampled/labelled the competition crops and GATE our external
data against that scheme (the user's insight: match the author's sampling, incl. sparse/'missing' labels).

The competition GT (train geffs) is not dense crops — it is a FEW cells (~1-2/frame) tracked FULLY over
~40-100 frames (96-99% linked), on top of a much denser DETECTION set (~82-438 cells/frame). So external
training data must match TWO profiles: the model-input density AND the label structure. This agent measures
the author's profile from the GT and compares our external (or box-sampled) data on each dimension, so we
never train on a mismatched distribution again (the density gap that already killed one transfer).

A BaseAgent subclass with its own data-wise test. Reusable/spec-driven:
{train_dir, external_gt, competition_profile (override), detection_cells}.
"""
from __future__ import annotations
import glob
import os
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent


def dataset_stages(datasets=None):
    """Shared accessor to the per-dataset zebrafish stage (S0..S4) + stage-path — the input for STAGE-AWARE
    agents (detector-select, arch-builder, math-master, xai). Reads the `sample-match`-produced artifacts;
    returns {dataset: {"stage": "S3", "phase": "...", "path": "S2-S3-S4"}} (filtered to `datasets` if given)."""
    import os
    from .base import COMP
    out = {}
    stg = COMP / "results" / "label_selection" / "dataset_zf_stage.parquet"
    pth = COMP / "results" / "label_selection" / "dataset_stage_path.parquet"
    if not os.path.exists(stg):
        return out
    import pandas as pd
    try:
        d = pd.read_parquet(stg)
        paths = pd.read_parquet(pth).set_index("dataset")["stage_path"].to_dict() if os.path.exists(pth) else {}
    except Exception:  # noqa: BLE001 — corrupt stage artifacts → return empty (callers treat stage as unknown)
        return out
    for _, r in d.iterrows():
        out[r["dataset"]] = {"stage": f"S{int(r['stage_idx'])}", "phase": r.get("phase", ""),
                             "path": paths.get(r["dataset"], f"S{int(r['stage_idx'])}")}
    if datasets is not None:
        out = {k: v for k, v in out.items() if k in set(datasets)}
    return out


def label_selection_discriminant(labeled, unlabeled):
    """PURE (data-wise tested). Out of ALL cells a detector finds, WHAT made the annotator pick the labeled
    ones? labeled/unlabeled = {feature: [values]} for cells that DID vs did NOT get a GT label. Per feature,
    Cohen's d = (mean_lab − mean_unlab)/pooled_std — a large |d| means that property SELECTS the labeled cells
    (e.g. brighter, more central, more isolated). Returns rows sorted by |d| desc (the selection criteria)."""
    import numpy as np
    out = []
    for f in labeled:
        a = np.asarray(labeled[f], float); b = np.asarray(unlabeled.get(f, []), float)
        if len(a) < 2 or len(b) < 2:
            continue
        sp = float(np.sqrt((a.var() + b.var()) / 2) + 1e-9)
        out.append({"feature": f, "cohens_d": round(float(a.mean() - b.mean()) / sp, 3),
                    "mean_labeled": round(float(a.mean()), 2), "mean_unlabeled": round(float(b.mean()), 2)})
    out.sort(key=lambda x: -abs(x["cohens_d"]))
    return out


class SampleMatch(BaseAgent):
    name = "sample-match"
    thread = "A"

    def _selection_features(self, ds, nframes=4, detector="cellpose-SAM", F=None, rows=None, noise=None,
                            frame_list=None):
        """Detect ALL cells, split LABELED (≤7µm from a GT node) vs UNLABELED, collect features into F (pooled)
        and per-cell `rows` (frame-wise). SKIPS frozen (bit-identical) frames — detects once, reuses the previous
        frame's centroids (saves the expensive Cellpose call; identical volume ⇒ identical detection). If `noise`
        is a list, records GT LABELS that have NO detected cell within 7µm — candidate annotation NOISE."""
        import sys, hashlib
        sys.path.insert(0, str(COMP)); sys.path.insert(0, str(COMP / "experiments" / "segment"))
        import numpy as np
        from scipy.spatial import cKDTree
        from src import io
        from src.config import Config
        from src.io import embryo_id
        from model_scratch.train_v0 import frames_of
        from experiments.segment.verify_external_detectors_cv import build_candidates
        scale = np.asarray(Config().SCALE)
        key = detector.lower().split("-")[0].split()[0]
        detect = next((c["detect"] for c in build_candidates() if key in c["name"].lower()), None)
        if detect is None:
            raise ValueError(f"detector '{detector}' not found")
        emb = embryo_id(ds)
        gn, _ = io.read_geff(COMP / "input/biohub-cell-tracking-during-development/train" / f"{ds}.geff")
        # a frame_list needs the FULL movie loaded (else frames_of caps T to nframes and filters the list away)
        ad, shape, dtype, T = frames_of(ds, None if frame_list else nframes)
        if F is None:
            F = {g: {"intensity": [], "z_depth": [], "radial": [], "isolation_um": []} for g in ("labeled", "unlabeled")}
        nf = T if nframes is None else min(int(nframes), T)     # None ⇒ ALL frames; else cap at T
        # frame_list = SPECIFIC indices (e.g. [0] golden, or [0,25,50,75,99] diverse); else contiguous 0..nf
        frames_to_do = [t for t in frame_list if 0 <= t < T] if frame_list else list(range(nf))
        prev_hash, prev_pk = None, None
        for t in frames_to_do:
            vol = io.load_volume(ad, shape, dtype, t).astype(np.float32)
            h = hashlib.md5(vol.tobytes()).hexdigest()
            if h == prev_hash and prev_pk is not None:         # FROZEN frame → reuse (skip Cellpose)
                pk = prev_pk
            else:
                pk = np.asarray(detect(vol), float).reshape(-1, 3)
                prev_hash, prev_pk = h, pk
            gf = gn[gn["t"] == t]
            gt_um = gf[["z", "y", "x"]].to_numpy(float) * scale if len(gf) else np.zeros((0, 3))
            # LABEL-NOISE: GT labels with no detected cell within 7µm (suspect annotations)
            if noise is not None and len(gt_um) and len(pk):
                dtree = cKDTree(np.asarray(pk, float) * scale)
                for j, gpos in enumerate(gt_um):
                    d = float(dtree.query(gpos)[0])
                    if d > 7.0:
                        gr = gf.iloc[j]
                        noise.append({"embryo": emb, "dataset": ds, "t": int(t), "z": float(gr["z"]),
                                      "y": float(gr["y"]), "x": float(gr["x"]), "nearest_det_um": round(d, 2)})
            if len(pk) == 0:
                continue
            pk = np.asarray(pk, float)
            cen = np.array([s / 2 for s in vol.shape]); pum_all = pk * scale
            tree = cKDTree(pum_all)
            # VECTORISED features for ALL cells at once (was O(N²) per-cell python loop → the bottleneck):
            d2, _ = tree.query(pum_all, k=2)                    # k=2: [self, nearest-other] → isolation
            iso_all = d2[:, 1] if d2.ndim == 2 and d2.shape[1] > 1 else np.zeros(len(pk))
            dens_all = np.array([len(n) - 1 for n in tree.query_ball_point(pum_all, 15.0)])  # one batched call
            zi = np.clip(pk[:, 0].astype(int), 0, vol.shape[0] - 1)
            yi = np.clip(pk[:, 1].astype(int), 0, vol.shape[1] - 1)
            xi = np.clip(pk[:, 2].astype(int), 0, vol.shape[2] - 1)
            inten_all = vol[zi, yi, xi].astype(float)
            rad_all = np.sqrt(((pk - cen) ** 2).sum(1))
            if len(gt_um):
                lab_all = (cKDTree(gt_um).query(pum_all)[0] <= 7.0).astype(int)
            else:
                lab_all = np.zeros(len(pk), int)
            for k in range(len(pk)):
                g = "labeled" if lab_all[k] else "unlabeled"
                F[g]["intensity"].append(float(inten_all[k])); F[g]["z_depth"].append(float(pk[k, 0]))
                F[g]["radial"].append(float(rad_all[k])); F[g]["isolation_um"].append(float(iso_all[k]))
                if rows is not None:
                    rows.append({"embryo": emb, "dataset": ds, "t": t, "z": float(pk[k, 0]), "y": float(pk[k, 1]),
                                 "x": float(pk[k, 2]), "intensity": round(float(inten_all[k]), 1),
                                 "z_depth": round(float(pk[k, 0]), 1), "radial": round(float(rad_all[k]), 1),
                                 "isolation_um": round(float(iso_all[k]), 2), "local_density": int(dens_all[k]),
                                 "labeled": int(lab_all[k])})
        return F

    def selection_csv(self, datasets, out_path, nframes=5, detector="DoG", frame_list=None):
        """FRAME-WISE per-cell TABLE (one row per detected cell: embryo,dataset,t,z,y,x,intensity,z_depth,radial,
        isolation_um,local_density,labeled) across all datasets — the structured input for eda/selector/math/xai/
        arch-builder. One-time + big (~10M rows for all frames) → saved as PARQUET (columnar, typed, compressed;
        CSV would be ~1GB). Writes per-dataset incrementally so a long run is crash-safe. A small CSV head for
        quick eyeballing; a per-frame aggregate parquet too. `out_path` may end .csv/.parquet — base is reused."""
        import os, pandas as pd
        base = out_path.rsplit(".", 1)[0]
        os.makedirs(os.path.dirname(base), exist_ok=True)
        part_dir = base + "_parts"; os.makedirs(part_dir, exist_ok=True)
        noise_dir = base + "_noise_parts"; os.makedirs(noise_dir, exist_ok=True)
        n_total = 0
        for i, ds in enumerate(datasets):
            if os.path.exists(os.path.join(part_dir, f"{ds}.parquet")):   # RESUME: skip already-done datasets
                continue
            rows, noise = [], []
            try:
                self._selection_features(ds, nframes, detector, F=None, rows=rows, noise=noise,
                                         frame_list=frame_list)
            except Exception:  # noqa: BLE001
                continue
            if rows:
                pd.DataFrame(rows).to_parquet(os.path.join(part_dir, f"{ds}.parquet"))  # crash-safe per dataset
                n_total += len(rows)
            if noise:
                pd.DataFrame(noise).to_parquet(os.path.join(noise_dir, f"{ds}.parquet"))
        # consolidate parts → one parquet + per-frame aggregate + a CSV head preview
        import glob as _g
        parts = [pd.read_parquet(p) for p in sorted(_g.glob(os.path.join(part_dir, "*.parquet")))]
        if not parts:
            return base + ".parquet", base + "_perframe.parquet", 0, 0
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(base + ".parquet")
        # RAW Cellpose detections — the precious one-time artifact (never re-run Cellpose; re-derive ANY feature
        # from these z,y,x centroids + the volumes + GT). Per-dataset parts in <base>_parts/ are also kept.
        df[["embryo", "dataset", "t", "z", "y", "x", "labeled"]].to_parquet(base + "_raw_detections.parquet")
        df.head(2000).to_csv(base + ".head.csv", index=False)
        gp = df.groupby(["embryo", "dataset", "t"])
        agg = gp.apply(lambda g: pd.Series({
            "n_detected": len(g), "n_labeled": int(g["labeled"].sum()),
            "iso_labeled": g.loc[g["labeled"] == 1, "isolation_um"].mean(),
            "iso_unlabeled": g.loc[g["labeled"] == 0, "isolation_um"].mean(),
            "int_labeled": g.loc[g["labeled"] == 1, "intensity"].mean(),
            "int_unlabeled": g.loc[g["labeled"] == 0, "intensity"].mean()}), include_groups=False).reset_index()
        agg.to_parquet(base + "_perframe.parquet")
        # consolidate LABEL-NOISE (GT labels with no nearby detection = suspect annotations)
        noise_parts = [pd.read_parquet(p) for p in sorted(_g.glob(os.path.join(noise_dir, "*.parquet")))]
        n_noise = 0
        if noise_parts:
            ndf = pd.concat(noise_parts, ignore_index=True)
            ndf.to_parquet(base + "_label_noise.parquet"); n_noise = len(ndf)
        return base + ".parquet", base + "_perframe.parquet", n_total, n_noise

    def selection_folder(self, datasets_by_emb, nframes=2, detector="cellpose-SAM", max_ds=6):
        """POOL the labeled-vs-unlabeled features across MULTIPLE datasets per embryo → one robust discriminant
        per embryo (not a single non-representative movie). Cellpose is slow → sample max_ds movies/embryo."""
        out = {}
        for emb, dss in datasets_by_emb.items():
            F = {g: {"intensity": [], "z_depth": [], "radial": [], "isolation_um": []} for g in ("labeled", "unlabeled")}
            used = 0
            for ds in dss[:max_ds]:
                try:
                    self._selection_features(ds, nframes, detector, F=F); used += 1
                except Exception:  # noqa: BLE001
                    continue
            out[emb] = {"datasets": used, "n_labeled": len(F["labeled"]["z_depth"]),
                        "n_unlabeled": len(F["unlabeled"]["z_depth"]),
                        "criteria": label_selection_discriminant(F["labeled"], F["unlabeled"])}
        return out

    def _lineage_structure(self, gn, ge):
        """One dataset → lineage fingerprint (connected components = lineages, singletons, track lengths)."""
        import collections, numpy as np
        ids = set(int(n) for n in gn["node_id"])
        adj = collections.defaultdict(set)
        for s, t in zip(ge.get("source_id", []), ge.get("target_id", [])):
            s, t = int(s), int(t)
            if s in ids and t in ids:
                adj[s].add(t); adj[t].add(s)
        seen, lens = set(), []
        for n in ids:
            if n in seen:
                continue
            stack, c = [n], 0
            while stack:
                x = stack.pop()
                if x in seen:
                    continue
                seen.add(x); c += 1; stack.extend(adj[x] - seen)
            lens.append(c)
        lens = np.array(lens) if lens else np.array([0])
        return {"n_labeled": len(ids), "n_lineages": len(lens),
                "singletons": int((lens == 1).sum()), "track_len_median": float(np.median(lens))}

    def folder_label_scan(self, train_dir, limit=None):
        """FULL-FOLDER label structure per embryo (user 2026-07: 'go through both embryos completely'). Aggregates
        lineage counts / singleton fraction / labeled fraction across EVERY train geff — fast (GT graph only)."""
        import glob, os, numpy as np, sys
        sys.path.insert(0, str(COMP))
        from src import io
        from src.io import embryo_id
        agg = {}
        geffs = sorted(glob.glob(os.path.join(train_dir, "*.geff")))
        if limit:
            geffs = geffs[:limit]
        for g in geffs:
            ds = os.path.basename(g)[:-5]; emb = embryo_id(ds)
            gn, ge = io.read_geff(g)
            estN = io.geff_estimated_nodes(g)
            st = self._lineage_structure(gn, ge)
            a = agg.setdefault(emb, {"n": 0, "lineages": [], "singleton_frac": [], "labeled_pct": [], "nodes": []})
            a["n"] += 1; a["lineages"].append(st["n_lineages"])
            a["singleton_frac"].append(st["singletons"] / max(st["n_lineages"], 1))
            a["labeled_pct"].append(100 * st["n_labeled"] / max(estN, 1)); a["nodes"].append(st["n_labeled"])
        out = {}
        for emb, a in agg.items():
            out[emb] = {"movies": a["n"], "lineages_median": int(np.median(a["lineages"])),
                        "lineages_min": int(np.min(a["lineages"])), "lineages_max": int(np.max(a["lineages"])),
                        "singleton_frac_median": round(float(np.median(a["singleton_frac"])), 3),
                        "labeled_pct_median": round(float(np.median(a["labeled_pct"])), 2),
                        "labeled_nodes_median": int(np.median(a["nodes"]))}
        return out

    def gt_patterns_full(self, train_dir, limit=None):
        """MORE label-selection patterns from GT, over ALL 199 movies × ALL frames (GT-only → fast): temporal
        bias (are labels early/mid/late?), founder fraction (lineages starting at t=0), division rate of labeled
        cells, and per-frame regularity. Complements the lineage-structure + detection-based discriminant."""
        import glob, os, numpy as np, collections, sys
        sys.path.insert(0, str(COMP))
        from src import io
        from src.io import embryo_id
        agg = {}
        geffs = sorted(glob.glob(os.path.join(train_dir, "*.geff")))
        if limit:
            geffs = geffs[:limit]
        for g in geffs:
            ds = os.path.basename(g)[:-5]; emb = embryo_id(ds)
            gn, ge = io.read_geff(g)
            T = int(gn["t"].max()) + 1 if len(gn) else 1
            ids = set(int(n) for n in gn["node_id"])
            # lineage components + founder frames
            adj = collections.defaultdict(set); tmin = {}
            for _, r in gn.iterrows():
                tmin[int(r["node_id"])] = int(r["t"])
            outdeg = collections.Counter()
            for s, t in zip(ge.get("source_id", []), ge.get("target_id", [])):
                s, t = int(s), int(t)
                if s in ids and t in ids:
                    adj[s].add(t); adj[t].add(s); outdeg[s] += 1
            seen, starts = set(), []
            for n in ids:
                if n in seen:
                    continue
                stack, comp = [n], []
                while stack:
                    x = stack.pop()
                    if x in seen:
                        continue
                    seen.add(x); comp.append(x); stack.extend(adj[x] - seen)
                starts.append(min(tmin.get(c, 0) for c in comp))
            div_parents = sum(1 for v in outdeg.values() if v >= 2)
            a = agg.setdefault(emb, {"n": 0, "frac_frames_labeled": [], "founder_t0_frac": [],
                                     "div_rate_labeled": [], "labels_first_half": []})
            a["n"] += 1
            a["frac_frames_labeled"].append(gn["t"].nunique() / max(T, 1))
            a["founder_t0_frac"].append(np.mean([s == 0 for s in starts]) if starts else 0.0)
            a["div_rate_labeled"].append(div_parents / max(len(ids), 1))
            a["labels_first_half"].append(float((gn["t"] < T / 2).mean()) if len(gn) else 0.5)
        out = {}
        for emb, a in agg.items():
            out[emb] = {"movies": a["n"],
                        "frames_labeled_frac": round(float(np.median(a["frac_frames_labeled"])), 3),
                        "founder_at_t0_frac": round(float(np.median(a["founder_t0_frac"])), 3),
                        "division_rate_labeled": round(float(np.median(a["div_rate_labeled"])), 4),
                        "labels_in_first_half": round(float(np.median(a["labels_first_half"])), 3)}
        return out

    # zebrafish developmental phases (E56 image-verified: nucleus size shrinks as count grows; the light-sheet
    # time-lapse spans gastrulation → segmentation). Names by DENSITY order (true cells/frame).
    ZF_PHASES = ["S0 early-gastrula (sparse, large nuclei)", "S1 gastrula", "S2 late-gastrula/epiboly",
                 "S3 early-segmentation", "S4 segmentation/somite (dense, small packed nuclei)"]

    def developmental_stages(self, train_dir=None, nstages=5):
        """Assign each of the 199 datasets a ZEBRAFISH developmental PHASE (S0..S4) from its TRUE cells/frame
        (`estimated_number_of_nodes`) via k-means on log-count (the established E56 staging), named by phase.
        Density = developmental stage (image-verified). Returns per-dataset stage + the embryo×stage cross-tab."""
        import glob, os, numpy as np, pandas as pd, sys
        sys.path.insert(0, str(COMP))
        from src import io
        from src.io import embryo_id
        train_dir = train_dir or str(COMP / "input" / "biohub-cell-tracking-during-development" / "train")
        rows = []
        for g in sorted(glob.glob(os.path.join(train_dir, "*.geff"))):
            ds = os.path.basename(g)[:-5]
            est = io.geff_estimated_nodes(g)                    # already per-frame (E56: 3.8k–79k cells/frame)
            rows.append({"embryo": embryo_id(ds), "dataset": ds, "cells_per_frame": float(est)})
        d = pd.DataFrame(rows)
        from sklearn.cluster import KMeans
        lc = np.log10(d["cells_per_frame"].clip(lower=1).values).reshape(-1, 1)
        km = KMeans(n_clusters=nstages, n_init=10, random_state=42).fit(lc)
        # order clusters by mean density → S0 (sparsest) .. S4 (densest)
        order = np.argsort([lc[km.labels_ == c].mean() for c in range(nstages)])
        remap = {int(old): new for new, old in enumerate(order)}
        d["stage_idx"] = [remap[int(c)] for c in km.labels_]
        d["phase"] = d["stage_idx"].map(lambda i: self.ZF_PHASES[i] if i < len(self.ZF_PHASES) else f"S{i}")
        xt = d.groupby(["stage_idx", "embryo"]).size().unstack(fill_value=0)
        rng = d.groupby("stage_idx")["cells_per_frame"].agg(["min", "max", "count"])
        return d, xt, rng

    def dataset_stage_path(self, parquet_path, train_dir=None):
        """Name each dataset by the STAGE PATH its frames traverse, e.g. 'S1-S2-S3' (user 2026-07). A movie
        progresses developmentally, so its early→late frames cross S0..S4 boundaries. Per frame: scale the
        Cellpose crop count to the dataset's TRUE cells/frame (× est/mean-crop), map to the S0..S4 bands (from
        `developmental_stages`), then compress consecutive-duplicate stages into the path label."""
        import pandas as pd, numpy as np
        d, xt, rng = self.developmental_stages(train_dir)                  # per-dataset true est + S0..S4 bands
        edges = [float(rng.loc[i, "max"]) for i in range(len(rng) - 1)]    # upper edge of S0..S3
        est = d.set_index(["embryo", "dataset"])["cells_per_frame"]
        df = pd.read_parquet(parquet_path)
        fc = df.groupby(["embryo", "dataset", "t"]).size().reset_index(name="crop")
        out = {}
        for (emb, ds), g in fc.groupby(["embryo", "dataset"]):
            g = g.sort_values("t")
            mc = max(float(g["crop"].mean()), 1.0)
            scale = float(est.get((emb, ds), mc)) / mc                     # crop-count → true cells/frame
            stages = ["S" + str(int(sum(t * scale > e for e in edges))) for t in g["crop"]]
            comp = [stages[0]]
            for s in stages[1:]:
                if s != comp[-1]:
                    comp.append(s)
            out[(emb, ds)] = {"path": "-".join(comp), "base_stage": f"S{int(d.set_index(['embryo','dataset']).loc[(emb,ds),'stage_idx'])}",
                              "n_stages": len(set(stages)), "frames": stages}
        return out

    def dataset_diversity(self, parquet_path, k=None):
        """IN-DEPTH per-DATASET diversity from the multi-frame table (frames [0,25,50,75,100]). Signature:
        density trajectory (mean/peak/growth/normalised-growth/temporal-CV) + mean isolation/local-density/
        intensity + labeled-fraction. Sanitises inf/outliers, RobustScaler (skewed cell counts), then k-means —
        with k CHOSEN by silhouette over a range if k is None. Returns (feat_with_cluster, representatives,
        cluster_profiles, silhouette). Representative = the dataset nearest each cluster centroid."""
        import pandas as pd, numpy as np
        df = pd.read_parquet(parquet_path)
        cnt = df.groupby(["embryo", "dataset", "t"]).size().reset_index(name="cells")
        piv = cnt.pivot_table(index=["embryo", "dataset"], columns="t", values="cells", fill_value=0)
        cols = list(piv.columns)
        feat = pd.DataFrame(index=piv.index)
        feat["mean_cells"] = piv.mean(1)
        feat["peak_cells"] = piv.max(1)
        feat["growth"] = piv[cols[-1]] - piv[cols[0]]                       # last − first frame
        feat["growth_frac"] = feat["growth"] / (feat["mean_cells"] + 1)     # normalised trajectory
        feat["cv_time"] = piv.std(1) / (piv.mean(1) + 1.0)                  # +1 (not 1e-9) so near-empty ≠ inf
        agg = df.groupby(["embryo", "dataset"]).agg(iso=("isolation_um", "mean"), dens=("local_density", "mean"),
                                                    inten=("intensity", "mean"), lab_frac=("labeled", "mean"))
        feat = feat.join(agg)
        # sanitise: kill inf, fill NaN with median, clip each column to its 1–99 percentile (kills outliers)
        feat = feat.replace([np.inf, -np.inf], np.nan)
        feat = feat.fillna(feat.median(numeric_only=True))
        for c in feat.columns:
            lo, hi = feat[c].quantile([0.01, 0.99])
            feat[c] = feat[c].clip(lo, hi)
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import silhouette_score
        X = np.nan_to_num(RobustScaler().fit_transform(feat.values), posinf=0.0, neginf=0.0)
        if k is None:                                              # choose k by silhouette over 4..12
            best = (-1, 6)
            for kk in range(4, 13):
                lab = KMeans(n_clusters=kk, n_init=10, random_state=42).fit_predict(X)
                s = silhouette_score(X, lab)
                if s > best[0]:
                    best = (s, kk)
            k = best[1]
        km = KMeans(n_clusters=int(k), n_init=10, random_state=42).fit(X)
        feat["cluster"] = km.labels_
        sil = float(silhouette_score(X, km.labels_))
        reps, profiles = [], {}
        for c in range(int(k)):
            idx = np.where(km.labels_ == c)[0]
            d = np.linalg.norm(X[idx] - km.cluster_centers_[c], axis=1)
            rep = tuple(feat.index[idx[int(d.argmin())]]); reps.append(rep)
            sub = feat.iloc[idx]
            profiles[c] = {"n": int(len(idx)), "rep": f"{rep[0]}/{rep[1]}",
                           "mean_cells": int(sub["mean_cells"].mean()), "growth": int(sub["growth"].mean()),
                           "iso": round(float(sub["iso"].mean()), 1), "dens": round(float(sub["dens"].mean()), 1),
                           "lab_frac": round(float(sub["lab_frac"].mean()), 3),
                           "embryos": sub.index.get_level_values(0).value_counts().to_dict()}
        return feat, reps, profiles, sil

    def frame_diversity(self, parquet_path, k=None):
        """FINER than per-dataset: cluster each (dataset, FRAME) point on its own signature (n_cells this frame,
        mean isolation/local-density/intensity, labeled fraction, z-spread). A dataset that grows/declines over
        time has its early vs late frames land in DIFFERENT clusters → a dataset SPANS multiple regimes (user
        2026-07: 'sub-parts of a dataset can have multiple clusters'). Returns (frame_feat_with_cluster,
        cluster_profiles, per-dataset spanned-clusters, silhouette)."""
        import pandas as pd, numpy as np
        df = pd.read_parquet(parquet_path)
        g = df.groupby(["embryo", "dataset", "t"])
        ff = g.agg(n_cells=("labeled", "size"), iso=("isolation_um", "mean"), dens=("local_density", "mean"),
                   inten=("intensity", "mean"), lab_frac=("labeled", "mean"), z_spread=("z", "std")).reset_index()
        cols = ["n_cells", "iso", "dens", "inten", "lab_frac", "z_spread"]
        X0 = ff[cols].replace([np.inf, -np.inf], np.nan)
        X0 = X0.fillna(X0.median())
        for c in cols:
            lo, hi = X0[c].quantile([0.01, 0.99]); X0[c] = X0[c].clip(lo, hi)
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import silhouette_score
        X = np.nan_to_num(RobustScaler().fit_transform(X0.values), posinf=0.0, neginf=0.0)
        if k is None:
            best = (-1, 6)
            for kk in range(4, 13):
                lab = KMeans(n_clusters=kk, n_init=10, random_state=42).fit_predict(X)
                s = silhouette_score(X, lab)
                if s > best[0]:
                    best = (s, kk)
            k = best[1]
        km = KMeans(n_clusters=int(k), n_init=10, random_state=42).fit(X)
        ff["cluster"] = km.labels_
        sil = float(silhouette_score(X, km.labels_))
        profiles = {}
        for c in range(int(k)):
            sub = ff[ff.cluster == c]
            profiles[c] = {"n_frames": int(len(sub)), "n_cells": int(sub.n_cells.mean()),
                           "iso": round(float(sub.iso.mean()), 1), "dens": round(float(sub.dens.mean()), 1),
                           "lab_frac": round(float(sub.lab_frac.mean()), 3),
                           "frames": sub.t.value_counts().sort_index().to_dict()}
        # per-dataset: how many DISTINCT clusters its frames span (>1 ⇒ traverses regimes)
        spans = ff.groupby(["embryo", "dataset"])["cluster"].agg(lambda s: sorted(set(s)))
        n_span = spans.apply(len)
        return ff, profiles, spans, n_span, sil

    def annotation_gaps(self, train_dir=None, limit=None):
        """CROSS-CHECK for missed data points (GT, all 199): (1) do tracks START/END at the volume EDGE (cells
        entering/leaving) or interior (author stopped)? (2) division SISTER geometry (daughter separation µm);
        (3) cross-dataset CONSISTENCY (IQR of n_tracks/ROI → is the scheme uniform or are there outliers?)."""
        import glob, os, collections, numpy as np, pandas as pd, sys
        sys.path.insert(0, str(COMP))
        from src import io
        from src.config import Config
        from src.io import embryo_id
        from model_scratch.train_v0 import frames_of
        scale = np.asarray(Config().SCALE)
        train_dir = train_dir or str(COMP / "input" / "biohub-cell-tracking-during-development" / "train")
        geffs = sorted(glob.glob(os.path.join(train_dir, "*.geff")))
        if limit:
            geffs = geffs[:limit]
        rows = []
        for g in geffs:
            ds = os.path.basename(g)[:-5]; emb = embryo_id(ds)
            gn, ge = io.read_geff(g)
            if len(gn) < 3:
                continue
            _, shape, _, _ = frames_of(ds, 1)
            Z, Y, X = shape if shape and len(shape) == 3 else (100, 256, 256)
            pos = {int(r.node_id): np.array([r.t, r.z, r.y, r.x], float) for r in gn.itertuples()}
            ids = set(pos); nxt = collections.defaultdict(list); indeg = collections.Counter()
            for s, t in zip(ge.get("source_id", []), ge.get("target_id", [])):
                s, t = int(s), int(t)
                if s in ids and t in ids:
                    nxt[s].append(t); indeg[t] += 1

            def edge_d(n):                                        # min voxel distance of a node to the volume face
                _, z, y, x = pos[n]
                return float(min(z, Z - z, y, Y - y, x, X - x))
            starts = [n for n in ids if indeg[n] == 0]            # track roots
            ends = [n for n in ids if not nxt[n]]                 # track leaves
            start_edge = np.mean([edge_d(n) < 5 for n in starts]) if starts else np.nan
            end_edge = np.mean([edge_d(n) < 5 for n in ends]) if ends else np.nan
            # division sisters: node with ≥2 children → distance between the two daughters
            sis = []
            for s, kids in nxt.items():
                if len(kids) >= 2:
                    a, b = pos[kids[0]][1:] * scale, pos[kids[1]][1:] * scale
                    sis.append(float(np.linalg.norm(a - b)))
            rows.append({"embryo": emb, "dataset": ds, "n_tracks": len(starts),
                         "start_at_edge_frac": start_edge, "end_at_edge_frac": end_edge,
                         "sister_dist_um": float(np.median(sis)) if sis else np.nan,
                         "roi_z_center": float(gn["z"].mean() / max(Z, 1))})
        df = pd.DataFrame(rows)
        numcols = ["n_tracks", "start_at_edge_frac", "end_at_edge_frac", "sister_dist_um", "roi_z_center"]
        for c in numcols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        agg = df.groupby("embryo")[numcols].median()
        return df, agg, df

    def label_selection_deep(self, train_dir=None, limit=None):
        """DEEP: why ~1%? Test the TRACKABILITY hypothesis — the author kept cells that (a) stay MUTUALLY separated
        (labelled cells never come close to each other → no track confusion), (b) are SPREAD out (sampling the
        region, not clustered), (c) move SMOOTHLY (low step-size variability). All GT-only, all 199. Each metric
        is compared to a null of the SAME #cells placed randomly in the labels' ROI, so we see deliberate choice."""
        import glob, os, collections, numpy as np, pandas as pd, sys
        sys.path.insert(0, str(COMP))
        from src import io
        from src.config import Config
        from src.io import embryo_id
        from scipy.spatial import cKDTree
        scale = np.asarray(Config().SCALE)
        rng = np.random.default_rng(42)
        train_dir = train_dir or str(COMP / "input" / "biohub-cell-tracking-during-development" / "train")
        geffs = sorted(glob.glob(os.path.join(train_dir, "*.geff")))
        if limit:
            geffs = geffs[:limit]
        rows = []
        for g in geffs:
            ds = os.path.basename(g)[:-5]; emb = embryo_id(ds)
            gn, ge = io.read_geff(g)
            if len(gn) < 4:
                continue
            # (a) mutual isolation + (b) spread vs random-in-ROI, per frame
            mut, spread_ratio = [], []
            for t, gf in gn.groupby("t"):
                if len(gf) < 3:
                    continue
                P = gf[["z", "y", "x"]].to_numpy(float) * scale
                d = cKDTree(P).query(P, k=2)[0][:, 1]
                mut.append(float(d.min()))                        # closest any two labelled cells get
                lab_mean = float(np.mean(d))
                lo = P.min(0); hi = P.max(0)                      # null: same #cells random in the labels' bbox
                R = rng.uniform(lo, hi, size=P.shape)
                rnd_mean = float(np.mean(cKDTree(R).query(R, k=2)[0][:, 1])) + 1e-9
                spread_ratio.append(lab_mean / rnd_mean)          # >1 ⇒ more spread than random (anti-clustered)
            # (c) motion smoothness: per-track step-size coefficient-of-variation (low ⇒ smooth/predictable)
            ids = set(int(n) for n in gn["node_id"])
            pos = {int(r.node_id): np.array([r.t, r.z, r.y, r.x], float) for r in gn.itertuples()}
            adj = collections.defaultdict(list)
            for s, t in zip(ge.get("source_id", []), ge.get("target_id", [])):
                s, t = int(s), int(t)
                if s in ids and t in ids:
                    adj[s].append(t)
            cvs = []
            seen = set()
            for n in ids:
                if n in seen or n not in adj:
                    continue
                steps, cur = [], n
                while cur in adj and adj[cur]:
                    nxt = adj[cur][0]; seen.add(cur)
                    steps.append(np.linalg.norm((pos[nxt][1:] - pos[cur][1:]) * scale)); cur = nxt
                if len(steps) >= 3:
                    steps = np.array(steps); cvs.append(float(steps.std() / (steps.mean() + 1e-9)))
            rows.append({"embryo": emb, "dataset": ds,
                         "mutual_min_um": float(np.median(mut)) if mut else np.nan,
                         "spread_vs_random": float(np.median(spread_ratio)) if spread_ratio else np.nan,
                         "motion_cv": float(np.median(cvs)) if cvs else np.nan})
        df = pd.DataFrame(rows)
        return df, df.groupby("embryo").median(numeric_only=True)

    def annotation_detail(self, train_dir=None, limit=None):
        """EXHAUSTIVE annotation-scheme audit across ALL 199 movies (GT-only → fast): lineage-tree structure
        (divisions, trees-vs-chains), track continuity (gaps, full-movie span), founder timing, per-frame count
        growth, and the SPATIAL pattern (is the ROI a sub-region? edge-avoidance? z-depth? spatial clustering?).
        So we miss no detail of HOW the author labeled. Returns per-embryo aggregates + overall."""
        import glob, os, collections, numpy as np, pandas as pd, sys
        sys.path.insert(0, str(COMP))
        from src import io
        from src.config import Config
        from src.io import embryo_id
        from model_scratch.train_v0 import frames_of
        scale = np.asarray(Config().SCALE)
        train_dir = train_dir or str(COMP / "input" / "biohub-cell-tracking-during-development" / "train")
        geffs = sorted(glob.glob(os.path.join(train_dir, "*.geff")))
        if limit:
            geffs = geffs[:limit]
        rows = []
        for g in geffs:
            ds = os.path.basename(g)[:-5]; emb = embryo_id(ds)
            gn, ge = io.read_geff(g)
            if not len(gn):
                continue
            T = int(gn["t"].max()) + 1
            ids = set(int(n) for n in gn["node_id"])
            pos = {int(r.node_id): (int(r.t), float(r.z), float(r.y), float(r.x)) for r in gn.itertuples()}
            # lineage graph + divisions (out-degree≥2)
            adj = collections.defaultdict(set); outdeg = collections.Counter()
            for s, t in zip(ge.get("source_id", []), ge.get("target_id", [])):
                s, t = int(s), int(t)
                if s in ids and t in ids:
                    adj[s].add(t); adj[t].add(s); outdeg[s] += 1
            n_div = sum(1 for v in outdeg.values() if v >= 2)
            seen, comps = set(), []
            for n in ids:
                if n in seen:
                    continue
                st, comp = [n], []
                while st:
                    x = st.pop()
                    if x in seen:
                        continue
                    seen.add(x); comp.append(x); st.extend(adj[x] - seen)
                comps.append(comp)
            # per-track: span, gaps, founder start, has-division
            spans, gaps, starts, has_div = [], 0, [], 0
            for comp in comps:
                ts = sorted(pos[c][0] for c in comp)
                spans.append(ts[-1] - ts[0] + 1)
                if len(set(ts)) < (ts[-1] - ts[0] + 1):           # a frame missing inside the track = gap
                    gaps += 1
                starts.append(ts[0])
                if any(outdeg[c] >= 2 for c in comp):
                    has_div += 1
            spans = np.array(spans)
            # per-frame labeled count growth (divisions add nodes)
            perf = gn.groupby("t").size()
            growth = perf.iloc[-1] / max(perf.iloc[0], 1) if len(perf) > 1 else 1.0
            # SPATIAL: volume shape → label ROI fraction, z-depth band, edge distance, spatial spread
            ad, shape, dtype, _ = frames_of(ds, 1)
            Z, Y, X = shape if shape and len(shape) == 3 else (100, 256, 256)
            zz, yy, xx = gn["z"].to_numpy(), gn["y"].to_numpy(), gn["x"].to_numpy()
            roi_frac = (((zz.max() - zz.min()) / max(Z, 1)) * ((yy.max() - yy.min()) / max(Y, 1)) *
                        ((xx.max() - xx.min()) / max(X, 1)))
            edge = float(np.min([zz.min(), Z - zz.max(), yy.min(), Y - yy.max(), xx.min(), X - xx.max()]))
            # spatial spread within a mid frame: mean nearest-labeled-neighbour (crowded vs spread)
            from scipy.spatial import cKDTree
            gm = gn[gn["t"] == int(perf.index[len(perf) // 2])]
            spread = np.nan
            if len(gm) > 2:
                p = gm[["z", "y", "x"]].to_numpy(float) * scale
                spread = float(np.median(cKDTree(p).query(p, k=2)[0][:, 1]))
            # (1) LINKING gate: displacement between consecutive-time nodes within a track (µm)
            disp = []
            for s, t in zip(ge.get("source_id", []), ge.get("target_id", [])):
                s, t = int(s), int(t)
                if s in pos and t in pos:
                    a1 = np.array(pos[s][1:]) * scale; b1 = np.array(pos[t][1:]) * scale
                    disp.append(float(np.sqrt(((a1 - b1) ** 2).sum())))
            link_motion = float(np.median(disp)) if disp else np.nan
            # (3) WHERE the ROI sits: normalised centroid of the labels in the volume (0..1 per axis)
            roi_cz = float(zz.mean() / max(Z, 1)); roi_cy = float(yy.mean() / max(Y, 1)); roi_cx = float(xx.mean() / max(X, 1))
            rows.append({"embryo": emb, "dataset": ds, "n_tracks": len(comps), "n_div": n_div,
                         "link_motion_um": link_motion, "roi_cz": roi_cz, "roi_cy": roi_cy, "roi_cx": roi_cx,
                         "div_per_track": n_div / max(len(comps), 1),
                         "tree_frac": has_div / max(len(comps), 1), "gap_frac": gaps / max(len(comps), 1),
                         "full_span_frac": float((spans >= 0.9 * T).mean()), "median_span": float(np.median(spans)),
                         "founder_t0_frac": float(np.mean([s == 0 for s in starts])),
                         "count_growth": float(growth), "roi_vol_frac": float(roi_frac),
                         "z_band_frac": float((zz.max() - zz.min()) / max(Z, 1)), "edge_dist_vox": edge,
                         "label_spread_um": spread})
        df = pd.DataFrame(rows)
        agg = df.groupby("embryo").median(numeric_only=True)
        return df, agg

    def _competition_profile(self, train_dir, limit=30):
        import numpy as np
        import sys
        sys.path.insert(0, str(COMP / "src")); sys.path.insert(0, str(COMP))
        from src import io
        npf, tl, divf, linked = [], [], [], []
        for g in sorted(glob.glob(os.path.join(train_dir, "*.geff")))[:limit]:
            gn, ge = io.read_geff(g)
            nf = gn["t"].nunique(); npf.append(len(gn) / max(nf, 1))
            tl.append(len(gn) / max(1, len(gn) - len(ge)))            # ~track length (frames per tracked cell)
            from collections import Counter
            out = Counter(int(s) for s, _ in ge[["source_id", "target_id"]].to_numpy())
            divf.append(sum(1 for v in out.values() if v >= 2) / max(len(gn), 1))
            linked.append(len(set(int(s) for s, _ in ge[["source_id", "target_id"]].to_numpy())) / max(len(gn), 1))
        return {"labelled_cells_per_frame": round(float(np.median(npf)), 1),
                "track_length_frames": round(float(np.median(tl)), 0),
                "division_frac": round(float(np.median(divf)), 4),
                "linked_frac": round(float(np.median(linked)), 2)}

    def run(self, q, worker):
        import numpy as np
        import pandas as pd
        spec = self.spec(q)
        # 0) label-SELECTION pattern: out of ALL cells Cellpose finds, WHAT distinguishes the author's chosen
        # labels? (user 2026-07-12 "how/what pattern chose the labels"). Optional (needs GPU) — spec.selection_ds.
        selection = {}
        if spec.get("selection_ds"):
            try:
                rows, counts = self._selection_analysis(spec["selection_ds"], int(spec.get("selection_frames", 4)),
                                                         spec.get("selection_detector", "cellpose-SAM"))
                selection = {"criteria": rows, **counts}
            except Exception as e:  # noqa: BLE001
                selection = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
        # 0b) COMPLETE label-annotation report over ALL 199 movies (user 2026-07: "complete training folder,
        # more patterns"). GT patterns run on all frames of all datasets; the detection discriminant uses DoG
        # (fast) across ALL datasets on a temporal frame sample (Cellpose can't do all frames).
        if spec.get("label_report"):
            import pandas as pd
            from src.io import embryo_id                        # glob, os are module-level (don't shadow)
            train_dir = spec.get("train_dir") or str(COMP / "input" / "biohub-cell-tracking-during-development" / "train")
            struct = self.folder_label_scan(train_dir)                 # lineage structure (all 199)
            gtp = self.gt_patterns_full(train_dir)                     # temporal/founder/division (all 199, all frames)
            all_ds = [os.path.basename(g)[:-5] for g in sorted(glob.glob(os.path.join(train_dir, "*.geff")))]
            sf = spec.get("sel_frames", 5)
            sf = None if sf in (None, "all", "None") else int(sf)  # None ⇒ ALL 100 frames per movie
            fl = spec.get("sel_frame_list")                       # SPECIFIC frames: [0] golden / [0,25,50,75,99] diverse
            det = spec.get("sel_detector", "DoG")
            tag = ("_cellpose" if "cell" in det.lower() else "") + (spec.get("out_tag", ""))
            # FRAME-WISE per-cell table (features + labeled flag + frame) — the shared input, saved as parquet
            csv, perframe, ncells, nnoise = self.selection_csv(
                all_ds, str(COMP / "results" / "label_selection" / f"cells{tag}.parquet"),
                nframes=sf, detector=det, frame_list=fl)
            # discriminant per embryo computed FROM the parquet table (detection runs once)
            df = pd.read_parquet(csv)
            feats = ["intensity", "z_depth", "radial", "isolation_um", "local_density"]
            disc = {}
            for emb, g in df.groupby("embryo"):
                lab = {f: g[g.labeled == 1][f].tolist() for f in feats}
                unl = {f: g[g.labeled == 0][f].tolist() for f in feats}
                disc[emb] = {"n_labeled": int((g.labeled == 1).sum()), "n_unlabeled": int((g.labeled == 0).sum()),
                             "criteria": label_selection_discriminant(lab, unl)}
            report = {"lineage_structure": struct, "gt_patterns": gtp, "selection_discriminant": disc,
                      "csv": csv, "perframe_csv": perframe, "n_cells": ncells, "n_label_noise": nnoise,
                      "raw_detections": csv.replace(".parquet", "_raw_detections.parquet"),
                      "label_noise": csv.replace(".parquet", "_label_noise.parquet")}
            self.save_state({"label_report": report})
            summ = "; ".join(f"{e}: {struct[e]['lineages_median']} lineages, {struct[e]['labeled_pct_median']}% labeled, "
                             f"founder@t0={gtp[e]['founder_at_t0_frac']}, div_rate={gtp[e]['division_rate_labeled']}"
                             for e in struct)
            self.log(f"label-report (all 199): {summ}", kind="finding",
                     recommendation="labels = complete lineages of easy-to-track (isolated/bright/peripheral) cells; "
                                    "recall-tilted detector on those matches the scheme.")
            return self.done({"label_report": report}, f"label-report (ALL 199 movies): {summ}")
        # 1) the author's profile (measured, or overridden for tests)
        prof = spec.get("competition_profile")
        if prof is None:
            train_dir = spec.get("train_dir") or str(COMP / "input" / "biohub-cell-tracking-during-development" / "train")
            if not glob.glob(os.path.join(train_dir, "*.geff")):
                return self.escalate(worker, "researcher", f"[{worker}] sample-match: no competition geffs in {train_dir}.")
            prof = self._competition_profile(train_dir)
        det_cells = float(spec.get("detection_cells", 250))          # model-input density (detections/frame)

        # 2) our external data profile
        ext = spec.get("external_gt") or str(COMP / "results" / "flow_gt" / "flow_node_gt_boxed.parquet")
        ext_density = ext_divfrac = None
        if os.path.exists(ext):
            try:
                df = pd.read_parquet(ext, columns=["embryo", "t", "is_division"])
                ext_density = float(np.nan_to_num(df.groupby(["embryo", "t"]).size().groupby("embryo").median().median()))
                ext_divfrac = round(float(np.nan_to_num(df["is_division"].mean())), 4)
            except Exception:  # noqa: BLE001 — corrupt/missing-column external parquet → treat as no external profile
                ext_density = ext_divfrac = None

        # 3) compare — the matchable model-input density is the key gate
        checks = {}
        if ext_density is not None:
            ratio = ext_density / max(det_cells, 1)
            checks["input_density"] = {"external": round(ext_density, 0), "competition": det_cells,
                                       "ratio": round(ratio, 2), "match": 0.4 <= ratio <= 2.5}
            # divisions are RARE biology (<~5% of cells); the competition per-dataset median is often 0, so
            # gate on an absolute rarity bound, not a relative-to-median-0 threshold.
            checks["division_frac"] = {"external": ext_divfrac, "competition": prof["division_frac"],
                                       "match": ext_divfrac is not None and ext_divfrac <= 0.05}
        # sister-ratio (division geometry the model keys on, per XAI) — SCALE-INVARIANT, so directly gated.
        comp_sr = float(spec.get("competition_sister_ratio", 1.60))
        ext_sr = spec.get("external_sister_ratio")
        if ext_sr is not None:
            ext_sr = float(ext_sr)
            checks["sister_ratio"] = {"external": round(ext_sr, 2), "competition": comp_sr,
                                      "match": abs(ext_sr - comp_sr) <= 0.5}
        all_match = all(c.get("match") for c in checks.values()) if checks else False

        self.save_state({"competition_profile": prof, "detection_cells": det_cells,
                         "external_density": ext_density, "checks": checks, "matched": all_match})
        self.log(summary=f"sample-match: competition labels ~{prof['labelled_cells_per_frame']}/frame over ~{prof['track_length_frames']:.0f} frames; external density {ext_density} vs input {det_cells} → {'MATCH' if all_match else 'MISMATCH'}",
                 detail=f"profile {prof}; checks {checks}", kind="verdict",
                 recommendation=("external sampling matches the author's scheme — safe to train" if all_match
                                 else "MISMATCH — box-sample/subsample external to the competition density before training"))
        rows = "\n".join(f"| {k} | {v.get('external')} | {v.get('competition')} | {'✅' if v.get('match') else '❌'} |"
                         for k, v in checks.items())
        msg = (f"[{worker}] **SAMPLE-MATCH** · author's scheme vs our external data\n"
               f"Author labels **~{prof['labelled_cells_per_frame']}/frame** tracked over **~{prof['track_length_frames']:.0f} frames** "
               f"({int(prof['linked_frac']*100)}% linked, div-frac {prof['division_frac']}); model input ~{det_cells:.0f} detections/frame.\n"
               f"| dimension | external | competition | match |\n|:-|--:|--:|:-:|\n{rows}\n"
               f"**{'✅ external matches — safe to train' if all_match else '❌ MISMATCH — re-sample external before training'}**")
        self.post(worker, "all", msg, routine=False, kind="verdict")
        return self.done({"competition_profile": prof, "external_density": ext_density,
                          "checks": checks, "matched": all_match, "label_selection": selection}, msg)


_AGENT = SampleMatch()


def run(q, worker):
    return _AGENT.run(q, worker)
