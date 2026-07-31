"""Worker for psf-deconv — runs under cellmot_venv (zarr/geff + torch/cuda + scipy/skimage). Prints one JSON line.

Grounds the feasibility question in the HOST IMAGING PROCESS (Tomer/Keller 2012 SiMView, docs/host_process):
the competition images are light-sheet OPTICAL SECTIONS that were fused + background-corrected but NOT
PSF-deconvolved. Light-sheet has an ELONGATED PSF in z, so in dense tissue (nearest-neighbour ~5.5µm) nuclei
BLUR/MERGE in z -> the merged-peak precision wall. QUESTION: does a light anisotropic PSF deconvolution
SEPARATE merged nuclei enough to matter, and does it fit the 2xT4/12h runtime?

Three gates (honest, per-embryo, NO golden-12 LB prediction):
  GATE A  merged-peak-resolution DIAGNOSTIC (detector-agnostic): at GT close-pairs (two GT nodes <=7µm apart in
          the SAME frame, i.e. the merged-peak regime), count distinct local maxima in a data-sized ROI, RAW vs
          DECONV. A pair is "resolved by deconv" if it went from 1 blob (raw) to >=2 peaks (deconv). This is the
          core signal — if peaks don't separate, z is too coarse to recover and it's an immediate NO-GO.
  GATE B  RUNTIME: measured s/frame for the deconv -> extrapolated ETA for the hidden test (199*100=19,900 frames)
          on 2xT4/12h (scaled from the local GPU by a conservative T4 factor).
  GATE C  DETECTOR precision (optional, default on): run the UNLEARNED classical DoG detector (src.detect, so there
          is NO train/test mismatch) on RAW vs DECONV dense frames, match nodes to GT per frame within 7µm ->
          node precision/recall/F1 per-embryo. Mismatch-free evidence of whether deconv helps DETECTION PRECISION.
          NOTE: the production pilkwang detector is LEARNED on RAW images -> deploying deconv there needs a RETRAIN
          (expensive); this DoG probe isolates the deconv effect from that mismatch.

XAI: every knob is derived from a MEASURED number (voxel scale, 7µm gate, PSF anisotropy from the z/xy sampling
ratio) — see the _why fields. Significance via math_master.paired_delta_report."""
import sys, os, glob, json, time, math
import numpy as np

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from src import io  # noqa: E402

TRAIN = os.path.join(COMP, "input", "biohub-cell-tracking-during-development", "train")
VOX = (1.625, 0.40625, 0.40625)          # z,y,x µm/voxel (measured from zarr multiscale attrs)
GATE_UM = 7.0                            # official one-to-one node match radius


# ---------------------------------------------------------------------------
# PSF + Richardson-Lucy deconvolution on GPU (anisotropic Gaussian, light-sheet)
# ---------------------------------------------------------------------------
def gaussian_psf(sigma_zyx, device):
    """Separable anisotropic 3D Gaussian PSF, normalised to sum 1. Broader in z = the light-sheet elongation."""
    import torch
    ks = [int(2 * math.ceil(3 * s) + 1) for s in sigma_zyx]
    axes = []
    for s, k in zip(sigma_zyx, ks):
        r = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2
        g = torch.exp(-(r ** 2) / (2 * s * s)); g = g / g.sum()
        axes.append(g)
    psf = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return (psf / psf.sum()), ks


def rl_deconv(vol, sigma_zyx, iters, device, amp_dtype="fp32"):
    """Richardson-Lucy deconvolution with an anisotropic Gaussian PSF. Returns a numpy volume (z,y,x)."""
    import torch
    import torch.nn.functional as F
    psf, ks = gaussian_psf(sigma_zyx, device)
    psf_m = torch.flip(psf, dims=(0, 1, 2))                       # mirrored PSF for the correlation step
    k = psf.view(1, 1, *psf.shape); km = psf_m.view(1, 1, *psf_m.shape)
    pad = [ks[2] // 2, ks[2] // 2, ks[1] // 2, ks[1] // 2, ks[0] // 2, ks[0] // 2]

    def conv(x, ker):
        return F.conv3d(F.pad(x, pad, mode="reflect"), ker)

    x = torch.from_numpy(np.ascontiguousarray(vol)).to(device).float()[None, None]
    u = x.clamp_min(1e-6).clone()
    eps = 1e-6
    for _ in range(int(iters)):
        blur = conv(u, k).clamp_min(eps)
        u = u * conv(x / blur, km)
        u = u.clamp_min(0)
    return u[0, 0].detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Peak counting (local maxima) — the detector-agnostic separability probe
# ---------------------------------------------------------------------------
def robust_thr(a):
    """Robust intensity threshold in an ROI: median + 0.5*(p99.5 - median). Data-derived, not eyeballed."""
    bg = float(np.median(a)); hi = float(np.percentile(a, 99.5))
    return bg + 0.5 * max(hi - bg, 1e-6)


def find_peaks(roi, min_dist_vox):
    """Discrete local-maxima coordinates above a robust threshold in an ROI (z,y,x). skimage peak_local_max
    is robust to smooth plateaus/ties (and to the checkerboard RL deconv can amplify); footprint enforces the
    ~half-nearest-neighbour physical separation so two peaks are a real nucleus-width apart."""
    from skimage.feature import peak_local_max
    if roi.size == 0:
        return np.empty((0, 3), int)
    thr = robust_thr(roi)
    fp = np.ones((max(2, min_dist_vox[0]), max(2, min_dist_vox[1]), max(2, min_dist_vox[2])), bool)
    return peak_local_max(roi, footprint=fp, threshold_abs=thr, exclude_border=False)


def count_peaks(roi, min_dist_vox):
    """# distinct nuclei-like local maxima in an ROI (used by the data-wise test / synthetic checks)."""
    return int(len(find_peaks(roi, min_dist_vox)))


def distinct_gt_peaks(roi, aloc, bloc, min_dist_vox, assign_um):
    """The honest merged-vs-resolved probe for ONE GT close-pair, robust to the dense UNLABELLED neighbours in
    the ROI: detect peaks, then count how many DISTINCT peaks are claimed by the two GT nodes A,B (each GT node
    claims its nearest peak within `assign_um`). Returns 0 (neither has a peak), 1 (a single peak serves both =
    MERGED), or 2 (A and B each own a separate peak = RESOLVED)."""
    pk = find_peaks(roi, min_dist_vox)
    if len(pk) == 0:
        return 0
    Pum = pk.astype(np.float64) * np.array(VOX)[None, :]
    claimed = []
    for gt in (aloc, bloc):
        gum = np.asarray(gt, float) * np.array(VOX)
        d = np.sqrt(((Pum - gum[None, :]) ** 2).sum(1))
        j = int(np.argmin(d))
        if d[j] <= assign_um:
            claimed.append(j)
    return len(set(claimed))


# ---------------------------------------------------------------------------
# Dense-frame + close-pair mining
# ---------------------------------------------------------------------------
def densest_datasets(n_per_embryo):
    rows = []
    for g in sorted(glob.glob(os.path.join(TRAIN, "*.geff"))):
        ds = os.path.basename(g)[:-5]; emb = ds.split("_")[0]
        try:
            nodes, _ = io.read_geff(g)
        except Exception:
            continue
        cpf = len(nodes) / max(1, nodes.t.nunique())
        rows.append((emb, ds, cpf))
    out = {}
    for emb in ("44b6", "6bba"):
        er = sorted([r for r in rows if r[0] == emb], key=lambda r: -r[2])
        out[emb] = [r[1] for r in er[:n_per_embryo]]
    return out


def close_pairs(nodes, t, gate_um):
    """GT close-pairs in frame t: node index pairs within gate_um (the merged-peak regime). Returns list of
    (coordA, coordB, dist_um, dz_um) with dz_um = z-only physical separation (the light-sheet-blur axis)."""
    f = nodes[nodes.t == t]
    if len(f) < 2:
        return []
    P = f[["z", "y", "x"]].to_numpy(np.float64)
    Pum = P * np.array(VOX)[None, :]
    out = []
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            d = float(np.sqrt(((Pum[i] - Pum[j]) ** 2).sum()))
            if d <= gate_um:
                dz = abs(Pum[i][0] - Pum[j][0])
                out.append((P[i], P[j], d, dz))
    return out


def run(cfg):
    import torch
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # hardware_tune governance for precision/tf32 (no hand-set dtype)
    hw = {}
    try:
        from fleet_agents import hardware_tune as HT
        hw = HT.load_config()
        if hw.get("allow_tf32"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:  # noqa: BLE001 — standalone worker: no hardware profile = safe defaults, not a crash
        pass

    # ---- XAI: derive every knob from a MEASURED number ----
    def _get(k, d):
        v = cfg.get(k); return d if v is None else v
    sigma_xy = float(_get("sigma_xy", 1.0))                  # ~in-plane PSF radius in voxels
    # z PSF broadening: light-sheet elongated in z; anchor to the physical z/xy sampling anisotropy (VOX ratio ~4x)
    sigma_z = float(_get("sigma_z", round(sigma_xy * VOX[0] / VOX[1], 2)))      # ~4x sigma_xy => ~4.0
    sigma_z = min(sigma_z, 3.0) if _get("cap_sigma_z", True) else sigma_z       # keep it LIGHT (task: 2-3px)
    iters = int(_get("rl_iters", 5))
    n_per_emb = int(_get("n_datasets_per_embryo", 1 if cfg.get("mode", "mini") == "mini" else 3))
    frames_per_ds = int(_get("frames_per_ds", 3 if cfg.get("mode", "mini") == "mini" else 8))

    # ROI half-size = the 7µm gate in voxels (must contain a <=7µm pair + margin); min peak-dist = half the
    # nearest-neighbour spacing (~5.5µm/2 = 2.75µm) so two peaks must be a real nucleus-width apart (this also
    # rejects the checkerboard local-maxima RL deconv can amplify). Floor 2 per axis (z=1 would be degenerate).
    HZ = int(math.ceil(GATE_UM / VOX[0])) + 1                 # ~5
    HXY = int(math.ceil(GATE_UM / VOX[1])) + 1                # ~18
    halfnn = float(cfg.get("min_sep_um", 2.75))
    md = (max(2, int(round(halfnn / VOX[0]))), max(2, int(round(halfnn / VOX[1]))), max(2, int(round(halfnn / VOX[1]))))

    dsel = densest_datasets(n_per_emb)
    run_detector = bool(cfg.get("run_detector", True))

    per_pair = []          # one row per GT close-pair (raw/deconv peak counts)
    det_rows = []          # per-frame detector node P/R (raw & deconv)
    deconv_times = []
    per_ds_resolved = {"44b6": [], "6bba": []}    # per-dataset resolved-fraction (paired A/B for significance)
    per_ds_rawres = {"44b6": [], "6bba": []}

    dog_cfg = None
    if run_detector:
        try:
            from src.config import Config
            from src import detect as DET
            dog_cfg = Config()
        except Exception:
            run_detector = False

    max_pair_frames = int(cfg.get("max_pair_frames", 30))
    for emb, dss in dsel.items():
        for ds in dss:
            nodes, _ = io.read_geff(os.path.join(TRAIN, ds + ".geff"))
            adir, shp, dt = io.read_array_meta(os.path.join(TRAIN, ds + ".zarr"))
            # GT close-pairs (<=7µm) are RARE in this sparse-lineage GT (mostly divisions) -> SCAN ALL frames
            # for pairs (the Gate-A merged regime), not just the densest, so the diagnostic has a real sample.
            pair_frames = [int(t) for t in sorted(nodes.t.unique()) if close_pairs(nodes, int(t), GATE_UM)]
            pair_frames = pair_frames[:max_pair_frames]
            # dense frames = the frames of THIS dataset with the most GT nodes (for the DoG detector gate)
            counts = nodes.groupby("t").size().sort_values(ascending=False)
            dense_frames = [int(t) for t in counts.index[:frames_per_ds]] if run_detector else []
            frames = sorted(set(pair_frames) | set(dense_frames))
            ds_pairs_raw2 = 0; ds_pairs_deconv2 = 0; ds_pairs_tot = 0
            for t in frames:
                pairs = close_pairs(nodes, t, GATE_UM)
                vol = io.load_volume(adir, shp, dt, t).astype(np.float32)
                # normalise to [0,1] robustly (per-frame), matching detector expectations
                lo = float(np.percentile(vol, 1.0)); hi = float(np.percentile(vol, 99.9))
                voln = np.clip((vol - lo) / max(1.0, hi - lo), 0, None)
                t_d = time.time()
                dec = rl_deconv(voln, (sigma_z, sigma_xy, sigma_xy), iters, device)
                deconv_times.append(time.time() - t_d)
                Z, Y, X = vol.shape
                # TIGHT ROI: just contain the pair (<=7µm) + a ~1 nucleus-radius (2.75µm) margin, so the probe
                # is about THESE two nuclei — not the dense unlabelled neighbours a full 7µm box would sweep in.
                marg = (int(math.ceil(2.75 / VOX[0])), int(math.ceil(2.75 / VOX[1])), int(math.ceil(2.75 / VOX[1])))
                assign_um = 3.5                                  # a GT node owns a peak within half the match gate
                for cA, cB, dist, dz in pairs:
                    plo = np.minimum(cA, cB).astype(int); phi = np.maximum(cA, cB).astype(int)
                    z0, z1 = max(0, plo[0] - marg[0]), min(Z, phi[0] + marg[0] + 1)
                    y0, y1 = max(0, plo[1] - marg[1]), min(Y, phi[1] + marg[1] + 1)
                    x0, x1 = max(0, plo[2] - marg[2]), min(X, phi[2] + marg[2] + 1)
                    off = np.array([z0, y0, x0])
                    r_raw = voln[z0:z1, y0:y1, x0:x1]; r_dec = dec[z0:z1, y0:y1, x0:x1]
                    npr = distinct_gt_peaks(r_raw, cA - off, cB - off, md, assign_um)
                    npd = distinct_gt_peaks(r_dec, cA - off, cB - off, md, assign_um)
                    per_pair.append(dict(emb=emb, ds=ds, t=t, dist_um=round(dist, 2), dz_um=round(dz, 2),
                                         raw_peaks=npr, deconv_peaks=npd,
                                         resolved=int(npr < 2 and npd >= 2)))
                    ds_pairs_tot += 1
                    ds_pairs_raw2 += int(npr >= 2); ds_pairs_deconv2 += int(npd >= 2)
                if run_detector and t in dense_frames:
                    for tag, v in (("raw", vol), ("deconv", dec * (hi - lo) + lo)):
                        try:
                            coords, scores = DET.detect_cells(v.astype(np.float32), dog_cfg)
                        except Exception:
                            coords = np.empty((0, 3))
                        det_rows.append(_node_pr(coords, nodes, t, emb, ds, tag))
            if ds_pairs_tot > 0:
                per_ds_resolved[emb].append(ds_pairs_deconv2 / ds_pairs_tot)
                per_ds_rawres[emb].append(ds_pairs_raw2 / ds_pairs_tot)

    # ---- GATE A: merged-peak-resolution diagnostic (per-embryo + z-separated subset) ----
    A = _diagnostic(per_pair, per_ds_rawres, per_ds_resolved)

    # ---- GATE B: runtime ----
    spf = float(np.mean(deconv_times)) if deconv_times else float("nan")
    # T4 conservative slowdown vs the local 5090 for a conv-bound FFT/deconv workload (~5x, single-GPU basis).
    t4_factor = float(cfg.get("t4_slowdown", 5.0))
    ngpu = 2
    frames_test = 199 * 100
    eta_h = (spf * t4_factor * frames_test) / ngpu / 3600.0 if spf == spf else float("nan")
    B = {"sec_per_frame_local": round(spf, 4), "t4_slowdown_assumed": t4_factor, "n_gpu": ngpu,
         "frames_hidden_test": frames_test, "eta_hours_2xT4": round(eta_h, 2),
         "fits_12h": bool(eta_h == eta_h and eta_h <= 12.0),
         "_why": "deconv-only ETA (add detector+linker separately); T4 factor conservative for conv/FFT-bound work"}

    # ---- GATE C: detector precision (DoG, mismatch-free) ----
    C = _detector_gate(det_rows) if run_detector else None

    res = {"mode": cfg.get("mode", "mini"), "sec": round(time.time() - t0, 1),
           "datasets": dsel, "n_close_pairs": len(per_pair),
           "psf": {"sigma_z_vox": sigma_z, "sigma_xy_vox": sigma_xy, "rl_iters": iters,
                   "_why_sigma_z": f"light-sheet z-elongation anchored to z/xy voxel anisotropy {VOX[0]}/{VOX[1]}~4x, "
                                   f"capped to <=3px (kept LIGHT per SiMView no-deconv note)"},
           "roi": {"HZ": HZ, "HXY": HXY, "min_dist_vox": list(md),
                   "_why": f"ROI half = ceil(7µm gate / voxel)+1; min peak-dist ~2µm (half nearest-neighbour)"},
           "device": device, "hw": {k: hw.get(k) for k in ("gpu", "amp_dtype", "allow_tf32")},
           "gateA_diagnostic": A, "gateB_runtime": B, "gateC_detector": C}
    res["verdict"] = _verdict(A, B, C)
    if cfg.get("out"):
        os.makedirs(os.path.dirname(cfg["out"]), exist_ok=True)
        json.dump(res, open(cfg["out"], "w"), indent=2)
    return res


def _node_pr(coords, nodes, t, emb, ds, tag):
    """Per-frame node precision/recall of predicted `coords` vs GT nodes at frame t (7µm one-to-one)."""
    from scipy.optimize import linear_sum_assignment
    g = nodes[nodes.t == t][["z", "y", "x"]].to_numpy(np.float64)
    n_gt = len(g); n_pred = len(coords)
    tp = 0
    if n_gt and n_pred:
        G = g * np.array(VOX)[None, :]; P = np.asarray(coords, float) * np.array(VOX)[None, :]
        D = np.sqrt(((G[:, None, :] - P[None, :, :]) ** 2).sum(2))
        cost = np.where(D <= GATE_UM, D, 1e9)
        ri, ci = linear_sum_assignment(cost)
        tp = int(sum(1 for r, c in zip(ri, ci) if cost[r, c] < 1e9))
    fp = n_pred - tp; fn = n_gt - tp
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return dict(emb=emb, ds=ds, t=int(t), tag=tag, tp=tp, fp=fp, fn=fn,
                prec=round(prec, 4), rec=round(rec, 4))


def _diagnostic(per_pair, raw_res, dec_res):
    """GATE A report: per-embryo resolved-pair counts + z-separated subset + math_master paired significance."""
    import numpy as _np
    out = {"overall": _pair_stats(per_pair)}
    out["by_embryo"] = {}
    for emb in ("44b6", "6bba"):
        rows = [p for p in per_pair if p["emb"] == emb]
        st = _pair_stats(rows)
        # z-dominated subset: pairs whose separation is mostly in z (the light-sheet blur axis)
        zrows = [p for p in rows if p["dz_um"] >= 0.6 * p["dist_um"] and p["dist_um"] > 0]
        st["z_separated"] = _pair_stats(zrows)
        # paired significance (per-dataset raw-2peak vs deconv-2peak fraction) via math_master
        try:
            from fleet_agents import math_master as MM
            a, b = raw_res.get(emb, []), dec_res.get(emb, [])
            st["paired_sig"] = MM.paired_delta_report(a, b) if len(a) >= 3 and len(a) == len(b) else {
                "n": len(a), "note": "too few datasets for paired test (need >=3)"}
        except Exception as e:
            st["paired_sig"] = {"error": str(e)[:80]}
        out["by_embryo"][emb] = st
    return out


def _pair_stats(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0, "resolved": 0, "resolved_frac": None, "raw_2peak_frac": None, "deconv_2peak_frac": None}
    resolved = sum(r["resolved"] for r in rows)
    raw2 = sum(1 for r in rows if r["raw_peaks"] >= 2); dec2 = sum(1 for r in rows if r["deconv_peaks"] >= 2)
    return {"n": n, "resolved": int(resolved), "resolved_frac": round(resolved / n, 3),
            "raw_2peak_frac": round(raw2 / n, 3), "deconv_2peak_frac": round(dec2 / n, 3),
            "mean_raw_peaks": round(float(np.mean([r["raw_peaks"] for r in rows])), 2),
            "mean_deconv_peaks": round(float(np.mean([r["deconv_peaks"] for r in rows])), 2)}


def _detector_gate(det_rows):
    """GATE C: DoG node precision/recall raw vs deconv, per-embryo, + paired significance (math_master)."""
    out = {"note": "UNLEARNED DoG detector -> mismatch-free; pilkwang (learned-on-RAW) would need a RETRAIN to deploy on deconv"}
    for emb in ("44b6", "6bba"):
        er = [r for r in det_rows if r["emb"] == emb]
        agg = {}
        for tag in ("raw", "deconv"):
            tr = [r for r in er if r["tag"] == tag]
            if not tr:
                agg[tag] = None; continue
            tp = sum(r["tp"] for r in tr); fp = sum(r["fp"] for r in tr); fn = sum(r["fn"] for r in tr)
            prec = tp / (tp + fp) if (tp + fp) else 0.0; rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            agg[tag] = {"prec": round(prec, 4), "rec": round(rec, 4), "f1": round(f1, 4),
                        "tp": tp, "fp": fp, "fn": fn}
        # paired per-frame precision significance
        sig = None
        try:
            from fleet_agents import math_master as MM
            key = {}
            for r in er:
                key.setdefault((r["ds"], r["t"]), {})[r["tag"]] = r["prec"]
            a = [v["raw"] for v in key.values() if "raw" in v and "deconv" in v]
            b = [v["deconv"] for v in key.values() if "raw" in v and "deconv" in v]
            sig = MM.paired_delta_report(a, b) if len(a) >= 3 else {"n": len(a), "note": "too few frames"}
        except Exception as e:
            sig = {"error": str(e)[:80]}
        out[emb] = {"raw": agg.get("raw"), "deconv": agg.get("deconv"), "precision_paired_sig": sig,
                    "d_precision": (round(agg["deconv"]["prec"] - agg["raw"]["prec"], 4)
                                    if agg.get("raw") and agg.get("deconv") else None)}
    return out


def _verdict(A, B, C):
    """Honest GO/NO-GO. GO only if deconv RESOLVES merged GT pairs (Gate A significant +), detector precision up
    BOTH embryos (or clearly would after recalibration), AND deconv fits 2xT4/12h (Gate B)."""
    emb = A.get("by_embryo", {})
    resolves = []
    for e in ("44b6", "6bba"):
        st = emb.get(e, {})
        sig = st.get("paired_sig", {})
        # a real separation gain: deconv finds 2 peaks more often than raw AND it is a positive, non-underpowered paired effect
        gain = (st.get("deconv_2peak_frac") or 0) - (st.get("raw_2peak_frac") or 0)
        pos = bool(sig.get("significant") and (sig.get("mean_delta") or 0) > 0) if "significant" in sig else (gain > 0.05)
        resolves.append((st.get("n", 0) >= 5) and gain > 0.02 and (pos or gain > 0.10))
    gateA = any(resolves) and all(r for r, st in zip(resolves, [emb.get("44b6", {}), emb.get("6bba", {})]) if st.get("n", 0) >= 5)
    gateA_any = any(resolves)

    gateB = bool(B.get("fits_12h"))

    detC = True; detC_note = "detector gate skipped"
    if C:
        dps = [C.get(e, {}).get("d_precision") for e in ("44b6", "6bba")]
        dps = [d for d in dps if d is not None]
        detC = all(d is not None and d >= -0.005 for d in dps) and any(d and d > 0 for d in dps) if dps else False
        detC_note = f"DoG Δprecision per-embryo={dps}"

    if not gateA_any:
        return {"decision": "NO-GO",
                "reason": "GATE A FAIL: deconv does NOT resolve merged GT close-pairs into separate peaks — z sampling "
                          "is too coarse to recover merged nuclei. A deconv+retrain would be wasted. Saved cost = WIN.",
                "gateA": False, "gateB": gateB, "gateC": detC, "gateC_note": detC_note}
    if not gateB:
        return {"decision": "NO-GO",
                "reason": f"GATE B FAIL: deconv ETA {B.get('eta_hours_2xT4')}h on 2xT4 exceeds the 12h budget "
                          f"(before detector+linker) — runtime-infeasible on the hidden test.",
                "gateA": gateA_any, "gateB": False, "gateC": detC, "gateC_note": detC_note}
    if C and not detC:
        return {"decision": "NO-GO",
                "reason": f"GATE C FAIL: even the mismatch-free DoG detector does not gain precision from deconv "
                          f"({detC_note}). The learned pilkwang detector (trained on RAW) would need an expensive "
                          f"retrain with no evidence of payoff — do NOT pursue.",
                "gateA": gateA_any, "gateB": True, "gateC": False, "gateC_note": detC_note}
    dec = "GO" if gateA else "WEAK-GO"
    return {"decision": dec,
            "reason": ("deconv resolves merged GT nuclei, improves mismatch-free DoG precision, and fits 2xT4/12h -> "
                       "worth a scaled test + (learned-detector) retrain-on-deconv" if dec == "GO" else
                       "positive on one embryo / borderline — scale (mode=full) before committing; a learned-detector "
                       "retrain is the real cost gate"),
            "gateA": gateA, "gateB": True, "gateC": detC, "gateC_note": detC_note}


if __name__ == "__main__":
    cfg = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"mode": "mini"}
    print(json.dumps(run(cfg)))
