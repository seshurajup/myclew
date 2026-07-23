"""psf-deconv verifier — data-wise self-test (runs under cellmot_venv: torch+scipy).

Checks (no heavy movie load):
  1. GATE LOGIC on synthetic gate tables — resolves+fits+precision-up -> GO; no peak resolution -> NO-GO;
     resolves but runtime-infeasible -> NO-GO; resolves+fits but DoG precision drops -> NO-GO.
  2. The PSF is genuinely ANISOTROPIC (z kernel broader than xy) — the light-sheet elongation.
  3. Peak counting + RL deconv on a SYNTHETIC merged-blob pair: two nuclei blurred together (1 blob raw) split
     into 2 peaks after an anisotropic RL deconv — the core Gate-A mechanism actually works on known ground truth.
  4. close_pairs geometry finds a <=7µm GT pair.
"""
import os, sys, importlib.util
import numpy as np

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))


def _load_worker():
    spec = importlib.util.spec_from_file_location("_psf_w", os.path.join(COMP, "fleet_agents", "_psf_deconv_worker.py"))
    w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)
    return w


def _tblA(e44_raw, e44_dec, e44_sig, e6_raw, e6_dec, e6_sig, n=20):
    def emb(raw, dec, sig):
        return {"n": n, "raw_2peak_frac": raw, "deconv_2peak_frac": dec,
                "paired_sig": sig}
    return {"by_embryo": {"44b6": emb(e44_raw, e44_dec, e44_sig), "6bba": emb(e6_raw, e6_dec, e6_sig)}}


def _sig(mean_delta, significant):
    return {"significant": significant, "mean_delta": mean_delta}


def _run():
    print("=== PSF-DECONV VERIFIER ===")
    w = _load_worker(); checks = {}

    # 1. GATE LOGIC ----------------------------------------------------------
    A_go = _tblA(0.20, 0.55, _sig(0.35, True), 0.25, 0.50, _sig(0.25, True))
    B_ok = {"fits_12h": True, "eta_hours_2xT4": 4.0}
    C_up = {"44b6": {"d_precision": 0.03}, "6bba": {"d_precision": 0.02}}
    v = w._verdict(A_go, B_ok, C_up)
    checks["resolve_fit_precup_is_GO"] = v["decision"] == "GO"

    A_no = _tblA(0.30, 0.31, _sig(0.01, False), 0.28, 0.29, _sig(0.0, False))
    v2 = w._verdict(A_no, B_ok, C_up)
    checks["no_resolution_is_NOGO"] = v2["decision"] == "NO-GO" and not v2["gateA"]

    B_slow = {"fits_12h": False, "eta_hours_2xT4": 40.0}
    v3 = w._verdict(A_go, B_slow, C_up)
    checks["runtime_infeasible_is_NOGO"] = v3["decision"] == "NO-GO" and v3["gateB"] is False

    C_down = {"44b6": {"d_precision": -0.05}, "6bba": {"d_precision": -0.04}}
    v4 = w._verdict(A_go, B_ok, C_down)
    checks["precision_drop_is_NOGO"] = v4["decision"] == "NO-GO" and v4["gateC"] is False

    # 2. PSF anisotropy ------------------------------------------------------
    import torch
    dev = "cpu"
    psf, ks = w.gaussian_psf((2.5, 1.0, 1.0), dev)
    checks["psf_z_broader_than_xy"] = ks[0] > ks[2] and abs(float(psf.sum()) - 1.0) < 1e-4

    # 3. Synthetic merged pair splits after RL deconv ------------------------
    # two nuclei 5 vox apart in x, each a small Gaussian; convolve with the anisotropic PSF to MERGE them,
    # then RL-deconv and check peak count goes 1 -> 2.
    Z, Y, X = 16, 32, 32
    zz, yy, xx = np.mgrid[0:Z, 0:Y, 0:X].astype(np.float32)
    def blob(cx): return np.exp(-(((zz - 8) ** 2) / 1.0 + ((yy - 16) ** 2) / 1.0 + ((xx - cx) ** 2) / 1.0))
    truth = blob(14) + blob(18)                       # two nuclei 4 vox apart in x
    # blur (merge) with a WIDE psf (realistic light-sheet blur) so the raw pair reads as ONE blob
    pblur, kb = w.gaussian_psf((3.0, 2.2, 2.2), dev)
    import torch.nn.functional as Ff
    tt = torch.from_numpy(truth)[None, None].float()
    padb = [kb[2] // 2, kb[2] // 2, kb[1] // 2, kb[1] // 2, kb[0] // 2, kb[0] // 2]
    blurred = Ff.conv3d(Ff.pad(tt, padb, mode="reflect"), pblur.view(1, 1, *pblur.shape))[0, 0].numpy()
    md = (3, 3, 3)
    raw_peaks = w.count_peaks(blurred, md)
    dec = w.rl_deconv(blurred / blurred.max(), (3.0, 2.2, 2.2), 30, dev)
    dec_peaks = w.count_peaks(dec, md)
    checks["merged_blob_resolves_after_deconv"] = raw_peaks < 2 and dec_peaks >= 2
    print(f"  (synthetic pair: raw_peaks={raw_peaks} -> deconv_peaks={dec_peaks})")

    # 4. close_pairs geometry ------------------------------------------------
    import pandas as pd
    nodes = pd.DataFrame({"node_id": [1, 2, 3], "t": [0, 0, 0],
                          "z": [10, 12, 10], "y": [50, 50, 200], "x": [50, 53, 50]})  # 1&2 close, 3 far
    cp = w.close_pairs(nodes, 0, 7.0)
    checks["close_pairs_finds_near_pair"] = len(cp) == 1 and cp[0][2] <= 7.0

    ok = all(checks.values())
    for k, val in checks.items():
        print(f"  [{'ok' if val else 'XX'}] {k}")
    print("RESULT", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
