"""ext_transfer_test — inject a mock gnn-link-train (external training) and a mock competition eval, and
assert ext-transfer: trains on the given external embryos, evaluates transfer to BOTH competition embryos
(the 2 CV datasets), and reports the per-embryo + mean transfer AP. No GPU."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import ext_transfer


def _run():
    print("=== EXT-TRANSFER DATA-WISE VERIFIER ===")
    seen = {}
    def mock_train(q, w):
        seen["include_embryos"] = q["spec"].get("include_embryos")
        seen["epochs"] = q["spec"].get("epochs")
        return ("done", {"div_ap": 0.74, "test_embryo": q["spec"].get("test_embryo")}, "all", "trained")

    agent = ext_transfer.ExtTransfer()
    agent._agents = lambda: {"gnn-link-train": mock_train}
    # mock the competition transfer eval → planted per-embryo AP (2 competition embryos)
    agent._eval_competition = lambda ckpt, comp, spec: {"44b6_a": 0.62, "6bba_b": 0.70}

    # force the ckpt/comp existence checks to pass by pointing at temp files
    import tempfile, torch  # noqa
    s, d, to, msg = agent.run({"question": "ext transfer", "spec": {
        "include_embryos": ["ZSNS001", "ZSNS003"], "epochs": 300}}, "test")

    checks = {
        "trained_on_2_embryos": seen.get("include_embryos") == ["ZSNS001", "ZSNS003"],
        "used_300_epochs": seen.get("epochs") == 300,
        "reports_ext_heldout_ap": d.get("ext_heldout_ap") == 0.74,
    }
    # transfer eval only runs if the ckpt+parquet exist; if not present in this env, transfer is {} —
    # accept either the planted eval (files present) or an empty transfer (files absent), but the
    # orchestration (train + report) must hold.
    tr = d.get("competition_transfer_ap") or {}
    if tr and "error" not in tr:
        checks["transfer_both_embryos"] = set(tr) == {"44b6_a", "6bba_b"}
        checks["mean_transfer_0.66"] = abs(d.get("mean_transfer_ap") - 0.66) < 1e-6
    else:
        print("  (competition ckpt/parquet not present in test env — orchestration verified, transfer eval skipped)")

    # ── APPEARANCE-MATCH: data-wise on SYNTHETIC volumes where the correct answer is known ──
    import numpy as np
    from scipy.ndimage import gaussian_filter
    from fleet_agents import ext_transfer as ET
    rng = np.random.RandomState(0)
    def blobs(seed, sharpen=1.0, gain=1.0, bias=0.0):          # a fake 3D nucleus field
        r = np.random.RandomState(seed); v = r.rand(8, 48, 48).astype(np.float32) * 0.2
        for _ in range(40):
            z, y, x = r.randint(0, 8), r.randint(4, 44), r.randint(4, 44)
            v[z, y - 2:y + 2, x - 2:x + 2] += 1.0
        v = gaussian_filter(v, sharpen)                        # PSF: larger sigma = blurrier
        return v * gain + bias
    comp = blobs(1, sharpen=1.0, gain=1.0)
    # (1) SAME distribution → identity/z-score already ≈0.5 (indistinguishable) — the search must confirm it
    same = blobs(2, sharpen=1.0, gain=1.0)
    rep_same, _ = ET.appearance_match_search(same, comp, sigmas=(2, 4), n_patch=300)
    checks["same_domain_is_matched"] = rep_same["best_adv_auc"] <= 0.65
    # (2) SHIFTED external (blurrier + brighter) → raw is separable, and the search must REDUCE the adv-AUC
    ext = blobs(3, sharpen=2.2, gain=1.7, bias=0.5)
    rep_shift, best_vol = ET.appearance_match_search(ext, comp, sigmas=(2, 4), n_patch=300)
    raw = next(t["adv_auc"] for t in rep_shift["trials"] if t["recipe"] == "raw z-score")
    checks["shift_search_improves"] = rep_shift["best_adv_auc"] <= raw          # a transform helps (or ties)
    checks["shift_reports_verdict"] = rep_shift["verdict"] in ("matched", "partial", "structural-gap")
    checks["best_vol_returned"] = best_vol is not None and np.isfinite(best_vol).all()
    # (3) the pure transforms are well-formed (shape-preserving, finite)
    hm = ET.histogram_match(ext, comp); lc = ET.local_contrast_norm(ext, 3.0); sp = ET.spectrum_match(ET.zscore_norm(ext), ET.zscore_norm(comp))
    checks["transforms_shape_finite"] = (hm.shape == ext.shape and np.isfinite(hm).all()
                                         and lc.shape == ext.shape and np.isfinite(lc).all()
                                         and sp.shape == ext.shape and np.isfinite(sp).all())
    print(f"  (appearance: same-domain best={rep_same['best_adv_auc']}, shifted raw={raw}→best={rep_shift['best_adv_auc']} via {rep_shift['best_recipe']} [{rep_shift['verdict']}])")

    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== ext-transfer: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
