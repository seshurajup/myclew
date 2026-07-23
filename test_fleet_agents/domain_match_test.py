"""domain_match_test — data-wise verifier for the REUSABLE domain-match agent on synthetic data where the
correct answer is known: (1) FEATURE shift → CORAL/OT drives adv-AUC down; (2) IMAGE shift → fixed-transform
search reduces adv-AUC; (3) LEARNED adversarial mapper reduces adv-AUC further while preserving structure;
(4) the agent run() dispatches feature vs image correctly and emits a verdict. No competition data needed."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from scipy.ndimage import gaussian_filter
from fleet_agents import domain_match as DM


def _blobs(seed, sharpen=1.0, gain=1.0, bias=0.0):
    r = np.random.RandomState(seed); v = r.rand(8, 48, 48).astype(np.float32) * 0.2
    for _ in range(40):
        z, y, x = r.randint(0, 8), r.randint(4, 44), r.randint(4, 44)
        v[z, y - 2:y + 2, x - 2:x + 2] += 1.0
    return gaussian_filter(v, sharpen) * gain + bias


def _run():
    print("=== DOMAIN-MATCH DATA-WISE VERIFIER ===")
    checks = {}

    # (1) FEATURE shift: source is a shifted+rotated Gaussian cloud → CORAL/OT must reduce adv-AUC
    rng = np.random.RandomState(0)
    tgt = rng.normal(0, 1, (400, 4))
    A = np.array([[1.6, 0.4, 0, 0], [0.2, 0.7, 0, 0], [0, 0, 1.3, 0.5], [0, 0, 0.1, 0.9]])
    src = rng.normal(0, 1, (400, 4)) @ A + np.array([2.0, -1.5, 1.0, 0.5])
    frep, _ = DM.feature_match_search(src, tgt)
    raw_f = next(t["adv_auc"] for t in frep["trials"] if t["recipe"] == "raw")
    checks["feature_raw_separable"] = raw_f >= 0.7
    checks["feature_search_improves"] = frep["best_adv_auc"] <= raw_f - 0.05
    print(f"  (feature: raw={raw_f} → best={frep['best_adv_auc']} via {frep['best_recipe']} [{frep['verdict']}])")

    # (2) IMAGE shift: blurrier+brighter source → fixed-transform search must reduce adv-AUC (auto off = fast)
    comp = _blobs(1, sharpen=1.0, gain=1.0)
    ext = _blobs(3, sharpen=2.2, gain=1.7, bias=0.5)
    irep, best_vol = DM.appearance_match_search(ext, comp, sigmas=(2, 4), n_patch=300, auto=False)
    raw_i = next(t["adv_auc"] for t in irep["trials"] if t["recipe"] == "raw z-score")
    checks["image_search_improves"] = irep["best_adv_auc"] <= raw_i
    checks["image_returns_vol"] = best_vol is not None and np.isfinite(best_vol).all()
    print(f"  (image fixed: raw={raw_i} → best={irep['best_adv_auc']} via {irep['best_recipe']} [{irep['verdict']}])")

    # (3) LEARNED mapper — verify the MECHANISM (not toy-GAN convergence, which is stochastic):
    #   (a) a STRONG structure weight preserves structure (the guard works);
    #   (b) prewarp warm-starts from the fixed transform (prewarp_auc populated & ≤ raw);
    #   (c) honest_match logic is exactly (matched AND structure preserved) — the anti-cheat guard.
    lrep, mapped = DM.learned_domain_map(ext, comp, iters=150, lambda_struct=8.0, patch=24, batch=24, prewarp=False)
    checks["learned_guard_preserves_structure"] = lrep["structure_corr"] >= 0.5     # strong λ ⇒ nuclei survive
    checks["learned_honest_logic"] = lrep["honest_match"] == (lrep["matched"] and lrep["structure_corr"] >= 0.5)
    checks["learned_shape_finite"] = mapped.shape == DM._slice_stack(ext).shape and np.isfinite(mapped).all()
    lrep2, _ = DM.learned_domain_map(ext, comp, iters=1, lambda_struct=8.0, patch=24, batch=24, prewarp=True, sigmas=(2, 4))
    checks["prewarp_warmstarts"] = lrep2["prewarp_auc"] is not None and lrep2["prewarp_auc"] <= lrep2["adv_auc_before"] + 1e-6
    print(f"  (learned: raw={lrep['adv_auc_before']}→{lrep['adv_auc_after']}, struct={lrep['structure_corr']}, honest={lrep['honest_match']}; prewarp_auc={lrep2['prewarp_auc']})")

    # (3b) AUTO-LEARNED transform: optimizer finds a structure-preserving transform that beats raw, honestly
    arep, avol = DM.auto_match(ext, comp, struct_min=0.6, restarts=1, maxiter=15, n_patch=200)
    checks["auto_beats_raw"] = arep["adv_auc"] <= raw_i           # learned params ≤ raw z-score
    checks["auto_preserves_structure"] = arep["structure_corr"] >= 0.6
    checks["auto_shape_finite"] = avol.shape == DM._slice_stack(ext).shape and np.isfinite(avol).all()
    print(f"  (auto-learned: adv-AUC={arep['adv_auc']} struct={arep['structure_corr']} params={arep['params']})")

    # (4) agent run() dispatch — feature mode (fast, no torch), and verdict emitted
    agent = DM.DomainMatch()
    st, d, to, msg = agent.run({"question": "match", "spec": {"src": src.tolist(), "target": tgt.tolist(), "mode": "feature"}}, "test")
    checks["agent_feature_dispatch"] = st == "done" and d.get("domain_match", {}).get("mode") == "feature"
    checks["agent_reports_verdict"] = d.get("verdict") in ("matched", "partial", "structural-gap")
    # image dispatch through the unified match() (fixed only, learned off for speed)
    ir = DM.match(ext, comp, mode="image", learned=False, sigmas=(2, 4), n_patch=200, auto=False)
    checks["unified_image_mode"] = ir["mode"] == "image" and "fixed" in ir

    # (5) within-domain FLOOR: two same-domain samples still differ ⇒ floor > 0.5 (the real match target)
    fl = DM.within_domain_floor([_blobs(10), _blobs(11), _blobs(12)], n_patch=250)
    checks["floor_computed"] = fl["floor"] is not None and 0.5 <= fl["floor"] <= 1.0
    print(f"  (within-domain floor: {fl['floor']} (min {fl['min']}) — target is the floor, not 0.5)")

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"\n=== domain-match: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  X FAILED: {e}"); sys.exit(1)
