"""gan_train_test — data-wise verifier for the REUSABLE gan-train agent on synthetic volumes: the GPU
adversarial trainer runs, the STRUCTURE guard works (strong λ preserves content), honest_match logic is
exact, apply_fn is shape-preserving/finite, augment mode runs, and the agent run() emits a verdict."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from scipy.ndimage import gaussian_filter
from fleet_agents import gan_train as GT


def _blobs(seed, sharpen=1.0, gain=1.0, bias=0.0):
    r = np.random.RandomState(seed); v = r.rand(8, 40, 40).astype(np.float32) * 0.2
    for _ in range(30):
        z, y, x = r.randint(0, 8), r.randint(4, 36), r.randint(4, 36)
        v[z, y - 2:y + 2, x - 2:x + 2] += 1.0
    return gaussian_filter(v, sharpen) * gain + bias


def _run():
    print("=== GAN-TRAIN DATA-WISE VERIFIER ===")
    checks = {}
    comp = _blobs(1, 1.0, 1.0); ext = _blobs(3, 2.0, 1.6, 0.4)

    # (1) translate mode with STRONG structure weight → content preserved (guard works), metrics well-formed
    apply_fn, m = GT.train_gan(ext, comp, mode="translate", iters=120, lambda_struct=8.0, patch=20, batch=20)
    checks["translate_runs"] = set(m) >= {"adv_auc_before", "adv_auc_after", "structure_corr", "honest_match", "device"}
    checks["guard_preserves_structure"] = m["structure_corr"] >= 0.5
    checks["honest_logic"] = m["honest_match"] == (m["matched"] and m["structure_corr"] >= 0.5)
    mapped = apply_fn(ext)
    checks["apply_shape_finite"] = mapped.shape[1:] == ext.shape[1:] and np.isfinite(mapped).all()
    print(f"  (translate: {m['adv_auc_before']}→{m['adv_auc_after']}, struct={m['structure_corr']}, honest={m['honest_match']}, dev={m['device']})")

    # (2) augment mode runs (noise-conditioned generator) and returns a usable apply_fn
    af2, m2 = GT.train_gan(comp, comp, mode="augment", iters=60, lambda_struct=1.0, patch=20, batch=20)
    out2 = af2(comp)
    checks["augment_runs"] = m2["mode"] == "augment" and out2.shape[1:] == comp.shape[1:] and np.isfinite(out2).all()

    # (3) agent run() emits a verdict
    agent = GT.GanTrain()
    st, d, to, msg = agent.run({"question": "gan", "spec": {"src": ext.tolist(), "target": comp.tolist(),
                                                            "iters": 60, "patch": 20, "batch": 20, "prewarp": False}}, "test")
    checks["agent_runs"] = st == "done" and "gan_metrics" in d

    # (4) domain_match.learned_domain_map now DELEGATES here (no dup) — still returns the expected contract
    from fleet_agents import domain_match as DM
    lrep, lmap = DM.learned_domain_map(ext, comp, iters=40, lambda_struct=8.0, patch=20, batch=20, prewarp=False)
    checks["domain_match_delegates"] = set(lrep) >= {"adv_auc_before", "adv_auc_after", "structure_corr", "honest_match"}

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"\n=== gan-train: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  X FAILED: {e}"); sys.exit(1)
