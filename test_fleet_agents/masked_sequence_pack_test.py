"""masked_sequence_pack_test — DATA-WISE verifier for the masked variable-length sequence ops (CMI 2nd).

The whole POINT of masked ops is leakage-freeness: appending garbage PAD timesteps (mask=0) must NOT change
any statistic or pooled vector. We build a padded batch, compute every op, and assert:
  • masked_zscore stats == numpy stats on the valid slice, and are INVARIANT to appended garbage padding;
  • normalized output is 0 on padding and ~unit-variance / ~zero-mean on the valid positions;
  • masked mean/max equal the plain mean/max of the valid slice, and ignore garbage padding;
  • masked attention weights sum to 1 over valid positions and are exactly 0 on padding.
"""
import os, sys
import numpy as np
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import masked_sequence_pack as M


def _run():
    print("=== MASKED-SEQUENCE-PACK DATA-WISE VERIFIER ===")
    checks = {}
    rng = np.random.RandomState(0)

    # 2 samples, 3 channels, valid lengths 4 and 2 in a padded L=6 batch
    N, C, L = 2, 3, 6
    x = rng.randn(N, C, L)
    mask = np.zeros((N, L)); mask[0, :4] = 1; mask[1, :2] = 1

    # --- norm: stats over valid positions only ---
    out, stats = M.masked_zscore(x, mask, return_stats=True)
    # reference: pool valid columns across both samples per channel
    valid_cols = np.concatenate([x[0, :, :4], x[1, :, :2]], axis=1)  # (C, 6 valid)
    ref_mean = valid_cols.mean(axis=1); ref_var = valid_cols.var(axis=1)
    checks["norm_mean_matches"] = np.allclose(stats["mean"], ref_mean)
    checks["norm_var_matches"] = np.allclose(stats["var"], ref_var)
    checks["norm_pad_zeroed"] = np.allclose(out[0, :, 4:], 0) and np.allclose(out[1, :, 2:], 0)
    # normalized valid values ~ zero mean / unit var per channel
    norm_valid = np.concatenate([out[0, :, :4], out[1, :, :2]], axis=1)
    checks["norm_zero_mean"] = np.allclose(norm_valid.mean(axis=1), 0, atol=1e-6)
    checks["norm_unit_var"] = np.allclose(norm_valid.var(axis=1), 1, atol=1e-3)

    # --- leakage-free: append GARBAGE padding columns (mask 0) → stats unchanged ---
    xg = np.concatenate([x, 999 * rng.randn(N, C, 3)], axis=2)      # L=9 now, 3 garbage cols
    mg = np.concatenate([mask, np.zeros((N, 3))], axis=1)
    _, stats_g = M.masked_zscore(xg, mg, return_stats=True)
    checks["norm_leakage_free"] = np.allclose(stats_g["mean"], stats["mean"]) and np.allclose(stats_g["var"], stats["var"])

    # --- pooling ---
    mean_pool = M.masked_mean_pool(x, mask)
    checks["mean_pool_sample0"] = np.allclose(mean_pool[0], x[0, :, :4].mean(axis=1))
    checks["mean_pool_sample1"] = np.allclose(mean_pool[1], x[1, :, :2].mean(axis=1))
    checks["mean_pool_leakage_free"] = np.allclose(M.masked_mean_pool(xg, mg), mean_pool)

    max_pool = M.masked_max_pool(x, mask)
    checks["max_pool_sample0"] = np.allclose(max_pool[0], x[0, :, :4].max(axis=1))
    # garbage padding is huge but masked → must NOT leak into max
    checks["max_pool_ignores_garbage"] = np.allclose(M.masked_max_pool(xg, mg), max_pool)

    # --- attention pooling ---
    scores = rng.randn(N, L)
    w = M.masked_softmax(scores, mask)
    checks["attn_weights_sum1"] = np.allclose(w.sum(axis=1), 1.0)
    checks["attn_pad_zero"] = np.allclose(w[0, 4:], 0) and np.allclose(w[1, 2:], 0)
    ap = M.masked_attention_pool(x, mask, scores)
    checks["attn_pool_shape"] = ap.shape == (N, C)

    # --- agent run() wrappers return the standard contract ---
    st, d, to, msg = M.run_norm({"spec": {"x": x.tolist(), "mask": mask.tolist()}}, "test")
    checks["norm_run_done"] = st == "done" and "normalized" in d
    st, d, to, msg = M.run_pool({"spec": {"x": x.tolist(), "mask": mask.tolist(), "how": "max"}}, "test")
    checks["pool_run_done"] = st == "done" and "pooled" in d

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== masked-sequence-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
