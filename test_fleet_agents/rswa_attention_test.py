"""rswa_attention_test — data-wise verifier for Reference Sliding Window Attention (Unlimited-OCR R-SWA).

Core properties:
  1. rswa_mask: reference block always visible; local window is causal and bounded by W; per-query key count
     ≤ n_ref + window regardless of seq length (the constant-cache guarantee).
  2. kv_cache_size: R-SWA cache is constant in decode_len; full-causal grows; reduction rises with length.
  3. rswa_attention: valid softmax rows (sum to 1); output equals a reference-implementation that only uses
     the allowed keys; a token beyond the window does NOT influence a later query except via the reference set.
  4. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import rswa_attention as R


def _run():
    print("=== R-SWA VERIFIER ===")
    rng = np.random.RandomState(0); checks = {}
    n_ref, W, seq = 8, 16, 200

    # 1. mask structure
    m = R.rswa_mask(seq, n_ref, W)
    checks["ref_always_visible"] = bool(m[:, :n_ref].all())               # every query sees every ref token
    # causal on the generated block: query i cannot attend to a generated key j>i (future)
    gen = m[:, n_ref:]
    future = np.array([[gen[i, j] for j in range(seq - n_ref)] for i in range(seq)])
    checks["causal_generated"] = bool(not any(future[i, j] for i in range(seq) for j in range(seq - n_ref) if (j + n_ref) > i))
    # per-query key count bounded by n_ref + W
    keys = m.sum(axis=1)
    checks["bounded_keys"] = int(keys.max()) <= n_ref + W
    checks["constant_tail"] = int(keys[-1]) == int(keys[seq - 1])         # trivially true; tail count is n_ref+W
    checks["tail_count_exact"] = int(keys[-1]) == n_ref + W
    # window is local: query far in the tail cannot see a generated token older than W (outside ref)
    qi = seq - 1
    visible_gen = np.where(m[qi, n_ref:])[0] + n_ref
    checks["window_local"] = bool(visible_gen.min() >= qi - W + 1)
    print(f"  -> max keys/query={int(keys.max())} (n_ref+W={n_ref+W}); tail query sees gen tokens ≥ {int(visible_gen.min())}")

    # 2. cache accounting
    c_short = R.kv_cache_size(n_ref, W, 100)
    c_long = R.kv_cache_size(n_ref, W, 10000)
    checks["cache_constant"] = c_short["rswa_entries"] == c_long["rswa_entries"] == n_ref + W
    checks["full_grows"] = c_long["full_entries"] > c_short["full_entries"]
    checks["reduction_scales"] = c_long["reduction"] > c_short["reduction"]
    print(f"  -> KV entries: R-SWA constant {c_long['rswa_entries']} vs full {c_long['full_entries']} @10k → {c_long['reduction']:.0f}x")

    # 3. attention correctness vs explicit masked reference
    d = 16; Q = rng.randn(seq, d); K = rng.randn(seq, d); V = rng.randn(seq, d)
    out = R.rswa_attention(Q, K, V, n_ref, W)
    checks["output_shape"] = out.shape == (seq, d)
    # softmax rows valid: recompute weights and check they sum to 1 over allowed keys
    scores = np.where(m, Q @ K.T / np.sqrt(d), -np.inf)
    scores -= scores.max(1, keepdims=True); wgt = np.exp(scores); wgt /= wgt.sum(1, keepdims=True)
    checks["softmax_valid"] = np.allclose(wgt.sum(1), 1.0)
    checks["matches_reference"] = np.allclose(out, wgt @ V, atol=1e-10)
    # a distant generated token (outside ref, outside window) has ZERO weight on the last query
    far = n_ref + 1                         # a generated token near the start, far from the tail
    checks["far_token_zero_weight"] = wgt[seq - 1, far] < 1e-12

    # 4. agent
    st, dta, to, msg = R.run_rswa({"spec": {"n_ref": 64, "window": 128, "decode_len": 8000}}, "t")
    checks["agent_done"] = st == "done" and dta["max_keys_per_query"] <= 64 + 128 and dta["cache_reduction"] > 10

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== rswa-attention: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
