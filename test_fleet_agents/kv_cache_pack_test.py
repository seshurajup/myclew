"""kv_cache_pack_test — verifier for the Gemma-4 long-context KV-cache agent (offline, no model)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import kv_cache_pack as K


def _run():
    print("=== KV-CACHE LONG-CONTEXT VERIFIER ===")
    checks = {}

    cfg = dict(seq_len=32768, n_layers=48, n_kv_heads=8, head_dim=128, dtype_bytes=1)

    # naive baseline is deterministic: 2 * n_layers * L * h * d * bytes
    exp_naive = 2 * 48 * 32768 * 8 * 128 * 1
    checks["naive_formula"] = abs(K.naive_kv_bytes(32768, 48, 8, 128, 1) - exp_naive) < 1.0

    # the Gemma-4 scheme uses less than naive (windowed local + shared/reused global)
    opt = K.kv_cache_bytes(**cfg, local_global_ratio=5, window=4096, global_share=1.0, values_as_keys=True)
    checks["opt_below_naive"] = opt["total_bytes"] < exp_naive
    checks["layer_split"] = opt["n_global"] >= 1 and (opt["n_global"] + opt["n_local"] == 48)

    # levers move the number the right way: more local:global ratio → fewer global layers → less memory
    hi_ratio = K.kv_cache_bytes(**cfg, local_global_ratio=5, window=4096)["total_bytes"]
    lo_ratio = K.kv_cache_bytes(**cfg, local_global_ratio=1, window=4096)["total_bytes"]
    checks["more_local_less_mem"] = hi_ratio < lo_ratio
    # smaller sliding window → less local memory
    small_w = K.kv_cache_bytes(**cfg, window=1024)["total_bytes"]
    big_w = K.kv_cache_bytes(**cfg, window=8192)["total_bytes"]
    checks["smaller_window_less_mem"] = small_w < big_w
    # values=keys halves the global tensor count → strictly less than separate K+V
    vk = K.kv_cache_bytes(**cfg, values_as_keys=True)["global_bytes"]
    kv = K.kv_cache_bytes(**cfg, values_as_keys=False)["global_bytes"]
    checks["values_as_keys_saves"] = abs(vk - kv / 2.0) < 1.0

    # global-KV reduction: dropping the whole V = 50%; report's 37.5% = reuse_fraction 0.75 (p-RoPE design)
    checks["vk_alone_50pct"] = abs(K.global_kv_reduction(1.0, True, reuse_fraction=1.0) - 0.5) < 1e-9
    r375 = K.global_kv_reduction(global_share=1.0, values_as_keys=True, reuse_fraction=0.75)
    checks["reproduces_37_5"] = abs(r375 - 0.375) < 1e-6
    # KV-sharing on top pushes the reduction beyond the reuse-only figure
    checks["sharing_adds"] = K.global_kv_reduction(1.6, True, 1.0) > 0.5
    print(f"  -> global-KV reduction: vk-only={K.global_kv_reduction(1.0,True):.3f}, tuned={r375:.3f}")

    # total reduction vs naive is a fraction in (0,1)
    red = K.reduction_vs_naive(**cfg, local_global_ratio=5, window=4096, global_share=1.0, values_as_keys=True)
    checks["reduction_fraction"] = 0.0 < red < 1.0
    print(f"  -> total KV cut vs naive: {red*100:.1f}%")

    # agent contract
    st, d, to, msg = K.run_kv({"spec": cfg}, "t")
    checks["agent_done"] = st == "done" and d["total_gb"] > 0 and 0 <= d["global_kv_reduction"] <= 1
    st2, d2, to2, msg2 = K.run_kv({"spec": {"seq_len": 1024}}, "t")   # missing keys → escalate
    checks["agent_escalates"] = st2 == "escalated"

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== kv-cache-longctx: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
