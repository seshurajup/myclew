"""moe_inference_pack_test — verifier for the Gemma-4 MoE inference-cost agent (offline, no model)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import moe_inference_pack as M


def _run():
    print("=== MoE INFERENCE-COST VERIFIER ===")
    checks = {}

    # exact accounting: 8 experts, top-1, 3B/expert, 2B shared → active 5B, total 26B
    r = M.moe_cost(n_experts=8, active_experts=1, expert_params=3.0e9, shared_params=2.0e9)
    checks["active_params"] = abs(r["active_params"] - 5.0e9) < 1e3
    checks["total_params"] = abs(r["total_params"] - 26.0e9) < 1e3
    checks["compute_ratio"] = abs(r["compute_ratio"] - (5.0/26.0)) < 1e-6
    checks["per_token_flops"] = abs(r["per_token_flops"] - 2.0*5.0e9) < 1e3
    checks["memory_is_total"] = abs(r["memory_params"] - r["total_params"]) < 1e-6
    checks["speedup_vs_dense"] = abs(r["speedup_vs_dense_total"] - 26.0/5.0) < 1e-6
    print(f"  -> 26B-A4B-like: active {r['active_params']/1e9:.1f}B / total {r['total_params']/1e9:.1f}B, "
          f"compute {r['compute_ratio']*100:.1f}%")

    # more active experts (top-k) → more compute but same memory
    r2 = M.moe_cost(8, 2, 3.0e9, 2.0e9)
    checks["more_active_more_compute"] = r2["active_params"] > r["active_params"]
    checks["memory_unchanged"] = abs(r2["total_params"] - r["total_params"]) < 1e-6

    # dense equivalent pays FLOPs on ALL params → strictly more than MoE active FLOPs
    checks["moe_cheaper_than_dense"] = M.dense_equivalent_flops(r["total_params"]) > r["per_token_flops"]

    # k clamped to E; k=E collapses to dense (active==total)
    rfull = M.moe_cost(4, 4, 1.0e9, 0.0)
    checks["k_eq_E_is_dense"] = abs(rfull["compute_ratio"] - 1.0) < 1e-9
    rclamp = M.moe_cost(4, 99, 1.0e9, 0.0)
    checks["k_clamped"] = abs(rclamp["active_params"] - rclamp["total_params"]) < 1e-6

    # agent contract
    st, d, to, msg = M.run_moe({"spec": {"n_experts": 8, "active_experts": 1,
                                         "expert_params": 3.0e9, "shared_params": 2.0e9}}, "t")
    checks["agent_done"] = st == "done" and abs(d["total_b"] - 26.0) < 0.01 and abs(d["active_b"] - 5.0) < 0.01
    st2, d2, to2, msg2 = M.run_moe({"spec": {"n_experts": 8}}, "t")   # missing keys → escalate
    checks["agent_escalates"] = st2 == "escalated"

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== moe-inference-cost: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
