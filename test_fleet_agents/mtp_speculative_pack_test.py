"""mtp_speculative_pack_test — verifier for the Gemma-4 MTP speculative-decoding agent (offline, no model)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import mtp_speculative_pack as M


def _run():
    print("=== MTP SPECULATIVE-DECODE VERIFIER ===")
    checks = {}

    # expected accepted: α→1 gives γ+1 tokens; α=0 gives exactly 1 (the bonus target token)
    checks["exp_alpha1"] = abs(M.expected_accepted(1.0, 4) - 5.0) < 1e-9
    checks["exp_alpha0"] = abs(M.expected_accepted(0.0, 4) - 1.0) < 1e-9
    # closed form α=0.5, γ=3: (1-0.5^4)/(1-0.5) = (1-0.0625)/0.5 = 1.875
    checks["exp_closed_form"] = abs(M.expected_accepted(0.5, 3) - 1.875) < 1e-6
    # monotone in α and in γ
    checks["exp_mono_alpha"] = M.expected_accepted(0.9, 4) > M.expected_accepted(0.5, 4)
    checks["exp_mono_gamma"] = M.expected_accepted(0.8, 6) > M.expected_accepted(0.8, 2)

    # speedup: free drafter (c=0) equals E[tokens]; a cost penalizes long drafts
    checks["speedup_free"] = abs(M.decode_speedup(0.8, 4, 0.0) - M.expected_accepted(0.8, 4)) < 1e-9
    s_cheap = M.decode_speedup(0.8, 4, 0.05); s_pricey = M.decode_speedup(0.8, 4, 0.5)
    checks["speedup_cost_hurts"] = s_pricey < s_cheap
    # good drafter beats standard decoding (>1×), bad+expensive drafter can drop below 1×
    checks["speedup_gt1_good"] = M.decode_speedup(0.9, 4, 0.05) > 1.0
    checks["speedup_lt1_bad"] = M.decode_speedup(0.1, 8, 1.0) < 1.0

    # optimal draft length: higher cost → shorter optimal γ
    g_lo, s_lo = M.optimal_draft_length(0.85, 0.02, 16)
    g_hi, s_hi = M.optimal_draft_length(0.85, 0.5, 16)
    checks["opt_shorter_when_pricey"] = g_hi <= g_lo
    checks["opt_speedup_ge1"] = s_lo >= 1.0
    print(f"  -> optimal γ cheap={g_lo}({s_lo:.2f}×) pricey={g_hi}({s_hi:.2f}×)")

    # agent contract
    st, d, to, msg = M.run_mtp({"spec": {"alpha": 0.85, "gamma": 4, "cost_ratio": 0.1}}, "t")
    checks["agent_done"] = st == "done" and d["speedup"] > 1.0 and "best_gamma" in d
    st2, d2, to2, msg2 = M.run_mtp({"spec": {}}, "t")   # missing alpha → escalate
    checks["agent_escalates"] = st2 == "escalated"

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== mtp-speculative-decode: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
