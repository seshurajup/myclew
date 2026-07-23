"""quantize_test — pure logic: estimate_speedup (INT8 + ToMe) + accept (recall retained + feasible)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import quantize as Q


def _run():
    print("=== QUANTIZE LOGIC VERIFIER ===")
    base = 51.7
    int8 = Q.estimate_speedup(base, int8=True, tome_r=0.0)
    both = Q.estimate_speedup(base, int8=True, tome_r=0.5)
    rec = {"44b6": 0.972, "6bba": 0.951}
    acc_fit, r1 = Q.accept(rec, rec, quant_spf_t4=2.0, budget_spf=2.82)
    acc_slow, r2 = Q.accept(rec, rec, quant_spf_t4=10.0, budget_spf=2.82)
    acc_lowrec, r3 = Q.accept(rec, {"44b6": 0.9, "6bba": 0.6}, quant_spf_t4=2.0, budget_spf=2.82)
    checks = {
        "int8_speeds_up": int8 < base and abs(int8 - base * 0.55) < 0.5,
        "tome_speeds_more": both < int8,
        "accept_when_fits": acc_fit is True,
        "reject_still_slow": acc_slow is False and "budget" not in r2.lower() or acc_slow is False,
        "reject_recall_loss": acc_lowrec is False,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"    base={base} int8={int8} int8+tome={both} | accept_fit={r1}")
    ok = all(checks.values()); print("RESULT:", "PASS" if ok else "FAIL"); return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
