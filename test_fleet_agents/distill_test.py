"""distill_test — pure logic: worth_distilling (teacher headroom) + accept_student (recovery + feasibility)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import distill as D


def _run():
    print("=== DISTILL LOGIC VERIFIER ===")
    teacher = {"44b6": 0.972, "6bba": 0.951}
    student = {"44b6": 0.806, "6bba": 0.738}
    worth, gap = D.worth_distilling(teacher, student)
    acc_ok, r1 = D.accept_student(teacher, {"44b6": 0.93, "6bba": 0.90}, t4_spf=0.8, budget_spf=2.82)
    acc_slow, r2 = D.accept_student(teacher, {"44b6": 0.93, "6bba": 0.90}, t4_spf=9.0, budget_spf=2.82)
    acc_low, r3 = D.accept_student(teacher, {"44b6": 0.80, "6bba": 0.70}, t4_spf=0.8, budget_spf=2.82)
    # direction-vs-endpoint decision (Direct-OPD, dopd05/dopd06)
    m_dir, _ = D.direction_transfer_worth(teacher_pre=0.24, teacher_post=0.54, student_ref=0.70)   # weak teacher, real shift
    m_end_strong, _ = D.direction_transfer_worth(teacher_pre=0.5, teacher_post=0.95, student_ref=0.70)  # teacher stronger
    m_end_flat, _ = D.direction_transfer_worth(teacher_pre=0.60, teacher_post=0.60, student_ref=0.70)   # no RL shift
    checks = {
        "worth_when_headroom": worth and gap == round(0.738 - 0.738, 4) or (worth and gap > 0),
        "gap_is_min_recall": gap == round(min(teacher.values()) - min(student.values()), 4),
        "accept_good_student": acc_ok is True,
        "reject_slow_student": acc_slow is False and "budget" in r2,
        "reject_low_recall": acc_low is False,
        "no_headroom_skips": D.worth_distilling({"44b6": 0.7, "6bba": 0.7}, {"44b6": 0.8, "6bba": 0.8})[0] is False,
        "dopd_direction_when_weak_teacher_shifts": m_dir == "direction",
        "dopd_endpoint_when_teacher_stronger": m_end_strong == "endpoint",
        "dopd_endpoint_when_no_rl_shift": m_end_flat == "endpoint",
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("    gap=", gap, "| accept_good=", r1)
    ok = all(checks.values()); print("RESULT:", "PASS" if ok else "FAIL"); return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
