"""keyframe_test — pure logic: keyframe_plan (sparse keyframes + interp + frozen) + feasible (T4 ETA)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import keyframe as K


def _run():
    print("=== KEYFRAME LOGIC VERIFIER ===")
    # T=10, frozen {5}, interval 3 → unique=[0,1,2,3,4,6,7,8,9], keyframes every 3rd = [0,3,7]
    plan = K.keyframe_plan(frozen={5}, T=10, interval=3)
    # heavy detector on keyframes only → feasible; all-frames heavy → infeasible
    feas_kf, eta_kf = K.feasible(plan["n_key"], big_spf_t4=51.7, n_interp=plan["n_interp"],
                                 cheap_spf=0.024, n_frozen=plan["n_frozen"])
    feas_all, eta_all = K.feasible(n_key=10, big_spf_t4=51.7, n_interp=0, cheap_spf=0.024)
    checks = {
        "keyframes_sparse": plan["keyframes"] == [0, 3, 7],
        "frozen_copied": plan["source_of"].get(5) == 4,
        "interp_from_keyframe": plan["source_of"].get(4) == 3,
        "counts_add_up": plan["n_key"] + plan["n_interp"] + plan["n_frozen"] == 10,
        "all_heavy_infeasible": feas_all is False,          # 51.7s/f on every frame = way over
        "keyframe_cuts_eta": eta_kf < eta_all,               # sparse keyframes = far less time
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"    keyframe eta={eta_kf}h vs all-heavy {eta_all}h | plan n_key={plan['n_key']}")
    ok = all(checks.values()); print("RESULT:", "PASS" if ok else "FAIL"); return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
