"""improve_loop_test — script a diagnose→apply sequence where the weakest lever MOVES between rounds and
CV improves twice then converges. Assert the loop routes by the current weakest link, keeps improving
rounds, and STOPS on no-improvement. No GPU — agents injected."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import improve_loop


def _run():
    print("=== IMPROVE-LOOP WORKFLOW VERIFIER ===")
    # scripted worlds: round0 weakest=node_recall → det-sweep gives cv 0.89 (↑ from 0.88);
    # round1 weakest=postproc → recipe-adopt gives cv 0.91 (↑); round2 weakest=postproc → recipe-adopt 0.905 (no ↑) → STOP
    diag_seq = [{"weakest": "node_recall"}, {"weakest": "postproc"}, {"weakest": "postproc"}]
    det_cv = [0.89]; adopt_cv = [0.91, 0.905]
    state = {"i_diag": 0, "i_det": 0, "i_adopt": 0}

    def pre(q, w):
        d = diag_seq[state["i_diag"]]; state["i_diag"] += 1; return ("done", d, "all", "")
    def det(q, w):
        cv = det_cv[state["i_det"]]; state["i_det"] += 1; return ("done", {"pick": {"cv": cv}}, "all", "")
    def adopt(q, w):
        cv = adopt_cv[state["i_adopt"]]; state["i_adopt"] += 1; return ("done", {"merged_cv": cv}, "all", "")

    agents = {"pre-analysis": pre, "det-sweep": det, "recipe-adopt": adopt}
    s, d, to, msg = improve_loop.ImproveLoop().run(
        {"question": "loop", "spec": {"agents": agents, "rounds": 4, "start_cv": 0.88, "eps": 0.001}}, "test")

    trace = d["trace"]
    checks = {
        "ran_3_rounds": d["rounds_run"] == 3,
        "round1_routed_detsweep": trace[0]["agent"] == "det-sweep" and trace[0]["improved"],
        "round2_routed_recipeadopt": trace[1]["agent"] == "recipe-adopt" and trace[1]["improved"],
        "round3_no_improve": trace[2]["improved"] is False,
        "stopped_converged": d["stopped"] == "converged (no improvement)",
        "best_cv_0.91": abs(d["best_cv"] - 0.91) < 1e-9,
    }
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== improve-loop: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
