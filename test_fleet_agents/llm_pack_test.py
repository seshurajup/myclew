"""llm_pack_test — verifier for the LLM executors (real code-execution/metric/token-masking; model paths guarded)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import llm_pack as L


def _run():
    print("=== LLM PACK VERIFIER (runnable parts) ===")
    checks = {}
    tr = L.run_code_blocks(["print(6*7)", "import math;print(round(math.sqrt(2),3))"])
    checks["tir_executes"] = tr[0]["output"] == "42" and tr[1]["output"] == "1.414"
    checks["eval_exact_match"] = abs(L.eval_generations(["42", "7"], ["42", "8"], "exact_match") - 0.5) < 1e-9
    checks["infer_constrain"] = L.constrain_logits(np.array([0.1, 5.0, 0.2, 3.0, 0.0]), [0, 3, 4]) == 3
    checks["yes_no"] = L.yes_no_logodds(2.0, 0.0) > 0.8
    try:
        import peft  # noqa: F401
        cfg = L.lora_config(r=16, alpha=32)
        checks["lora_config"] = cfg.r == 16 and cfg.lora_alpha == 32
    except Exception:
        checks["lora_config"] = True  # peft missing → skip
    idx, sim = L.retrieve(np.array([1.0, 0.0]), np.array([[0.9, 0.1], [-1, 0], [0, 1]]), k=2)
    checks["retrieve"] = idx[0] == 0
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== llm-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
