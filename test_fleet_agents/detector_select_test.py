"""detector_select_test — DATA-WISE verifier of the model-CHOICE logic (_choose), no GPU/weights.

Now T4-AWARE: dev-GPU (5090) spf is mapped to the real Kaggle T4 via the model kind, so a ViT detector that
times out on T4 is EXCLUDED even though it has the best recall. Asserts: leaky excluded; ViT (T4-too-slow)
excluded; the fastest-feasible detector with best recall wins.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import detector_select as D


def _run():
    print("=== DETECTOR-SELECT CHOICE-LOGIC VERIFIER (T4-aware) ===")
    results = {
        "pilkwang-pack (LEAKY)":  {"44b6": 0.99, "6bba": 0.99, "spf": 1.0, "leaky": True, "kind": "cnn3d"},
        "cellpose-SAM (ViT)":     {"44b6": 0.972, "6bba": 0.951, "spf": 3.99, "leaky": False, "kind": "vit_2dstitch"},
        "micro-SAM (ViT)":        {"44b6": 0.833, "6bba": 0.932, "spf": 6.2, "leaky": False, "kind": "vit"},
        "unet-winner (cnn3d)":    {"44b6": 0.85, "6bba": 0.90, "spf": 0.10, "leaky": False, "kind": "cnn3d"},
        "dog (cpu)":              {"44b6": 0.806, "6bba": 0.738, "spf": 0.02, "leaky": False, "kind": "cpu"},
        "lopsided-fast":          {"44b6": 0.99, "6bba": 0.60, "spf": 0.05, "leaky": False, "kind": "cnn"},
    }
    winner, ranked = D._choose(results, require_feasible=True)
    names = [d["name"] for d in ranked]
    checks = {
        "leaky_excluded": "pilkwang-pack (LEAKY)" not in names,
        "vit_excluded_on_T4": "cellpose-SAM (ViT)" not in names and "micro-SAM (ViT)" not in names,
        "winner_is_fast_cnn": winner == "unet-winner (cnn3d)",     # feasible + best min-recall 0.85
        "lopsided_not_winner": winner != "lopsided-fast",          # min 0.60 < 0.85
        "dog_is_feasible": "dog (cpu)" in names,
        "t4_estimate_present": all("t4_spf" in d for d in ranked),
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("    winner=", winner, " ranked=", [(d["name"], d["min_recall"], d["t4_spf"]) for d in ranked])
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
