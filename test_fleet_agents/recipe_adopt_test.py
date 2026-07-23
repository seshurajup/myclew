"""recipe_adopt_test — plant a base config, a public recipe, and a MOCK canonical scorer with a KNOWN
winning subset. Assert recipe-adopt keeps exactly the load-bearing knobs (positive Δ), drops the rest,
and reports the merged config/CV. No GPU — the scorer is injected via spec.score_fn."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import recipe_adopt


def _run():
    print("=== RECIPE-ADOPT DATA-WISE VERIFIER ===")
    base = {"BIOHUB_DET": 0.90, "BIOHUB_POOL": 5.0, "BIOHUB_MTL": 4}
    recipe = {"BIOHUB_DET": 0.97, "BIOHUB_POOL": 3.0, "BIOHUB_MTL": 6}   # abhijith-like
    # ground truth: DET=0.97 helps (+0.05), MTL=6 helps (+0.02), POOL=3.0 HURTS (-0.03)
    KNOWN = {"BIOHUB_DET": +0.05, "BIOHUB_POOL": -0.03, "BIOHUB_MTL": +0.02}
    BASE_CV = 0.8837

    def mock_score(cfg):
        cv = BASE_CV
        for k, gain in KNOWN.items():
            if cfg.get(k) == recipe[k]:      # a knob set to the recipe value contributes its known gain
                cv += gain
        return cv

    s, data, to, msg = recipe_adopt.RecipeAdopt().run(
        {"question": "graft abhijith", "spec": {"base": base, "recipe": recipe, "score_fn": mock_score}}, "test")

    kept = set(data.get("kept") or [])
    checks = {
        "keeps_DET_and_MTL": kept == {"BIOHUB_DET", "BIOHUB_MTL"},
        "drops_POOL": "BIOHUB_POOL" not in kept,
        "merged_uses_kept_values": data["merged_config"]["BIOHUB_DET"] == 0.97
                                   and data["merged_config"]["BIOHUB_MTL"] == 6
                                   and data["merged_config"]["BIOHUB_POOL"] == 5.0,   # POOL reverted to base
        "merged_cv_is_base_plus_gains": abs(data["merged_cv"] - (BASE_CV + 0.05 + 0.02)) < 1e-6,
        "ranked_best_first": data["results"][0]["knob"] == "BIOHUB_DET",   # +0.05 biggest
    }
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== recipe-adopt: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
