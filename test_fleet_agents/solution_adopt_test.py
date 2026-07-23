"""solution_adopt_test — verify the adopted workflow reproduces the REAL winning levers per comp archetype.

For each archetype (QWK tabular, survival tabular, RMSE tabular, LLM-ranking, ARC reasoning, agentic-security,
audio pseudo-label), assert the emitted workflow contains the grounded steps the actual 2025-26 winners used.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import comp_config as CC
from fleet_agents import solution_adopt as SA


def _agents(cfg, inv=None):
    return [s["agent"] for s in SA.adopt(cfg, inv)]


def _run():
    print("=== SOLUTION-ADOPT DATA-WISE VERIFIER (grounded workflows) ===")
    checks = {}

    # QWK tabular (child-mind): must regress+round-optimize
    a = _agents(CC.CompConfig(slug="qwk", modality="tabular", metric="quadratic_weighted_kappa"))
    checks["qwk_target_transform"] = "target-transform" in a
    checks["qwk_post_round"] = "post-optimize" in a and "tab-fe" in a and "pseudo-label" in a
    checks["qwk_blend"] = "blend-optimize" in a

    # survival tabular (equity): must factorize
    a = _agents(CC.CompConfig(slug="equity", modality="tabular", metric="stratified_concordance_index"))
    checks["survival_factorize"] = "target-transform" in a

    # RMSE tabular (s5e4): must clip-guard
    a = _agents(CC.CompConfig(slug="rmse", modality="tabular", metric="rmse"))
    checks["rmse_clip"] = "post-optimize" in a
    a_auc = _agents(CC.CompConfig(slug="auc", modality="tabular", metric="roc_auc"))
    checks["auc_calibrate"] = "calibrate" in a_auc

    # LLM ranking (eedi): retrieve-rerank + quantize + cascade
    a = _agents(CC.CompConfig(slug="eedi", modality="text", paradigm="predictive", metric="map_at_k"),
                inv={"t": "retrieve then rerank cascade MAP@25"})
    checks["llm_finetune"] = "llm-finetune" in a
    checks["llm_retrieve_rerank"] = "llm-retrieve-rerank" in a
    checks["llm_quantize_infer"] = "quantize" in a and "llm-infer" in a

    # reasoning (ARC): test-time-training is THE lever
    a = _agents(CC.CompConfig(slug="arc", modality="grid-reasoning", paradigm="reasoning", metric="exact_match"))
    checks["arc_ttc"] = "ttc" in a and "program-search" in a and "reason-dsl" in a

    # agentic-security (JED): sec-* + replay + imitation
    a = _agents(CC.CompConfig(slug="aas", modality="agent-env", paradigm="agentic", task="attack"))
    checks["sec_attack"] = "sec-attack" in a and "sec-eval" in a
    checks["agentic_replay_imitation"] = "lb-replay-mine" in a and "imitation-learn" in a and "agent-selfplay" in a

    # audio (birdclef): pseudo-label self-training present
    a = _agents(CC.CompConfig(slug="bird", modality="image", paradigm="predictive", metric="roc_auc"))
    checks["vision_pseudo_label"] = "pseudo-label" in a and "aug-find" in a

    # universal tail: human-gated submit always last-ish
    a = _agents(CC.CompConfig(slug="x", modality="tabular", metric="rmse"))
    checks["human_gate"] = "submit-guard" in a and "beat-bar" in a and "nb-preflight" in a

    # agent run() contract
    st, d, to, msg = SA.run({"spec": {"slug": "qwk", "modality": "tabular",
                                      "metric": "quadratic_weighted_kappa"}}, "test")
    checks["agent_run_done"] = st == "done" and d["n_steps"] > 8

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== solution-adopt: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
