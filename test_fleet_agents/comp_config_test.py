"""comp_config_test — data-wise verifier for the CompConfig contract (MUST actually run).

Constructs the CompConfig for EACH real example competition the user named, and asserts:
  • it routes to the correct pack (tabular→tab, biohub→biohub, arc→reason, ai-agent-security→agent/sec, neurogolf→unknown),
  • the metric-registry scores real numbers correctly (accuracy/rmse/auc/qwk/exact_match vs hand-computed),
  • round-trip to_dict/from_dict + save/load is lossless,
  • the Scorer interface REFUSES an untagged eval-set (the anti-subset-overcredit guard).
"""
import os, sys, json, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import comp_config as CC


# the real example comps the user gave → expected routing
EXAMPLES = [
    dict(slug="playground-series-s6e7", modality="tabular", paradigm="predictive", task="classification",
         metric="roc_auc", expect_pack="tab"),
    dict(slug="rogii-wellbore-geology-prediction", modality="sequence", paradigm="predictive",
         task="classification", metric="accuracy", domain="geology", cv_scheme="grouped-sequence", expect_pack="tab"),
    dict(slug="biohub-cell-tracking-during-development", modality="volume-time", paradigm="predictive",
         task="tracking", metric="edge_jaccard", expect_pack="biohub"),
    dict(slug="arc-prize-2026-arc-agi-3", modality="grid-reasoning", paradigm="reasoning",
         task="program-synthesis", metric="exact_match", expect_pack="reason"),
    dict(slug="ai-agent-security-multi-step-tool-attacks", modality="agent-env", paradigm="agentic",
         task="attack", metric="unknown", expect_pack="agent/sec"),
    dict(slug="pokemon-tcg-ai-battle", modality="agent-env", paradigm="agentic", task="policy",
         metric="unknown", expect_pack="agent"),
    dict(slug="autonomous-agent-prediction-beta", modality="agent-config", paradigm="prompt-program",
         task="classification", metric="roc_auc", expect_pack="prompt"),
    dict(slug="neurogolf-2026", modality="unknown", paradigm="predictive", task="unknown",
         metric="unknown", expect_pack="unknown"),
]


def _run():
    print("=== COMP_CONFIG CONTRACT DATA-WISE VERIFIER ===")
    checks = {}

    # 1. routing for every real example comp
    for e in EXAMPLES:
        exp = e.pop("expect_pack")
        cfg = CC.CompConfig(**e)
        got = cfg.pack()
        checks[f"route[{cfg.slug}]->{exp}"] = (got == exp)
        if got != exp:
            print(f"  X {cfg.slug}: expected {exp} got {got}")

    # 2. metric-registry correctness vs hand-computed
    y = np.array([0, 1, 1, 0, 1]); p = np.array([0, 1, 0, 0, 1])
    checks["accuracy"] = abs(CC.score("accuracy", y, p) - 0.8) < 1e-9
    checks["f1_binary"] = abs(CC.score("f1", y, p) - (2 * 2 / (2 + 3))) < 1e-9  # tp2 fp0 fn1 → P1 R.667 → F1 .8? recompute below
    # recompute f1 exactly: tp=2 (idx1,4), fp=0, fn=1 (idx2) → prec=1, rec=2/3 → f1=0.8
    checks["f1_binary"] = abs(CC.score("f1", y, p) - 0.8) < 1e-9
    yr = np.array([1.0, 2.0, 3.0]); pr = np.array([1.0, 2.0, 5.0])
    checks["rmse"] = abs(CC.score("rmse", yr, pr) - np.sqrt(4 / 3)) < 1e-9
    checks["mae"] = abs(CC.score("mae", yr, pr) - (2 / 3)) < 1e-9
    # auc: scores rank perfectly separable
    ya = np.array([0, 0, 1, 1]); pa = np.array([0.1, 0.2, 0.8, 0.9])
    checks["roc_auc_perfect"] = abs(CC.score("roc_auc", ya, pa) - 1.0) < 1e-9
    checks["roc_auc_alias"] = abs(CC.score("auc", ya, pa) - 1.0) < 1e-9  # alias resolves
    # qwk perfect agreement = 1
    yo = np.array([0, 1, 2, 3]); po = np.array([0, 1, 2, 3])
    checks["qwk_perfect"] = abs(CC.score("qwk", yo, po) - 1.0) < 1e-9
    # exact_match on grids
    g1 = [np.zeros((2, 2)), np.ones((2, 2))]; g2 = [np.zeros((2, 2)), np.zeros((2, 2))]
    checks["exact_match_half"] = abs(CC.score("exact_match", g1, g2) - 0.5) < 1e-9

    # 2b. metrics harvested from REAL 2025-26 top solutions
    checks["average_precision_perfect"] = abs(CC.score("average_precision", ya, pa) - 1.0) < 1e-9
    checks["ap_alias_map"] = abs(CC.score("map", ya, pa) - 1.0) < 1e-9
    checks["f2_binary"] = abs(CC.score("f2", y, p) - (5 * 1.0 * (2/3) / (4 * 1.0 + 2/3))) < 1e-9  # tp2 fp0 fn1
    checks["smape_zero"] = abs(CC.score("smape", yr, yr) - 0.0) < 1e-9
    # concordance: perfect SURVIVAL-score ordering (higher pred = longer survival, standard C-index) → 1.0
    surv = np.array([[5, 1], [10, 1], [15, 1]], float)  # (time, event)
    surv_score = np.array([1.0, 2.0, 3.0])               # longer time → higher score (concordant)
    checks["concordance_perfect"] = abs(CC.score("concordance_index", surv, surv_score) - 1.0) < 1e-9
    checks["partial_auc_present"] = CC.metric_spec("isic")["key"] == "partial_auc"
    # fn=None metrics still resolve direction (scored by their pack's agent)
    checks["stratified_cindex_dir"] = CC.metric_spec("stratified_concordance_index")["direction"] == "max"
    checks["tm_score_dir"] = CC.metric_spec("tm_score")["direction"] == "max"
    checks["map_at_k_alias"] = CC.metric_spec("map@25")["key"] == "map_at_k"
    # gap-scan metrics (arc-2025/aimo/ariel/mitsui/vesuvius)
    mu_sig = np.column_stack([yr, np.ones(3)])                 # perfect mean, unit sigma
    checks["gaussian_nll_min"] = CC.score("gaussian_nll", yr, mu_sig) < CC.score("gaussian_nll", yr, np.column_stack([yr + 2, np.ones(3)]))
    checks["pass_at_k"] = abs(CC.score("pass_at_k", ["a", "b"], [["x", "a"], ["y", "z"]]) - 0.5) < 1e-9
    checks["maj_at_k"] = abs(CC.score("maj_at_k", ["a"], [["a", "a", "b"]]) - 1.0) < 1e-9
    checks["spearman_sharpe_dir"] = CC.metric_spec("mitsui")["direction"] == "max" and CC.metric_spec("mitsui")["key"] == "spearman_sharpe"
    checks["surface_dice_dir"] = CC.metric_spec("vesuvius")["key"] == "surface_dice"

    # 3. direction lookup
    checks["auc_maximize"] = CC.metric_spec("roc_auc")["direction"] == "max"
    checks["rmse_minimize"] = CC.metric_spec("rmse")["direction"] == "min"

    # 4. round-trip + save/load
    cfg = CC.CompConfig(slug="playground-series-s6e7", modality="tabular", metric="roc_auc",
                        id_col="id", target_cols=["target"], n_folds=5)
    checks["roundtrip_dict"] = CC.CompConfig.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict()
    tmp = tempfile.mktemp(suffix=".json"); cfg.save(tmp)
    checks["roundtrip_file"] = CC.CompConfig.load(tmp).to_dict() == cfg.to_dict()

    # 5. Scorer interface requires an eval-set tag (anti-subset-overcredit)
    sc = CC.Scorer()
    try:
        sc.score(cfg, y, p, eval_set="")
        checks["scorer_requires_evalset"] = False
    except ValueError:
        checks["scorer_requires_evalset"] = True
    res = sc.score(cfg, ya, pa, eval_set="fold0")
    checks["scorer_returns_value"] = abs(res["value"] - 1.0) < 1e-9 and res["eval_set"] == "fold0"

    # 6. register_metric hook (comp-specific fn injects cleanly)
    CC.register_metric("dummy_biohub", lambda a, b: 0.42, direction="max", aliases=["dbio"])
    checks["register_metric"] = abs(CC.score("dbio", [0], [0]) - 0.42) < 1e-9

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== comp_config: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
