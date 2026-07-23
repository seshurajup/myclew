"""comp_onboard_test — data-wise verifier for the comp-onboard front door (MUST actually run, offline).

Two things must hold:
  (A) KNOWN comps onboard exactly (the memorized table) → correct pack.
  (B) COLD-START inference works from a synthetic file-manifest + evaluation text ALONE (no table, no
      network) → a brand-new comp still fingerprints to the right modality/metric/pack. This is the proof
      the fleet generalizes (the neurogolf-class requirement), not just recites.
Also: the unknown route must produce a gap REPORT, never crash; and the agent run() must persist a config.
"""
import os, sys, json, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import comp_onboard as A


def _run():
    print("=== COMP-ONBOARD DATA-WISE VERIFIER (offline) ===")
    checks = {}

    # (A) KNOWN comps → exact routing (via table)
    known = {
        "playground-series-s6e7": "tab",
        "biohub-cell-tracking-during-development": "biohub",
        "arc-prize-2026-arc-agi-3": "reason",
        "ai-agent-security-multi-step-tool-attacks": "agent/sec",
        "pokemon-tcg-ai-battle": "agent",
    }
    for slug, exp in known.items():
        cfg = A.infer_config(slug)
        checks[f"known[{slug}]->{exp}"] = (cfg.pack() == exp)

    # (B) COLD-START — inference from synthetic manifest + eval text, slug NOT in the table
    # B1: a fresh tabular playground-like comp
    cfg = A.infer_config("brand-new-tabular-2027",
                         files=["train.csv", "test.csv", "sample_submission.csv"],
                         eval_text="Submissions are evaluated on the area under the ROC curve.",
                         sample_header=["id", "target"])
    checks["cold_tabular_modality"] = cfg.modality == "tabular"
    checks["cold_tabular_metric"] = cfg.metric == "roc_auc"
    checks["cold_tabular_task"] = cfg.task == "classification"
    checks["cold_tabular_pack"] = cfg.pack() == "tab"
    checks["cold_tabular_schema"] = cfg.id_col == "id" and cfg.target_cols == ["target"]

    # B2: a fresh regression comp
    cfg = A.infer_config("house-prices-x", files=["train.csv", "test.csv"],
                         eval_text="Root Mean Squared Error (RMSE) between the predicted and actual price.")
    checks["cold_regression"] = cfg.metric == "rmse" and cfg.task == "regression" and cfg.pack() == "tab"

    # B3: a fresh 3D+time bio comp (geff/zarr) — should hit volume-time → biohub
    cfg = A.infer_config("some-new-cell-comp", files=["embryo01.zarr", "gt_tracks.geff"],
                         eval_text="Tracking accuracy via edge jaccard over cell lineages.")
    checks["cold_volume_time"] = cfg.modality == "volume-time" and cfg.pack() == "biohub"

    # B4: a fresh reasoning comp
    cfg = A.infer_config("puzzle-synth-2027", files=["tasks.json"],
                         eval_text="Percentage of test grids solved with an exact match; abstraction and reasoning.")
    checks["cold_reasoning"] = cfg.paradigm == "reasoning" and cfg.metric == "exact_match" and cfg.pack() == "reason"

    # B5: a fresh agentic comp
    cfg = A.infer_config("robo-arena-2027", files=["env.py", "starter.ipynb"],
                         eval_text="Your agent plays episodes against an opponent in the simulator; reward per episode.")
    checks["cold_agentic"] = cfg.paradigm == "agentic" and cfg.pack() == "agent"

    # B5b: prompt-program — submission is an authored ADK agent bundle (autonomous-agent-prediction-beta)
    cfg = A.infer_config("autonomous-agent-prediction-beta")  # known table
    checks["prompt_program_known"] = cfg.modality == "agent-config" and cfg.paradigm == "prompt-program" and cfg.pack() == "prompt"
    # cold-start: a NEW agent-authoring comp detected from its manifest + eval text alone
    cfg = A.infer_config("new-agent-authoring-2027",
                         files=["agent.yaml", "prompts/system.md", "skills/my-skill/SKILL.md", "data/train.csv"],
                         eval_text="Submit a submission.zip with an agent.yaml (ADK agent config), system prompt, "
                                   "custom tools and skills; scored by AUC ROC.")
    checks["prompt_program_coldstart"] = cfg.modality == "agent-config" and cfg.paradigm == "prompt-program" and cfg.pack() == "prompt"

    # B6: an unknown comp → route 'unknown', gap report, NO crash
    cfg = A.infer_config("neurogolf-2026", files=["weird.bin"], eval_text="a novel scoring rule")
    checks["cold_unknown_route"] = cfg.pack() == "unknown"

    # agent run() offline path: passes manifest via spec, persists config, returns gap report on unknown
    st, res, to, msg = A.run({"spec": {"slug": "neurogolf-2026", "files": ["weird.bin"],
                                       "eval_text": "novel", "offline": True}}, "test")
    checks["run_unknown_escalates"] = (st == "escalated") and isinstance(res, dict) and "gap_report" in res
    checks["run_persists_config"] = os.path.exists(res.get("config_file", ""))

    # agent run() offline happy path (tabular) → done + pack tab
    st, res, to, msg = A.run({"spec": {"slug": "brand-new-tab", "offline": True,
                                       "files": ["train.csv", "test.csv", "sample_submission.csv"],
                                       "eval_text": "area under the ROC curve",
                                       "sample_header": ["id", "target"]}}, "test")
    checks["run_tab_done"] = (st == "done") and res.get("pack") == "tab"

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== comp-onboard: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
