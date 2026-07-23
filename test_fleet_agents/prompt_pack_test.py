"""prompt_pack_test — data-wise verifier for the PROMPT-PROGRAM pack (autonomous-agent-prediction-beta).

Proves the whole thing end-to-end, offline:
  • skill-build writes a runnable deterministic AutoML floor,
  • agent-author writes a valid ADK bundle (required manifest present, !include + skill wired),
  • agent-package validates + zips CONTENTS AT ROOT (agent.yaml at zip root, not nested),
  • agent-config-eval ACTUALLY RUNS the skill on synthetic hidden-label tasks and gets real AUC > 0.75
    (the champion's exact validation discipline — the skill is a genuinely competitive floor, not a stub),
  • prompt-optimize ranks variants by hidden AUC.
"""
import os, sys, tempfile, zipfile, subprocess
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import prompt_skill_build as SB
from fleet_agents import prompt_agent_author as AU
from fleet_agents import prompt_agent_eval as EV


def _run():
    print("=== PROMPT-PROGRAM PACK DATA-WISE VERIFIER (offline, real skill run) ===")
    checks = {}
    d = tempfile.mkdtemp(prefix="aap_bundle_")

    # author the full bundle
    bundle = AU.author(d)
    for f in AU.REQUIRED:
        checks[f"has[{f}]"] = os.path.exists(os.path.join(bundle, f))
    checks["agent_yaml_include"] = "!include prompts/system.md" in open(os.path.join(bundle, "agent.yaml")).read()
    checks["skill_wired"] = "skills/tabular-autopilot" in open(os.path.join(bundle, "agent.yaml")).read()

    # (a) read-only reviewer sub-agent tool — mounted ALONGSIDE data_analyst
    agent_yaml = open(os.path.join(bundle, "agent.yaml")).read()
    checks["reviewer_mounted"] = "tools/reviewer.yaml" in agent_yaml and "tools/data_analyst.yaml" in agent_yaml
    rv_yaml = open(os.path.join(bundle, "tools", "reviewer.yaml")).read()
    rv_md = open(os.path.join(bundle, "prompts", "reviewer.md")).read()
    checks["reviewer_temp0"] = "temperature: 0" in rv_yaml
    checks["reviewer_request_enum"] = "REQUEST_TYPE" in rv_md
    checks["reviewer_action_enum"] = "NEXT_FLASH_ACTION" in rv_md and all(
        a in rv_md for a in ("USE_CURRENT", "APPLY_FIX_AND_RUN", "STOP_AND_SELECT"))
    checks["reviewer_read_only"] = not any(t in rv_yaml for t in ("write_file", "submit_predictions", "select_submission"))

    # (b) the two new deterministic skill scripts are emitted and run on tiny synthetic data
    scripts = os.path.join(bundle, "skills", "tabular-autopilot", "scripts")
    checks["has[candidate_similarity.py]"] = os.path.exists(os.path.join(scripts, "candidate_similarity.py"))
    checks["has[shift_profile.py]"] = os.path.exists(os.path.join(scripts, "shift_profile.py"))
    syn = tempfile.mkdtemp(prefix="aap_syn_")
    import numpy as _np, pandas as _pd
    _rng = _np.random.default_rng(0); _n = 120
    _base = _rng.random(_n)
    for _fn, _v in (("a.csv", _base + 0.02 * _rng.random(_n)), ("b.csv", _base + 0.02 * _rng.random(_n)),
                    ("c.csv", _rng.random(_n))):
        _pd.DataFrame({"id": range(_n), "pred": _np.clip(_v, 0, 1)}).to_csv(os.path.join(syn, _fn), index=False)
    _y = (_rng.random(_n) > 0.5).astype(int)
    _pd.DataFrame({"f1": _rng.normal(0, 1, _n), "cat": _rng.choice(["x", "y"], _n), "target": _y}).to_csv(
        os.path.join(syn, "train.csv"), index=False)
    _pd.DataFrame({"f1": _rng.normal(0.5, 1, _n), "cat": _rng.choice(["x", "z"], _n)}).to_csv(
        os.path.join(syn, "test.csv"), index=False)
    r1 = subprocess.run([sys.executable, os.path.join(scripts, "candidate_similarity.py"), "--candidates",
                         os.path.join(syn, "a.csv"), os.path.join(syn, "b.csv"), os.path.join(syn, "c.csv")],
                        capture_output=True, text=True)
    checks["candidate_similarity_runs"] = "CANDSIM_OK" in r1.stdout and "MOST_COMPLEMENTARY" in r1.stdout and "N_VALID 3" in r1.stdout
    r2 = subprocess.run([sys.executable, os.path.join(scripts, "shift_profile.py"), "--train",
                         os.path.join(syn, "train.csv"), "--test", os.path.join(syn, "test.csv")],
                        capture_output=True, text=True)
    checks["shift_profile_runs"] = "SHIFTPROF_OK" in r2.stdout and "NUM_DRIFT" in r2.stdout and "CAT_UNSEEN" in r2.stdout

    # validate + package (zip contents at root)
    v = AU.validate(bundle); checks["validate_ok"] = v["ok"]
    zp = AU.package(bundle)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
    checks["zip_root_agent_yaml"] = "agent.yaml" in names          # at ROOT, not bundle/agent.yaml
    checks["zip_has_skill"] = any("skills/tabular-autopilot/scripts/run_pipeline.py" in n for n in names)

    # THE proof: run the deterministic skill on synthetic hidden-label tasks → real AUC
    skill_dir = os.path.join(bundle, "skills", "tabular-autopilot")
    res = EV.eval_skill(skill_dir)
    checks["eval_ran_all_tasks"] = res["n_ok"] == 4
    checks["skill_auc_competitive"] = res["mean_auc"] is not None and res["mean_auc"] > 0.75
    print(f"  -> skill floor mean hidden-label AUC = {res['mean_auc']} over {res['n_ok']} tasks; per={[round(p['auc'],3) if p['auc'] else None for p in res['per_task']]}")

    # prompt-optimize ranks variants (here: two copies → both valid, best chosen)
    st, out, to, msg = EV.run_optimize({"spec": {"variants": [
        {"name": "floor_v1", "skill_dir": skill_dir}, {"name": "floor_v2", "skill_dir": skill_dir}]}}, "test")
    checks["prompt_optimize_ranks"] = st == "done" and out["best"]["mean_auc"] is not None

    # agent-config-eval agent contract
    st, out, to, msg = EV.run({"spec": {"skill_dir": skill_dir}}, "test")
    checks["eval_agent_solid"] = st == "done" and out["solid"] is True

    for k, val in checks.items():
        print(f"  {'OK' if val else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== prompt-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
