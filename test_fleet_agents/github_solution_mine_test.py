"""github_solution_mine_test — offline verifier (stubbed gh) for the GitHub winner-code miner.

Asserts: repo links are extracted from writeup markdown (and library deps skipped); the tree→key-file
prioritization keeps ML modules (train/model/loss) and README; fetched files are saved; the whole mine()
indexes correctly with a stubbed gh API. Also confirms the real gh CLI exists (live path wired).
"""
import os, sys, json, tempfile, shutil
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import github_solution_mine as A


def _run():
    print("=== GITHUB-SOLUTION-MINE VERIFIER (offline stub) ===")
    checks = {}

    # 1. extract repos from writeup markdown, skip library deps
    wd = tempfile.mkdtemp(prefix="wu_"); os.makedirs(os.path.join(wd, "comp1"))
    open(os.path.join(wd, "comp1", "rank1.md"), "w").write(
        "Our code: https://github.com/winner/kaggle-cryoet-1st-place and we used "
        "https://github.com/albumentations-team/albumentations and https://github.com/winner/kaggle-cryoet-1st-place again.")
    repos = A.repos_from_writeups(wd)
    checks["extract_repo"] = ("winner", "kaggle-cryoet-1st-place") in repos
    checks["skip_library"] = ("albumentations-team", "albumentations") not in repos
    checks["dedup"] = len(repos) == 1

    # 2. key-file prioritization keeps ML modules + README, drops noise
    paths = ["README.md", "src/train.py", "src/model.py", "src/loss.py", "notebooks/scratch.ipynb",
             "docs/index.html", "setup.py", "src/dataset.py"]
    kf = A.key_files(paths)
    checks["keeps_readme"] = "README.md" in kf
    checks["keeps_ml"] = all(f in kf for f in ["src/train.py", "src/model.py", "src/loss.py", "src/dataset.py"])
    checks["drops_noise"] = "docs/index.html" not in kf and "notebooks/scratch.ipynb" not in kf

    # 3. mine() with stubbed gh → indexes + saves
    def fake_gh(args, timeout=45):
        ep = args[1] if len(args) > 1 else ""
        if ep.startswith("repos/") and ep.count("/") == 2 and "trees" not in ep and "contents" not in ep:
            return json.dumps({"default_branch": "main"})
        if "git/trees" in ep:
            return json.dumps({"tree": [{"path": "train.py", "type": "blob"},
                                        {"path": "model.py", "type": "blob"},
                                        {"path": "README.md", "type": "blob"}]})
        if "contents/" in ep:
            import base64
            return json.dumps({"content": base64.b64encode(b"# real winning code\nimport torch\n").decode()})
        return ""
    orig = A._gh; A._gh = fake_gh
    try:
        out = tempfile.mkdtemp(prefix="ghm_")
        idx = A.mine([("winner", "kaggle-cryoet-1st-place")], out_dir=out, per_repo=8)
        checks["index_built"] = "winner/kaggle-cryoet-1st-place" in idx
        checks["files_saved"] = idx["winner/kaggle-cryoet-1st-place"]["saved"] >= 2
        # agent run() offline
        st, d, to, msg = A.run({"spec": {"repos": ["winner/kaggle-cryoet-1st-place"], "out_dir": out}}, "test")
        checks["agent_run_done"] = st == "done" and d["modules_fetched"] >= 2
        shutil.rmtree(out, ignore_errors=True)
    finally:
        A._gh = orig
        shutil.rmtree(wd, ignore_errors=True)

    # 4. real gh CLI exists
    import shutil as sh
    checks["gh_available"] = sh.which(A.GH) is not None

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== github-solution-mine: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
