import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import gm_repo_distill as GD


def _run():
    print("=== GM-REPO-DISTILL VERIFIER (offline, injected clone/scan) ===")
    checks = {}
    # injected clone (no network) + scan returning known techniques incl a GAP (ema, "")
    def fake_clone(repo, dest): return repo != "bad/repo"
    def fake_scan(dest): return {"exponential moving average (EMA)": "", "pseudo-labeling / self-training": "pseudo-label"}
    import tempfile
    gd = tempfile.mkdtemp(prefix="ghtest_")
    man = os.path.join(gd, "manifest.json")
    res = GD.distill(["a/x", "b/y", "bad/repo"], github_dir=gd, clone=fake_clone, scan=fake_scan, manifest_path=man)
    checks["all_processed"] = set(res) == {"a/x", "b/y", "bad/repo"}
    checks["good_done"] = res["a/x"]["status"] == "done" and "gaps" in res["a/x"]
    checks["gap_detected"] = "exponential moving average (EMA)" in res["a/x"]["gaps"]
    checks["covered_not_gap"] = "pseudo-labeling / self-training" not in res["a/x"]["gaps"]
    checks["clone_fail_handled"] = res["bad/repo"]["status"] == "clone-failed"
    # cleanup: clones deleted (keep=False default) — dest dirs should not remain
    import glob
    checks["clones_deleted"] = len(glob.glob(os.path.join(gd, "*__*"))) == 0
    # manifest skip: a second run skips already-done repos
    res2 = GD.distill(["a/x"], github_dir=gd, clone=fake_clone, scan=fake_scan, manifest_path=man)
    checks["manifest_skips_done"] = res2 == {} or "a/x" not in res2
    # real lexicon has entries + GAP markers
    checks["lexicon_has_gaps"] = any(a == "" for _, a in GD.TECH.values()) and len(GD.TECH) >= 15
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"\n=== gm-repo-distill: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
