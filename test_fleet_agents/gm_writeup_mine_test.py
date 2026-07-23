"""gm_writeup_mine_test — data-wise verifier for the gm-writeup-mine pipeline (OFFLINE, stubbed network).

Stubs the nvidia-kaggle bearer scripts so no network/token is needed, then asserts the fetch→save→index
logic: writeup-URL JSON is parsed, top-N markdown is written to docs/gm_writeups/<slug>/, and the summary
counts are correct. Also confirms the real bearer scripts EXIST on disk (so a live run is wired).
"""
import os, sys, json, tempfile, shutil
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import gm_writeup_mine as A


def _run():
    print("=== GM-WRITEUP-MINE DATA-WISE VERIFIER (offline stub) ===")
    checks = {}

    # stub the script runner: fetch_leaderboard_writeups → JSON list; fetch_writeup → markdown
    def fake_run(script, args, timeout=90):
        if script == "fetch_leaderboard_writeups.py":
            return json.dumps([
                {"rank": "1", "team": "a", "writeup_url": "https://kaggle.com/c/x/writeups/1st-place"},
                {"rank": "2", "team": "b", "writeup_url": "https://kaggle.com/c/x/writeups/2nd-place"},
                {"rank": "3", "team": "c", "writeup_url": "https://kaggle.com/c/x/writeups/3rd-place"},
            ])
        if script == "fetch_writeup.py":
            return "# Solution\n\n" + ("This is a real writeup with plenty of content. " * 20)
        return ""
    orig = A._run_script
    A._run_script = fake_run
    try:
        tmp = tempfile.mkdtemp(prefix="gmwm_")
        summary = A.mine(["fake-comp"], top_n=2, out_dir=tmp)
        checks["summary_count"] = summary["fake-comp"]["n"] == 2                 # top_n=2 honored
        files = os.listdir(os.path.join(tmp, "fake-comp"))
        checks["files_written"] = len(files) == 2 and all(f.endswith(".md") for f in files)
        checks["file_nonempty"] = all(os.path.getsize(os.path.join(tmp, "fake-comp", f)) > 200 for f in files)

        # idempotent: re-mine skips existing (still counts them)
        summary2 = A.mine(["fake-comp"], top_n=2, out_dir=tmp)
        checks["idempotent"] = summary2["fake-comp"]["n"] == 2 and len(os.listdir(os.path.join(tmp, "fake-comp"))) == 2

        # agent run() contract
        st, d, to, msg = A.run({"spec": {"slugs": ["fake-comp"], "top_n": 2, "out_dir": tmp}}, "test")
        checks["agent_run_done"] = st == "done" and d["total_writeups"] == 2
        shutil.rmtree(tmp, ignore_errors=True)
    finally:
        A._run_script = orig

    # the real bearer scripts must exist on disk (live path is wired)
    checks["bearer_scripts_exist"] = os.path.isfile(os.path.join(A._SCRIPTS, "fetch_leaderboard_writeups.py")) \
        and os.path.isfile(os.path.join(A._SCRIPTS, "fetch_writeup.py"))

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== gm-writeup-mine: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
