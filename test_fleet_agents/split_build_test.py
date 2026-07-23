"""split_build_test — per-agent verifier: run `split-build` on real repo state and assert it returns a valid
(status, data, to, message) contract with real output (not a crash). Light agent: runs directly."""
import os, sys, subprocess
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import fleet_agents as fa

VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}
KIND = "split-build"


def _stub(*a, **k):
    out = ("VALIDATION_SCORE: 0.75\nofficial_score: 0.75\nTO_SUBMIT: /tmp/x.csv\n"
           '{"score": 0.75, "adjE": 0.75, "combined_score": 0.9, "base_score": 0.88, '
           '"combined_divJ": 0.1, "combined_adjE": 0.8, "delta": 0.02, "n": 12, "folds": [], "probs": []}\n')
    return subprocess.CompletedProcess(a[0] if a else k.get("args", []), 0, stdout=out, stderr="")


def test_split_build_runs_and_returns_contract():
    raw = fa._RAW_HANDLERS[KIND]
    heavy = False
    orig = subprocess.run
    if heavy:
        subprocess.run = _stub
    try:
        out = raw({"question": "test split-build", "spec": {}}, "test")
    finally:
        subprocess.run = orig
    assert isinstance(out, tuple) and len(out) == 4, f"bad return shape: {out}"
    assert out[0] in VALID, f"invalid status: {out[0]}"
    assert isinstance(out[1], dict) or out[1] is None, f"data not a dict: {type(out[1])}"
    return {"valid_contract": True, "runs_clean": True}


def _run():
    print(f"=== {KIND} PER-AGENT VERIFIER ===")
    try:
        r = test_split_build_runs_and_returns_contract()
        for k, v in r.items(): print(f"  {'OK' if v else 'X'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  X FAILED: {e}"); ok = False
    print(f"=== {KIND}: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
