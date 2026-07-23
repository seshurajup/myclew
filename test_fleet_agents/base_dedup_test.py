"""base_dedup_test — verify BaseAgent.post() dedup: an agent must NOT re-post a message that is the same as
(or near-identical to) one of the last few thread entries. Prevents the board spam the user flagged."""
import json
import os
import sys
import tempfile
from pathlib import Path
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import fleet_agents as fa


def test_dedup_skips_repeats():
    agent = fa.AGENTS["xai"]
    # signature: identical / trailing-only-different messages collapse; genuinely different ones don't
    a = "[w] **SCOREBOARD** best 0.8803 gap -0.011 · 74 recipes"
    b = "[w] **SCOREBOARD** best 0.8803 gap -0.011 · 74 recipes"           # identical
    c = "[w] **TRICK-GATE** verdict: adopt gap2-recovery Δ+0.002"           # different
    assert agent._msg_sig(a) == agent._msg_sig(b), "identical messages must share a signature"
    assert agent._msg_sig(a) != agent._msg_sig(c), "different messages must differ"

    # _is_duplicate against a planted recent thread
    from researchpapers.fleet import post as _p
    with tempfile.TemporaryDirectory() as d:
        thr = Path(d) / "thread.jsonl"
        thr.write_text("\n".join(json.dumps({"content": m}) for m in [
            "old message 1", "old message 2", a]) + "\n")
        orig = _p.runtime_dir
        _p.runtime_dir = lambda: Path(d)
        try:
            dup = agent._is_duplicate(b)          # b == a which is in the last 3 → duplicate
            fresh = agent._is_duplicate(c)         # c not in recent → not a duplicate
        finally:
            _p.runtime_dir = orig
    assert dup is True, "a repeat of a recent message must be flagged as duplicate"
    assert fresh is False, "a genuinely new message must NOT be flagged as duplicate"
    return {"repeat_deduped": dup, "new_allowed": not fresh}


def _run():
    print("=== BASEAGENT dedup DATA-WISE VERIFIER ===")
    try:
        r = test_dedup_skips_repeats()
        for k, v in r.items(): print(f"  {'OK' if v else 'X'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  X FAILED: {e}"); ok = False
    print(f"=== base-dedup: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
