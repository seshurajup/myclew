"""context_offload_test — data-wise verifier for the context-offload agent.

Ground-truth properties (the whole point of the agent):
  1. A large output is WRITTEN to disk verbatim (nothing lost), under output/run_artifacts/.
  2. The returned STUB is COMPACT — far smaller than the input — and carries the path + a head/tail preview
     (so the board thread stays small; this is the deep-agent offload contract).
  3. RP_COMP routes the artifact dir to the active competition (reusable across comps).
  4. read_slice round-trips a slice of the offloaded file (offset/limit), so detail is recoverable on demand.
  5. Agent contract: run(q, worker) returns a valid (status, data, to, message) 4-tuple for offload AND read,
     and is harmless on an empty spec (smoke path).
"""
import os
import sys
import tempfile

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import context_offload as C

VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}


def _run():
    print("=== CONTEXT-OFFLOAD VERIFIER ===")
    checks = {}
    big = "\n".join(f"row {i}: some measured detail value={i * 3.14159:.4f}" for i in range(5000))

    with tempfile.TemporaryDirectory() as td:
        # 1. write verbatim + compact stub
        res = C.offload(big, label="score table", summary="golden CV dump", out_dir=td)
        p = res["path"]
        checks["file_written"] = os.path.exists(p)
        checks["verbatim"] = open(p).read() == big
        checks["lines_counted"] = res["lines"] == 5000
        stub = res["stub"]
        checks["stub_compact"] = len(stub) < len(big) // 5          # thread carries a fraction, not the dump
        checks["stub_has_path"] = p in stub
        checks["stub_has_preview"] = "truncated" in stub and "row 0:" in stub and "row 4999:" in stub
        checks["label_sanitized"] = "score_table" in os.path.basename(p)  # space → underscore
        print(f"  -> input {len(big)} bytes / 5000 lines  →  stub {len(stub)} bytes  (path {os.path.basename(p)})")

        # 4. read_slice round-trip
        sl = C.read_slice(p, offset=10, limit=3)
        checks["read_offset"] = sl["text"].splitlines()[0] == "row 10: some measured detail value=31.4159"
        checks["read_returned"] = sl["returned"] == 3 and sl["more"] is True
        bad = C.read_slice(os.path.join(td, "nope.md"))
        checks["read_missing_graceful"] = "error" in bad
        print(f"  -> read_slice(offset=10,limit=3): first='{sl['text'].splitlines()[0]}' more={sl['more']}")

    # 3. RP_COMP routing
    old = os.environ.get("RP_COMP")
    try:
        os.environ["RP_COMP"] = "some-other-comp"
        d = C.artifacts_dir()
        checks["rpcomp_routes"] = d.as_posix().endswith("some-other-comp/output/run_artifacts")
    finally:
        if old is None:
            os.environ.pop("RP_COMP", None)
        else:
            os.environ["RP_COMP"] = old
    print(f"  -> RP_COMP='some-other-comp' → {C.artifacts_dir() if False else d}")

    # 5. agent contract — offload, read, and empty-spec smoke
    with tempfile.TemporaryDirectory() as td:
        out = C.run({"question": "hdr", "spec": {"text": "a\nb\nc", "label": "t", "out_dir": td}}, "tester")
        checks["contract_offload"] = isinstance(out, tuple) and len(out) == 4 and out[0] in VALID
        wrote = out[1].get("path")
        rd = C.run({"question": "", "spec": {"mode": "read", "path": wrote, "offset": 0, "limit": 2}}, "tester")
        checks["contract_read"] = isinstance(rd, tuple) and rd[0] == "done" and rd[1]["returned"] == 2
        sm = C.run({"question": "smoke text", "spec": {}}, "tester")
        checks["contract_smoke"] = isinstance(sm, tuple) and len(sm) == 4 and sm[0] in VALID
    print(f"  -> contract: offload={checks['contract_offload']} read={checks['contract_read']} smoke={checks['contract_smoke']}")

    # 6. truncate_args — LOWER-tier in-place clip (deepagents _should_truncate_args), MEASURED shrink
    assert C.TRUNCATE_LIMIT < C.OFFLOAD_LIMIT, "two-tier: truncate threshold must be BELOW offload threshold"
    checks["two_tier_thresholds"] = C.TRUNCATE_LIMIT < C.OFFLOAD_LIMIT
    big_arg = "x" * 50_000
    clipped = C.truncate_args(big_arg, limit=4_000)
    checks["trunc_str_shrinks"] = len(clipped) < len(big_arg) and len(clipped) < 4_500  # ~limit + marker
    checks["trunc_keeps_head_tail"] = clipped.startswith("x") and clipped.endswith("x")
    checks["trunc_marker"] = "chars elided" in clipped and "if offloaded" in clipped
    checks["trunc_path_named"] = "docs/big.log" in C.truncate_args("y" * 9000, limit=1000, path="docs/big.log")
    checks["trunc_small_untouched"] = C.truncate_args("short", limit=4_000) == "short"  # below limit → verbatim
    # dict: oversized string VALUES clipped IN PLACE, small ones + non-str untouched
    d = {"cmd": "grep", "out": "z" * 30_000, "n": 5}
    ret = C.truncate_args(d, limit=2_000)
    checks["trunc_dict_inplace"] = ret is d and len(d["out"]) < 2_500 and d["cmd"] == "grep" and d["n"] == 5
    shrink_pct = 100 * (1 - len(clipped) / len(big_arg))
    print(f"  -> truncate_args: str 50000→{len(clipped)} chars ({shrink_pct:.1f}% shrink); "
          f"dict['out'] 30000→{len(d['out'])} chars; thresholds {C.TRUNCATE_LIMIT}<{C.OFFLOAD_LIMIT}")
    # 6b. agent contract for the truncate verb
    tv = C.run({"question": "hdr", "spec": {"mode": "truncate", "text": "w" * 20_000, "limit": 3_000}}, "tester")
    checks["contract_truncate"] = (isinstance(tv, tuple) and tv[0] == "done"
                                   and tv[1]["shrunk"] and tv[1]["after_chars"] < 20_000)

    ok = all(checks.values())
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    print(f"=== context-offload: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
