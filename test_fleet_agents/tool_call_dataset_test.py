"""Regression for the verified tool-call SFT dataset (fleet_agents/prompt_dataset.build_tool_call_dataset).

The dataset teaches a local Gemma-class model the fleet's two-tool protocol. Its whole claim to quality is
that `execute_capability` — the real validator — vets every example before it is kept, so this test's job is
to prove the vetting is real: correct examples must validate, expected-error examples must reproduce their
exact error code, and the quality gates must be reported rather than assumed.

Offline: no LLM and no network. Uses the live capability index and a stub dispatch.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "researchpapers"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fleet_agents import agent_routing as AR  # noqa: E402
from fleet_agents import prompt_dataset as PD  # noqa: E402


def _stub(q, worker):
    return ("done", {"stub": True}, None, "stub")


def _parse(target):
    """Targets are the COMPACT wire form by default (44% fewer tokens than JSON, 4 fragile syntax
    characters instead of 28). Accept either form so the test covers both `format` settings."""
    return AR.from_compact(target) or json.loads(target)


def _run():
    checks = {}
    d = PD.build_tool_call_dataset({"n": 320, "seed": 3, "hard_frac": 0.5})
    ex, rep = d["examples"], d["report"]

    checks["the builder produces a non-trivial corpus"] = len(ex) > 150
    checks["all four protocol skills are present"] = all(
        rep["by_kind"].get(k, 0) > 10
        for k in ("search", "execute", "recover_unknown", "recover_args"))
    checks["it covers many distinct capabilities"] = rep["capabilities_covered"] > 80
    checks["train/val are disjoint"] = not (
        {e["input"] + e["gold"] for e in d["train"]} & {e["input"] + e["gold"] for e in d["val"]})

    # THE PROMPT-LEAK GATE. Measured defect: these corpora REPEAT prompts (the same fleet state recurs), so
    # a random example-level split put 21.3% of val inputs — 60.6% of `next_agent` — into train as well.
    # `next_agent` then read 100% while the actual decision skill was never tested. The split is now by
    # PROMPT GROUP, and disjointness on (input+gold) is too weak to catch a regression: check the input.
    checks["no val PROMPT appears in train (not just no exact pair)"] = not (
        {e["input"] for e in d["train"]} & {e["input"] for e in d["val"]})
    dh_leak = PD.build_history_tool_dataset({"direct_per_agent": 4, "history": 3})
    checks["the history builder splits by prompt too"] = not (
        {e["input"] for e in dh_leak["train"]} & {e["input"] for e in dh_leak["val"]})
    checks["no duplicate (kind, request, gold) triples"] = len(
        {(e["kind"], e["input"], e["gold"]) for e in ex}) == len(ex)

    # every example is chat-formatted with the protocol in the system turn and JSON as the target
    checks["every example is chat-formatted"] = all(
        e["messages"][0]["role"] == "system" and e["messages"][-1]["role"] == "assistant"
        for e in ex)
    checks["the system turn carries the two-tool contract"] = all(
        "search_capabilities" in e["messages"][0]["content"]
        and "execute_capability" in e["messages"][0]["content"] for e in ex)
    checks["every target parses and names a tool"] = all(
        _parse(e["messages"][-1]["content"]).get("tool") in
        ("search_capabilities", "execute_capability") for e in ex)
    checks["targets are in the COMPACT wire form by default"] = all(
        e["messages"][-1]["content"].split("|", 1)[0].upper() in ("SEARCH", "EXEC", "FINAL")
        for e in ex)

    # --- THE POINT OF THE WHOLE THING: re-validate the labels with the real validator
    bad_exec = bad_search = 0
    for e in ex:
        g = _parse(e["gold"])
        if g["tool"] == "execute_capability":
            r = AR.execute_capability(g["name"], g.get("spec"),
                                      schema_digest_echo=g.get("schema_digest"), dispatch=_stub)
            if not r.get("ok"):
                bad_exec += 1
        else:
            hits = [m["name"] for m in AR.search_capabilities(g["query"], limit=8)["matches"]]
            if e["capability"] not in hits:
                bad_search += 1
    checks["EVERY execute label passes the real validator"] = bad_exec == 0
    checks["EVERY search label actually retrieves its target"] = bad_search == 0

    # --- the recovery examples must contain a genuine, correctly-typed error in their transcript
    unk = [e for e in ex if e["kind"] == "recover_unknown"]
    arg = [e for e in ex if e["kind"] == "recover_args"]
    checks["unknown-capability transcripts contain that exact error"] = all(
        "unknown_capability" in e["messages"][-2]["content"] for e in unk)
    checks["and instruct a re-search, per the protocol"] = all(
        _parse(e["gold"])["tool"] == "search_capabilities" for e in unk)
    checks["bad-argument transcripts contain that exact error"] = all(
        "invalid_capability_arguments" in e["messages"][-2]["content"] for e in arg)
    checks["and instruct corrected ARGUMENTS, not a re-search"] = all(
        _parse(e["gold"])["tool"] == "execute_capability" for e in arg)
    checks["the anti-loop flag is visible in the transcript"] = all(
        "same_arguments_retryable" in e["messages"][-2]["content"] for e in arg)
    checks["every execute label echoes a digest"] = all(
        _parse(e["gold"]).get("schema_digest")
        for e in ex if _parse(e["gold"])["tool"] == "execute_capability")

    # both wire formats must round-trip, and the JSON path must still be selectable for comparison runs
    dj = PD.build_tool_call_dataset({"n": 120, "seed": 3, "format": "json"})
    checks["format='json' still produces JSON targets"] = all(
        json.loads(e["messages"][-1]["content"]).get("tool") for e in dj["examples"])
    checks["compact round-trips types exactly"] = AR.from_compact(
        AR.to_compact({"tool": "execute_capability", "name": "x", "spec": {"b": 4, "f": 0.5, "t": True},
                       "schema_digest": "d"}))["spec"] == {"b": 4, "f": 0.5, "t": True}
    checks["compact survives fences and prose"] = AR.from_compact(
        "Sure!\n```\nSEARCH|quantize int8\n```")["query"] == "quantize int8"

    # --- quality gates are REPORTED (their values are informational; their presence is required)
    for k in ("verified_frac", "mean_request_summary_overlap", "overlap_hard", "overlap_easy",
              "idf_top1_disagrees_frac", "unroutable_no_summary", "reject_reasons"):
        checks[f"the report exposes `{k}`"] = k in rep
    checks["the verified fraction is high"] = rep["verified_frac"] > 0.9
    checks["rejects are KEPT for audit, not silently dropped"] = isinstance(d["rejected"], list) and (
        len(d["rejected"]) == 0 or "why" in d["rejected"][0])
    checks["the anti-leakage path measurably lowers overlap"] = rep["overlap_hard"] < rep["overlap_easy"]
    checks["a real share of examples are genuine hard negatives"] = rep["idf_top1_disagrees_frac"] > 0.02

    # THE POSITIONAL-SHORTCUT GATE. Measured defect: the execute shortlist is built by searching the
    # target's own summary, so the gold sat at rank 1 in 94.8% of examples — "always pick the first entry"
    # scored 94.8% and a 2B model learned exactly that, reporting a meaningless 100%. The shortlist is now
    # shuffled; this check makes the regression impossible to reintroduce silently.
    checks["the gold is NOT concentrated at shortlist rank 1"] = rep["gold_rank1_frac"] < 0.30
    checks["and sits roughly uniformly across the shortlist"] = 0.03 < rep["gold_rank1_frac"] < 0.30
    checks["the gate reports how many examples it covers"] = rep["shortlisted_examples"] > 0

    # --- failure mining must reinforce the RIGHT task (a real bug: `direct` failures mined SEARCH
    # examples, so the weakest kind — weights-only recall at 60.9% — could never improve from its own
    # mistakes). And for `direct` there is no shortlist to hide a hard negative in, so the confused
    # sibling and the name-family are trained alongside instead.
    md = PD.mine_failures([{"kind": "direct", "capability": "tracker-postproc",
                            "predicted": "tracker-predict"}], {"mult": 2})
    checks["a DIRECT failure mines direct examples, not search"] = bool(md["examples"]) and all(
        e["kind"] == "direct" and _parse(e["gold"])["tool"] == "execute_capability"
        for e in md["examples"])
    mined_names = {e["capability"] for e in md["examples"]}
    checks["mining trains the target AND the agent it was confused with"] = {
        "tracker-postproc", "tracker-predict"} <= mined_names
    checks["and pulls in name-family siblings for contrast"] = len(
        {n for n in mined_names if n.startswith("tracker-")}) >= 3
    checks["direct mining carries the recall MODE marker"] = all(
        e["input"].startswith("MODE: recall") for e in md["examples"])

    me = PD.mine_failures([{"kind": "execute", "capability": "quantize",
                            "predicted": "lowbit-qat"}], {"mult": 1})
    checks["an EXECUTE failure still mines shortlist examples"] = bool(me["examples"]) and all(
        len(e["messages"]) == 4 for e in me["examples"])
    checks["every mined example passes the real validator"] = all(
        AR.execute_capability(g["name"], g["spec"], schema_digest_echo=g["schema_digest"],
                              dispatch=_stub).get("ok")
        for e in md["examples"] + me["examples"]
        for g in [_parse(e["gold"])] if g["tool"] == "execute_capability")

    # --- the two prompt MODES must be distinguishable, or the model cannot tell recall from discovery
    dh = PD.build_history_tool_dataset({"direct_per_agent": 2, "history": 2})
    dirs = [e for e in dh["examples"] if e["kind"] == "direct"]
    nxt = [e for e in dh["examples"] if e["kind"] == "next_agent"]
    checks["direct prompts are marked MODE: recall"] = bool(dirs) and all(
        e["input"].startswith("MODE: recall") for e in dirs)
    checks["discovery prompts are marked MODE: discover"] = bool(nxt) and all(
        e["input"].startswith("MODE: discover") for e in nxt)
    checks["recall targets EXEC, discovery targets SEARCH"] = (
        all(_parse(e["gold"])["tool"] == "execute_capability" for e in dirs)
        and all(_parse(e["gold"])["tool"] == "search_capabilities" for e in nxt))

    # --- TRAINING/SERVING SHAPE. Measured defect: local_pilot inserted a "Competition `x`: N experiments
    # logged, best private ..." line that training never contains, and that ONE unseen line pushed the model
    # out of the protocol — it answered FINAL, the pilot got no capability, while the benchmark (fed
    # training-shaped prompts) happily read 85%. The pilot's request and the training input must agree.
    from fleet_agents import local_pilot as LP  # noqa: PLC0415
    import re as _re
    try:
        _req, _st, _n, _u = LP.build_state_request("biohub-cell-tracking-during-development", 3)
        _dh2 = PD.build_history_tool_dataset({"history": 3, "kinds": ["next_agent"]})
        _tin = _dh2["examples"][0]["input"]

        def _shape(t):
            return [_re.sub(r"`[^`]*`", "`X`", ln.split(":")[0])[:44]
                    for ln in t.split("\n") if ln.strip()]
        checks["the pilot's prompt HEAD matches the training prompt"] = _shape(_req)[:2] == _shape(_tin)[:2]
        checks["the pilot's prompt carries the discover MODE marker"] = _req.startswith("MODE: discover")
        # THE SELF-FEEDBACK GATE. The pilot's decisions land in the same decision log it READS, so its
        # history filled with its own prior picks: every run changed the next run's prompt and, at
        # temperature 0, identical state gave 3 different answers with 2 runs producing NO capability
        # (3/5 usable) while the bench read 99.5% dispatchable. Excluding self-entries -> 5/5, deterministic.
        checks["the pilot never reads its OWN output back as history"] = (
            "agent `local-pilot`" not in _req)
        checks["both end with the same instruction line"] = (
            _req.strip().split("\n")[-1] == _tin.strip().split("\n")[-1])
    except (FileNotFoundError, ValueError, IndexError) as _e:
        checks[f"pilot/training shape check ran (skipped: {type(_e).__name__})"] = False

    # --- determinism, so a training run is reproducible
    d2 = PD.build_tool_call_dataset({"n": 320, "seed": 3, "hard_frac": 0.5})
    checks["the same seed reproduces the same corpus"] = [e["gold"] for e in d2["examples"]] == [
        e["gold"] for e in ex]

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  corpus: n={rep['n']} verified={rep['verified_frac']:.1%} "
          f"kinds={rep['by_kind']} covered={rep['capabilities_covered']}")
    print(f"  {sum(1 for v in checks.values() if v)}/{len(checks)} passed")
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
