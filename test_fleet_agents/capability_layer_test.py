"""Regression for the capability layer adopted from different-ai/openwork.

repo: https://github.com/different-ai/openwork (v0.18.11 @ ed16748) · clone: research/openwork_repo
contract read: ee/apps/den-api/src/mcp/agent.ts · adopted in fleet_agents/agent_routing.py

Everything here runs OFFLINE against the `dummy` backend, so CI never depends on Ollama being up or on an
API key existing. The live proof (Gemma 4B via Ollama driving all 320 agents) is a separate manual run —
this file guards the contract, not the model.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "researchpapers"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fleet_agents import agent_routing as AR  # noqa: E402
from fleet_agents import llm_backend as LB  # noqa: E402


def _run():
    checks = {}

    idx = AR.capability_index()
    checks["the whole fleet is indexed as capabilities"] = len(idx) > 200
    checks["every capability carries a name and a schema digest"] = all(
        c.get("name") and c.get("schema_digest") for c in idx.values())

    # exactly two tools, in every provider's shape
    for flavor, key in (("anthropic", "input_schema"), ("openai", "function"), ("plain", "properties")):
        ts = AR.llm_tool_schemas(flavor)
        checks[f"{flavor}: exactly two tools"] = len(ts) == 2
        checks[f"{flavor}: the schema is shaped for that provider"] = all(key in t for t in ts)
    names = {t["name"] for t in AR.llm_tool_schemas("anthropic")}
    checks["the two tools are search + execute"] = names == {"search_capabilities", "execute_capability"}
    checks["search is declared read-only, execute destructive"] = (
        AR.SEARCH_ANNOTATIONS["read_only"] and AR.EXECUTE_ANNOTATIONS["destructive"])

    # discovery
    r = AR.search_capabilities("quantize a model to int8 for faster inference")
    checks["search finds the quantize agent"] = "quantize" in [m["name"] for m in r["matches"]]
    checks["and reports how many capabilities exist"] = r["total_capabilities"] > 200
    checks["a match carries the schema the model must fill"] = all(
        "spec_schema" in m and "schema_digest" in m for m in r["matches"])
    miss = AR.search_capabilities("zzzz qqqq no such thing")
    checks["a miss returns a hint, not just an empty list"] = bool(miss.get("hint")) or bool(miss["matches"])

    # the retry protocol — the part that stops a model looping
    unknown = AR.execute_capability("no-such-agent", {})
    checks["unknown capability is named as such"] = unknown["error"] == "unknown_capability"
    checks["and tells the model to search again"] = unknown["retry"]["action"] == "search_capabilities"
    checks["and forbids an identical retry"] = unknown["same_arguments_retryable"] is False
    near = AR.execute_capability("quantise", {})
    checks["a near-miss NAME gets suggestions"] = "quantize" in near["did_you_mean"]

    stale = AR.execute_capability("cv-build", {}, schema_digest_echo="deadbeef")
    checks["a stale schema digest is rejected"] = stale["error"] == "stale_schema_digest"
    checks["and sends the model back to search"] = stale["retry"]["searchRequired" if
                                                                 "searchRequired" in stale["retry"]
                                                                 else "search_required"]

    bad = AR.execute_capability("cv-build", {"k": "two"})
    checks["a wrong argument TYPE is caught before the agent runs"] = (
        bad["error"] == "invalid_capability_arguments")
    checks["with the offending path named"] = bad["issues"][0]["path"] == "spec.k"
    checks["and the model told to correct arguments"] = bad["retry"]["action"] == "correct_arguments"
    notdict = AR.execute_capability("cv-build", ["not", "a", "dict"])
    checks["a non-object spec is rejected"] = notdict["error"] == "invalid_capability_arguments"

    # execution really dispatches (a stub handler proves the wiring without running a real agent)
    seen = {}

    def _stub(q, worker):
        seen.update(q)
        return ("done", {"echo": q["spec"]}, None, "stub ran")

    good = AR.execute_capability("cv-build", {"k": 2}, dispatch=_stub)
    checks["execute dispatches to the handler"] = good["ok"] and seen.get("kind") == "cv-build"
    checks["passing the spec through unchanged"] = seen.get("spec") == {"k": 2}
    checks["and returning the agent's own message"] = good["message"] == "stub ran"

    def _boom(q, worker):
        raise RuntimeError("agent exploded")

    crash = AR.execute_capability("cv-build", {}, dispatch=_boom)
    checks["an agent crash is a RESULT, not an exception"] = crash["error"] == "capability_failed"
    checks["carrying the real error text"] = "agent exploded" in crash["message"]

    # the text-protocol parser (for models with no native tool calling, e.g. gemma3n:e4b)
    o = AR._first_json_object('sure! {"tool": "search_capabilities", "query": "int8"} hope that helps')
    checks["JSON is extracted from surrounding prose"] = o == {"tool": "search_capabilities",
                                                               "query": "int8"}
    nested = AR._first_json_object('{"tool":"execute_capability","spec":{"k":2,"s":"}"}}')
    checks["nested objects and braces in strings parse"] = nested["spec"] == {"k": 2, "s": "}"}
    checks["prose with no JSON returns None"] = AR._first_json_object("no json here") is None
    checks["a broken object does not raise"] = AR._first_json_object('{"a": ') is None

    # the whole loop, offline
    out = AR.capability_loop("quantize a model to int8", model="dummy/echo", max_steps=3)
    tools_used = [s.get("tool") for s in out["steps"]]
    checks["the loop runs end to end with no network"] = "search_capabilities" in tools_used
    checks["and reaches a final answer"] = out["answer"] is not None
    checks["execution is OPT-IN (a plan is returned, nothing runs)"] = not any(
        s.get("tool") == "execute_capability" and not s.get("planned") for s in out["steps"])

    # llm_backend: the transports the loop rides on
    checks["dummy is always available"] = "dummy" in LB.available_providers()
    checks["tool calls normalise from the OpenAI shape"] = LB.tool_calls(
        {"choices": [{"message": {"tool_calls": [
            {"id": "1", "function": {"name": "search_capabilities", "arguments": '{"query":"x"}'}}]}}]}
    ) == [{"id": "1", "name": "search_capabilities", "arguments": {"query": "x"}}]
    checks["and from the Anthropic shape"] = LB.tool_calls(
        {"content": [{"type": "tool_use", "id": "a", "name": "execute_capability", "input": {"name": "q"}}]}
    ) == [{"id": "a", "name": "execute_capability", "arguments": {"name": "q"}}]
    checks["a malformed argument string is kept, not crashed on"] = LB.tool_calls(
        {"choices": [{"message": {"tool_calls": [
            {"function": {"name": "x", "arguments": "{not json"}}]}}]}
    )[0]["arguments"] == {"_unparsed": "{not json"}
    checks["the openwork link is in the adopting module"] = "different-ai/openwork" in (
        AR.__doc__ or "") or "different-ai/openwork" in open(AR.__file__).read()

    # THE EXACT-NAME GATE. Measured defect: `_terms` kept hyphens, so a query for an agent's own name

    # stayed one token (`tracker-postproc`) while names are INDEXED split (`{tracker, postproc}`) --

    # they could never intersect and `search_capabilities("tracker-postproc")` returned an EMPTY list.

    # Only 7.5% of agents were findable by their own name; hyphenated ones (most of the roster) were

    # unreachable to Claude and the local pilot alike. Now 90%.

    import random as _rnd

    _idx = AR.capability_index()

    _names = sorted(_idx)

    _rnd.Random(7).shuffle(_names)

    _sample = _names[:80]

    _found = [n for n in _sample

              if n in [m["name"] for m in AR.search_capabilities(n, limit=8)["matches"]]]

    checks["an agent is findable by its EXACT name"] = len(_found) / len(_sample) > 0.80

    checks["hyphenated names retrieve (the regression that broke every one)"] = all(

        n in [m["name"] for m in AR.search_capabilities(n, limit=8)["matches"]]

        for n in ("tracker-postproc", "lowbit-qat", "nb-preflight"))


    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n  {sum(1 for v in checks.values() if v)}/{len(checks)} passed")
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
