"""llm_backend_test — data-wise verifier for the multi-provider LLM backend (omnigent-pattern, stdlib-only).

Core properties (all OFFLINE — no network, no keys, via the dummy/echo provider):
  1. dummy/echo round-trips the last user message deterministically.
  2. provider/model prefix parsing selects the provider; explicit provider= overrides.
  3. available_providers always includes dummy; reflects env (OLLAMA_HOST/OPENROUTER/ANTHROPIC) when set.
  4. an unconfigured provider raises LLMBackendUnavailable (so callers can escalate).
  5. agent contract runs offline."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import llm_backend as B


def _run():
    print("=== LLM-BACKEND VERIFIER ===")
    checks = {}

    # 1. dummy echo round-trip
    r = B.chat([{"role": "user", "content": "hello world"}], model="dummy/echo")
    checks["echo_roundtrip"] = "hello world" in r["text"] and r["provider"] == "dummy"
    print(f"  -> echo: {r['text']!r} via {r['provider']}")

    # 2. prefix parsing + explicit override
    r2 = B.chat([{"role": "user", "content": "hi"}], provider="dummy", model="anything")
    checks["explicit_provider"] = r2["provider"] == "dummy" and r2["model"] == "anything"
    r3 = B.chat([{"role": "user", "content": "hi"}], model="echo/foo")
    checks["prefix_parse"] = r3["provider"] == "echo"

    # 3. available providers reflects env
    provs = B.available_providers()
    checks["dummy_always"] = "dummy" in provs
    old = os.environ.get("OLLAMA_HOST")
    os.environ["OLLAMA_HOST"] = "http://localhost:11434"
    checks["ollama_detected"] = "ollama" in B.available_providers()
    if old is None:
        del os.environ["OLLAMA_HOST"]
    else:
        os.environ["OLLAMA_HOST"] = old
    print(f"  -> providers (base): {provs}")

    # 4. unconfigured provider escalates
    saved = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        B.chat([{"role": "user", "content": "hi"}], provider="anthropic")
        checks["unconfigured_raises"] = False
    except B.LLMBackendUnavailable:
        checks["unconfigured_raises"] = True
    finally:
        if saved is not None:
            os.environ["ANTHROPIC_API_KEY"] = saved

    # 5. agent contract (offline)
    st, dta, to, msg = B.run_llmbackend({"spec": {"model": "dummy/echo", "prompt": "ping"}}, "t")
    checks["agent_done"] = st == "done" and dta["ok"] and "dummy" in dta["providers"]

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== llm-backend: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
