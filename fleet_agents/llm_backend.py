"""llm_backend — a tiny, dependency-light multi-provider LLM client so fleet agents are NOT Claude-only.
Design pattern lifted from omnigent's llms/ SDK (get_adapter registry + provider/model routing + an
OpenAI-compatible adapter covering many backends), but reimplemented sync and stdlib-only (urllib) — no
httpx/litellm/openai/anthropic packages required. One `chat()` entry dispatches to:

  • openai-compatible HTTP  — covers OpenRouter, Ollama (/v1), vLLM, LM Studio, Groq, DeepSeek, together, …
  • anthropic Messages API   — Claude
  • local HF transformers     — offline model fallback (import-guarded; escalates if no model/GPU)
  • dummy / echo              — ZERO-network deterministic backend for unit tests

Providers/keys come from env so the whole fleet can be pointed at free/local LLMs with no code change:
  OPENROUTER_API_KEY   → https://openrouter.ai/api/v1        (hundreds of models incl. free tiers)
  OLLAMA_HOST          → $OLLAMA_HOST/v1  (default http://localhost:11434/v1, no key) — fully local/free
  OPENAI_BASE_URL[+KEY]→ any OpenAI-compatible endpoint (vLLM/LM Studio/self-host)
  ANTHROPIC_API_KEY    → Claude
Model strings may be prefixed "provider/model" (e.g. "ollama/llama3.1", "openrouter/meta-llama/…",
"anthropic/claude-…", "dummy/echo"); unprefixed → auto-select the first configured provider.

Public API:
  • chat(messages, model=, provider=, temperature=, max_tokens=, timeout=) -> {"text","provider","model","raw"}
  • available_providers() -> list[str]         — which backends are configured/usable right now.
  • class LLMBackendUnavailable(Exception)     — raised when nothing is configured (callers escalate()).
"""
from __future__ import annotations
import json
import os
import urllib.request
import urllib.error
from .base import BaseAgent


class LLMBackendUnavailable(Exception):
    pass


# ---------------------------------------------------------------- provider config from env
def _openai_compat_targets():
    """Ordered (provider, base_url, api_key) for every configured OpenAI-compatible endpoint."""
    out = []
    if os.environ.get("OPENROUTER_API_KEY"):
        out.append(("openrouter", "https://openrouter.ai/api/v1", os.environ["OPENROUTER_API_KEY"]))
    host = os.environ.get("OLLAMA_HOST")
    if host:
        out.append(("ollama", host.rstrip("/") + ("" if host.rstrip("/").endswith("/v1") else "/v1"), None))
    if os.environ.get("OPENAI_BASE_URL"):
        out.append(("openai", os.environ["OPENAI_BASE_URL"].rstrip("/"), os.environ.get("OPENAI_API_KEY")))
    return out


def available_providers():
    """Providers usable right now (dummy is always available; local-hf reported only if transformers imports)."""
    provs = [p for p, _, _ in _openai_compat_targets()]
    if os.environ.get("ANTHROPIC_API_KEY"):
        provs.append("anthropic")
    try:
        import transformers  # noqa: F401
        provs.append("local-hf")
    except Exception:  # noqa: BLE001
        pass
    provs.append("dummy")
    return provs


# ---------------------------------------------------------------- HTTP helper (stdlib only)
def _post_json(url, payload, headers, timeout):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


# ---------------------------------------------------------------- backends
def _openai_compat(messages, model, base_url, api_key, temperature, max_tokens, timeout):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    raw = _post_json(base_url.rstrip("/") + "/chat/completions", body, headers, timeout)
    return raw["choices"][0]["message"]["content"], raw


def _anthropic(messages, model, temperature, max_tokens, timeout):
    key = os.environ["ANTHROPIC_API_KEY"]
    sys_txt = "\n".join(m["content"] for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") != "system"]
    body = {"model": model, "messages": turns, "max_tokens": max_tokens, "temperature": temperature}
    if sys_txt:
        body["system"] = sys_txt
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    raw = _post_json("https://api.anthropic.com/v1/messages", body, headers, timeout)
    return "".join(b.get("text", "") for b in raw.get("content", [])), raw


def _local_hf(messages, model, temperature, max_tokens, timeout):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise LLMBackendUnavailable(f"local-hf needs transformers+torch: {e}")
    tok = AutoTokenizer.from_pretrained(model)
    mdl = AutoModelForCausalLM.from_pretrained(model)
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt").to(mdl.device)
    out = mdl.generate(**ids, max_new_tokens=max_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-5))
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, {"model": model}


def _dummy(messages, model, **_):
    """Zero-network deterministic echo backend for tests: returns the last user message content."""
    last = next((m["content"] for m in reversed(messages) if m.get("role") != "system"), "")
    return f"[echo:{model}] {last}", {"provider": "dummy", "model": model}


# ---------------------------------------------------------------- dispatch
def chat(messages, *, model="dummy/echo", provider=None, temperature=0.0, max_tokens=512, timeout=60):
    """Send a chat to an LLM. `messages` = [{"role","content"}, …]. Returns
    {"text","provider","model","raw"}. Provider is taken from `provider=`, else the "provider/" model prefix,
    else auto-selected from configured env (anthropic → openrouter → ollama → openai-compat → local-hf → dummy).
    Raises LLMBackendUnavailable if the chosen/auto provider is not configured (caller should escalate)."""
    if provider is None and "/" in model:
        provider, model = model.split("/", 1)
    compat = {p: (u, k) for p, u, k in _openai_compat_targets()}

    if provider is None:                                    # auto-select first configured
        if os.environ.get("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif compat:
            provider = next(iter(compat))
        else:
            provider = "dummy"; model = model if model != "dummy/echo" else "echo"

    if provider in ("dummy", "echo"):
        text, raw = _dummy(messages, model)
    elif provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise LLMBackendUnavailable("anthropic selected but ANTHROPIC_API_KEY unset")
        text, raw = _anthropic(messages, model, temperature, max_tokens, timeout)
    elif provider == "local-hf":
        text, raw = _local_hf(messages, model, temperature, max_tokens, timeout)
    elif provider in compat:
        base, key = compat[provider]
        text, raw = _openai_compat(messages, model, base, key, temperature, max_tokens, timeout)
    else:
        raise LLMBackendUnavailable(
            f"provider {provider!r} not configured; available={available_providers()}")
    return {"text": text, "provider": provider, "model": model, "raw": raw}


# ---------------------------------------------------------------- agent
class LLMBackend(BaseAgent):
    name = "llm-backend"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        provs = available_providers()
        prompt = s.get("prompt", "ping")
        try:
            r = chat([{"role": "user", "content": prompt}], model=s.get("model", "dummy/echo"),
                     provider=s.get("provider"))
            ok = True; text = r["text"]; used = r["provider"]
        except LLMBackendUnavailable as e:
            ok = False; text = str(e); used = "none"
        msg = (f"llm-backend: providers configured={provs}; test call via '{used}' → "
               f"{text[:80]!r}. Point the fleet at free/local LLMs (Ollama/OpenRouter/vLLM) or Claude with no "
               f"code change — set OLLAMA_HOST / OPENROUTER_API_KEY / ANTHROPIC_API_KEY")
        self.log(msg, kind="finding",
                 recommendation="agents needing an LLM call llm_backend.chat(...); dummy/echo keeps tests offline; "
                                "escalate on LLMBackendUnavailable")
        return self.done({"providers": provs, "ok": ok, "provider": used}, msg)


_AGENT = LLMBackend()


def run_llmbackend(q, worker):
    return _AGENT.run(q, worker)
