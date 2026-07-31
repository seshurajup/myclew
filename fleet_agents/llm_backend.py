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
_OLLAMA_PROBE = {}


def _ollama_alive(timeout=1.0):
    """Is a local Ollama serving right now? Probed once per process.

    `OLLAMA_HOST` is often left unexported even when the daemon is running, which used to make the fleet
    fall back to `dummy` while a perfectly good local model sat idle. A 1-second probe removes that class
    of silent downgrade.
    """
    if "alive" in _OLLAMA_PROBE:
        return _OLLAMA_PROBE["alive"]
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=timeout):
            _OLLAMA_PROBE["alive"] = True
    except Exception:  # noqa: BLE001
        _OLLAMA_PROBE["alive"] = False
    return _OLLAMA_PROBE["alive"]


def ollama_models(timeout=3.0):
    """Model names a local Ollama can serve — so a caller can pick one instead of guessing a tag."""
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=timeout) as r:
            return [m["name"] for m in json.loads(r.read().decode()).get("models", [])]
    except Exception:  # noqa: BLE001
        return []


def _openai_compat_targets():
    """Ordered (provider, base_url, api_key) for every configured OpenAI-compatible endpoint."""
    out = []
    if os.environ.get("OPENROUTER_API_KEY"):
        out.append(("openrouter", "https://openrouter.ai/api/v1", os.environ["OPENROUTER_API_KEY"]))
    host = os.environ.get("OLLAMA_HOST")
    if not host and _ollama_alive():
        host = "http://localhost:11434"        # running locally but unexported — the common case
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        # a bare "HTTP Error 400: Bad Request" is undebuggable; providers put the actual reason
        # ("model does not support tools", "unknown model", a schema complaint) in the BODY
        try:
            detail = e.read().decode()[:600]
        except Exception:  # noqa: BLE001
            detail = ""
        raise LLMBackendUnavailable(f"{url} -> HTTP {e.code}: {detail or e.reason}") from None


# ---------------------------------------------------------------- backends
def _openai_compat(messages, model, base_url, api_key, temperature, max_tokens, timeout, tools=None):
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
    raw = _post_json(base_url.rstrip("/") + "/chat/completions", body, headers, timeout)
    return (raw["choices"][0]["message"].get("content") or ""), raw


def _anthropic(messages, model, temperature, max_tokens, timeout, tools=None):
    key = os.environ["ANTHROPIC_API_KEY"]
    sys_txt = "\n".join(m["content"] for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") != "system"]
    body = {"model": model, "messages": turns, "max_tokens": max_tokens, "temperature": temperature}
    if sys_txt:
        body["system"] = sys_txt
    if tools:
        body["tools"] = tools
    headers = {"x-api-key": key, "anthropic-version": "2023-06-01"}
    raw = _post_json("https://api.anthropic.com/v1/messages", body, headers, timeout)
    return "".join(b.get("text", "") for b in raw.get("content", [])), raw


_HF_CACHE = {}


def _local_hf(messages, model, temperature, max_tokens, timeout, adapter=None, bits=4):
    """Serve a LOCAL HF causal LM, optionally with a LoRA adapter — the deployment path for whatever
    `llm-tool-train` produced.

    Two things this must get right or the pilot is unusable:
      * CACHE the model. The first version rebuilt a 9.6 GB model on every call, which is minutes per
        decision — fine for a one-shot test, useless for a loop that runs every few minutes;
      * load in 4-bit by default, matching how the adapter was TRAINED. Serving a QLoRA adapter on a bf16
        base is a silent quality loss, not an error.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch as _t
    except Exception as e:  # noqa: BLE001
        raise LLMBackendUnavailable(f"local-hf needs transformers+torch: {e}")

    key = (model, adapter, bits)
    if key not in _HF_CACHE:
        tok = AutoTokenizer.from_pretrained(model)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        kw = {"dtype": _t.bfloat16, "device_map": "auto"}
        if bits in (4, 8) and _t.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig
                kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=(bits == 4), load_in_8bit=(bits == 8),
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=_t.bfloat16)
            except Exception:  # noqa: BLE001 — bitsandbytes missing: bf16 still works
                pass
        try:
            mdl = AutoModelForCausalLM.from_pretrained(model, **kw)
        except TypeError:
            kw["torch_dtype"] = kw.pop("dtype")
            mdl = AutoModelForCausalLM.from_pretrained(model, **kw)
        if adapter:
            from peft import PeftModel
            mdl = PeftModel.from_pretrained(mdl, adapter)
            mdl.eval()
        _HF_CACHE[key] = (mdl, tok)
    mdl, tok = _HF_CACHE[key]

    try:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:  # noqa: BLE001 — a base model may ship no template
        prompt = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages) + "\nassistant:"
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to(mdl.device)
    with __import__("torch").no_grad():
        out = mdl.generate(**ids, max_new_tokens=max_tokens, do_sample=temperature > 0,
                           temperature=max(temperature, 1e-5),
                           pad_token_id=tok.pad_token_id or tok.eos_token_id)
    text = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
    return text, {"model": model, "adapter": adapter, "bits": bits, "device": str(mdl.device),
                  "cached": len(_HF_CACHE)}


def _dummy(messages, model, **_):
    """Zero-network deterministic backend for tests: echoes the last user message.

    One special case earns its keep: when the system prompt is the capability protocol (the two-tool
    interface adopted from openwork), a bare echo cannot drive the loop, so tool-using code paths would be
    untestable offline. Here it emits a well-formed `search_capabilities` call on the first turn and a
    `final` on the next — enough to exercise the whole loop with no network and no GPU.
    """
    sys_txt = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
    turns = [m for m in messages if m.get("role") != "system"]
    last = turns[-1]["content"] if turns else ""
    if "search_capabilities" in sys_txt:
        already = any("search_capabilities" in (m.get("content") or "")
                      for m in turns if m.get("role") == "assistant")
        if already:
            return (json.dumps({"tool": "final", "answer": "search returned matches (dummy backend)"}),
                    {"provider": "dummy", "model": model})
        first = turns[0]["content"] if turns else ""
        return (json.dumps({"tool": "search_capabilities", "query": first[:120]}),
                {"provider": "dummy", "model": model})
    return f"[echo:{model}] {last}", {"provider": "dummy", "model": model}


# ---------------------------------------------------------------- dispatch
def chat(messages, *, model="dummy/echo", provider=None, temperature=0.0, max_tokens=512,
         timeout=60, tools=None, adapter=None, bits=4):
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
        text, raw = _anthropic(messages, model, temperature, max_tokens, timeout, tools=tools)
    elif provider == "local-hf":
        text, raw = _local_hf(messages, model, temperature, max_tokens, timeout,
                              adapter=adapter, bits=bits)
    elif provider in compat:
        base, key = compat[provider]
        text, raw = _openai_compat(messages, model, base, key, temperature, max_tokens, timeout,
                                   tools=tools)
    else:
        raise LLMBackendUnavailable(
            f"provider {provider!r} not configured; available={available_providers()}")
    return {"text": text, "provider": provider, "model": model, "raw": raw,
            "tool_calls": tool_calls(raw, provider)}


def tool_calls(raw, provider=None):
    """Normalise a provider's tool-call payload to [{"id","name","arguments"}].

    Anthropic returns `content` blocks of type `tool_use`; OpenAI-compatible servers (including Ollama's
    /v1) return `choices[0].message.tool_calls` with a JSON *string* of arguments. Callers get one shape so
    the capability loop is written once, not three times.
    """
    out = []
    if not isinstance(raw, dict):
        return out
    for b in raw.get("content", []) or []:                # anthropic
        if isinstance(b, dict) and b.get("type") == "tool_use":
            out.append({"id": b.get("id", ""), "name": b.get("name", ""),
                        "arguments": b.get("input", {}) or {}})
    for ch in raw.get("choices", []) or []:               # openai-compatible / ollama
        for tc in ((ch.get("message") or {}).get("tool_calls") or []):
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except Exception:                         # noqa: BLE001 — a malformed arg string is data
                    args = {"_unparsed": args}
            out.append({"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args or {}})
    return out


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
