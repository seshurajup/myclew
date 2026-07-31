"""coworker-backend — REUSABLE multi-provider LLM routing + risk/approval gating spec distilled from
andrewyng/openworker (aisuite-based multi-provider coworker with MCP connectors + risk approval).
PURE, data-wise tested.

openworker value we keep (the *policy*, not the service): (1) provider-agnostic model routing —
one `provider:model` string picks the backend, with capability fallback; (2) a risk classifier that
tags every proposed action WRITE_LOCAL / EXEC / EXTERNAL and decides auto-run vs human-approval,
exactly the approval gate this repo's browser/agent rules already require. We fold this into the
fleet's agent_routing / llm-backend concepts as a callable policy so any agent that dispatches an
LLM or a side-effecting action gets consistent backend selection + approval gating — instead of each
agent re-inventing it. [[fleet_external_repo_adoptions_2026_07]] [[agent_routing]] [[nooa_agents_2607_20709]]
"""
from __future__ import annotations

try:
    from .base import BaseAgent
except Exception:  # noqa: BLE001
    BaseAgent = object

# openworker aisuite providers (config.py) → default model. Latest/most-capable Claude first per policy.
PROVIDERS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o",
    "gemini": "gemini-2.5-pro",
    "groq": "llama-3.3-70b",
    "ollama": "llama3.2",
}

# openworker risk.py RiskClass. Higher tier ⇒ needs approval. Mirrors this repo's action-category rules.
RISK = {"READ": 0, "WRITE_LOCAL": 1, "EXEC": 2, "EXTERNAL": 3}
# EXTERNAL = anything that leaves the machine / is hard to reverse ⇒ ALWAYS human approval (send, publish,
# purchase, delete, post). EXEC/WRITE_LOCAL auto-run below the configured autonomy threshold.
_EXTERNAL_KW = ("send", "email", "post", "publish", "purchase", "buy", "transfer", "delete", "deploy", "tweet")
_EXEC_KW = ("run", "exec", "shell", "bash", "install", "pip", "subprocess", "compile")
_WRITE_KW = ("write", "save", "edit", "create", "mkdir", "chmod", "overwrite")


def route(model: str | None = None, provider: str | None = None) -> dict:
    """Resolve a `provider:model` request to a concrete backend, aisuite-style. Accepts 'anthropic',
    'anthropic:claude-opus-5', or a bare model; falls back to the anthropic default (policy: latest
    Claude) when unknown, so a missing/renamed backend never hard-fails a dispatch."""
    if model and ":" in model:
        provider, model = model.split(":", 1)
    if provider in PROVIDERS:
        return {"provider": provider, "model": model or PROVIDERS[provider], "resolved": True}
    for prov, dflt in PROVIDERS.items():             # bare model name → owning provider
        if model and model == dflt:
            return {"provider": prov, "model": model, "resolved": True}
    return {"provider": "anthropic", "model": PROVIDERS["anthropic"], "resolved": False}


def classify(action: str) -> str:
    """openworker risk.classify(): map an action description to its RiskClass. EXTERNAL wins over EXEC
    over WRITE_LOCAL over READ (most-severe keyword present decides)."""
    a = (action or "").lower()
    if any(k in a for k in _EXTERNAL_KW):
        return "EXTERNAL"
    if any(k in a for k in _EXEC_KW):
        return "EXEC"
    if any(k in a for k in _WRITE_KW):
        return "WRITE_LOCAL"
    return "READ"


def needs_approval(action: str, autonomy: int = 1) -> dict:
    """Gate an action. `autonomy` = highest RiskClass tier that may auto-run (0=READ only … 3=all).
    EXTERNAL is capped: it always needs approval regardless of autonomy (matches this repo's
    'explicit permission required' + 'prohibited' action rules)."""
    cls = classify(action)
    tier = RISK[cls]
    approve = tier > autonomy or cls == "EXTERNAL"
    return {"action": action, "risk": cls, "tier": tier, "autonomy": autonomy,
            "needs_approval": approve, "auto_run": not approve}


class CoworkerBackend(BaseAgent):
    name = "coworker-backend"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        rep: dict = {}
        if spec.get("model") or spec.get("provider"):
            rep["route"] = route(spec.get("model"), spec.get("provider"))
        actions = spec.get("actions") or ([spec["action"]] if spec.get("action") else [])
        if actions:
            rep["gate"] = [needs_approval(a, int(spec.get("autonomy", 1))) for a in actions]
            rep["approvals_required"] = [g["action"] for g in rep["gate"] if g["needs_approval"]]
        parts = []
        if "route" in rep:
            parts.append(f"→ {rep['route']['provider']}:{rep['route']['model']}"
                         + ("" if rep["route"]["resolved"] else " (fallback)"))
        if "gate" in rep:
            parts.append(f"{len(rep['approvals_required'])}/{len(rep['gate'])} need approval")
        msg = f"[{worker}] **coworker-backend** · " + " · ".join(parts or ["idle"])
        if hasattr(self, "done"):
            self.save_state(rep)
            self.post(worker, "leader", msg, routine=False, kind="verdict")
            return self.done(rep, msg, to="leader")
        return rep


_AGENT = CoworkerBackend()


def run(q, worker):
    return _AGENT.run(q, worker)
