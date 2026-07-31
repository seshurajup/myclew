"""agent-routing — the single source of truth for AGENT ↔ DOMAIN grouping + per-competition routing, expressed
in the KAGGLE competition-modality taxonomy (the same one comp_config/comp-onboard produce for a real comp:
tabular / sequence / image / video / pointcloud / volume-time / text / agent-env / agent-config / grid-reasoning).
So when the board selects an ACTIVE KAGGLE PROJECT, its comp-onboard modality filters the agents directly — no
abstract "module" layer.

Every agent is MULTI-TAGGED with the Kaggle modalities it serves (a training-trick → tabular+image+volume-time+…),
derived from its DOMAIN pack (below). Routing by a competition's modality surfaces the COMPLETE working set;
no agent is left untagged/unreachable. Pure/registry-driven; no side effects.
"""
from __future__ import annotations

# the KAGGLE competition modalities (== comp_config.MODALITIES minus 'unknown'). This is the ONLY taxonomy used.
try:
    from .comp_config import MODALITIES as _CFG_MODS
    MODALITIES = [m for m in _CFG_MODS if m != "unknown"]
except Exception:  # noqa: BLE001
    MODALITIES = ["tabular", "sequence", "image", "audio", "video", "pointcloud", "volume-time", "text",
                  "multimodal", "graph", "agent-env", "agent-config", "grid-reasoning"]
_ALL = tuple(MODALITIES)
# modalities that TRAIN a model (everything a training-trick / quantizer applies to; excludes the pure
# agent-env/agent-config/grid-reasoning orchestration comps). audio ⊂ trainable (CNN/transformer training).
# multimodal (fusion nets) + graph (GNNs) both train models → trainable.
_TRAINABLE = ("tabular", "sequence", "image", "audio", "video", "pointcloud", "volume-time", "text",
              "multimodal", "graph")

# a Kaggle modality -> the pipeline_completeness.PIPELINE key that holds its ordered onboard→submit skeleton.
KAGGLE_TO_PIPELINE = {
    "tabular": "tabular", "sequence": "timeseries_forecast",
    "image": "vision", "audio": "vision", "video": "detection_tracking", "pointcloud": "vision", "volume-time": "detection_tracking",
    "text": "nlp_llm", "agent-env": "agentic", "agent-config": "agentic", "grid-reasoning": "reasoning_code",
    "multimodal": "vision",             # best onboard→submit skeleton for image+text fusion comps
    "graph": "detection_tracking",      # GNN/link-prediction agents (gnn-link-train/gnn-probe) live in this pack
}

# DOMAIN/pack -> the KAGGLE modalities it serves. "ALL" = cross-cutting (every competition type).
DOMAIN_MODALITIES = {
    # cross-cutting utilities / research / orchestration / meta — used in every competition
    "Generic CORE": "ALL", "Meta/Mining": "ALL", "Gap toolkit": "ALL", "Submission": "ALL",
    # adversarial metric-vulnerability probing applies to ANY comp that has a scorer — but is sharpest on the
    # graph/volume-time metrics (the biohub division/edge-jaccard degeneracies). Tag broadly.
    "Metric/Validation security": "ALL",
    # techniques that plug into ANY trainable model
    "2026 frontier": "ALL", "Arch/Config search": "ALL", "Compression/Quantization": "ALL", "Optimization": "ALL",
    "Training tricks": _TRAINABLE, "Training heads/regularizers": _TRAINABLE, "GM toolkit": _TRAINABLE,
    # WBF/TTA/snapshot — the image-family + tabular; multimodal comps carry image sub-inputs too.
    "Inference tricks": ("image", "audio", "video", "pointcloud", "volume-time", "tabular", "multimodal"),
    # modality-specific packs (Kaggle taxonomy). "multimodal" is added to EVERY pack that serves image/text/
    # audio/video (a multimodal comp pulls the union); "graph" to the GNN pack + tabular-adjacent graph FE.
    "Tabular": ("tabular", "sequence", "graph"),
    "Domain FE": ("tabular", "sequence"),
    "Forecast/Finance/Sports": ("sequence", "tabular"),
    "Audio": ("audio", "multimodal"),
    "Graph": ("graph", "multimodal"),
    # video-SPECIFIC frame-sampling + temporal-aggregation + motion agents (the cross-cutting Detection&Tracking
    # / Vision packs already serve video; this pack is the video↔multimodal fusion working set).
    "Video": ("video", "multimodal"),
    # multimodal-SPECIFIC feature-level fusion agents (the cross-cutting packs already carry the single
    # modalities image/text/tabular; this pack only serves the fusion/multimodal comp type).
    "Multimodal": ("multimodal",),
    "Detection & Tracking": ("volume-time", "video", "image", "audio", "multimodal", "graph"),
    "Vision/3D-seg": ("image", "audio", "video", "pointcloud", "volume-time", "multimodal"),
    "External-data transfer": ("image", "audio", "video", "volume-time", "tabular", "multimodal"),
    "LLM": ("text", "agent-config", "grid-reasoning", "multimodal"),
    "Prompt-program": ("text", "agent-config", "agent-env", "grid-reasoning", "multimodal"),
    "Reasoning/Code": ("grid-reasoning", "text", "multimodal"),
    # network-golf ONNX-emit/verify/cost + idiom catalogue + worker-context — the grid-reasoning working set
    "Grid-reasoning (ONNX-golf)": ("grid-reasoning",),
    "Agentic": ("agent-env", "agent-config"),
    "BIOHUB (3D+time)": ("volume-time",),
}


def _domain_modalities(pack):
    """The KAGGLE modalities a DOMAIN/pack serves (list). Unknown pack -> [] (still domain-tagged, just unrouted)."""
    v = DOMAIN_MODALITIES.get(pack)
    if v == "ALL":
        return list(_ALL)
    return list(v or [])


def domains(handlers=None):
    """{domain/pack -> [agents]} — the grouping the dashboard shows. Uses coverage-audit's live classification."""
    from . import HANDLERS
    from . import coverage_audit as CA
    return CA.audit(list(handlers if handlers is not None else HANDLERS))


def domain_of(agent, handlers=None):
    """Which domain/pack an agent belongs to (or 'UNCLASSIFIED')."""
    for pack, members in domains(handlers).items():
        if agent in members:
            return pack
    return "UNCLASSIFIED"


def agents_for_modality(kaggle_modality, handlers=None):
    """The pipeline-STAGED skeleton for a KAGGLE modality, grouped by stage (onboard→submit). Translates the
    Kaggle modality to its pipeline key. `route()` returns the FULL working set (skeleton + domain-tagged)."""
    from . import HANDLERS
    from . import pipeline_completeness as PC
    H = set(handlers if handlers is not None else HANDLERS)
    pkey = KAGGLE_TO_PIPELINE.get(kaggle_modality)
    stages = PC.PIPELINE.get(pkey, {}) if pkey else {}
    return {stage: [a for a in stages.get(stage, []) if a in H] for stage in PC.STAGES}


def route(kaggle_modality, handlers=None):
    """Flat, de-duplicated COMPLETE working set for a KAGGLE modality: the pipeline skeleton PLUS every agent whose
    DOMAIN serves this modality (cross-cutting core/techniques + the modality's own packs). This is what lets the
    board initiate exactly the relevant agents for the SELECTED COMPETITION instead of all of them."""
    from . import HANDLERS
    H = list(handlers if handlers is not None else HANDLERS)
    seen, out = set(), []
    for stage, agents in agents_for_modality(kaggle_modality, H).items():   # ordered skeleton first
        for a in agents:
            if a not in seen:
                seen.add(a); out.append(a)
    for pack, members in domains(H).items():                                # then every domain-tagged agent
        if kaggle_modality in _domain_modalities(pack):
            for a in members:
                if a not in seen:
                    seen.add(a); out.append(a)
    return out


def agents_for_competition(kaggle_modality, handlers=None):
    """The complete working set for a KAGGLE modality, grouped by DOMAIN/pack (what the dashboard shows when a
    competition is picked). {pack -> [agents that serve this modality]}."""
    from . import HANDLERS
    from . import pipeline_completeness as PC
    H = list(handlers if handlers is not None else HANDLERS)
    pkey = KAGGLE_TO_PIPELINE.get(kaggle_modality)
    staged = {a for agents in PC.PIPELINE.get(pkey, {}).values() for a in agents} if pkey else set()
    out = {}
    for pack, members in domains(H).items():
        serves = kaggle_modality in _domain_modalities(pack)
        picked = [a for a in members if serves or a in staged]
        if picked:
            out[pack] = picked
    return out


# ----------------------------- intent routing (CoT-rewrite + calibration) -----------------------------
# Adopted from chrishayuk/virtual-experts: routing a FREE-TEXT request straight to an agent is brittle
# ("fuse my boxes" vs "combine detections from several models" scatter). The fix is two-stage — first
# NORMALIZE the request into a canonical action, then route on that stable form. Here the canonical form
# is the agent roster's own capability DESCRIPTION (the calibration target), so varied phrasings that
# mean the same thing land on the same agent. Deterministic (token overlap), no LLM required; an LLM
# CoT-rewrite can front this by emitting the normalized {intent, modality} it scores against.
_INTENT_STOP = set("""the a an of to and or for with in on at by from as is are be this that these those it
its into via using use used based our we you your can will not but if then than model models data train
run make build get set need want how do does i me my agent""".split())
_WORD = __import__("re").compile(r"[a-z][a-z0-9+#-]{2,}")


def _roster():
    """[(name, description)] from the single (thread, name, description, spec) roster in __init__.SEED —
    the calibration corpus for intent routing."""
    try:
        from . import SEED
    except Exception:  # noqa: BLE001
        return []
    return [(row[1], row[2]) for row in SEED if len(row) >= 3]


def _terms(text):
    """Query/description terms, WITH hyphen-and-underscore parts.

    Measured defect: `_WORD` keeps hyphens, so a query for an agent's exact name stayed a single token
    (`tracker-postproc`) while `normalize_request` indexes names SPLIT (`{tracker, postproc}`). The two
    could never intersect, so searching an agent by its own name returned an EMPTY list for every
    hyphenated agent — 92.5% of the roster. Emitting the parts alongside the whole token fixes retrieval
    for every caller (the local pilot, Claude, and `search_capabilities` alike).
    """
    raw = {w for w in _WORD.findall((text or "").lower()) if w not in _INTENT_STOP}
    out = set(raw)
    for w in raw:
        if "-" in w or "_" in w:
            out |= {p for p in w.replace("_", "-").split("-") if p and p not in _INTENT_STOP}
    return out


_IDF_CACHE = {}


def _idf():
    """Inverse document frequency of each term over the roster descriptions, so discriminative words
    (fusion, quantize, boxes) outweigh common ones (model, detection, training) when matching."""
    if _IDF_CACHE:
        return _IDF_CACHE
    roster = _roster()
    if not roster:
        return {}
    import math
    n = len(roster)
    df = {}
    for _name, desc in roster:
        for t in _terms(desc):
            df[t] = df.get(t, 0) + 1
    for t, c in df.items():
        _IDF_CACHE[t] = math.log((n + 1) / (c + 1)) + 1.0
    return _IDF_CACHE


def normalize_request(text, modality=None, top=5, handlers=None):
    """CoT-rewrite step: map a free-text request to a ranked canonical action list
    [{'agent', 'score', 'matched':[terms]}], scored by IDF-weighted term overlap against each agent's
    capability description (optionally restricted to agents that serve `modality`). The top entry is the
    normalized action to route to; ties/low scores signal 'ask the leader' rather than mis-route."""
    q = _terms(text)
    if not q:
        return []
    idf = _idf()
    allow = None
    if modality is not None:
        allow = set(route(modality, handlers))
    ranked = []
    for name, desc in _roster():
        if allow is not None and name not in allow:
            continue
        dt = _terms(desc); nt = _terms(name.replace("-", " "))
        matched = (q & dt) | (q & nt)
        if not matched:
            continue
        # IDF-weighted overlap; a direct NAME-token match is the strongest calibration signal (×2)
        score = sum(idf.get(t, 1.0) for t in (q & dt)) + 2.0 * sum(idf.get(t, 1.0) for t in (q & nt))
        ranked.append({"agent": name, "score": round(score, 2), "matched": sorted(matched)})
    ranked.sort(key=lambda r: -r["score"])
    return ranked[:top]


def route_request(text, modality=None, handlers=None, margin=1.5):
    """Route a free-text request to a single agent via the normalized action list. Returns
    {'agent', 'confident', 'candidates'}: confident=False when the top two are within `margin`
    (ambiguous → the caller should escalate to the leader rather than guess), per virtual-experts'
    calibration-gap principle."""
    cands = normalize_request(text, modality=modality, handlers=handlers)
    if not cands:
        return {"agent": None, "confident": False, "candidates": []}
    confident = len(cands) == 1 or (cands[0]["score"] - cands[1]["score"]) >= margin
    return {"agent": cands[0]["agent"], "confident": confident, "candidates": cands}


def tag_map(handlers=None):
    """{agent -> {'domain': pack, 'modalities': [KAGGLE comp-types it serves]}} — the per-agent MULTI-TAG the
    dashboard and the leader use to filter/group. modalities come from the agent's DOMAIN pack, so EVERY agent is
    tagged with the full set of Kaggle competition types it serves (a training-trick → tabular+image+volume-time+…)."""
    from . import HANDLERS
    H = list(handlers if handlers is not None else HANDLERS)
    dmap = {}
    for pack, members in domains(H).items():
        dmods = _domain_modalities(pack)
        for a in members:
            dmap[a] = {"domain": pack, "modalities": sorted(dmods)}
    return dmap


# ---------------------------------------------------------------- capability layer
# Adopted from different-ai/openwork (https://github.com/different-ai/openwork, v0.18.11 @ ed16748,
# clone: research/openwork_repo, contract: ee/apps/den-api/src/mcp/agent.ts).
#
# The problem it solves for us, exactly: we have 321 agents in HANDLERS. Handing an LLM 321 tool
# definitions is not a design — it is a context-window bill, and the model still picks wrong. OpenWork's
# answer is to expose **exactly two** tools and put the registry behind them: `search_capabilities` to
# discover, `execute_capability` to run. Four of its details are load-bearing and reproduced here:
#
#   1. two tools, with honest annotations — search is read-only/idempotent, execute is destructive and not;
#   2. `schema_digest` — search returns a digest of the argument schema and execute must echo it back, which
#      is how you catch a model that invented a spec or is working from a stale search result;
#   3. a STRUCTURED retry protocol — the error says what to do next (`correct_arguments` vs
#      `search_capabilities`) and sets `same_arguments_retryable=False`, so a model cannot loop on identical
#      arguments (openwork: "never retry the same arguments unchanged");
#   4. an instruction block telling the model to try 2-4 keyword variants before concluding a capability
#      does not exist — cheap, and it removes most false "you don't support that" answers.
#
# We do NOT adopt its transport (a TS MCP server over HTTP/OAuth). These are plain Python functions, so the
# same capability pair drives Claude, Ollama, any OpenAI-compatible endpoint and a local HF model through
# `llm_backend` — see `llm_tool_schemas()` for the per-provider tool shapes.

import hashlib as _hashlib
import json as _json

CAPABILITY_INSTRUCTIONS = "\n".join([
    "This fleet intentionally exposes exactly two tools: search_capabilities and execute_capability.",
    "There are hundreds of agents behind them; you discover an agent instead of being handed all of them.",
    "Always call search_capabilities first, with 2-4 keyword variants, before concluding that something is "
    "unavailable.",
    "Use execute_capability only with an exact `name` returned by search_capabilities.",
    "Copy the match's `schema_digest` into execute_capability. It proves your arguments came from a real "
    "search rather than from memory.",
    "`spec` must be an object shaped like the match's `spec_schema`. Extra keys are allowed; wrong types "
    "are not.",
    "If execute_capability returns unknown_capability, call search_capabilities again before retrying.",
    "If it returns invalid_capability_arguments, correct the listed issues and retry ONCE with CHANGED "
    "arguments. Never retry identical arguments — they cannot succeed.",
    "A successful search_capabilities call proves the fleet is reachable. Never report the fleet as down "
    "because one agent failed.",
])

SEARCH_ANNOTATIONS = {"read_only": True, "destructive": False, "idempotent": True, "open_world": True}
EXECUTE_ANNOTATIONS = {"read_only": False, "destructive": True, "idempotent": False, "open_world": True}
EXECUTE_TIMEOUT_S = 180


def _spec_schema(example):
    """An example spec dict from the SEED roster → a minimal JSON-schema-ish description of it.

    The roster already carries a working spec per agent, which is a better schema source than a hand-written
    one: it cannot drift from what the agent actually accepts.
    """
    def kind(v):
        if isinstance(v, bool):
            return "boolean"
        if isinstance(v, int):
            return "integer"
        if isinstance(v, float):
            return "number"
        if isinstance(v, (list, tuple)):
            return "array"
        if isinstance(v, dict):
            return "object"
        return "string"
    return {k: kind(v) for k, v in (example or {}).items()}


def schema_digest(schema):
    """A short stable digest of a spec schema (openwork's `schemaDigest`).

    Echoed back by `execute_capability`, so a stale or invented argument set is caught before the agent
    runs rather than after it has done something.
    """
    return _hashlib.sha256(_json.dumps(schema or {}, sort_keys=True).encode()).hexdigest()[:12]


_CAP_CACHE = {}


def capability_index(handlers=None):
    """{name -> capability descriptor} over the whole fleet: summary, example spec, spec schema, digest,
    domain and the Kaggle modalities it serves. One row per registered agent."""
    key = "default" if handlers is None else id(handlers)
    if key in _CAP_CACHE:
        return _CAP_CACHE[key]
    from . import HANDLERS
    H = set(handlers if handlers is not None else HANDLERS)
    tags = {}
    try:
        tags = tag_map(handlers)
    except Exception:  # noqa: BLE001 — tagging is metadata; a capability must still be discoverable
        tags = {}
    index = {}
    try:
        from . import SEED
    except Exception:  # noqa: BLE001
        SEED = []
    examples, summaries = {}, {}
    for row in SEED:
        if len(row) < 3:
            continue
        nm, desc = row[1], row[2]
        summaries.setdefault(nm, desc)
        if len(row) >= 4 and isinstance(row[3], dict) and row[3] and nm not in examples:
            examples[nm] = row[3]
    # 28 registered agents have no SEED row, so they had an EMPTY summary and were unroutable by ANY
    # model — and untrainable for weights-only recall. Their MODULES document themselves perfectly well,
    # so fall back to the module docstring's first line. Automatic, and it covers every future agent too.
    def _doc_summary(kind):
        import inspect
        import sys as _s
        raw = globals().get("_RAW_FOR_DOC")
        fn = (raw or {}).get(kind)
        if fn is None:
            return ""
        mod = _s.modules.get(getattr(fn, "__module__", ""), None)
        doc = (inspect.getdoc(mod) or "").strip()
        if not doc:
            return ""
        first = doc.split("\n\n")[0].replace("\n", " ")
        first = first.split(" — ", 1)[-1] if " — " in first[:60] else first
        return " ".join(first.split())[:220]

    try:
        from . import _RAW_HANDLERS as _raw_h
        globals()["_RAW_FOR_DOC"] = _raw_h
    except Exception:  # noqa: BLE001 — the fallback is a nicety; an import hiccup must not break routing
        globals()["_RAW_FOR_DOC"] = {}

    for nm in sorted(H):
        sch = _spec_schema(examples.get(nm))
        summ = summaries.get(nm, "") or _doc_summary(nm)
        index[nm] = {"name": nm, "summary": summ, "spec_example": examples.get(nm, {}),
                     "spec_schema": sch, "schema_digest": schema_digest(sch),
                     "domain": tags.get(nm, {}).get("domain", ""),
                     "modalities": tags.get(nm, {}).get("modalities", [])}
    _CAP_CACHE[key] = index
    return index


def search_capabilities(query, limit=8, modality=None, handlers=None):
    """Tool 1 of 2 — discover agents by free text. Read-only, idempotent.

    Scoring reuses `normalize_request` (the IDF-weighted matcher already calibrated on the roster), then
    each match is dressed in the fields a model needs to call it: the exact `name`, its `spec_schema` and
    the `schema_digest` it must echo back. A miss returns a `hint` telling the model to try other keywords
    rather than an empty list it will interpret as "unsupported".
    """
    index = capability_index(handlers)
    ranked = normalize_request(query, modality=modality, top=max(int(limit), 1) * 3, handlers=handlers)
    matches = []
    for r in ranked:
        cap = index.get(r["agent"])
        if not cap:
            continue
        matches.append({"name": cap["name"], "score": r["score"], "summary": cap["summary"],
                        "matched": r["matched"], "spec_schema": cap["spec_schema"],
                        "spec_example": cap["spec_example"], "schema_digest": cap["schema_digest"],
                        "domain": cap["domain"], "modalities": cap["modalities"],
                        "has_spec": bool(cap["spec_schema"])})
        if len(matches) >= int(limit):
            break
    out = {"matches": matches, "total_capabilities": len(index)}
    if not matches:
        out["hint"] = ("No capability matched those words. Try 2-4 different keyword variants (a task noun, "
                       "a metric, a model family) before concluding the fleet cannot do this.")
    return out


def execute_capability(name, spec=None, schema_digest_echo=None, handlers=None, worker="cli",
                       dispatch=None, require_digest=False):
    """Tool 2 of 2 — run one discovered agent. NOT read-only, NOT idempotent.

    Returns either `{"ok": True, "status", "data", "message"}` or a structured error carrying the retry
    protocol, which is the part worth having: `retry.action` tells the model whether to fix its arguments
    or search again, and `same_arguments_retryable=False` forbids the identical-retry loop that otherwise
    burns a whole context window.
    """
    index = capability_index(handlers)
    if name not in index:
        near = [m["name"] for m in search_capabilities(name, limit=5, handlers=handlers)["matches"]]
        if not near:
            # a model that guessed a plausible-looking NAME gets name-similarity suggestions; the word
            # matcher above only helps when it guessed a plausible-looking DESCRIPTION
            import difflib
            near = difflib.get_close_matches(str(name), list(index), n=5, cutoff=0.4)
        return {"ok": False, "error": "unknown_capability",
                "message": f'No capability named "{name}". Call search_capabilities for a valid name.',
                "did_you_mean": near, "same_arguments_retryable": False,
                "retry": {"action": "search_capabilities", "search_required": True}}
    cap = index[name]
    if (require_digest or schema_digest_echo) and schema_digest_echo != cap["schema_digest"]:
        return {"ok": False, "error": "stale_schema_digest",
                "message": ("The schema_digest does not match this capability's current schema, so these "
                            "arguments were not built from a current search."),
                "expected_digest": cap["schema_digest"], "received_digest": schema_digest_echo,
                "same_arguments_retryable": False,
                "retry": {"action": "search_capabilities", "search_required": True}}
    spec = {} if spec is None else spec
    if not isinstance(spec, dict):
        return {"ok": False, "error": "invalid_capability_arguments",
                "message": "`spec` must be an object.", "issues": [{"path": "spec", "keyword": "type",
                                                                    "message": "expected object"}],
                "same_arguments_retryable": False,
                "retry": {"action": "correct_arguments", "search_required": False}}
    issues = []
    want = cap["spec_schema"]
    checkers = {"integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
                "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
                "boolean": lambda v: isinstance(v, bool),
                "array": lambda v: isinstance(v, (list, tuple)),
                "object": lambda v: isinstance(v, dict),
                "string": lambda v: isinstance(v, str)}
    for k, t in want.items():
        if k in spec and not checkers.get(t, lambda _v: True)(spec[k]):
            issues.append({"path": f"spec.{k}", "keyword": "type", "message": f"expected {t}"})
    if issues:
        return {"ok": False, "error": "invalid_capability_arguments",
                "message": "Correct the listed issues and retry once with CHANGED arguments.",
                "issues": issues, "spec_schema": want, "same_arguments_retryable": False,
                "retry": {"action": "correct_arguments", "search_required": False}}
    if dispatch is None:
        from . import HANDLERS
        dispatch = HANDLERS.get(name) if hasattr(HANDLERS, "get") else None
    if dispatch is None:
        return {"ok": False, "error": "capability_not_dispatchable",
                "message": f'"{name}" is registered but has no handler in this process.',
                "same_arguments_retryable": False,
                "retry": {"action": "search_capabilities", "search_required": True}}
    try:
        res = dispatch({"kind": name, "spec": spec}, worker)
    except Exception as e:  # noqa: BLE001 — an agent crash is a RESULT, not a fleet outage
        return {"ok": False, "error": "capability_failed", "capability": name,
                "message": f"{type(e).__name__}: {str(e)[:300]}",
                "same_arguments_retryable": False,
                "retry": {"action": "correct_arguments", "search_required": False}}
    status, data, to, msg = (list(res) + [None] * 4)[:4] if isinstance(res, (list, tuple)) else (
        "done", res, None, str(res)[:400])
    return {"ok": True, "capability": name, "status": status, "data": data, "message": msg}


def llm_tool_schemas(flavor="anthropic"):
    """The same two tools, in the shape each provider wants — so one capability layer serves them all.

    `flavor`: "anthropic" (Claude Messages `tools`), "openai" (chat-completions `tools`, which is also what
    Ollama's /v1 and every OpenAI-compatible server accepts), or "plain" (a JSON description to paste into
    a prompt for a model with no native tool-calling, e.g. a small local HF model).
    """
    search_props = {"query": {"type": "string", "description": "keywords describing the task"},
                    "limit": {"type": "integer", "description": "max matches (default 8)"},
                    "modality": {"type": "string",
                                 "description": "optional Kaggle competition type to restrict to"}}
    exec_props = {"name": {"type": "string", "description": "exact name from search_capabilities"},
                  "spec": {"type": "object", "description": "arguments matching the match's spec_schema"},
                  "schema_digest": {"type": "string",
                                    "description": "copy the match's schema_digest here"}}
    tools = [("search_capabilities",
              "Discover fleet agents by free text. Call this FIRST, with 2-4 keyword variants.",
              search_props, ["query"], SEARCH_ANNOTATIONS),
             ("execute_capability",
              "Run one agent discovered by search_capabilities. Not idempotent.",
              exec_props, ["name"], EXECUTE_ANNOTATIONS)]
    if flavor == "anthropic":
        return [{"name": n, "description": d,
                 "input_schema": {"type": "object", "properties": p, "required": r}}
                for n, d, p, r, _a in tools]
    if flavor == "openai":
        return [{"type": "function",
                 "function": {"name": n, "description": d,
                              "parameters": {"type": "object", "properties": p, "required": r}}}
                for n, d, p, r, _a in tools]
    return [{"name": n, "description": d, "properties": p, "required": r, "annotations": a}
            for n, d, p, r, a in tools]


def _first_json_object(text):
    """The first balanced {...} in a model's prose → dict, or None.

    Needed because plenty of useful local models have no native tool-calling (Ollama rejects tools for
    `gemma3n:e4b` outright: "does not support tools"). Rather than exclude them, we let them emit a JSON
    object and parse it — the same two capabilities, over text instead of a tool API.
    """
    if not text:
        return None
    start = None
    depth = 0
    in_str = esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return _json.loads(text[start:i + 1])
                except Exception:  # noqa: BLE001 — keep scanning for a later, well-formed object
                    start = None
    return None


TEXT_PROTOCOL = (
    "You have exactly two capabilities. Reply with ONE JSON object and nothing else.\n"
    'To discover:  {"tool": "search_capabilities", "query": "<keywords>"}\n'
    'To run:       {"tool": "execute_capability", "name": "<exact name>", "spec": {...}, '
    '"schema_digest": "<from the search match>"}\n'
    'When you are done, reply {"tool": "final", "answer": "<what you found or ran>"}'
)


# ---------------------------------------------------------------- compact wire format (measured)
# A small local model pays for every brace. Measured on 265 real targets with this tokenizer:
#     JSON     48.4 tokens/target, 28 fragile syntax characters ({ } [ ] " : ,)
#     COMPACT  27.1 tokens/target,  4 fragile characters (| = ,)      -> 44% fewer tokens
# Fewer tokens is cheaper AND fewer places to go wrong: unbalanced braces, trailing commas and markdown
# fences are the classic small-model JSON failures, and none of them exist in a pipe-delimited line.
# The format is also trivially grammar-constrainable at decode time, which JSON is not.
#
#     SEARCH|<query words>
#     EXEC|<agent-name>|<schema_digest>|<k=v,k=v,...>
#     FINAL|<answer>
COMPACT_SPEC = (
    "Reply with EXACTLY ONE line, no prose and no code fences:\n"
    "  SEARCH|<keywords>                       to discover capabilities\n"
    "  EXEC|<name>|<schema_digest>|<k=v,k=v>   to run one, using an exact name from a search\n"
    "  FINAL|<answer>                          when done\n"
    "Use | as the separator. Omit the trailing section if there are no arguments."
)


def to_compact(gold):
    """{'tool':…} -> the one-line wire form."""
    t = gold.get("tool")
    if t == "search_capabilities":
        return "SEARCH|" + str(gold.get("query", "")).strip()
    if t == "execute_capability":
        kv = ",".join(f"{k}={v}" for k, v in (gold.get("spec") or {}).items())
        return f"EXEC|{gold.get('name','')}|{gold.get('schema_digest','')}|{kv}"
    return "FINAL|" + str(gold.get("answer", ""))


def from_compact(text):
    """The one-line wire form -> {'tool':…}, or None. Tolerant of fences, prose and stray whitespace,
    because a model that is 95% obedient still needs its 5% parsed."""
    if not text:
        return None
    for raw in str(text).replace("```", "\n").splitlines():
        ln = raw.strip().strip("`").strip()
        if not ln:
            continue
        # Accept `:` as well as `|` after the verb. Observed in the wild: a fine-tuned model emitted
        # `SEARCH: fuse motion relink gap close` — right intent, wrong delimiter — and the parser returned
        # None, turning a near-miss into a total failure. Normalise the verb separator before splitting.
        for _v in ("SEARCH", "EXEC", "FINAL"):
            if ln.upper().startswith(_v + ":"):
                ln = _v + "|" + ln[len(_v) + 1:].lstrip()
                break
        up = ln.upper()
        if up.startswith("SEARCH|"):
            return {"tool": "search_capabilities", "query": ln.split("|", 1)[1].strip()}
        if up.startswith("FINAL|"):
            return {"tool": "final", "answer": ln.split("|", 1)[1].strip()}
        if up.startswith("EXEC|"):
            parts = ln.split("|")
            name = parts[1].strip() if len(parts) > 1 else ""
            digest = parts[2].strip() if len(parts) > 2 else ""
            spec = {}
            if len(parts) > 3 and parts[3].strip():
                # split on commas that are NOT inside brackets. Measured bug this fixes: `cap=[12, 0]`
                # naively split into `cap=[12` and `0]`, so an array value silently became the string
                # "[12" and the real validator rejected it. Bracket depth keeps list values intact.
                seg, depth, buf = [], 0, ""
                for ch in parts[3]:
                    if ch in "[{(":
                        depth += 1
                    elif ch in "]})":
                        depth = max(0, depth - 1)
                    if ch == "," and depth == 0:
                        seg.append(buf); buf = ""
                    else:
                        buf += ch
                if buf:
                    seg.append(buf)
                for pair in seg:
                    if "=" not in pair:
                        continue
                    k, v = pair.split("=", 1)
                    k, v = k.strip(), v.strip()
                    if v.startswith("[") or v.startswith("{"):
                        try:
                            spec[k] = _json.loads(v.replace("'", '"'))
                        except ValueError:
                            spec[k] = v
                    elif v.lower() in ("true", "false"):       # restore the types the schema expects
                        spec[k] = v.lower() == "true"
                    elif v.lstrip("-").isdigit():
                        spec[k] = int(v)
                    else:
                        try:
                            spec[k] = float(v)
                        except ValueError:
                            spec[k] = v
            return {"tool": "execute_capability", "name": name, "spec": spec,
                    "schema_digest": digest}
    return None


def capability_loop(request, model="dummy/echo", max_steps=6, native_tools=None, handlers=None,
                    execute=False, chat_fn=None, temperature=0.0, max_tokens=700, worker="cli",
                    adapter=None, bits=4):
    """Drive the fleet's two capabilities from ANY LLM — the openwork pattern, provider-agnostic.

    `native_tools=None` auto-detects: try the provider's tool API and fall back to `TEXT_PROTOCOL` when it
    refuses (Ollama 400s for models without tool support). That fallback is the whole reason a small local
    model can still drive 320 agents.

    `execute=False` by default. Discovery is read-only, but `execute_capability` is annotated destructive,
    so actually running an agent because a language model asked is opt-in — the model's plan is returned for
    inspection instead. This is a deliberate difference from openwork, which is a user-driven desktop app;
    ours is unattended.

    Returns {"steps": [...], "answer", "used_native_tools", "provider", "model"}.
    """
    if chat_fn is None:
        from . import llm_backend as _lb
        chat_fn = _lb.chat
    tools = llm_tool_schemas("openai") if native_tools is not False else None
    sys_msg = CAPABILITY_INSTRUCTIONS
    steps, used_native = [], bool(tools)
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": request}]
    provider = None

    def _ask(with_tools):
        kw = {"model": model, "temperature": temperature, "max_tokens": max_tokens}
        if adapter:                        # serve the fine-tuned adapter, not just the base model
            kw["adapter"] = adapter; kw["bits"] = bits
        if with_tools:
            kw["tools"] = tools
        return chat_fn(msgs, **kw)

    for _ in range(int(max_steps)):
        try:
            r = _ask(used_native)
        except Exception as e:  # noqa: BLE001
            if used_native and ("does not support tools" in str(e) or "tools" in str(e).lower()):
                used_native = False                       # the model has no tool API — switch to text
                msgs[0] = {"role": "system", "content": sys_msg + "\n\n" + TEXT_PROTOCOL}
                continue
            steps.append({"error": str(e)[:300]})
            break
        provider = r.get("provider", provider)
        calls = r.get("tool_calls") or []
        if not calls:
            # accept EITHER wire format: compact first (it is what we fine-tune on), then JSON
            obj = from_compact(r.get("text", "")) or _first_json_object(r.get("text", ""))
            if obj and obj.get("tool"):
                calls = [{"name": obj["tool"], "arguments": {k: v for k, v in obj.items() if k != "tool"}}]
            elif used_native:
                # tools were accepted but unused: the model answered in prose. Re-ask over text once.
                used_native = False
                msgs[0] = {"role": "system", "content": sys_msg + "\n\n" + TEXT_PROTOCOL}
                continue
            else:
                steps.append({"tool": "final", "answer": (r.get("text") or "")[:800]})
                break
        call = calls[0]
        name, args = call.get("name"), call.get("arguments") or {}
        if name == "final":
            steps.append({"tool": "final", "answer": str(args.get("answer", ""))[:800]})
            break
        if name == "search_capabilities":
            res = search_capabilities(args.get("query") or request, limit=int(args.get("limit") or 6),
                                      modality=args.get("modality"), handlers=handlers)
            steps.append({"tool": name, "query": args.get("query"),
                          "matches": [m["name"] for m in res["matches"]], "result": res})
        elif name == "execute_capability":
            if not execute:
                steps.append({"tool": name, "planned": True, "name": args.get("name"),
                              "spec": args.get("spec"), "note": "execute=False — plan returned, not run"})
                break
            res = execute_capability(args.get("name"), args.get("spec"),
                                     schema_digest_echo=args.get("schema_digest"),
                                     handlers=handlers, worker=worker)
            steps.append({"tool": name, "name": args.get("name"), "result": res})
            if res.get("ok"):
                break
        else:
            steps.append({"tool": name, "error": "unknown tool"})
            break
        msgs.append({"role": "assistant", "content": _json.dumps({"tool": name, **args})})
        msgs.append({"role": "user",
                     "content": "Tool result:\n" + _json.dumps(steps[-1].get("result", steps[-1]),
                                                               default=str)[:2500]})
    answer = next((s.get("answer") for s in reversed(steps) if s.get("tool") == "final"), None)
    return {"steps": steps, "answer": answer, "used_native_tools": used_native,
            "provider": provider, "model": model}
