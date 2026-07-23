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
