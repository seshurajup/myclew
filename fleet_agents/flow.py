"""flow — the composition glue so EVERY agent loops together cleanly.

Agents all share the (status, data, to, msg) contract, but they NAME their outputs differently
(det-sweep→{pick:{cv}}, recipe-adopt→{merged_cv}, scorer→{score}, saliency-detect→{coords}). For a
workflow to chain arbitrary agents, one agent's output must land on the next agent's INPUT key even
when the names differ. This module canonicalises outputs and re-emits them under EVERY input alias a
downstream agent might read, so pipeline/beat-bar/improve-loop/campaign can wire ANY agent to ANY agent.

  canon(data)              → {cv, config, nodes, recommend, predicted_lb, lever} pulled from any agent's output
  carry_spec(spec, data)   → next spec = spec + canonical values under ALL known input aliases (explicit spec wins)
  cv_of(data)              → the single CV any agent produced (or None)
"""
from __future__ import annotations

# canonical output → the output keys agents actually write it under (first match wins)
_OUT = {
    "cv": ["cv", "merged_cv", "combined_score", "score", "golden_cv", "canonical", "official_score", "best_cv"],
    "config": ["merged_config", "best_config", "config"],
    "recommend": ["recommend"],
    "predicted_lb": ["predicted_lb"],
    "lever": ["weakest", "lever", "bucket"],
    "nodes": ["coords", "nodes", "new_candidates"],
    "decoupled": ["decoupled"],
    "slope": ["slope"], "intercept": ["intercept"],
}
# canonical → the input-spec keys downstream agents READ it under (re-emit under all of these)
_IN = {
    "cv": ["candidate_cv", "start_cv", "cv"],
    "config": ["base", "config"],
    "nodes": ["existing_nodes", "nodes"],
    "lever": ["lever", "weakest"],
    "recommend": ["recommend"],
    "predicted_lb": ["predicted_lb"],
    "decoupled": ["decoupled"],
    "slope": ["slope"], "intercept": ["intercept"],
}


def canon(data) -> dict:
    """Pull canonical fields from ANY agent's output dict (handles nested {pick:{cv}})."""
    if not isinstance(data, dict):
        return {}
    out = {}
    for ck, keys in _OUT.items():
        for k in keys:
            v = data.get(k)
            if v is not None and not isinstance(v, (dict, list)):
                out[ck] = v
                break
    if "cv" not in out:                                       # det-sweep-style {pick:{cv}}
        pick = data.get("pick")
        if isinstance(pick, dict) and isinstance(pick.get("cv"), (int, float)):
            out["cv"] = pick["cv"]
    return out


def cv_of(data):
    return canon(data).get("cv")


def carry_spec(spec, data, extra_aliases=None) -> dict:
    """Next agent's spec = the given spec + prior canonical outputs re-emitted under EVERY input alias
    (so the next agent finds the value under the key IT reads). An explicit spec value always wins.

    extra_aliases: optional {canonical_key: [extra input-spec keys]} to widen re-emission for a new agent."""
    c = canon(data)
    merged = dict(spec or {}) if isinstance(spec, dict) else {}
    extra = extra_aliases if isinstance(extra_aliases, dict) else {}
    for ck, val in c.items():
        for alias in list(_IN.get(ck, [ck])) + list(extra.get(ck, [])):
            merged.setdefault(alias, val)
    return merged
