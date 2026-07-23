"""data-wise test for agent-routing — proves every registered agent is (a) domain-tagged and (b) multi-tagged
with the competition modalities it serves, so the leader can initiate exactly the relevant agents for a comp
type instead of all of them. Guards against a future pack being added without a domain→modality mapping."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fleet_agents as F
from fleet_agents import agent_routing as R
from fleet_agents import coverage_audit as CA

fails = []
def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c: fails.append(n)

H = list(F.HANDLERS)
tm = R.tag_map(H)

# 1) EVERY agent is domain-tagged and has >=1 modality (nothing unreachable by competition type)
check("every agent in tag_map", len(tm) == len(H))
check("every agent has a domain", all(tm[a]["domain"] and tm[a]["domain"] != "UNCLASSIFIED" for a in H))
no_mod = [a for a in H if not tm[a]["modalities"]]
check("every agent has >=1 modality (0 unreachable)", not no_mod)
if no_mod:
    print("   unreachable:", no_mod[:20])

# 2) tags are MULTI-valued and in the KAGGLE taxonomy (an agent can belong to tabular+image+volume-time+...)
check("MODALITIES is the Kaggle taxonomy", "volume-time" in R.MODALITIES and "agent-config" in R.MODALITIES and "text" in R.MODALITIES)
# 'audio' is a first-class Kaggle data-type (birdclef) — added to the taxonomy + trainable set; routes like image
check("audio is in the Kaggle taxonomy", "audio" in R.MODALITIES)
check("audio is trainable (training tricks apply)", "audio" in R._TRAINABLE)
check("route(audio) includes a training-trick (train-tricks)", "train-tricks" in set(R.route("audio", H)))

# --- the two NEW Kaggle data-type modalities (multimodal + graph) are first-class + trainable ---
check("multimodal is in the Kaggle taxonomy", "multimodal" in R.MODALITIES)
check("graph is in the Kaggle taxonomy", "graph" in R.MODALITIES)
check("multimodal is trainable", "multimodal" in R._TRAINABLE)
check("graph is trainable", "graph" in R._TRAINABLE)
check("multimodal has a pipeline skeleton", R.KAGGLE_TO_PIPELINE.get("multimodal") == "vision")
check("graph has a pipeline skeleton", R.KAGGLE_TO_PIPELINE.get("graph") == "detection_tracking")
for m in ("multimodal", "graph"):
    rt = set(R.route(m, H))
    check(f"route({m}) non-trivial (>30)", len(rt) > 30)
    check(f"route({m}) includes cross-cutting core (math-master)", "math-master" in rt)
    check(f"route({m}) includes a training-trick (train-tricks)", "train-tricks" in rt)
# multimodal = the UNION → a multimodal comp pulls image-family + text/LLM agents
mm = set(R.route("multimodal", H))
check("route(multimodal) pulls a vision inference trick (multi-tta)", "multi-tta" in mm)
check("route(multimodal) pulls an LLM agent (llm-infer)", "llm-infer" in mm)
# graph = GNN comps → the graph link agents are reachable
gr = set(R.route("graph", H))
check("route(graph) includes gnn-link-train", "gnn-link-train" in gr)
check("route(graph) includes gnn-probe", "gnn-probe" in gr)
# multi-tag spot checks via tag_map: an agent lists ALL modalities it serves
check("multi-tta is tagged multimodal (image-family fusion applies)", "multimodal" in tm["multi-tta"]["modalities"])
check("wbf-fusion is tagged multimodal", "multimodal" in tm["wbf-fusion"]["modalities"])
check("gnn-link-train is tagged graph", "graph" in tm["gnn-link-train"]["modalities"])
check("gnn-probe is tagged graph", "graph" in tm["gnn-probe"]["modalities"])
# a vision+text-capable pack agent must carry 'multimodal' alongside its base modalities
check("llm-infer (text) carries multimodal", "multimodal" in tm["llm-infer"]["modalities"])
check("tab-train carries graph (graph-tabular) but not image", "graph" in tm["tab-train"]["modalities"] and "image" not in tm["tab-train"]["modalities"])
multi = [a for a in H if len(tm[a]["modalities"]) > 1]
check("many agents are multi-modality", len(multi) > 50)
# a cross-cutting technique serves ALL Kaggle modalities
check("muon-optimizer serves all modalities", set(tm.get("muon-optimizer", {}).get("modalities", [])) == set(R.MODALITIES))
# a training trick spans trainable modalities incl tabular AND image AND volume-time (the user's example)
tt = set(tm.get("train-tricks", {}).get("modalities", []))
check("train-tricks spans tabular+image+volume-time", {"tabular", "image", "volume-time"} <= tt)
# a tabular agent is tabular-scoped (not image)
check("tab-train is tabular, not image", "tabular" in tm["tab-train"]["modalities"] and "image" not in tm["tab-train"]["modalities"])

# 3) EVERY pack coverage-audit produces has a domain→modality mapping (guard against a new untagged pack)
packs = set(CA.audit(H).keys())
unmapped = [p for p in packs if p not in R.DOMAIN_MODALITIES]
check("every pack has a modality mapping", not unmapped)
if unmapped:
    print("   packs missing from DOMAIN_MODALITIES:", unmapped)

# 4) route(kaggle_modality) is a FOCUSED but COMPLETE working set: subset of all, non-trivial, includes both a
#    cross-cutting core agent and the modality's own pack agents
for m in R.MODALITIES:
    r = R.route(m, H)
    rs = set(r)
    check(f"route({m}) has no dups", len(r) == len(rs))
    check(f"route({m}) ⊆ all agents", rs <= set(H))
    check(f"route({m}) is focused (< all)", len(r) < len(H))
    check(f"route({m}) is non-trivial (>30)", len(r) > 30)
    check(f"route({m}) includes cross-cutting core (math-master)", "math-master" in rs)
check("route(tabular) includes tab-train", "tab-train" in set(R.route("tabular", H)))
check("route(image) includes an inference trick (multi-tta)", "multi-tta" in set(R.route("image", H)))
check("route(volume-time) includes biohub tracker (mh-ilp)", "mh-ilp" in set(R.route("volume-time", H)))
check("route(text) includes llm-infer", "llm-infer" in set(R.route("text", H)))
check("route(tabular) EXCLUDES a pure-LLM agent (llm-infer)", "llm-infer" not in set(R.route("tabular", H)))

# 5) agents_for_competition groups by pack and only includes serving packs
comp = R.agents_for_competition("tabular", H)
check("agents_for_competition returns packs", isinstance(comp, dict) and "Tabular" in comp)
check("agents_for_competition tabular excludes Agentic pack", "Agentic" not in comp)

print("=== agent-routing: " + ("PASS" if not fails else "FAIL " + ",".join(fails)) + " ===")
sys.exit(1 if fails else 0)
