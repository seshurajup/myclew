"""kaggle_modality_test — DATA-WISE verifier for kaggle-modality (OFFLINE, no network).

Proves: (A) resolve_modality maps the Kaggle tag vocabulary (data-type > task > keyword) to our modalities,
and (B) build_map JOINS a tiny FIXTURE Meta Kaggle table (Competitions↔CompetitionTags↔Tags written to a
temp dir, download=False) → the right modality + source for each slug, including the KNOWN_COMPS fallback
for an untagged comp. No Kaggle CLI / network is touched.
"""
import os, sys, csv, json, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import kaggle_modality as KM
from fleet_agents import comp_config as CC

fails = []
def check(n, c):
    print(("PASS " if c else "FAIL ") + n)
    if not c: fails.append(n)


# ---------------------------------------------------------------- (A) resolve_modality tag vocabulary
def _r(tags, cat=None):
    return KM.resolve_modality("slug", tags=tags, category=cat)

check("data-type tag: tabular", _r(["tabular"]) == "tabular")
check("data-type tag: time series → sequence", _r(["time series data"]) == "sequence")
check("data-type tag: image", _r(["image"]) == "image")
check("data-type tag: audio", _r(["audio"]) == "audio")
check("data-type tag: text", _r(["text"]) == "text")
check("task tag: audio-event-classification → audio", _r(["multilabel classification", "audio event classification"]) == "audio")
check("task tag: translation → text", _r(["history", "translation"]) == "text")
check("task tag: reinforcement learning → agent-env", _r(["games", "reinforcement learning"]) == "agent-env")
check("task tag: general-knowledge-and-reasoning → text", _r(["deep learning", "general knowledge and reasoning"]) == "text")
check("keyword: reasoning tag → grid-reasoning", _r(["reasoning"]) == "grid-reasoning")
check("data-type beats task: text tag wins even with a task tag", _r(["translation", "text"]) == "text")
check("no signal → unknown", _r([], cat="Research") == "unknown")
check("audio is a real modality in the taxonomy", "audio" in CC.MODALITIES)

# --- the two NEW Kaggle data-type modalities (multimodal + graph) ---
check("multimodal is a real modality", "multimodal" in CC.MODALITIES)
check("graph is a real modality", "graph" in CC.MODALITIES)
check("data-type tag: multimodal → multimodal", _r(["multimodal"]) == "multimodal")
check("data-type tag: multimodal data → multimodal", _r(["multimodal data"]) == "multimodal")
check("data-type tag: graph → graph", _r(["graph"]) == "graph")
check("categorical is a tabular sub-type", _r(["categorical"]) == "tabular")
check("bigquery is a tabular sub-type", _r(["bigquery"]) == "tabular")
# a multimodal DATA-TYPE tag WINS even when image/text task tags are also present
check("multimodal data-type beats image+text task tags",
      _r(["multimodal", "image classification", "text classification"]) == "multimodal")
# the newly-mapped modality-implying TASK tags
check("task tag: object detection → image", _r(["object detection"]) == "image")
check("task tag: pose detection → image", _r(["pose detection"]) == "image")
check("task tag: token classification → text", _r(["token classification"]) == "text")
check("task tag: summarization → text", _r(["summarization"]) == "text")
check("task tag: speech-to-text → audio", _r(["speech-to-text"]) == "audio")
check("task tag: audio-to-audio → audio", _r(["audio-to-audio"]) == "audio")
check("task tag: video generation → video", _r(["video generation"]) == "video")
check("task tag: tabular classification → tabular", _r(["tabular classification"]) == "tabular")
check("task tag: evaluation → text", _r(["evaluation"]) == "text")
check("task tag: math → text", _r(["math"]) == "text")
# task-ONLY tags (no data modality) must NOT force a modality — they resolve via the data-type tag / unknown
check("task-only: binary classification → unknown (no modality)", _r(["binary classification"]) == "unknown")
check("task-only: regression → unknown (no modality)", _r(["regression"]) == "unknown")
check("task-only: clustering → unknown (no modality)", _r(["clustering"]) == "unknown")
check("task-only tag does NOT block a data-type tag: [regression, tabular] → tabular",
      _r(["regression", "tabular"]) == "tabular")

# --- coverage_report: the never-miss self-check over Kaggle's ENTIRE data-type + task vocabulary ---
cov = KM.coverage_report()          # reads the cached real Tags.csv (offline)
cc = cov["counts"]
check("coverage: every DATA-TYPE tag mapped or ignored (0 unmapped)", cc["data_type_unmapped"] == 0)
check("coverage: 0 unmapped modality-implying TASK tags", cc["task_unmapped"] == 0)
check("coverage: all 9 real data-type modalities mapped", cc["data_type_mapped"] == 9)
check("coverage: synthetic/root data-type ignored (provenance, not a modality)", cc["data_type_ignored"] >= 1)
check("coverage: modality-implying task tags mapped", cc["task_mapped"] >= 40)
check("coverage: task-only tags acknowledged as ''", cc["task_intentional_empty"] >= 10)
check("coverage report structure", set(cov["data_type"]) == {"mapped", "ignored", "unmapped"})


# ---------------------------------------------------------------- (B) build_map over a FIXTURE Meta Kaggle table
def _write_fixture(d):
    with open(os.path.join(d, "Competitions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Id", "Slug", "HostSegmentTitle"])
        w.writerow(["1", "fixture-birds", "Research"])
        w.writerow(["2", "fixture-translate", "Featured"])
        w.writerow(["3", "fixture-untagged-tracking", "Research"])   # no tags → KNOWN_COMPS / name-keyword fallback
    with open(os.path.join(d, "Tags.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Id", "ParentTagId", "Name", "Slug", "FullPath"])
        w.writerow(["10", "", "audio event classification", "aec", "task > audio-event-classification"])
        w.writerow(["11", "", "text", "text", "data type > text"])
        w.writerow(["12", "", "translation", "translation", "task > translation"])
    with open(os.path.join(d, "CompetitionTags.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["Id", "CompetitionId", "TagId"])
        w.writerow(["100", "1", "10"])                 # birds → audio event classification
        w.writerow(["101", "2", "11"]); w.writerow(["102", "2", "12"])  # translate → text + translation


with tempfile.TemporaryDirectory() as d:
    _write_fixture(d)
    m = KM.build_map(slugs=["fixture-birds", "fixture-translate", "fixture-untagged-tracking"],
                     meta_dir=d, download=False, cache=False)
    check("build_map: audio-event → audio (kaggle-tags)",
          m["fixture-birds"]["modality"] == "audio" and m["fixture-birds"]["source"] == "kaggle-tags")
    check("build_map: text data-type → text (kaggle-tags)",
          m["fixture-translate"]["modality"] == "text" and m["fixture-translate"]["source"] == "kaggle-tags")
    # untagged comp → name-keyword fallback (slug contains 'tracking' → volume-time)
    check("build_map: untagged tracking → name-keyword fallback",
          m["fixture-untagged-tracking"]["modality"] == "volume-time"
          and m["fixture-untagged-tracking"]["source"] in ("name-keyword", "known-comps"))
    check("build_map: kaggle_tags carried through", m["fixture-birds"]["kaggle_tags"] == ["audio event classification"])
    check("build_map: category carried through", m["fixture-translate"]["category"] == "Featured")

# a real KNOWN_COMPS slug with NO tags in the fixture → known-comps fallback
with tempfile.TemporaryDirectory() as d:
    _write_fixture(d)
    m2 = KM.build_map(slugs=["biohub-cell-tracking-during-development"], meta_dir=d, download=False, cache=False)
    v = m2["biohub-cell-tracking-during-development"]
    check("build_map: KNOWN_COMPS fallback (biohub → volume-time)", v["modality"] == "volume-time" and v["source"] in ("known-comps", "name-keyword"))


# ---------------------------------------------------------------- (C) agent contract (offline, cached report)
st, res, to, msg = KM.run({"spec": {}}, "test")
check("agent run() returns valid contract", st == "done" and isinstance(res, dict) and "map" in res)

print("=== kaggle_modality: " + ("PASS" if not fails else "FAIL " + ",".join(fails)) + " ===")
sys.exit(1 if fails else 0)
