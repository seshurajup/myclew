"""research_search_test — DATA-WISE verifier of the ranking logic (_rank) + feasibility hint, no network.
Asserts: a one-pass 3D StarDist model ranks above a Cellpose 2d-stitch model (slow penalty); tag-overlap
with the prefer set boosts rank; feasibility hint labels fast vs slow families correctly.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import research_search as R


def _run():
    print("=== RESEARCH-SEARCH RANK-LOGIC VERIFIER ===")
    cands = [
        {"name": "someone/cellpose-2d-stitch-nuclei", "tags": ["cellpose", "2d"], "downloads": 9000, "likes": 50},
        {"name": "KapoorLabs/xenopus-stardist3d-nuclei", "tags": ["stardist3d", "3d", "nuclei"], "downloads": 300, "likes": 5},
        {"name": "rando/unrelated-llm", "tags": ["text-generation"], "downloads": 500000, "likes": 999},
    ]
    ranked = R._rank(cands, prefer={"3d", "nuclei", "zebrafish"}, min_signal=0)
    names = [d["name"] for d in ranked]
    stardist = next(d for d in ranked if "stardist3d" in d["name"])
    cellpose = next(d for d in ranked if "cellpose" in d["name"])
    checks = {
        "stardist_beats_cellpose": names.index(stardist["name"]) < names.index(cellpose["name"]),
        "cellpose_flagged_slow": "SLOW" in cellpose["feasibility"],
        "stardist_flagged_fast": "fast" in stardist["feasibility"],
        "stardist_has_tag_overlap": stardist["tag_overlap"] >= 2,
        "feas_hint_fast": "fast" in R._feasibility_hint("x/unet3d-nuclei", ["3d"]),
        "feas_hint_slow": "SLOW" in R._feasibility_hint("x/cellpose-sam", ["cellpose"]),
    }
    # date filter: keep only in-range; undated candidates are KEPT (do-not-lose-data rule)
    dated = [
        {"name": "old", "date": "2019-05-01", "tags": [], "downloads": 1},
        {"name": "recent", "date": "2025-11-01", "tags": [], "downloads": 1},
        {"name": "undated", "date": "", "tags": [], "downloads": 1},
    ]
    kept = {c["name"] for c in R._within_dates(dated, since="2024-01-01", until="2026-12-31")}
    checks["date_filters_old_out"] = "old" not in kept
    checks["date_keeps_recent"] = "recent" in kept
    checks["date_keeps_undated"] = "undated" in kept

    # BM25 internal ranker: the doc containing the query terms ranks first; irrelevant doc scores ~0
    corpus = [
        R._tokenize("stardist 3d nuclei segmentation zebrafish light-sheet"),
        R._tokenize("cellpose 2d membrane histology stain"),
        R._tokenize("large language model text generation"),
    ]
    br = R._bm25_rank("zebrafish nuclei 3d", corpus)
    checks["bm25_ranks_relevant_first"] = br[0][0] == 0 and br[0][1] > 0
    checks["bm25_irrelevant_scores_zero"] = dict(br).get(2, 0) == 0
    checks["bm25_empty_query"] = R._bm25_rank("", corpus) == []

    # full-text passage extractors (pure, no network) — the agent-native paper-reading path
    bioc = [{"documents": [{"passages": [
        {"infons": {"section_type": "METHODS"}, "text": "Sparse random nuclear labeling was used in zebrafish."},
        {"infons": {"section_type": "METHODS"}, "text": "Cells were imaged on a light-sheet platform."},
        {"infons": {"section_type": "RESULTS"}, "text": "Accuracy improved over the baseline tracker."},
    ]}]}]
    allp = R._bioc_passages(bioc)
    hits = R._bioc_passages(bioc, keywords=["sparse", "labeling"])
    checks["bioc_reads_all"] = len(allp) == 3 and allp[0]["section"] == "METHODS"
    checks["bioc_keyword_filters"] = len(hits) == 1 and "Sparse" in hits[0]["text"]
    checks["bioc_empty_safe"] = R._bioc_passages({}) == [] and R._bioc_passages([]) == []
    xmlp = R._xml_passages("<abstract><p>Dual-channel <i>sparse</i> labeling in embryo.</p></abstract>",
                           keywords=["sparse"])
    checks["xml_extracts_and_strips"] = len(xmlp) == 1 and "sparse labeling" in xmlp[0]["text"].lower() \
        and "<i>" not in xmlp[0]["text"]
    checks["xml_keyword_miss_empty"] = R._xml_passages("<p>unrelated text</p>", keywords=["sparse"]) == []

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("    order=", names, "| date-kept=", kept, "| bm25=", br)
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
