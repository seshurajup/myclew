"""lit_search_test — DATA-WISE verifier of lit-search result-shaping (license extraction, tag filtering, arXiv
year filter) with MOCKED APIs (no network). Closes the fleet's one missing test.
"""
import os, sys
import huggingface_hub
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import lit_search as L


class _M:
    def __init__(self, i, tags): self.id = i; self.tags = tags; self.downloads = 100; self.likes = 3
    pipeline_tag = "image-segmentation"


def _run():
    print("=== LIT-SEARCH LOGIC VERIFIER (mocked APIs) ===")
    checks = {}
    # hf_models: license extracted from tags, license:/region: filtered out of the tag list
    L.list_models = lambda **kw: [_M("org/stardist3d", ["3d", "nuclei", "license:bsd-3-clause", "region:us"])]
    huggingface_hub.list_models = L.list_models
    hm = L.hf_models("stardist", 5)
    checks["hf_model_id"] = hm and hm[0]["id"] == "org/stardist3d"
    checks["license_extracted"] = hm[0]["license"] == "bsd-3-clause"
    checks["meta_tags_filtered"] = "license:bsd-3-clause" not in hm[0]["tags"] and "region:us" not in hm[0]["tags"]
    checks["real_tags_kept"] = "nuclei" in hm[0]["tags"]

    # arxiv year filter: an old paper is dropped, a recent one kept.
    # arxiv_papers does `import requests` INSIDE the fn → patch the REAL requests.get (same module object).
    import requests

    class _R:
        text = ('<feed xmlns="http://www.w3.org/2005/Atom">'
                '<entry><title>New 3D tracking</title><published>2025-03-01T00:00:00Z</published>'
                '<id>http://arxiv/1</id><summary>x</summary><author><name>A</name></author></entry>'
                '<entry><title>Old method</title><published>2019-01-01T00:00:00Z</published>'
                '<id>http://arxiv/2</id><summary>y</summary></entry></feed>')
    requests.get = lambda *a, **k: _R()
    ap = L.arxiv_papers("tracking", 12, since_year=2024)
    titles = [p["title"] for p in ap]
    checks["arxiv_recent_kept"] = "New 3D tracking" in titles
    checks["arxiv_old_dropped"] = "Old method" not in titles
    checks["arxiv_schema"] = ap and set(("title", "published", "url", "authors", "summary")) <= set(ap[0])

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values()); print("RESULT:", "PASS" if ok else "FAIL"); return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
