"""research_search_integration_test — SOLID coverage of the network/PARSING layer with MOCKED API responses
(no live network). Each source's fetch is monkeypatched to return a canned payload; we assert the method maps
it to the correct candidate schema {name, src, url, date, tags, downloads}. This catches an API field-rename or
a parse regression that the pure-logic test can't see.
"""
import os, sys
import urllib.request
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import research_search as R


class _FakeResp:
    def __init__(self, data): self.data = data
    def read(self): return self.data
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _run():
    print("=== RESEARCH-SEARCH INTEGRATION (mocked APIs) ===")
    a = R.ResearchSearch()
    checks = {}

    # bioimage.io
    a._get_json = lambda url, **kw: {"collection": [
        {"type": "model", "name": "StarDist3D", "nickname": "affable-shark", "description": "3d nuclei seg",
         "tags": ["3d", "nuclei"], "download_count": 123, "rdf_source": "http://bi/x"}]}
    r = a._search_bioimage("nuclei 3d")
    checks["bioimage_schema"] = (r and r[0]["src"] == "bioimage" and r[0]["name"] == "affable-shark"
                                 and r[0]["downloads"] == 123 and r[0]["url"] == "http://bi/x")

    # zenodo
    a._get_json = lambda url, **kw: {"hits": {"hits": [
        {"metadata": {"title": "Z model", "publication_date": "2024-05-01", "keywords": ["nuclei"],
                      "resource_type": {"type": "software"}},
         "stats": {"views": 50}, "links": {"html": "http://z/1"}}]}}
    r = a._search_zenodo("x")
    checks["zenodo_schema"] = (r[0]["src"] == "zenodo" and r[0]["date"] == "2024-05-01"
                               and r[0]["url"] == "http://z/1" and r[0]["downloads"] == 50)

    # github
    a._get_json = lambda url, **kw: {"items": [
        {"full_name": "org/repo", "pushed_at": "2025-01-02T00:00:00Z", "topics": ["stardist"],
         "language": "Python", "stargazers_count": 200, "forks_count": 9, "html_url": "http://g/1"}]}
    r = a._search_github("stardist", 3)
    good = [x for x in r if not str(x["name"]).startswith("<")]
    checks["github_schema"] = (good and good[0]["src"] == "github" and good[0]["name"] == "org/repo"
                               and good[0]["date"] == "2025-01-02" and good[0]["downloads"] == 200)

    # europepmc
    a._get_json = lambda url, **kw: {"resultList": {"result": [
        {"title": "EPMC paper", "firstPublicationDate": "2024-03-01", "source": "PPR", "pubType": "preprint",
         "citedByCount": 7, "doi": "10.1/xy", "id": "9"}]}}
    r = a._search_europepmc("nucleus", 3)
    checks["europepmc_schema"] = (r[0]["src"] == "europepmc" and r[0]["date"] == "2024-03-01"
                                  and r[0]["url"] == "https://doi.org/10.1/xy" and r[0]["downloads"] == 7)

    # figshare (POST)
    a._post_json = lambda url, body, **kw: [
        {"title": "F item", "published_date": "2024-06-01", "defined_type_name": "software",
         "url_public_html": "http://f/1"}]
    r = a._search_figshare("nuclei", 3)
    checks["figshare_schema"] = (r[0]["src"] == "figshare" and r[0]["date"] == "2024-06-01"
                                 and r[0]["url"] == "http://f/1")

    # HF (patch HfApi listers + _get_json for papers)
    from huggingface_hub import HfApi
    _orig = (HfApi.list_models, HfApi.list_datasets, HfApi.list_spaces)

    class _M:
        def __init__(self, i): self.id = i; self.tags = ["3d"]; self.downloads = 10; self.likes = 1
        last_modified = "2024-02-02"
    HfApi.list_models = lambda self, **kw: [_M("org/model")]
    HfApi.list_datasets = lambda self, **kw: [_M("org/dataset")]
    HfApi.list_spaces = lambda self, **kw: [_M("org/space")]
    a._get_json = lambda url, **kw: [{"paper": {"id": "2401.1", "title": "HF paper", "publishedAt": "2024-01-01",
                                                "upvotes": 5}}]
    r = a._search_hf("nuclei", None, None, 2)
    srcs = {x["src"] for x in r if not str(x["name"]).startswith("<")}
    checks["hf_all_types"] = {"hf-models", "hf-datasets", "hf-spaces", "hf-papers"} <= srcs
    HfApi.list_models, HfApi.list_datasets, HfApi.list_spaces = _orig

    # arxiv (patch urllib.urlopen with canned Atom XML)
    _origopen = urllib.request.urlopen
    xml = (b'<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"><entry>'
           b'<title>Arxiv paper</title><id>http://arxiv.org/abs/2405.1</id>'
           b'<published>2024-05-05T00:00:00Z</published><summary>abstract</summary>'
           b'<category term="cs.CV"/></entry></feed>')
    urllib.request.urlopen = lambda url, timeout=30: _FakeResp(xml)
    r = a._search_arxiv("tracking", ["cs.CV"], 3)
    checks["arxiv_schema"] = (r[0]["src"] == "arxiv" and r[0]["date"] == "2024-05-05"
                              and r[0]["url"] == "http://arxiv.org/abs/2405.1")
    urllib.request.urlopen = _origopen

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
