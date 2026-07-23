"""lit-search — DETERMINISTIC python literature/model search over Hugging Face (models + datasets) and arXiv.
No LLM: hits the HF Hub API (huggingface_hub) and the arXiv Atom API (plain HTTP) directly, so results are exact,
fast, and free. Complements `deep-research` (LLM synthesis) and `paper-research` (curated catalog): this one
FINDS the raw candidates — downloadable weights, datasets, and recent papers — for OUR domain (zebrafish 3D
light-sheet nuclei detection + division-aware tracking), then RECORDS them to the journal.

Reusable/spec-driven: {query, sources:[hf_models,hf_datasets,arxiv], limit, since_year, hf_filter}. A BaseAgent
subclass with a data-wise test (mocks both APIs). Use it to keep a live, tracked inventory of in-domain assets.
"""
from __future__ import annotations
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent


def hf_models(query, limit=15, task=None):
    from huggingface_hub import list_models
    out = []
    limit = max(1, int(limit))
    for m in list_models(search=query, limit=limit, filter=task, sort="downloads"):  # hf_hub 1.21: no `direction`
        tags = list(getattr(m, "tags", []) or [])
        lic = next((t.split(":", 1)[1] for t in tags if t.startswith("license:")), None)
        out.append({"id": m.id, "downloads": getattr(m, "downloads", None), "likes": getattr(m, "likes", None),
                    "pipeline": getattr(m, "pipeline_tag", None), "license": lic,
                    "tags": [t for t in tags if not t.startswith(("license:", "region:"))][:8]})
    return out


def hf_datasets(query, limit=10):
    from huggingface_hub import list_datasets
    out = []
    for d in list_datasets(search=query, limit=limit, sort="downloads"):  # hf_hub 1.21: no `direction`
        out.append({"id": d.id, "downloads": getattr(d, "downloads", None), "likes": getattr(d, "likes", None),
                    "tags": [t for t in (getattr(d, "tags", []) or []) if not t.startswith("region:")][:8]})
    return out


def arxiv_papers(query, limit=12, since_year=2024, timeout=30):
    """timeout (s): cap the arXiv HTTP fetch so a hung endpoint can't stall the agent."""
    import requests, xml.etree.ElementTree as ET
    r = requests.get("http://export.arxiv.org/api/query",
                     params={"search_query": f"all:{query}", "start": 0, "max_results": max(1, int(limit)),
                             "sortBy": "submittedDate", "sortOrder": "descending"}, timeout=max(1, int(timeout)))
    ns = {"a": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(r.text); out = []
    for e in root.findall("a:entry", ns):
        pub = (e.findtext("a:published", "", ns) or "")[:10]
        if since_year and pub[:4] and pub[:4].isdigit() and int(pub[:4]) < since_year:
            continue
        out.append({"title": " ".join((e.findtext("a:title", "", ns) or "").split()),
                    "published": pub, "url": (e.findtext("a:id", "", ns) or ""),
                    "authors": [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)][:3],
                    "summary": " ".join((e.findtext("a:summary", "", ns) or "").split())[:220]})
    return out


class LitSearch(BaseAgent):
    name = "lit-search"
    thread = "A"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        query = spec.get("query") or "3D nucleus detection light-sheet microscopy"
        # OPTIONAL: 'max_results' aliases 'limit'; 'timeout' (s) caps the arXiv fetch. Robust to bad ints.
        try:
            lim = max(1, int(spec.get("limit", spec.get("max_results", 12))))
        except Exception:  # noqa: BLE001
            lim = 12
        try:
            yr = int(spec.get("since_year", 2024))
        except Exception:  # noqa: BLE001
            yr = 2024
        to = spec.get("timeout", 30)
        sources = spec.get("sources") or ["hf_models", "hf_datasets", "arxiv"]
        if isinstance(sources, str):                         # tolerate a single-source string
            sources = [sources]
        res, errs = {}, {}
        if "hf_models" in sources:
            try: res["hf_models"] = hf_models(query, lim, spec.get("hf_filter"))
            except Exception as e: errs["hf_models"] = str(e)[:120]  # noqa: BLE001
        if "hf_datasets" in sources:
            try: res["hf_datasets"] = hf_datasets(query, lim)
            except Exception as e: errs["hf_datasets"] = str(e)[:120]  # noqa: BLE001
        if "arxiv" in sources:
            try: res["arxiv"] = arxiv_papers(query, lim, yr, timeout=to)
            except Exception as e: errs["arxiv"] = str(e)[:120]  # noqa: BLE001

        nm = len(res.get("hf_models", [])); nd = len(res.get("hf_datasets", [])); na = len(res.get("arxiv", []))
        self.save_state({"query": query, "n_models": nm, "n_datasets": nd, "n_arxiv": na, "results": res, "errors": errs})
        from . import ledger
        top_m = ", ".join(m["id"] for m in res.get("hf_models", [])[:5])
        top_a = "; ".join(f"{p['title'][:50]} ({p['published'][:7]})" for p in res.get("arxiv", [])[:3])
        ledger.log("lit-search", summary=f"lit-search '{query}': {nm} HF models, {nd} datasets, {na} arXiv(≥{yr})",
                   detail=f"models: {top_m} | arxiv: {top_a}", kind="finding",
                   recommendation="download in-domain weights / read recent papers surfaced")
        mrows = "\n".join(f"| {m['id']} | {m.get('downloads') or 0} | {m.get('license') or '?'} | {m.get('pipeline') or ''} |"
                          for m in res.get("hf_models", [])[:10])
        arows = "\n".join(f"- {p['published'][:7]} · [{p['title'][:70]}]({p['url']})" for p in res.get("arxiv", [])[:8])
        msg = (f"[{worker}] **LIT-SEARCH** · '{query}' → {nm} HF models · {nd} datasets · {na} arXiv (≥{yr})"
               + (f" · errors: {errs}" if errs else "") + "\n\n"
               f"**HF models** (by downloads):\n| repo | dl | license | task |\n|---|--:|---|---|\n{mrows}\n\n"
               f"**arXiv (recent)**:\n{arows}")
        self.post(worker, "all", msg, routine=False, kind="finding")
        return self.done({"query": query, "results": res, "errors": errs,
                          "counts": {"models": nm, "datasets": nd, "arxiv": na}}, msg)


_AGENT = LitSearch()


def run(q, worker):
    return _AGENT.run(q, worker)
