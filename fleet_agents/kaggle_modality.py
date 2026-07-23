"""kaggle-modality — ground a competition's DATA-MODALITY in Kaggle's OWN metadata (tags + category) instead
of a hardcoded table, so ANY active competition resolves its modality automatically.

THE SOURCE OF TRUTH = the Meta Kaggle dataset (`kaggle/meta-kaggle`), only the 3 small files we need:
  • Competitions.csv     — Slug + HostSegmentTitle (category) + numeric Id
  • CompetitionTags.csv  — CompetitionId → TagId
  • Tags.csv             — TagId → Name / FullPath   (the Names/FullPaths ARE Kaggle's real taxonomy:
        "data type > text", "task > translation", "task > audio-event-classification", "task >
        reinforcement-learning", "task > evaluation > general-knowledge-and-reasoning", ...)

We map that vocabulary → our `comp_config.MODALITIES`. resolve_modality() prefers the MOST SPECIFIC signal
(data-type tag > task tag > keyword/category > 'unknown'). build_map() joins Competitions↔CompetitionTags↔
Tags for the user's ACTIVE competitions (sibling dirs of the comp root), CACHES the result to
docs/kaggle_modality_map.json so comp-onboard and the :7788 dashboard read the Kaggle-grounded modality
OFFLINE (no re-hitting Kaggle). A tiny fleet-agent wrapper (`kaggle-modality`) refreshes/reports it.

Pure stdlib. Never raises out of the read paths. If the Meta Kaggle download fails (network/quota),
build_map falls back to KNOWN_COMPS then a competition-name keyword inference and records source honestly.
"""
from __future__ import annotations
import csv
import json
import os
import subprocess
from pathlib import Path

from .base import BaseAgent, COMP

KAGGLE = os.environ.get("KAGGLE_BIN", "/home/seshu/miniconda3/envs/llm/bin/kaggle")

_META_CACHE = COMP / "config" / "_auto" / "meta_kaggle"          # where the 3 Meta Kaggle CSVs are cached
_MAP_PATH = COMP / "docs" / "kaggle_modality_map.json"           # the cached slug→modality map (offline read)
_META_FILES = ("Competitions.csv", "CompetitionTags.csv", "Tags.csv")

# comp dirs that are NOT competitions (the same denylist the dashboard uses)
_DENYLIST = {"kaggle_ai", "unsloth_compiled_cache", "amio"}

# --------------------------------------------------------------------------- the tag → modality vocabulary
# Kaggle DATA-TYPE tags (FullPath "data type > ...") — the MOST SPECIFIC signal. Covers ALL 11 of Kaggle's real
# data-type tags: the 9 that name a modality, plus the 2 provenance tags (synthetic, root) which are IGNORED
# (they describe HOW the data was made, not WHAT it is). categorical + bigquery are tabular sub-types.
_DATATYPE_TAG = {
    "tabular": "tabular", "categorical": "tabular", "bigquery": "tabular",
    "time series": "sequence", "time series data": "sequence", "sequential": "sequence",
    "image": "image", "images": "image",
    "audio": "audio",
    "video": "video",
    "text": "text",
    "multimodal": "multimodal", "multimodal data": "multimodal",
    "graph": "graph",
    "point cloud": "pointcloud", "lidar": "pointcloud",
}
# DATA-TYPE tags that are NOT a modality (provenance / root) — acknowledged so coverage_report treats them as
# intentionally ignored rather than "unmapped".
_IGNORE_DATATYPE = {"synthetic", "data type"}

# Kaggle TASK tags (FullPath "task > ...") — strong modality signal when there is no data-type tag.
# Value "" means the task tag does NOT imply a data modality (the data-type tag decides) — it is explicitly
# acknowledged (so a genuinely new task tag surfaces as unmapped, but these do not). Priority below is
# data-type > modality-implying task > keyword, so a "" task tag never blocks a real signal.
_TASK_TAG = {
    # --- image family ---
    "image classification": "image", "image segmentation": "image", "object detection": "image",
    "image generator": "image", "image-to-text": "image", "image super resolution": "image",
    "image augmentation": "image", "image text detection": "image", "image text recognition": "image",
    "image style transfer": "image", "image classification logits": "image", "pose detection": "image",
    "aesthetic quality": "image", "segmentation": "image", "denoising": "image",
    # legacy/alias image task names (backward-compat):
    "semantic segmentation": "image", "instance segmentation": "image", "image generation": "image",
    # --- text family ---
    "text classification": "text", "text generation": "text", "question answering": "text",
    "summarization": "text", "translation": "text", "token classification": "text",
    "text pre-processing": "text", "text conversation": "text", "retrieval question answering": "text",
    "text-to-text generation": "text", "text sequence alignment": "text",
    "multilingual and cross-lingual capabilities": "text", "coding": "text", "math": "text",
    # LLM reasoning EVALUATION (nvidia-nemotron) is a text/LLM comp, not abstract grid program-synthesis:
    "general knowledge and reasoning": "text", "evaluation": "text",
    # legacy/alias text task names (backward-compat):
    "machine translation": "text", "named entity recognition": "text", "nlp": "text",
    # --- audio family ---
    "audio command detection": "audio", "audio event classification": "audio",
    "audio classification": "audio", "audio synthesis": "audio", "speech-to-text": "audio",
    "automatic speech recognition": "audio", "audio-to-audio": "audio",
    # legacy/alias audio task names (backward-compat):
    "speech recognition": "audio", "sound event detection": "audio", "music": "audio",
    # --- video family ---
    "video classification": "video", "video generation": "video", "video text": "video",
    "action recognition": "video",   # legacy/alias
    # --- tabular / agent ---
    "tabular classification": "tabular",
    "reinforcement learning": "agent-env",
    # abstract program-synthesis / ARC-style reasoning → our grid-reasoning modality:
    "abstraction and reasoning": "grid-reasoning", "program synthesis": "grid-reasoning",
    # --- task tags that DO NOT imply a data modality (data-type tag decides) → "" (acknowledged, not a modality) ---
    "binary classification": "", "regression": "", "multiclass classification": "",
    "multilabel classification": "", "multitask classification": "", "classification": "",
    "logistic regression": "", "linear regression": "", "clustering": "", "feature extraction": "",
    "retrieval/ranking": "", "distance": "", "other": "",
}
# generic KEYWORD fallback (matched as substrings of a tag name OR the category / slug) — LEAST specific.
_KEYWORD = [
    ("audio", "audio"), ("speech", "audio"), ("sound", "audio"), ("music", "audio"), ("bird", "audio"),
    ("point cloud", "pointcloud"), ("lidar", "pointcloud"),
    ("computer vision", "image"), ("segmentation", "image"), ("object detection", "image"),
    ("image", "image"),
    ("video", "video"),
    ("machine translation", "text"), ("translation", "text"), ("language", "text"),
    ("nlp", "text"), ("text", "text"),
    ("reinforcement", "agent-env"), ("agent", "agent-env"),
    ("arc-agi", "grid-reasoning"), ("abstraction", "grid-reasoning"), ("reasoning", "grid-reasoning"),
    ("time series", "sequence"),
    ("tabular", "tabular"),
]

# name-keyword fallback for the offline / no-Meta-Kaggle path (matched against the competition SLUG).
_SLUG_KEYWORD = [
    ("birdclef", "audio"), ("audio", "audio"), ("speech", "audio"), ("sound", "audio"),
    ("translation", "text"), ("nlp", "text"), ("language", "text"), ("nemotron", "text"),
    ("cell-tracking", "volume-time"), ("tracking", "volume-time"),
    ("security", "agent-env"), ("agent", "agent-env"), ("orbit", "agent-env"), ("tcg", "agent-env"),
    ("arc-agi", "grid-reasoning"), ("arc-prize", "grid-reasoning"),
]


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def resolve_modality(slug, tags=None, category=None):
    """Kaggle tags/category → our comp_config modality string.
    Priority: explicit DATA-TYPE tag > TASK tag > keyword (tag names + category) > 'unknown'.
    `tags` = a list of Kaggle tag NAMES (e.g. ["text", "translation"]). `category` = HostSegmentTitle."""
    names = [_norm(t) for t in (tags or []) if _norm(t)]
    # 1) explicit data-type tag (most specific)
    for n in names:
        if n in _DATATYPE_TAG:
            return _DATATYPE_TAG[n]
    # 2) modality-implying task tag (skip task-only tags mapped to "" — they don't decide a modality)
    for n in names:
        m = _TASK_TAG.get(n)
        if m:
            return m
    # 3) keyword substrings over tag names then the category
    hay = names + ([_norm(category)] if category else [])
    for kw, mod in _KEYWORD:
        if any(kw in h for h in hay):
            return mod
    return "unknown"


# --------------------------------------------------------------------------- Meta Kaggle join
def _ensure_meta(meta_dir: Path, download=True) -> bool:
    """Ensure the 3 Meta Kaggle CSVs live in meta_dir. Download only the missing ones. Returns True if all present."""
    meta_dir.mkdir(parents=True, exist_ok=True)
    for f in _META_FILES:
        if (meta_dir / f).exists():
            continue
        if not download:
            return False
        try:
            subprocess.run([KAGGLE, "datasets", "download", "kaggle/meta-kaggle", "-f", f,
                            "-p", str(meta_dir)], capture_output=True, text=True, timeout=600,
                           env={**os.environ})
            # the CLI may leave a .zip — unzip if so
            z = meta_dir / (f + ".zip")
            if z.exists():
                import zipfile
                with zipfile.ZipFile(z) as zz:
                    zz.extractall(meta_dir)
                z.unlink()
        except Exception:  # noqa: BLE001
            pass
    return all((meta_dir / f).exists() for f in _META_FILES)


def _load_meta(meta_dir=None, download=True) -> dict:
    """Join Competitions↔CompetitionTags↔Tags → {slug: {'id', 'category', 'tags':[names], 'fullpaths':[...]}}.
    Empty dict if the files are unavailable (offline + no cache)."""
    meta_dir = Path(meta_dir) if meta_dir else _META_CACHE
    if not _ensure_meta(meta_dir, download=download):
        return {}
    try:
        tags = {}
        with open(meta_dir / "Tags.csv", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                tags[row["Id"]] = {"name": row.get("Name", ""), "full": row.get("FullPath", "")}
        comp_tags = {}
        with open(meta_dir / "CompetitionTags.csv", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                comp_tags.setdefault(row["CompetitionId"], []).append(row["TagId"])
        out = {}
        with open(meta_dir / "Competitions.csv", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                cid, slug = row.get("Id", ""), row.get("Slug", "")
                if not slug:
                    continue
                tids = comp_tags.get(cid, [])
                out[slug] = {
                    "id": cid,
                    "category": row.get("HostSegmentTitle", ""),
                    "tags": [tags.get(t, {}).get("name", "") for t in tids if tags.get(t, {}).get("name")],
                    "fullpaths": [tags.get(t, {}).get("full", "") for t in tids],
                }
        return out
    except Exception:  # noqa: BLE001
        return {}


def active_slugs() -> list:
    """The user's ACTIVE competitions = sibling dirs of the comp root (minus the denylist / dotfiles)."""
    root = COMP.parent
    out = []
    try:
        for d in sorted(root.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and d.name not in _DENYLIST:
                out.append(d.name)
    except Exception:  # noqa: BLE001
        pass
    return out


def _known_modality(slug) -> str:
    """High-confidence override from comp-onboard's KNOWN_COMPS table (late import → no cycle)."""
    try:
        from .comp_onboard import KNOWN_COMPS
        return (KNOWN_COMPS.get(slug) or {}).get("modality", "") or ""
    except Exception:  # noqa: BLE001
        return ""


def _name_keyword(slug) -> str:
    s = _norm(slug)
    for kw, mod in _SLUG_KEYWORD:
        if kw in s:
            return mod
    return ""


def build_map(slugs=None, meta_dir=None, download=True, cache=True) -> dict:
    """{slug: {modality, kaggle_tags, category, source}} for the ACTIVE comps, grounded in Meta Kaggle tags.
    source ∈ {kaggle-tags, known-comps, name-keyword, unknown}. Caches to docs/kaggle_modality_map.json.
    Set download=False (test/offline) to only read pre-supplied meta_dir CSVs."""
    slugs = list(slugs) if slugs else active_slugs()
    meta = _load_meta(meta_dir, download=download)
    out = {}
    for slug in slugs:
        info = meta.get(slug) or {}
        tags = info.get("tags", [])
        category = info.get("category", "")
        mod = resolve_modality(slug, tags=tags, category=category)
        source = "kaggle-tags"
        if mod == "unknown":
            km = _known_modality(slug)          # KNOWN_COMPS fills gaps for untagged comps (biohub/ai-agent-security)
            if km:
                mod, source = km, "known-comps"
            else:
                nk = _name_keyword(slug)
                if nk:
                    mod, source = nk, "name-keyword"
                else:
                    source = "unknown"
        out[slug] = {"modality": mod, "kaggle_tags": tags, "category": category, "source": source}
    if cache:
        try:
            _MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
            _MAP_PATH.write_text(json.dumps(out, indent=2, sort_keys=True))
        except Exception:  # noqa: BLE001
            pass
    return out


def load_map() -> dict:
    """Read the cached Kaggle-grounded modality map (offline). {} if not yet built."""
    try:
        return json.loads(_MAP_PATH.read_text())
    except Exception:  # noqa: BLE001
        return {}


def cached_modality(slug) -> str:
    """The cached Kaggle-tag-derived modality for a slug, or '' if unknown/not cached (never raises)."""
    return (load_map().get(slug) or {}).get("modality", "") or ""


# --------------------------------------------------------------------------- coverage self-check ("never miss")
def _tag_toplevel(fullpath: str) -> str:
    return (fullpath or "").split(">")[0].strip().lower()


def coverage_report(meta_dir=None) -> dict:
    """Audit our tag→modality map against Kaggle's ENTIRE data-type + task tag vocabulary (Tags.csv).

    For every tag whose FullPath top-level is 'data type' or 'task' with CompetitionCount>0, report whether
    resolve_modality maps it. Guarantee: a NEW Kaggle tag added in future that we do NOT yet map surfaces in
    `unmapped` (caught, not silently missed). DATA-TYPE tags must ALL be mapped or explicitly ignored
    (synthetic, root). TASK tags: modality-implying ones mapped; task-only ones intentionally "" (acknowledged).

    Returns {'data_type': {mapped, ignored, unmapped}, 'task': {mapped, intentional_empty, unmapped}, 'counts':…}.
    Offline (reads the cached Tags.csv); returns the empty skeleton if the file is unavailable (never raises)."""
    meta_dir = Path(meta_dir) if meta_dir else _META_CACHE
    rep = {"data_type": {"mapped": [], "ignored": [], "unmapped": []},
           "task": {"mapped": [], "intentional_empty": [], "unmapped": []}}
    try:
        with open(meta_dir / "Tags.csv", encoding="utf-8", errors="replace") as f:
            rows = list(csv.DictReader(f))
    except Exception:  # noqa: BLE001
        rows = []
    for r in rows:
        try:
            cc = int(r.get("CompetitionCount") or 0)
        except (TypeError, ValueError):
            cc = 0
        if cc <= 0:
            continue
        name = _norm(r.get("Name", ""))
        top = _tag_toplevel(r.get("FullPath", ""))
        if top == "data type":
            if name in _IGNORE_DATATYPE:
                rep["data_type"]["ignored"].append(name)
            elif name in _DATATYPE_TAG:
                rep["data_type"]["mapped"].append(name)
            else:
                rep["data_type"]["unmapped"].append(name)
        elif top == "task":
            if name in _TASK_TAG:
                bucket = "mapped" if _TASK_TAG[name] else "intentional_empty"
                rep["task"][bucket].append(name)
            else:
                rep["task"]["unmapped"].append(name)
    rep["counts"] = {
        "data_type_mapped": len(rep["data_type"]["mapped"]),
        "data_type_ignored": len(rep["data_type"]["ignored"]),
        "data_type_unmapped": len(rep["data_type"]["unmapped"]),
        "task_mapped": len(rep["task"]["mapped"]),
        "task_intentional_empty": len(rep["task"]["intentional_empty"]),
        "task_unmapped": len(rep["task"]["unmapped"]),
    }
    return rep


# --------------------------------------------------------------------------- fleet agent wrapper
class KaggleModality(BaseAgent):
    name = "kaggle-modality"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        slugs = spec.get("slugs")
        if slugs:                                    # build/refresh (may hit Meta Kaggle)
            m = build_map(slugs)
            action = "built"
        else:                                        # empty → REPORT the cached map (read-only, no download)
            m = load_map()
            action = "cached" if m else "empty"
        parts = ", ".join(f"{s}={v.get('modality')}({v.get('source')})" for s, v in sorted(m.items()))
        cov = coverage_report()                      # never-miss self-check over Kaggle's ENTIRE tag vocabulary
        c = cov.get("counts", {})
        dt_unmapped = cov.get("data_type", {}).get("unmapped", [])
        task_unmapped = cov.get("task", {}).get("unmapped", [])
        cov_msg = (f"tag-coverage: data-type {c.get('data_type_mapped', 0)} mapped / "
                   f"{c.get('data_type_ignored', 0)} ignored / {c.get('data_type_unmapped', 0)} UNMAPPED; "
                   f"task {c.get('task_mapped', 0)} mapped / {c.get('task_intentional_empty', 0)} task-only / "
                   f"{c.get('task_unmapped', 0)} UNMAPPED"
                   + (f" → NEW unmapped tags {dt_unmapped + task_unmapped}" if (dt_unmapped or task_unmapped) else ""))
        msg = (f"kaggle-modality: {action} — {len(m)} active comp(s) grounded in Kaggle tags → {parts or '(none)'}. "
               f"{cov_msg}. Cache: {_MAP_PATH}")
        self.log(msg, kind="finding",
                 recommendation="comp-onboard + :7788 dashboard read docs/kaggle_modality_map.json for real modality; "
                                "any UNMAPPED tag above = a new Kaggle tag to add to kaggle_modality maps")
        return self.done({"map": m, "path": str(_MAP_PATH), "action": action, "n": len(m), "coverage": cov}, msg)


_AGENT = KaggleModality()


def run(q, worker):
    return _AGENT.run(q, worker)
