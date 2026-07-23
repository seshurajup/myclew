"""arc-idioms — DETERMINISTIC idiom-catalogue tool for network-golf / grid-reasoning (ARC-AGI-ONNX) comps.
PURE PYTHON — no LLM. It parses the mined construction playbook (patterns.md, 9th place — the per-score-band
ONNX idioms, ARC rule families, cost-saving rules, and worked exemplars) into a queryable catalogue, and
`query(signature)` returns the candidate idioms + their ACHIEVED score band for a task's rule family. This is
the reusable-construction MEMORY the top teams shared; the live researcher agent calls it via fleet_dispatch
to pick a target band + idiom before asking arc-onnx-golf to emit.

Also accepts the 5th-place YAML idiom format ({title, description, summary, votes}) so a growing team idiom
catalogue can be merged in.

Sources (default bundled): fleet_agents/data/arc_patterns.md. Spec: {patterns_path, query|task, top_k,
yaml_idioms_path}. Empty spec → parse the bundled catalogue and return counts. Pure text parsing (no onnx).
Data-wise test: test_fleet_agents/arc_idioms_test.py (synthetic patterns.md + the bundled real one).
"""
from __future__ import annotations
import re
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
_BUNDLED = COMP / "fleet_agents" / "data" / "arc_patterns.md"

# ONNX op vocabulary — recognised inside `backticks` so idioms carry the ops they use.
ONNX_OPS = {
    "Einsum", "Conv", "ConvTranspose", "ConvInteger", "QLinearConv", "QLinearMatMul", "MatMulInteger",
    "Gather", "GatherND", "GatherElements", "Scatter", "ScatterND", "ScatterElements", "Slice", "Pad",
    "Transpose", "Reshape", "Tile", "Resize", "RoiAlign", "MaxRoiPool", "GridSample", "MaxPool", "LpPool",
    "AveragePool", "TopK", "ArgMax", "OneHot", "Range", "CumSum", "Trilu", "Mod", "Sign", "Hardmax",
    "Shrink", "BitShift", "BitwiseAnd", "BitwiseOr", "QuantizeLinear", "DequantizeLinear", "Cast", "Split",
    "Concat", "Equal", "Where", "ReduceSum", "ReduceL2", "Sqrt", "LayerNormalization", "TfIdfVectorizer",
    "Identity", "Constant", "MatMul", "Mul", "Add",
}
_BAND_RE = re.compile(r"^##\s+(?:Band\s+)?[`\"]?(\d+)-?(\d+)?[`\"]?\s*$")
_EXACT_RE = re.compile(r"^##\s+Exact\s+[`\"]?(\d+)[`\"]?", re.I)
_SUBSEC_RE = re.compile(r"^###\s+(.+?)\s*$")
_EXEMPLAR_RE = re.compile(r"^\|\s*[`\"]?(task\d+)[`\"]?\s*\|\s*([\d.]+)\s*\|\s*(\d+)\s*\|\s*(.+?)\s*\|")
_OPS_IN = re.compile(r"`([A-Za-z][A-Za-z0-9]*)")   # op name right after a backtick (may be followed by (attrs))


def _ops(text):
    return sorted({m for m in _OPS_IN.findall(text) if m in ONNX_OPS})


def _bullet_title(text):
    """`- **Title.** description` → (title, description)."""
    m = re.match(r"\s*[-*]\s+\*\*(.+?)\.?\*\*\s*(.*)$", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = re.match(r"\s*[-*]\s+(.*)$", text)
    if m:
        d = m.group(1).strip()
        return d.split(".")[0][:60], d
    return None, None


def parse_patterns(path=None):
    """Parse patterns.md → list of idiom records. Each: {kind, band, band_lo, band_hi, title, description,
    ops, task, score, cost}. kind ∈ {onnx_idiom, arc_family, cost_rule, exemplar, idea}."""
    p = Path(path) if path else _BUNDLED
    if not p.exists():
        return []
    out, band, lo, hi, sub = [], None, None, None, None
    for raw in p.read_text().splitlines():
        line = raw.rstrip()
        mb = _BAND_RE.match(line)
        me = _EXACT_RE.match(line)
        if mb:
            lo = int(mb.group(1)); hi = int(mb.group(2)) if mb.group(2) else lo
            band = f"{lo}-{hi}" if hi != lo else str(lo); sub = None; continue
        if me:
            lo = hi = int(me.group(1)); band = f"exact-{lo}"; sub = None; continue
        ms = _SUBSEC_RE.match(line)
        if ms:
            sub = ms.group(1).lower(); continue
        mx = _EXEMPLAR_RE.match(line)
        if mx and band is not None:
            out.append({"kind": "exemplar", "band": band, "band_lo": lo, "band_hi": hi,
                        "task": mx.group(1), "score": float(mx.group(2)), "cost": int(mx.group(3)),
                        "title": mx.group(4), "description": mx.group(4), "ops": _ops(mx.group(4))})
            continue
        if band is None or sub is None or not line.strip().startswith(("-", "*")):
            continue
        title, desc = _bullet_title(line)
        if not title:
            continue
        kind = {"onnx idioms": "onnx_idiom", "recurring arc families": "arc_family",
                "cost-saving rules": "cost_rule", "additional useful ideas": "idea"}.get(sub)
        if kind is None:
            continue
        out.append({"kind": kind, "band": band, "band_lo": lo, "band_hi": hi, "title": title,
                    "description": desc, "ops": _ops(line), "task": None, "score": None, "cost": None})
    return out


def parse_yaml_idioms(path):
    """Parse a 5th-place-style YAML idiom file/dir ({title, description, summary, votes}) → idiom records.
    Guarded: needs pyyaml; returns [] if unavailable. Votes are summarised to an evidence count."""
    try:
        import yaml  # noqa
    except Exception:  # noqa: BLE001
        return []
    p = Path(path)
    files = sorted(p.glob("*.y*ml")) if p.is_dir() else ([p] if p.exists() else [])
    recs = []
    for f in files:
        try:
            docs = list(yaml.safe_load_all(f.read_text()))
        except Exception:  # noqa: BLE001
            continue
        for d in docs:
            if not isinstance(d, dict) or "title" not in d:
                continue
            votes = d.get("votes") or {}
            ev = sum(1 for v in votes.values() if isinstance(v, dict) and (v.get("vote") or 0) > 0)
            recs.append({"kind": "team_idiom", "band": None, "band_lo": None, "band_hi": None,
                         "title": str(d.get("title")), "description": str(d.get("summary") or d.get("description") or ""),
                         "ops": _ops(str(d.get("description", "")) + " " + str(d.get("summary", ""))),
                         "task": None, "score": None, "cost": None, "evidence": ev})
    return recs


# ---- task signature (from train pairs) → keyword tags to query with -----------------------------------
def signature_from_task(train):
    """Cheap keyword signature of a task's rule family from its train pairs (no onnx). Feeds query()."""
    import numpy as np
    tags = []
    ins = [np.asarray(i) for i, _ in train]
    outs = [np.asarray(o) for _, o in train]
    if not ins:
        return {"tags": [], "text": ""}
    same = all(i.shape == o.shape for i, o in zip(ins, outs))
    tags.append("same shape" if same else "shape changes")
    if all(i.shape[0] == i.shape[1] for i in ins):
        tags.append("square")
    if same and all(set(np.unique(i)) == set(np.unique(o)) is False or True for i, o in zip(ins, outs)):
        # palette-ish: same geometry, colours move
        if all(i.shape == o.shape for i, o in zip(ins, outs)) and \
           any(not np.array_equal(i, o) for i, o in zip(ins, outs)):
            tags.append("palette recolor")
    if all(o.size > i.size for i, o in zip(ins, outs)):
        tags.append("grid enhancement upsample tile")
    if all(o.size < i.size for i, o in zip(ins, outs)):
        tags.append("crop compression")
    if len({o.shape for o in outs}) == 1 and all(np.array_equal(outs[0], o) for o in outs):
        tags.append("constant fixed output")
    if all(np.array_equal(i.T, o) for i, o in zip(ins, outs) if i.shape == o.T.shape):
        tags.append("mirror reflection transpose")
    return {"tags": tags, "text": " ".join(tags)}


def query(catalogue, signature, top_k=8):
    """Rank idiom records by relevance to a signature (free-text string, list of tags, or dict with
    'text'/'tags'/'ops'). Returns top_k with their achieved score band so the worker can pick a target."""
    if isinstance(signature, dict):
        text = (signature.get("text") or "") + " " + " ".join(signature.get("tags") or [])
        want_ops = set(signature.get("ops") or [])
    elif isinstance(signature, (list, tuple)):
        text = " ".join(str(x) for x in signature); want_ops = set()
    else:
        text = str(signature or ""); want_ops = set()
    words = {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}
    scored = []
    for rec in catalogue:
        hay = f"{rec.get('title','')} {rec.get('description','')}".lower()
        hw = set(re.findall(r"[a-z]+", hay))
        overlap = len(words & hw)
        op_overlap = len(want_ops & set(rec.get("ops") or []))
        # prefer concrete idioms/exemplars over generic rules, and higher-band (cheaper) exemplars
        kind_w = {"onnx_idiom": 2, "exemplar": 2, "arc_family": 1, "team_idiom": 2, "idea": 1, "cost_rule": 1}
        s = overlap * 2 + op_overlap * 3 + kind_w.get(rec.get("kind"), 0)
        if rec.get("kind") == "exemplar" and rec.get("score"):
            s += rec["score"] / 25.0
        if s > 0:
            scored.append((s, rec))
    scored.sort(key=lambda t: (-t[0], -(t[1].get("band_hi") or 0)))
    return [r for _, r in scored[:top_k]]


def summarize(catalogue):
    by_kind, by_band = {}, {}
    for r in catalogue:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + 1
        if r.get("band"):
            by_band[r["band"]] = by_band.get(r["band"], 0) + 1
    return {"total": len(catalogue), "by_kind": by_kind, "by_band": by_band}


class ArcIdioms(BaseAgent):
    name = "arc-idioms"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        spec = self.spec(q)
        cat = parse_patterns(spec.get("patterns_path"))
        if spec.get("yaml_idioms_path"):
            cat = cat + parse_yaml_idioms(spec["yaml_idioms_path"])
        if not cat:
            return self.escalate(worker, "researcher",
                                 f"[{worker}] arc-idioms: no idioms parsed (patterns.md missing at "
                                 f"{spec.get('patterns_path') or _BUNDLED}).")
        summ = summarize(cat)
        sig = spec.get("query")
        if sig is None and spec.get("task"):
            t = spec["task"]
            sig = signature_from_task([(e["input"], e["output"]) for e in t.get("train", [])])
        if sig is None:
            msg = (f"[{worker}] arc-idioms: {summ['total']} idioms parsed — {summ['by_kind']} across bands "
                   f"{sorted(summ['by_band'])}. Pass spec.query or spec.task to rank candidates.")
            return self.done({"summary": summ, "n_idioms": summ["total"]}, msg)
        top = query(cat, sig, top_k=int(spec.get("top_k", 8)))
        cand = [{"kind": r["kind"], "band": r.get("band"), "title": r["title"], "ops": r.get("ops"),
                 "task": r.get("task"), "cost": r.get("cost"), "score": r.get("score")} for r in top]
        bands = sorted({r.get("band") for r in top if r.get("band")})
        msg = (f"[{worker}] arc-idioms: {len(cand)} candidate idioms for signature (bands {bands}); "
               f"top: {cand[0]['title'] if cand else 'none'}")
        self.log(summary=msg, kind="finding")
        return self.done({"candidates": cand, "summary": summ, "signature": sig}, msg, to="researcher")


_AGENT = ArcIdioms()


def run(q, worker):
    return _AGENT.run(q, worker)
