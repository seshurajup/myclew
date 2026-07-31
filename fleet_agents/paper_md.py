"""paper-md — turn ANY PDF paper into a well-designed Markdown document + real assets, reusably.

WHY: to *learn* a paper (every formula, figure, table) we need it as text we can quote, diff and teach
from — not a PDF. The fleet already had `research-search._read_pdf` (keyword passages, for triage);
this agent is the other half: a full, structure-preserving conversion we can read and turn into lessons.

BACKENDS — the choice is MEASURED, not assumed (`action="bench"` scores every available backend on the
same PDF and returns the table):
  pymupdf   the always-available in-house pipeline. Font-driven headings (bold LinLibertineTB @14.3/12.0
            → h2/h3), math-font equation blocks, and REAL LaTeX rebuilt from span geometry: NFKC
            un-mangles PDF math alphanumerics (𝑊𝑡+1 → W), span size+baseline recover sub/superscripts
            (→ `W_{t+1}`), symbol fonts are repaired by glyph table (txsys `M` → \\mathcal{M},
            txexs `Í` → \\sum, txsym `R` → \\mathbb{R}). Raster figures are extracted LOSSLESSLY and
            bound to their "Figure N:" caption; `find_tables()` → markdown tables; and every equation /
            figure / table also gets a CROP PNG, so a formula can never be silently lost.
  docling   IBM docling + LOCAL formula enrichment — the PREFERRED model backend: it decodes formulas to
            real LaTeX (`$$W_{t+1} = W_t - \\eta_t\\nabla_{W_t}\\mathcal{L}(W_t; x_t), && (1)$$`) and
            needs NO docker. Runs out-of-process (`PAPER_MD_DOCLING_PY`) and caches `_docling.md`.
  marker    datalab marker-pdf, in an ISOLATED venv (`PAPER_MD_MARKER_PY`). OPT-IN ONLY: in the current
            release both `--mode balanced` and `--mode fast` route OCR through surya → vLLM → a docker
            container, which we deliberately do not use. Kept wired for a future docker-free release.
  hybrid    DEFAULT when a model backend exists: model prose+LaTeX spliced with pymupdf's assets and
            equation crops (best of both). Falls back to pymupdf alone when no model backend is present.

Everything is spec-driven so it works for any paper in any competition:
    {"kind": "paper-md", "spec": {"pdf": "<url|path>", "slug": "nested-learning",
                                  "backend": "auto|pymupdf|marker|docling|hybrid", "action": "convert|bench",
                                  "outdir": "docs/papers", "dpi": 220, "eq_images": true}}
Outputs `<outdir>/<slug>/<slug>.md`, `assets/{fig,eq,tab}/*.png`, `equations.json`, `manifest.json`.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unicodedata
from pathlib import Path

from .base import BaseAgent, COMP

# ---------------------------------------------------------------- font knowledge (measured on real papers)
# Fonts whose spans are MATH (a block dominated by these is a display equation, not prose).
MATH_FONTS = ("LibertineMath", "txmi", "txsy", "txex", "txsym", "MSAM", "NewTXMI", "CMMI", "CMSY", "CMEX")
# Script/blackboard fonts: the *letter* is the glyph, so it needs a LaTeX wrapper, not the raw char.
BB_FONTS = ("txsym",)                       # blackboard-bold: R -> \mathbb{R}  (checked FIRST: txsym
SCRIPT_FONTS = ("txsys", "CMSY")            # would also match a "txsy" script prefix)
# Big-operator / extensible-delimiter fonts encode PIECES of a tall glyph; keep the real operators,
# drop the assembly pieces (they render as control chars / stray letters and only add noise).
BIG_OP = {"Í": r"\sum", "∑": r"\sum", "Ë": r"\prod", "Ö": r"\int", "√": r"\sqrt"}
DROP_GLYPHS = set("\x00\x01\x02\x03\x10\x11\x12\x13\x14\x15\x16\x17\x1c\x1d\x1e\x1f\ufe01\ufe02\ufe03"
                  "\uf8f0\uf8f1\uf8f2\uf8f3\uf8f4\uf8f5\uf8f6\uf8f7")

GREEK_TEX = {
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta", "ε": r"\epsilon", "ζ": r"\zeta",
    "η": r"\eta", "θ": r"\theta", "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho", "σ": r"\sigma", "τ": r"\tau",
    "υ": r"\upsilon", "φ": r"\phi", "ϕ": r"\phi", "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda", "Ξ": r"\Xi", "Π": r"\Pi",
    "Σ": r"\Sigma", "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega", "∂": r"\partial",
}
SYMBOL_TEX = {
    "∇": r"\nabla ", "⊤": r"^{\top}", "∥": r"\|", "⟨": r"\langle ", "⟩": r"\rangle ",
    "⊙": r"\odot ", "⊗": r"\otimes ", "∈": r"\in ", "≻": r"\succ ", "⊁": r"\nsucc ",
    "≡": r"\equiv ", "≈": r"\approx ", "≤": r"\le ", "≥": r"\ge ", "≠": r"\neq ",
    "→": r"\to ", "←": r"\gets ", "×": r"\times ", "·": r"\cdot ", "−": "-", "∼": r"\sim ",
    "⌈": r"\lceil ", "⌉": r"\rceil ", "⌊": r"\lfloor ", "⌋": r"\rfloor ", "□": r"\square ",
    "∞": r"\infty ", "≔": r":=", "∀": r"\forall ", "∃": r"\exists ", "∅": r"\emptyset ",
    "⊆": r"\subseteq ", "∪": r"\cup ", "∩": r"\cap ", "±": r"\pm ", "≫": r"\gg ", "≪": r"\ll ",
}
CAPTION_RE = re.compile(r"^\s*(Figure|Table|Algorithm)\s+(\d+)\s*[:.]", re.I)
EQNUM_RE = re.compile(r"^\((\d+)\)$")


# ---------------------------------------------------------------- pure helpers (data-wise testable)
def slugify(s: str) -> str:
    """`Nested Learning: The Illusion...` → `nested-learning-the-illusion` (stable asset/dir names)."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return re.sub(r"-{2,}", "-", s)[:60] or "paper"


def unmangle(text: str) -> str:
    """PDF math alphanumerics are private-use-ish codepoints (𝑊, 𝜽, 𝒙); NFKC maps them to real
    letters (W, θ, x) — the single highest-value fix for any LaTeX-authored paper."""
    return unicodedata.normalize("NFKC", text or "")


def tex_atoms(text: str) -> str:
    """Greek + operator symbols → LaTeX commands (already-unmangled text)."""
    out = []
    for ch in text:
        if ch in DROP_GLYPHS:
            continue
        out.append(GREEK_TEX.get(ch) or SYMBOL_TEX.get(ch) or BIG_OP.get(ch) or ch)
    return "".join(out)


def is_math_font(font: str) -> bool:
    return any(font.startswith(f) or f in font for f in MATH_FONTS)


BOLD_MARKS = ("bold", "medi", "semib", "black", "heavy", "-bd", "tb", "bd")


def is_bold(span: dict) -> bool:
    """PyMuPDF's bold flag (bit 4) OR a bold-ish family name — a "B" in the name is not enough:
    an arXiv template writes its section headings in `NimbusRomNo9L-Medi`, which has no B at all."""
    if int(span.get("flags", 0)) & 16:
        return True
    f = span.get("font", "").lower()
    return any(m in f for m in BOLD_MARKS)


def horizontal(line: dict) -> bool:
    """False for rotated text — the arXiv stamp down the left margin is 90°-rotated marginalia and would
    otherwise be picked up as the paper's title (it is set in a larger font than the title)."""
    dx, dy = line.get("dir", (1, 0))
    return abs(dx) > 0.9 and abs(dy) < 0.1


def span_tex(span: dict) -> str:
    """One span → LaTeX, honouring the font's meaning (script/blackboard/big-op) then symbols."""
    font, txt = span.get("font", ""), unmangle(span.get("text", ""))
    if any(font.startswith(f) for f in BB_FONTS):
        core = "".join(ch for ch in txt if ch not in DROP_GLYPHS)
        return "".join(rf"\mathbb{{{ch}}}" if ch.isalpha() else tex_atoms(ch) for ch in core)
    if any(font.startswith(f) for f in SCRIPT_FONTS):
        core = "".join(ch for ch in txt if ch not in DROP_GLYPHS)
        return "".join(rf"\mathcal{{{ch}}}" if ch.isalpha() else tex_atoms(ch) for ch in core)
    if font.startswith("txex"):
        return "".join(BIG_OP.get(ch, "" if ch in DROP_GLYPHS or ch.isalpha() else ch) for ch in txt)
    return tex_atoms(txt)


def baseline_bands(spans: list, body_size: float, tol: float = 3.0) -> list:
    """Group a display-math block's spans into baseline BANDS (one per visual row of the formula).

    PyMuPDF hands back sub/superscripts as their own "lines", so a per-line baseline is wrong: a
    subscript-only line looks like a full row and its `t` would be emitted as `t`, not `_{t}`. Anchoring
    on the block's full-size baselines and attaching small spans to the nearest anchor fixes that.
    Returns [(anchor_y, [spans sorted by x]), ...] ordered top→bottom.
    """
    anchors = sorted({round(s["origin"][1], 1) for s in spans if s["size"] >= 0.92 * body_size})
    merged: list = []
    for a in anchors:                                        # collapse anchors of the same visual row
        if merged and a - merged[-1] <= tol:
            continue
        merged.append(a)
    if not merged:
        merged = [round(min(s["origin"][1] for s in spans), 1)]
    bands: dict = {a: [] for a in merged}
    for s in spans:
        y = s["origin"][1]
        bands[min(merged, key=lambda a: abs(a - y))].append(s)
    return [(a, sorted(bands[a], key=lambda s: s["origin"][0])) for a in merged if bands[a]]


def line_to_latex(spans: list, body_size: float, base_y: float | None = None) -> str:
    """Rebuild one baseline band as LaTeX, recovering sub/superscripts from span geometry.

    A smaller span sitting BELOW the band's baseline is a subscript, ABOVE is a superscript — that is
    what turns the flat `Wt+1` of naive text extraction into `W_{t+1}`.
    """
    spans = [s for s in spans if s.get("text", "").strip() or s.get("text") == " "]
    if not spans:
        return ""
    if base_y is None:
        base_y = max((s["origin"][1] for s in spans if s["size"] >= 0.92 * body_size),
                     default=spans[0]["origin"][1])
    out, run, run_kind = [], [], None

    def flush():
        nonlocal run, run_kind
        if run:
            inner = "".join(run).strip()
            if inner:
                out.append(f"_{{{inner}}}" if run_kind == "sub" else f"^{{{inner}}}")
            run, run_kind = [], None

    for s in spans:
        small = s["size"] < 0.9 * body_size
        kind = None
        if small:
            kind = "sub" if s["origin"][1] > base_y + 0.15 else "sup"
        tex = span_tex(s)
        if kind is None:
            flush()
            out.append(tex)
        else:
            if run_kind and kind != run_kind:
                flush()
            run_kind = kind
            run.append(tex)
    flush()
    s = "".join(out)
    s = re.sub(r"\s{2,}", " ", s).strip()
    return s


def dehyphenate(lines: list) -> str:
    """Join a paragraph's lines: `neuro-\\nplasticity` → `neuroplasticity`, else a single space."""
    buf = ""
    for ln in lines:
        ln = ln.rstrip()
        if not buf:
            buf = ln
        elif buf.endswith("-") and not buf.endswith("--"):
            buf = buf[:-1] + ln.lstrip()
        else:
            buf += " " + ln.lstrip()
    return re.sub(r"\s{2,}", " ", buf).strip()


def score_md(md: str) -> dict:
    """Quality proxy used by `action="bench"` — how much STRUCTURE survived the conversion."""
    return {
        "chars": len(md),
        "headings": len(re.findall(r"(?m)^#{1,4} ", md)),
        "equations": len(re.findall(r"\$\$", md)) // 2 + len(re.findall(r"(?m)^\\\[", md)),
        "eq_tags": len(re.findall(r"\\tag\{", md)),
        "images": len(re.findall(r"!\[", md)),
        "tables": len(re.findall(r"(?m)^\|.*\|$", md)),
        "latex_cmds": len(re.findall(r"\\[a-zA-Z]{2,}", md)),
        "mojibake": len(re.findall(r"[\ue000-\uf8ff\x00-\x08]", md)),
    }


def bind_captions(figures: list, captions: list) -> list:
    """Attach each 'Figure N:' caption to the image whose bbox is nearest ABOVE it on the same page."""
    for cap in captions:
        best, best_d = None, 1e9
        for fig in figures:
            if fig["page"] != cap["page"] or fig.get("caption"):
                continue
            d = abs(cap["y0"] - fig["y1"])
            if d < best_d:
                best, best_d = fig, d
        if best is not None:
            best["caption"] = cap["text"]
            best["label"] = cap["label"]
    return figures


# ---------------------------------------------------------------- I/O: fetch
def fetch_pdf(pdf: str, dest_dir: Path) -> Path:
    """Accept a URL or a local path (docs/papers/*.pdf included); return a local PDF path."""
    if re.match(r"^https?://", pdf):
        dest_dir.mkdir(parents=True, exist_ok=True)
        out = dest_dir / "source.pdf"
        if not out.exists() or out.stat().st_size == 0:
            import urllib.request
            req = urllib.request.Request(pdf, headers={"User-Agent": "Mozilla/5.0 (paper-md)"})
            with urllib.request.urlopen(req, timeout=120) as r, open(out, "wb") as f:
                f.write(r.read())
        return out
    p = Path(pdf)
    for cand in (p, COMP / pdf, COMP / "docs" / "papers" / pdf):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"PDF not found: {pdf!r}")


# ---------------------------------------------------------------- backend: pymupdf (in-house)
def extract_pymupdf(pdf: Path, out: Path, dpi: int = 220, eq_images: bool = True,
                    tables: bool = True) -> dict:
    """Structure-preserving extraction: sections, paragraphs, LaTeX equations, figures, tables, crops."""
    import fitz

    doc = fitz.open(pdf)
    (out / "assets" / "fig").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "eq").mkdir(parents=True, exist_ok=True)
    (out / "assets" / "tab").mkdir(parents=True, exist_ok=True)
    zoom = dpi / 72.0

    # PASS 1 — body font size (the most common) and the sizes headings actually use in THIS template.
    # Absolute thresholds do not transfer: NL's sections are 14.3pt on a 10pt body, an arXiv template's
    # are 12pt on the same body, so the heading level has to come from the observed distribution.
    sizes: dict = {}
    for page in doc:
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                for sp in ln["spans"]:
                    sizes[round(sp["size"], 1)] = sizes.get(round(sp["size"], 1), 0) + len(sp["text"])
    body_size = max(sizes, key=sizes.get) if sizes else 10.0
    head_pages: dict = {}
    for pno_, page in enumerate(doc, start=1):
        for blk in page.get_text("dict")["blocks"]:
            for ln in blk.get("lines", []):
                if not horizontal(ln):
                    continue
                for sp in ln["spans"]:
                    if is_bold(sp) and sp["size"] > body_size + 0.4 and len(sp["text"].strip()) > 2:
                        head_pages.setdefault(round(sp["size"], 1), set()).add(pno_)
    # A §-heading size RECURS across pages; the title occurs once, in a larger font. Taking the max
    # would make the title define the scale and demote every real section (seen on a DeepMind template:
    # title 18.9pt bold, sections 13.0pt bold).
    recurring = [z for z, pages in head_pages.items() if len(pages) >= 2]
    top_size = max(recurring) if recurring else (max(head_pages) if head_pages else body_size + 3.5)

    md, equations, figures, captions, tbls, sections = [], [], [], [], [], []
    title = ""
    for pno, page in enumerate(doc, start=1):
        pending_nums: list = []          # right-margin "(19)" blocks, bound to their formula per page
        # --- raster figures, extracted losslessly, with their bbox for caption binding
        for i, info in enumerate(page.get_image_info(xrefs=True)):
            xref = info.get("xref")
            if not xref:
                continue
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if pix.width < 60 or pix.height < 60:
                    continue
                rel = f"assets/fig/p{pno:02d}_{i}.png"
                pix.save(out / rel)
                bb = info["bbox"]
                figures.append(dict(page=pno, path=rel, y0=bb[1], y1=bb[3], caption="", label="",
                                    w=pix.width, h=pix.height))
            except Exception:  # noqa: BLE001
                continue

        blocks = sorted([b for b in page.get_text("dict")["blocks"] if b.get("lines")],
                        key=lambda b: (round(b["bbox"][1], 1), round(b["bbox"][0], 1)))
        for blk in blocks:
            lines = [ln for ln in blk["lines"] if horizontal(ln)]     # drop rotated marginalia
            if not lines:
                continue
            flat = [(("".join(sp["text"] for sp in ln["spans"])), ln) for ln in lines]
            text = unmangle(" ".join(t for t, _ in flat)).strip()
            if not text or re.fullmatch(r"\d{1,3}", text):          # page number / empty
                continue
            spans = [sp for ln in lines for sp in ln["spans"]]
            big_bold = [sp for sp in spans if is_bold(sp) and sp["size"] > body_size + 0.4]
            mathish = sum(len(sp["text"]) for sp in spans if is_math_font(sp["font"]))
            total = max(1, sum(len(sp["text"]) for sp in spans))

            if not title and any(sp["size"] > top_size + 0.6 for sp in spans) and len(text) > 15:
                title = text
                md.append(f"# {text}\n")
                continue
            if big_bold:                                             # ---- section heading
                # heading text = the BIG-BOLD spans only: a block can merge an epigraph with the
                # heading that follows it ("… — Albert Einstein  1 Introduction").
                parts, prev = [], None
                for sp in big_bold:                                  # re-insert the spaces the PDF drops
                    if prev is not None and (sp["bbox"][0] - prev["bbox"][2] > 1.0
                                             or sp["bbox"][1] - prev["bbox"][1] > 1.0):   # wrapped line
                        parts.append(" ")
                    parts.append(unmangle(sp["text"]))
                    prev = sp
                htext = re.sub(r"\s{2,}", " ", "".join(parts)).strip()
                # a 1-2 character "heading" is a dropped cap or a stray label (an arXiv template sets
                # ABSTRACT as a bold A + small caps), never a section
                if not htext or len(htext) > 140 or len(htext.strip(" .")) < 3:
                    md.append(dehyphenate([unmangle(t) for t, _ in flat]) + "\n")
                    continue
                # LEVEL BY FONT SIZE, not by the numbering: `1 Introduction` and `B Adam …` are both
                # top-level (14.3pt) while `4.2 …` is a subsection (12pt) — numbering alone gets the
                # lettered appendices and any un-numbered top section wrong.
                top = max(sp["size"] for sp in big_bold) >= top_size - 0.3
                num = re.match(r"^([0-9]+(?:\.[0-9]+)*|[A-Z])\.?\s+(.*)$", htext)   # '1.' and 'A.' too
                md.append(f"\n{'##' if top else '###'} {htext}\n")
                sections.append(dict(level=1 if top else 2, page=pno,
                                     num=(num.group(1) if num else ""),
                                     title=(num.group(2) if num else htext).strip(), heading=htext))
                continue
            cap = CAPTION_RE.match(text)
            if cap:
                captions.append(dict(page=pno, y0=blk["bbox"][1], text=text,
                                     label=f"{cap.group(1).title()} {cap.group(2)}"))
                continue
            if EQNUM_RE.match(text):                                # a lone "(19)" = a right-margin
                pending_nums.append((blk["bbox"][1], blk["bbox"][3], text[1:-1]))   # equation number
                continue
            if mathish / total > 0.30:                              # ---- display equation block
                tag = ""
                for t, _ln in flat:                                 # …or the number sits inside the block
                    t_clean = unmangle(t).strip()
                    m = EQNUM_RE.match(t_clean) or re.search(r"\((\d+)\)\s*$", t_clean)
                    if m:
                        tag = m.group(1)
                tex = " ".join(x for x in (line_to_latex(band, body_size, anchor)
                                           for anchor, band in baseline_bands(spans, body_size)) if x)
                if tag:
                    tex = re.sub(r"\s*\(" + tag + r"\)\s*$", "", tex).strip() + rf" \tag{{{tag}}}"
                img_rel = ""
                if eq_images:
                    img_rel = f"assets/eq/eq_p{pno:02d}_{len(equations):03d}.png"
                    clip = fitz.Rect(blk["bbox"]) + (-6, -4, 6, 4)
                    page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip).save(out / img_rel)
                equations.append(dict(n=tag or None, page=pno, latex=tex, raw=text, image=img_rel,
                                      y0=blk["bbox"][1], y1=blk["bbox"][3]))
                md.append(f"\n$$\n{tex}\n$$\n")
                if img_rel:
                    md.append(f'<sub>[rendered]({img_rel})</sub>\n')
                continue
            md.append(dehyphenate([unmangle(t) for t, _ in flat]) + "\n")

        # bind each right-margin number to the formula it sits beside (vertical-centre nearest match)
        for y0, y1, num in pending_nums:
            mid = (y0 + y1) / 2
            cands = [e for e in equations if e["page"] == pno and not e["n"]]
            if not cands:
                continue
            best = min(cands, key=lambda e: abs((e["y0"] + e["y1"]) / 2 - mid))
            if abs((best["y0"] + best["y1"]) / 2 - mid) < 60:
                best["n"] = num
                best["latex"] = best["latex"].rstrip() + rf" \tag{{{num}}}"

        if tables:                                                  # ---- tables via layout analysis
            try:
                for ti, tb in enumerate(page.find_tables().tables):
                    rows = tb.extract()
                    if not rows or len(rows) < 2:
                        continue
                    rel = f"assets/tab/p{pno:02d}_{ti}.png"
                    page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=fitz.Rect(tb.bbox)).save(out / rel)
                    hdr = [unmangle(c or "") for c in rows[0]]
                    body = [[unmangle(c or "").replace("|", "\\|") for c in r] for r in rows[1:]]
                    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
                    lines += ["| " + " | ".join(r) + " |" for r in body]
                    tbls.append(dict(page=pno, rows=len(rows), cols=len(hdr), image=rel))
                    md.append("\n" + "\n".join(lines) + f"\n\n<sub>[rendered]({rel})</sub>\n")
            except Exception:  # noqa: BLE001
                pass

    figures = bind_captions(figures, captions)
    fig_md = ["\n## Figures\n"]
    for f in figures:
        cap = f.get("caption") or f"page {f['page']}"
        fig_md.append(f"\n![{cap[:80]}]({f['path']})\n\n*{cap}*\n")
    body_md = "".join(x if x.endswith("\n") else x + "\n" for x in md)
    toc = ["\n## Contents\n"] + [
        f"{'  ' * (s['level'] - 1)}- {s['num']} {s['title']} <sub>p{s['page']}</sub>" for s in sections] + [""]
    return dict(title=title, markdown="\n".join(toc) + body_md + "".join(fig_md), equations=equations,
                figures=figures, tables=tbls, sections=sections, pages=doc.page_count, body_size=body_size)


# ---------------------------------------------------------------- backends: model-based (isolated venv)
def _marker_py() -> Path:
    return Path(os.environ.get("PAPER_MD_MARKER_PY")
                or (COMP.parent / "external" / "venv-paper2md" / "bin" / "python"))


def marker_available() -> bool:
    py = _marker_py()
    if not py.exists():
        return False
    r = subprocess.run([str(py), "-c", "import marker"], capture_output=True, timeout=180)
    return r.returncode == 0


def marker_convert(pdf: Path, out: Path, timeout: int = 7200, mode: str = "balanced") -> str:
    """marker-pdf in its own venv → markdown text (LaTeX equations). Never imported in-process.

    `mode=balanced` is marker's GPU path (VLM layout model + full-page OCR) — the quality that makes
    equations come out as real LaTeX. Cached: a second call reuses the produced .md.
    """
    py = _marker_py()
    work = out / "_marker"
    work.mkdir(parents=True, exist_ok=True)
    hits = sorted(work.rglob("*.md"))
    if hits:
        return hits[0].read_text(errors="replace")
    cli = py.parent / "marker_single"
    cmd = ([str(cli)] if cli.exists() else [str(py), "-m", "marker.scripts.convert_single"])
    cmd += [str(pdf), "--output_dir", str(work), "--output_format", "markdown", "--mode", mode]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    hits = sorted(work.rglob("*.md"))
    if not hits:
        raise RuntimeError(f"marker produced no markdown (rc={r.returncode}): {r.stderr[-400:]}")
    return hits[0].read_text(errors="replace")


def _docling_py() -> Path:
    """Interpreter that owns docling (kept out-of-process for the same ABI reason as marker)."""
    return Path(os.environ.get("PAPER_MD_DOCLING_PY")
                or "/home/seshu/miniconda3/envs/llm/bin/python")


def docling_available() -> bool:
    py = _docling_py()
    if not py.exists():
        try:
            import docling  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False
    return subprocess.run([str(py), "-c", "import docling"], capture_output=True, timeout=180).returncode == 0


def docling_convert(pdf: Path, out: Path | None = None, formulas: bool = True,
                    timeout: int = 7200) -> str:
    """docling with LOCAL formula enrichment — the no-docker path to real LaTeX.

    MEASURED on NL.pdf (2026-07-29): plain docling leaves `<!-- formula-not-decoded -->` for all 121
    numbered equations; with `do_formula_enrichment` (CodeFormula runs locally, no vLLM/docker) it emits
    `$$W_{t+1} = W_t - \\eta_t\\nabla_{W_t}\\mathcal{L}(W_t; x_t), && (1)$$` — 0 undecoded. marker's
    `balanced`/`fast` modes both route OCR through surya→vLLM→docker, so they are NOT used here.
    """
    def _inproc() -> str:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
        opts = PdfPipelineOptions()
        opts.do_formula_enrichment = formulas
        opts.do_table_structure = True
        conv = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)})
        return conv.convert(str(pdf)).document.export_to_markdown()

    py = _docling_py()
    if not py.exists():
        return _inproc()
    cache = (out / "_docling.md") if out else None
    if cache and cache.exists() and cache.stat().st_size > 1000:
        return cache.read_text(errors="replace")
    script = ("import sys;from docling.datamodel.base_models import InputFormat;"
              "from docling.datamodel.pipeline_options import PdfPipelineOptions;"
              "from docling.document_converter import DocumentConverter,PdfFormatOption;"
              f"o=PdfPipelineOptions();o.do_formula_enrichment={bool(formulas)};o.do_table_structure=True;"
              "c=DocumentConverter(format_options={InputFormat.PDF:PdfFormatOption(pipeline_options=o)});"
              "open(sys.argv[2],'w').write(c.convert(sys.argv[1]).document.export_to_markdown())")
    dst = cache or (pdf.parent / "_docling.md")
    r = subprocess.run([str(py), "-c", script, str(pdf), str(dst)], capture_output=True, text=True,
                       timeout=timeout)
    if not dst.exists() or dst.stat().st_size < 100:
        raise RuntimeError(f"docling produced nothing (rc={r.returncode}): {r.stderr[-300:]}")
    return dst.read_text(errors="replace")


def tag_equations(md: str) -> str:
    """docling writes an equation number as LaTeX alignment junk (`, && ( 1 )`); make it a real
    `\\tag{1}` so the number survives into MathJax and `score_md` can count it."""
    md = re.sub(r",?\s*&\s*&\s*\(\s*(\d+)\s*\)", lambda m: rf" \tag{{{m.group(1)}}}", md)   # `, & & ( 1 )`
    md = re.sub(r"\\q?quad\s*\(\s*(\d+)\s*\)", lambda m: rf" \tag{{{m.group(1)}}}", md)     # `\quad (1)`
    return re.sub(r"\\tag\{(\d+)\}\s*\\\\", lambda m: rf"\tag{{{m.group(1)}}}", md)


def hybrid_merge(model_md: str, mu: dict) -> str:
    """Model prose+LaTeX first (best equations), then our lossless asset appendix (nothing lost).

    Only NUMBERED formulas are linked in the appendix — every block still has a crop on disk, but the
    paper's own numbering is what a reader navigates by.
    """
    parts = [tag_equations(model_md).rstrip(), "\n\n---\n\n## Assets extracted losslessly (paper-md/pymupdf)\n"]
    for f in mu["figures"]:
        cap = f.get("caption") or f"page {f['page']}"
        parts.append(f"\n![{cap[:80]}]({f['path']})\n\n*{cap}*\n")
    numbered = [e for e in mu["equations"] if e.get("n") and e.get("image")]
    if numbered:
        parts.append(f"\n### Equation crops — every numbered formula as rendered ({len(numbered)})\n")
        for e in numbered:
            parts.append(f"\n**({e['n']})** p{e['page']} ![eq]({e['image']})\n")
    return "".join(parts)


# ---------------------------------------------------------------- the agent
class PaperMd(BaseAgent):
    name = "paper-md"
    thread = "S"
    kind = "finding"

    DEFAULT_OUT = "docs/papers"

    def run(self, q, worker):
        spec = self.spec(q)
        pdf_in = spec.get("pdf") or spec.get("url") or spec.get("path")
        if not pdf_in:
            return self.escalate(worker, "leader",
                                 f"[{worker}] paper-md needs `spec.pdf` (a URL or a local PDF path).")
        action = spec.get("action", "convert")
        backend = spec.get("backend", "auto")
        dpi = int(spec.get("dpi", 220))
        eq_images = bool(spec.get("eq_images", True))
        outroot = Path(spec.get("outdir") or self.DEFAULT_OUT)
        outroot = outroot if outroot.is_absolute() else COMP / outroot
        slug = slugify(spec.get("slug") or Path(str(pdf_in)).stem)
        out = outroot / slug
        out.mkdir(parents=True, exist_ok=True)
        pdf = fetch_pdf(str(pdf_in), out)

        mu = extract_pymupdf(pdf, out, dpi=dpi, eq_images=eq_images,
                             tables=bool(spec.get("tables", True)))
        avail = {"pymupdf": True, "marker": marker_available(), "docling": docling_available()}

        if action == "bench":
            scores = {"pymupdf": score_md(mu["markdown"])}
            for name, fn in (("docling", lambda: docling_convert(pdf, out)),
                             ("marker", lambda: marker_convert(pdf, out))):
                if not avail[name]:
                    continue
                try:
                    scores[name] = score_md(fn())
                except Exception as e:  # noqa: BLE001
                    scores[name] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            (out / "bench.json").write_text(json.dumps(scores, indent=2))
            best = max((k for k in scores if "error" not in scores[k]),
                       key=lambda k: (scores[k]["eq_tags"], scores[k]["equations"], scores[k]["headings"]))
            rows = " · ".join(f"{k}(eq={v.get('equations','?')},tags={v.get('eq_tags','?')},"
                              f"head={v.get('headings','?')})" for k, v in scores.items())
            msg = (f"[{worker}] PAPER-MD bench on `{pdf.name}` ({mu['pages']}p): {rows} → **best={best}**. "
                   f"Scores cached at {out.relative_to(COMP) if str(out).startswith(str(COMP)) else out}/bench.json.")
            self.post(worker, "all", msg)
            return self.done({"scores": scores, "best": best, "outdir": str(out)}, msg)

        chosen = backend
        if backend == "auto":
            chosen = "hybrid" if (avail["docling"] or avail["marker"]) else "pymupdf"
        md = mu["markdown"]
        used = "pymupdf"
        if chosen in ("marker", "docling", "hybrid"):
            model_md, used_model = "", ""
            order = {"marker": ["marker"], "docling": ["docling"]}.get(chosen, ["docling", "marker"])
            for name in order:                       # docling FIRST: it needs no docker (marker does)
                if not avail.get(name):
                    continue
                try:
                    model_md = marker_convert(pdf, out) if name == "marker" else docling_convert(pdf, out)
                    used_model = name
                    break
                except Exception as e:  # noqa: BLE001
                    self.post(worker, "all", f"[{worker}] paper-md {name} backend failed "
                                             f"({type(e).__name__}: {str(e)[:100]}) — falling back.", routine=True)
            if model_md:
                md = hybrid_merge(model_md, mu) if chosen == "hybrid" else model_md
                used = f"{used_model}+pymupdf-assets" if chosen == "hybrid" else used_model

        head = (f"> **Source:** `{pdf.name}` · {mu['pages']} pages · "
                f"{len(mu['figures'])} figures · {len(mu['equations'])} display equations · "
                f"{len(mu['tables'])} tables · converted by fleet `paper-md` (backend=**{used}**)\n")
        md_path = out / f"{slug}.md"
        md_path.write_text(head + "\n" + md)
        (out / "equations.json").write_text(json.dumps(mu["equations"], indent=2))
        manifest = dict(slug=slug, source=str(pdf_in), pdf=str(pdf), title=mu["title"], backend=used,
                        pages=mu["pages"], figures=mu["figures"], tables=mu["tables"],
                        sections=mu["sections"], equations=len(mu["equations"]),
                        numbered=sum(1 for e in mu["equations"] if e["n"]),
                        md=str(md_path), equations_json=str(out / "equations.json"),
                        available=avail, score=score_md(md))
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        st = self.load_state({}) or {}
        st[slug] = {k: manifest[k] for k in ("backend", "pages", "equations", "numbered", "md")}
        self.save_state(st)

        rel = md_path.relative_to(COMP) if str(md_path).startswith(str(COMP)) else md_path
        msg = (f"[{worker}] PAPER-MD ✅ `{mu['title'][:70] or pdf.name}` → `{rel}` "
               f"(backend **{used}**, {mu['pages']}p, {len(mu['figures'])} figures, "
               f"{len(mu['equations'])} equations of which {manifest['numbered']} numbered, "
               f"{len(mu['tables'])} tables; assets in `{out.name}/assets/`).")
        self.post(worker, "all", msg)
        self.log(f"paper-md: {slug}", detail=json.dumps(manifest["score"]), kind="finding")
        return self.done(manifest, msg)


_AGENT = PaperMd()


def run(q, worker):
    return _AGENT.run(q, worker)


if __name__ == "__main__":                     # CLI: python -m fleet_agents.paper_md <pdf> [slug] [backend]
    argv = sys.argv[1:]
    sp = {"pdf": argv[0] if argv else ""}
    if len(argv) > 1:
        sp["slug"] = argv[1]
    if len(argv) > 2:
        sp["backend"] = argv[2]
    if len(argv) > 3:
        sp["action"] = argv[3]
    print(run({"spec": sp}, "cli")[3])
