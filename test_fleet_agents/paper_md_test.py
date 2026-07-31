"""paper_md_test — DATA-WISE verifier for the PURE conversion logic (no PDF, no network).

The value of paper-md is that a LaTeX-authored PDF survives extraction: math alphanumerics get
un-mangled (𝑊 → W), the script/blackboard fonts keep their meaning (txsys M → \\mathcal{M}, txsym R →
\\mathbb{R}), extensible-delimiter junk is dropped, and sub/superscripts are recovered from span
GEOMETRY so `W t+1` becomes `W_{t+1}`. Each of those is asserted here on synthetic spans that mimic
the real font/size/baseline pattern measured in the paper.
"""
import os
import sys

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

from fleet_agents import paper_md as P  # noqa: E402


def _span(text, font="LinLibertineT", size=10.0, x=0.0, y=100.0):
    return {"text": text, "font": font, "size": size, "origin": (x, y)}


def _run():
    print("=== PAPER-MD PURE-LOGIC VERIFIER ===")
    body = 10.0
    # W_{t+1} = W_t - eta_t * grad ... : subscripts are SMALLER spans sitting BELOW the baseline
    eq = [_span("W", "LibertineMathMI", 10.0, 0, 100.0),
          _span("t+1", "LibertineMathMI7", 7.3, 8, 101.6),
          _span(" = ", "txmiaX", 10.0, 20, 100.0),
          _span("W", "LibertineMathMI", 10.0, 30, 100.0),
          _span("t", "LibertineMathMI7", 7.3, 38, 101.6),
          _span("−", "txsys", 10.0, 44, 100.0),
          _span("𝜂", "LibertineMathMI", 10.0, 52, 100.0),
          _span("t", "LibertineMathMI7", 7.3, 59, 101.6),
          _span("∇", "txsys", 10.0, 66, 100.0),
          _span("L", "txsys", 10.0, 74, 100.0)]
    tex = P.line_to_latex(eq, body)
    # a superscript sits ABOVE the baseline (smaller y in PDF coords)
    sup = P.line_to_latex([_span("x", "LibertineMathMI", 10.0, 0, 100.0),
                           _span("2", "LibertineMathMI7", 7.3, 8, 97.0)], body)
    # a whole band that is subscript-only must still be a subscript (block-level anchoring)
    bands = P.baseline_bands(eq + [_span("i=1", "LibertineMathMI7", 7.3, 66, 108.0)], body)
    checks = {
        "unmangle_NFKC": P.unmangle("𝑊𝑡+1 = 𝜽") == "Wt+1 = θ",
        "greek_to_tex": P.tex_atoms("η θ α") == r"\eta \theta \alpha",
        "nabla_transpose": P.tex_atoms("∇⊤") == r"\nabla ^{\top}",
        "script_font_mathcal": P.span_tex(_span("M", "txsys")) == r"\mathcal{M}",
        "blackboard_font": P.span_tex(_span("R", "txsym")) == r"\mathbb{R}",
        "bigop_sum": P.span_tex(_span("Í", "txexs")) == r"\sum",
        "delimiter_junk_dropped": P.span_tex(_span("\x00\x01", "txexs")) == "",
        "subscript_recovered": "W_{t+1}" in tex,
        "second_subscript": "W_{t}" in tex,
        "eta_subscript": r"\eta_{t}" in tex,
        "superscript_recovered": sup == "x^{2}",
        "no_mojibake_in_tex": all(ord(c) < 0x2000 or c in "∥⟨⟩" for c in tex),
        "bands_sorted_top_down": [round(a) for a, _ in bands] == sorted(round(a) for a, _ in bands),
        "subscript_only_band_merged": len(bands) == 1,
        "dehyphenate_joins": P.dehyphenate(["neuro-", "plasticity is"]) == "neuroplasticity is",
        "dehyphenate_spaces": P.dehyphenate(["one", "two"]) == "one two",
        "slugify": P.slugify("Nested Learning: The Illusion!") == "nested-learning-the-illusion",
        "is_math_font": P.is_math_font("LibertineMathMI7") and not P.is_math_font("LinLibertineT"),
    }
    # caption binding: the caption below an image on the same page wins
    figs = [{"page": 3, "y0": 50, "y1": 200, "caption": "", "label": ""},
            {"page": 4, "y0": 50, "y1": 200, "caption": "", "label": ""}]
    caps = [{"page": 3, "y0": 205, "text": "Figure 2: the NL paradigm", "label": "Figure 2"}]
    bound = P.bind_captions(figs, caps)
    checks["caption_bound_same_page"] = bound[0]["label"] == "Figure 2" and bound[1]["caption"] == ""

    sc = P.score_md("# T\n\n$$\nx \\tag{1}\n$$\n\n![f](a.png)\n\n| a | b |\n")
    checks["score_counts"] = (sc["headings"], sc["equations"], sc["eq_tags"], sc["images"]) == (1, 1, 1, 1)
    checks["score_mojibake_zero"] = sc["mojibake"] == 0

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  latex → {tex}")
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
