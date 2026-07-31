"""paper_learn_test — DATA-WISE verifier for the paper→lessons logic (no PDF, no network).

What must hold for a lesson series to be trustworthy:
  * every numbered formula of the paper lands in exactly one lesson (`coverage.missing == []`),
  * the PACK's hand-checked LaTeX wins over the extractor's best-effort text,
  * a formula with a proof gets a runnable code cell; one without gets a VISIBLE TODO (never silence),
  * the shared HEADER is injected once per lesson, not once per cell,
  * `cells_ran` detects a code cell that produced no output (the honesty check),
  * `learner.render_lesson` emits the hub's exact template (`@ id/order/title` then `--- note`),
  * `curriculum_add` appends a section and is idempotent on re-run.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

from fleet_agents import learner, paper_learn as P  # noqa: E402


def _run():
    print("=== PAPER-LEARN PURE-LOGIC VERIFIER ===")
    equations = [dict(n="1", page=2, latex=r"a=b", image="assets/eq/e1.png"),
                 dict(n="3", page=4, latex=r"c=d", image="assets/eq/e3.png"),
                 dict(n=None, page=4, latex=r"junk", image="")]
    manifest = dict(slug="demo", title="Demo paper", pages=6, sections=[
        dict(level=1, num="1", title="Intro", page=1), dict(level=1, num="2", title="Method", page=3)],
        figures=[dict(page=2, path="assets/fig/p02_0.png", caption="Figure 1: a fig", label="Figure 1")])
    EQ = {1: dict(name="First", latex=r"x=1", why="because", code="print('proof 1')"),
          2: dict(name="Second", latex=r"x=2", why="because")}          # 2 has NO proof
    total = 3
    pages = P.eq_pages(equations, total)
    plan = learner.plan_from_paper(manifest, "dm", order_base=100)
    assigned = set()
    for it in plan:
        p0, p1 = it["pages"]
        it["eqs"] = [n for n, pg in pages.items() if p0 <= pg <= p1 and n not in assigned]
        assigned |= set(it["eqs"])
    leftover = [n for n in range(1, total + 1) if n not in assigned]
    if leftover:
        plan[-1]["eqs"] = sorted(plan[-1]["eqs"] + leftover)
    cov = P.coverage(manifest, equations, EQ, plan, total)

    crops = {int(e["n"]): e for e in equations if e.get("n")}
    cells = P.eq_cells(plan[0]["eqs"] + plan[1]["eqs"], EQ, crops, "", "learning/assets/demo")
    figs = P.fig_cells(manifest["figures"], "learning/assets/demo")
    hdr_cells = P.with_header([{"note": "n0"}, {"note": "n1", "code": "a = 1"},
                               {"note": "n2", "code": "b = 2"}], "import torch")

    checks = {
        "eq_pages_covers_every_number": sorted(pages) == [1, 2, 3],
        "eq_pages_interpolates_gap": pages[2] >= pages[1],
        "coverage_nothing_missing": cov["missing"] == [] and cov["placed"] == total,
        "coverage_counts_taught": cov["taught"] == 2 and cov["untaught"] == [3],
        "coverage_counts_proven": cov["proven"] == 1 and cov["unproven"] == [2, 3],
        "coverage_counts_crops": cov["cropped"] == 2,
        "pack_latex_overrides_extractor": "x=1" in cells[0]["note"] and "a=b" not in cells[0]["note"],
        "extractor_latex_used_when_pack_silent": any("c=d" in c["note"] for c in cells),
        "proof_becomes_a_code_cell": cells[0].get("code") == "print('proof 1')",
        "missing_proof_is_visible": all(c.get("code") is None for c in cells[1:]),
        "missing_why_is_a_TODO": any("TODO" in c["note"] for c in cells[1:]),
        "crop_attached_per_equation": "e1.png" in (cells[0].get("image") or ""),
        "figure_cell_keeps_caption": "Figure 1" in figs[0]["note"] and "p02_0.png" in figs[0]["image"],
        "header_injected_once": hdr_cells[1]["code"].startswith("import torch") and
                                not hdr_cells[2]["code"].startswith("import torch"),
        "header_not_added_to_note_only_cell": "code" not in hdr_cells[0] or hdr_cells[0].get("code") is None,
    }

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # the hub's template: header first, then blocks — and a lesson must be re-parseable
        doc = learner.render_lesson("t1", "Title", "note body", "print(1)", order=5,
                                    subtitle="sub", source="src.md")
        checks["render_has_meta_header"] = doc.startswith("@ id: t1\n@ order: 5\n@ title: Title")
        checks["render_keeps_note_and_code"] = "--- note\nnote body" in doc and "--- code\nprint(1)" in doc
        checks["render_cells_form"] = "--- image" in learner.render_lesson(
            "t2", "T", "", "", cells=[{"note": "n", "image": "a.png"}])

        # cells_ran: a code block with no following output must be reported as silent
        ok_file = td / "ok.learning"
        ok_file.write_text("@ id: x\n\n--- note\nn\n\n--- code\nprint(1)\n\n--- output\n```\n1\n```\n")
        bad_file = td / "bad.learning"
        bad_file.write_text("@ id: x\n\n--- note\nn\n\n--- code\nprint(1)\n\n--- note\nnext\n")
        checks["cells_ran_detects_output"] = P.cells_ran(ok_file) == {"code_cells": 1, "with_output": 1, "silent": 0}
        checks["cells_ran_detects_silence"] = P.cells_ran(bad_file)["silent"] == 1

        # curriculum_add: append, then be idempotent
        cur = td / "curriculum.yml"
        cur.write_text("sections:\n  - title: Old\n    lessons: [a1]\n")
        learner.curriculum_add("New sec", ["n1", "n2"], path=cur)
        first = cur.read_text()
        learner.curriculum_add("New sec", ["n1", "n2", "n3"], path=cur)
        second = cur.read_text()
        checks["curriculum_appends"] = "  - title: New sec\n    lessons: [n1, n2]\n" in first
        checks["curriculum_keeps_existing"] = "- title: Old" in second
        checks["curriculum_updates_in_place"] = second.count("- title: New sec") == 1 and "n3]" in second

        # config_row: a published architecture, normalised for comparison (no weights needed)
        row = P.config_row({"repo": "org/m", "model_type": "moe", "hidden_size": 2048,
                            "num_hidden_layers": 24, "num_attention_heads": 16, "num_key_value_heads": 4,
                            "n_routed_experts": 64, "num_experts_per_tok": 8, "vocab_size": 32000})
        checks["config_row_normalises_experts"] = row["experts"] == 64 and row["active_experts"] == 8
        checks["config_row_estimates_params"] = row["params_est_B"] > 0
        mc = P.models_cells([row], "docs/papers/demo/models")
        checks["models_cell_is_a_table"] = mc[0]["note"].count("|") > 6 and "config.json" in mc[0]["note"]

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  coverage → {json.dumps({k: cov[k] for k in ('total', 'placed', 'taught', 'proven', 'cropped')})}")
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
