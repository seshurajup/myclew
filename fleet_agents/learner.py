"""Learner agent — capture NEW knowledge as a Pattern-B lesson (.py pure code + .learning) and refresh.

The rule (user): whenever the team learns something new — a kept experiment, a finding, or a user
'teach me X' request — it becomes a lesson in the established Pattern B: a `.py` (PURE runnable code,
no prose) next to a `.learning` (all explanation + the code shown + REAL captured outputs), same
basename. This agent writes both and refreshes the `.learning` via learning/lessonkit.py so the :7777
hub shows it. Deterministic (no LLM writes the lesson content — it's handed in via the spec).

Two entry points:
  action="lesson"          {id,title,note,code}                → one Pattern-B pair (+ curriculum entry)
  action="paper-scaffold"  {manifest, prefix, section}          → a WHOLE well-designed lesson SERIES
        scaffolded from a `paper-md` conversion: one lesson per paper section, in paper order, each
        carrying that section's REAL numbered formulas (as `$$…$$` + the rendered crop) and its figures,
        with a runnable code slot per formula — then registered in learning/curriculum.yml so the hub
        renders the series. The deterministic half (structure, every formula, images, ordering,
        registration, refresh) is the agent's; the teaching prose + PyTorch stays authored.

Pattern B format is NOT free-form — the hub parses it (`_parse_learning`): an `@ key: value` header
(id/order/title/subtitle/source) then `--- note|code|output|image|shape` blocks. A lesson written
without that header loses its title and cannot be ordered, so `add_lesson()` emits the header itself.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent
LEARN_DIR = COMP / "learning" / "fleet_lessons"
LESSONKIT = COMP / "learning" / "lessonkit.py"
CURRICULUM = COMP / "learning" / "curriculum.yml"
KV = "/home/seshu/miniconda3/envs/kaggle_vision/bin/python"


def _refresh(learning: Path, py: str = KV) -> bool:
    """Run every `--- code` block in place and rewrite `--- output` with the REAL captured output."""
    if not LESSONKIT.exists():
        return False
    r = subprocess.run([py, str(LESSONKIT), str(learning)], cwd=str(COMP),
                       capture_output=True, text=True, timeout=600)
    return r.returncode == 0


def render_lesson(lesson_id: str, title: str, note: str, code: str, order: int = 999,
                  subtitle: str = "", source: str = "", cells: list | None = None) -> str:
    """Pure: build a valid Pattern-B `.learning` document (the hub's exact template).

    `cells` (optional) is the general form: [{note, code, image, shape}, ...]; `note`/`code` are the
    one-cell shorthand. The `@` header comes first — without it the hub has no id/order/title.
    """
    head = [f"@ id: {lesson_id}", f"@ order: {order}", f"@ title: {title}"]
    if subtitle:
        head.append(f"@ subtitle: {subtitle}")
    if source:
        head.append(f"@ source: {source}")
    body = []
    for c in (cells if cells is not None else [{"note": note, "code": code}]):
        body += ["--- note", (c.get("note") or "").strip(), ""]
        if c.get("code"):
            body += ["--- code", c["code"].rstrip(), ""]
        if c.get("image"):
            body += ["--- image", c["image"].strip(), ""]
        if c.get("shape"):
            body += ["--- shape", c["shape"].strip(), ""]
    return "\n".join(head) + "\n\n" + "\n".join(body) + "\n"


def add_lesson(lesson_id: str, title: str, note: str, code: str, order: int = 999,
               subtitle: str = "", source: str = "", cells: list | None = None,
               out_dir: Path | None = None, refresh: bool = True) -> dict:
    """Write the Pattern-B pair (<id>.py PURE code + <id>.learning) and refresh the `.learning`."""
    d = Path(out_dir) if out_dir else LEARN_DIR
    d.mkdir(parents=True, exist_ok=True)
    py = d / f"{lesson_id}.py"
    learning = d / f"{lesson_id}.learning"
    pure = code if cells is None else "\n\n".join(c["code"].rstrip() for c in cells if c.get("code"))
    py.write_text((pure or "").rstrip() + "\n")                       # PURE code only, no prose
    learning.write_text(render_lesson(lesson_id, title, note, code, order=order, subtitle=subtitle,
                                      source=source, cells=cells))
    return {"py": str(py), "learning": str(learning),
            "refreshed": bool(refresh and _refresh(learning))}


# ------------------------------------------------------------------ curriculum registration
def curriculum_add(section_title: str, lesson_ids: list, path: Path | None = None) -> dict:
    """Append/refresh a section in learning/curriculum.yml — the ONE ordering source the hub reads.

    Text-level edit on purpose: curriculum.yml is hand-maintained and commented; a yaml round-trip
    would drop those comments.
    """
    p = Path(path) if path else CURRICULUM
    ids = "[" + ", ".join(lesson_ids) + "]"
    block = f"  - title: {section_title}\n    lessons: {ids}\n"
    if not p.exists():
        p.write_text("sections:\n" + block)
        return {"created": True, "section": section_title, "lessons": len(lesson_ids)}
    txt = p.read_text()
    pat = re.compile(r"(?m)^  - title: " + re.escape(section_title) + r"\n    lessons: \[[^\]]*\]\n")
    if pat.search(txt):
        txt = pat.sub(block, txt)
        action = "updated"
    else:
        txt = txt.rstrip("\n") + "\n" + block
        action = "appended"
    p.write_text(txt)
    return {action: True, "section": section_title, "lessons": len(lesson_ids)}


# ------------------------------------------------------------------ paper → lesson series
def plan_from_paper(manifest: dict, prefix: str, min_eqs: int = 0, order_base: int = 1000,
                    step: int = 10, level: int = 1) -> list:
    """Pure: a `paper-md` manifest → an ordered lesson PLAN (one lesson per paper section).

    Each planned lesson owns the pages between its heading and the next heading of the same-or-higher
    level, and therefore owns every formula/figure on those pages: nothing in the paper is unassigned.
    """
    secs = [s for s in manifest.get("sections", []) if s.get("level", 1) <= level] or \
           manifest.get("sections", [])
    pages = int(manifest.get("pages") or 0)
    figs = manifest.get("figures", [])
    plan = []
    for i, s in enumerate(secs):
        p0 = int(s.get("page") or 1)
        p1 = (int(secs[i + 1].get("page") or p0) - 1) if i + 1 < len(secs) else pages
        p1 = max(p0, p1)
        lid = f"{prefix}{i + 1:02d}"
        plan.append(dict(id=lid, order=order_base + step * i, num=s.get("num", ""),
                         title=s.get("title") or s.get("heading", ""), pages=[p0, p1],
                         figures=[f for f in figs if p0 <= int(f.get("page", 0)) <= p1]))
    if min_eqs:                                                  # optional: drop formula-free sections
        plan = [x for x in plan if len(x.get("equations", [])) >= min_eqs] or plan
    return plan


def paper_cells(item: dict, equations: list, paper_title: str, md_rel: str, asset_rel: str) -> list:
    """Pure: the `--- note/code/image` cells for one planned lesson — every formula of its pages, in
    order, each as `$$latex$$` + its rendered crop + a runnable code slot to fill in."""
    p0, p1 = item["pages"]
    eqs = [e for e in equations if p0 <= int(e.get("page", 0)) <= p1]
    numbered = [e for e in eqs if e.get("n")]
    head = (f"## {item['num']} {item['title']}\n"
            f"**Paper:** {paper_title} · §{item['num']} · pages {p0}–{p1} · "
            f"{len(numbered)} numbered equation(s), {len(item['figures'])} figure(s).\n\n"
            f"> TODO(author): the *why* of this section in 3–5 lines, then teach each formula below.\n"
            f"> Source markdown: `{md_rel}`")
    cells = [{"note": head}]
    for f in item["figures"]:
        cells.append({"note": f"### {f.get('label') or 'Figure'} — as printed in the paper",
                      "image": f"{asset_rel}/{Path(f['path']).name}\n{f.get('caption', '')}"})
    for e in eqs:
        tag = f"Equation ({e['n']})" if e.get("n") else "Unnumbered formula"
        note = [f"### {tag}", "", f"$$\n{e.get('latex', '').replace(chr(10), ' ')}\n$$", "",
                "> TODO(author): what each symbol is, what it *does*, and why the paper needs it."]
        cell = {"note": "\n".join(note),
                "code": f"# TODO(author): the {tag.lower()} in runnable PyTorch, then assert it matches the paper.\n"}
        if e.get("image"):
            cell["image"] = f"{asset_rel}/eq/{Path(e['image']).name}\n{tag} as rendered in the PDF"
        cells.append(cell)
    cells.append({"note": "**[Recap]** TODO(author): 3 bullets — the claim, the mechanism, the "
                          "consequence for our own models."})
    return cells


def scaffold_from_paper(manifest_path, prefix: str, section_title: str, out_dir=None,
                        order_base: int = 1000, level: int = 1, refresh: bool = False,
                        assets_into: str | None = None) -> dict:
    """paper-md conversion → a complete, well-designed lesson SERIES on disk + registered in the
    curriculum. Returns the plan so the caller (or a researcher agent) knows exactly what to author."""
    mp = Path(manifest_path)
    manifest = json.loads(mp.read_text())
    eq_path = Path(manifest.get("equations_json") or (mp.parent / "equations.json"))
    equations = json.loads(eq_path.read_text()) if eq_path.exists() else []
    out = Path(out_dir) if out_dir else (COMP / "learning" / "annotated")
    slug = manifest.get("slug", prefix)
    # copy the paper's assets under the competition root so the hub can serve them via /asset/<slug>/…
    asset_root = Path(assets_into) if assets_into else (COMP / "learning" / "assets" / slug)
    asset_root.mkdir(parents=True, exist_ok=True)
    (asset_root / "eq").mkdir(exist_ok=True)
    import shutil
    src = mp.parent / "assets"
    for sub, dst in (("fig", asset_root), ("eq", asset_root / "eq"), ("tab", asset_root)):
        for f in sorted((src / sub).glob("*.png")) if (src / sub).exists() else []:
            shutil.copy2(f, dst / f.name)
    asset_rel = str(asset_root.relative_to(COMP))

    plan = plan_from_paper(manifest, prefix, order_base=order_base, level=level)
    md_rel = str(Path(manifest.get("md", "")).relative_to(COMP)) if manifest.get("md") else ""
    written = []
    for item in plan:
        cells = paper_cells(item, equations, manifest.get("title", slug), md_rel, asset_rel)
        res = add_lesson(item["id"], f"{item['num']} {item['title']}".strip(), "", "",
                         order=item["order"], subtitle=f"{manifest.get('title', slug)} §{item['num']}",
                         source=md_rel, cells=cells, out_dir=out, refresh=refresh)
        written.append({**item, **res})
    reg = curriculum_add(section_title, [x["id"] for x in plan])
    return {"lessons": len(written), "plan": [{k: x[k] for k in ("id", "num", "title", "pages")} for x in written],
            "assets": asset_rel, "curriculum": reg, "out_dir": str(out)}


# ------------------------------------------------------------------ fleet handler
def learn(q, worker):
    """Fleet handler — add a lesson, scaffold a whole paper series, or report the learner is ready."""
    spec = q.get("spec", {}) if isinstance(q, dict) else {}
    action = spec.get("action") or ("paper-scaffold" if spec.get("manifest") else
                                    ("lesson" if spec.get("title") and spec.get("code") else "status"))

    if action == "paper-scaffold":
        res = scaffold_from_paper(spec["manifest"], spec.get("prefix", "pp"),
                                  spec.get("section", "Paper study"),
                                  out_dir=spec.get("out_dir"), order_base=int(spec.get("order_base", 1000)),
                                  level=int(spec.get("level", 1)), refresh=bool(spec.get("refresh", False)))
        rows = ", ".join(f"{x['id']}(§{x['num']}, p{x['pages'][0]}–{x['pages'][1]})" for x in res["plan"][:6])
        return ("done", res, "all",
                f"[{worker}] LEARNER scaffolded **{res['lessons']} Pattern-B lessons** from the paper "
                f"({rows}{', …' if len(res['plan']) > 6 else ''}); every numbered formula placed as $$LaTeX$$ + "
                f"its PDF crop, figures copied to `{res['assets']}`, section registered in curriculum.yml "
                f"(shows on :7777). Author the TODO notes + PyTorch cells to finish.")

    if action == "lesson":
        res = add_lesson(spec.get("id", "fleet_lesson"), spec["title"], spec.get("note", ""), spec["code"],
                         order=int(spec.get("order", 999)), subtitle=spec.get("subtitle", ""),
                         source=spec.get("source", ""), out_dir=spec.get("out_dir"))
        if spec.get("section"):
            curriculum_add(spec["section"], [spec.get("id", "fleet_lesson")])
        return ("done", res, "all",
                f"[{worker}] LEARNER: added lesson '{spec['title']}' → {Path(res['learning']).name} "
                f"(Pattern B: pure .py + .learning{', refreshed' if res['refreshed'] else ''}; shows on :7777).")

    n = len(list(LEARN_DIR.glob("*.learning"))) if LEARN_DIR.exists() else 0
    n_all = len(list((COMP / "learning").rglob("*.learning")))
    return ("done", {"lessons": n, "lessons_all": n_all}, "all",
            f"[{worker}] LEARNER ready: any NEW finding → a Pattern-B lesson (pure .py + .learning with real "
            f"outputs, refreshed via lessonkit, shown on :7777). {n_all} lesson(s) in learning/ so far. "
            f"Hand me {{id,title,note,code}} for one lesson, or {{manifest,prefix,section}} from a `paper-md` "
            f"conversion and I'll scaffold the whole paper as a lesson series.")
