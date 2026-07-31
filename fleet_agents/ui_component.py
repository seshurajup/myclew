"""ui_component — design-token → self-contained HTML component/dashboard generator, the fleet's answer to
facebook/astryx (a React + StyleX design system). We do NOT adopt astryx's TS/React runtime (a Node stack is
foreign to a Python ML fleet); we lift its LOAD-BEARING idea: a theme expressed as CSS custom properties
(design tokens) that a small set of composable components consume, so light/dark and restyling are one-line
token swaps. This produces a single dependency-free .html (inline CSS) — deterministic, offline, no Node —
for rendering a competition leaderboard, CV/experiment dashboard, or metric report the user can open or share.

Primitives (stdlib, no deps):
  • tokens(theme)                 — the CSS-custom-property design tokens (light/dark), astryx-style.
  • stat_card(label, value, ...)  — a KPI tile component (value + delta).
  • table(headers, rows)          — a styled data table (leaderboard / ledger).
  • capability_browser(caps, ...)  — openwork's UI argument: with hundreds of capabilities the
                                  interface is a SEARCH BOX, not a list. Offline, ~15 lines of inline JS.
  • repo_page(manifest, ...)  — an ADOPTED GitHub repo -> one page, driven entirely by the manifest
                                  paper-learn's repo mode already writes (so the next repo needs no code).
  • dashboard(title, cards, ...)  — compose cards + tables + a theme into one self-contained HTML string.
"""
from __future__ import annotations
import html as _html
from .base import BaseAgent

_THEMES = {
    "light": {"bg": "#ffffff", "fg": "#1a1a2e", "muted": "#6b7280", "accent": "#4f46e5",
              "card": "#f8f9fb", "border": "#e5e7eb", "pos": "#059669", "neg": "#dc2626"},
    "dark":  {"bg": "#0f1117", "fg": "#e5e7eb", "muted": "#9ca3af", "accent": "#818cf8",
              "card": "#1a1d29", "border": "#2d3140", "pos": "#34d399", "neg": "#f87171"},
}


def tokens(theme="light"):
    """Design tokens as a CSS-custom-property block (`:root{--bg:…}`), astryx theme-as-vars pattern."""
    t = _THEMES.get(theme, _THEMES["light"])
    return ":root{" + "".join(f"--{k}:{v};" for k, v in t.items()) + "}"


def stat_card(label, value, delta=None):
    """A KPI tile. delta (optional): signed number rendered green/red."""
    d = ""
    if delta is not None:
        cls = "pos" if float(delta) >= 0 else "neg"
        d = f'<div class="delta {cls}">{"+" if float(delta)>=0 else ""}{delta}</div>'
    return (f'<div class="card"><div class="label">{_html.escape(str(label))}</div>'
            f'<div class="value">{_html.escape(str(value))}</div>{d}</div>')


def table(headers, rows):
    """A styled data table (leaderboard/ledger). headers: list[str], rows: list[list]."""
    h = "".join(f"<th>{_html.escape(str(x))}</th>" for x in headers)
    body = "".join("<tr>" + "".join(f"<td>{_html.escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'

_CSS = """
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--fg);
font-family:system-ui,-apple-system,sans-serif;padding:24px;max-width:1100px;margin:auto}
h1{font-size:1.6rem;margin-bottom:16px}.row{display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px}
.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px 20px;
min-width:160px;flex:1}.label{color:var(--muted);font-size:.8rem;text-transform:uppercase;letter-spacing:.04em}
.value{font-size:1.8rem;font-weight:700;margin-top:4px}.delta{font-size:.85rem;margin-top:4px}
.pos{color:var(--pos)}.neg{color:var(--neg)}table{width:100%;border-collapse:collapse;
background:var(--card);border:1px solid var(--border);border-radius:12px;overflow:hidden}
th,td{padding:10px 14px;text-align:left;border-bottom:1px solid var(--border)}
th{color:var(--muted);font-size:.78rem;text-transform:uppercase}tr:last-child td{border-bottom:none}
td:first-child{font-weight:600;color:var(--accent)}
"""


def dashboard(title, cards=None, tables=None, theme="light"):
    """Compose a full self-contained HTML dashboard: theme tokens + KPI card row + tables. Returns HTML string.
    cards: list of (label,value[,delta]); tables: list of (title, headers, rows)."""
    card_html = "".join(stat_card(*c) for c in (cards or []))
    tbl_html = "".join(f'<h1>{_html.escape(tt)}</h1>{table(hd, rw)}' for (tt, hd, rw) in (tables or []))
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<style>{tokens(theme)}{_CSS}</style></head><body>"
            f"<h1>{_html.escape(title)}</h1><div class='row'>{card_html}</div>{tbl_html}</body></html>")


# ---------------------------------------------------------------- adopted-repo page (reusable)
def repo_page(manifest, units=None, results=None, lessons=None, theme="light", learn_url=None):
    """An adopted GitHub repo → one self-contained page, from the SAME design tokens as every other card.

    Reusable by construction: everything comes from the `manifest.json` that `paper-learn`'s repo mode
    already writes (`docs/repos/<slug>/manifest.json`), so any repo we adopt next gets a page with no new
    code. `results` is an optional list of (title, headers, rows) — for tabfm those are the repo's OWN
    benchmark parquets, which is why the caller labels them as the authors' numbers, not ours.
    """
    slug = manifest.get("slug", "repo")
    url = manifest.get("source", "")
    secs = manifest.get("sections", [])
    cards = [("units taught", str(len(units or []))),
             ("sections", str(len(secs))),
             ("lessons", str(len(lessons or []))),
             ("source", "GitHub")]
    tables = []
    if secs:
        tables.append(("Sections — each one a lesson",
                       ["#", "section", "units"],
                       [[s.get("num", ""), s.get("title", ""), s.get("units", "")] for s in secs]))
    if units:
        tables.append(("Every unit — an API plus the invariant we assert by calling it",
                       ["unit", "name"], [[u.get("n", ""), u.get("name", "")] for u in units]))
    for t in (results or []):
        tables.append(t)
    body = dashboard(manifest.get("title", slug), cards=cards, tables=tables, theme=theme)
    link_bar = (f"<div style='margin:-8px 0 18px'>"
                f"<a href='{_html.escape(url)}' rel='noopener' style='color:var(--accent);font-weight:600'>"
                f"{_html.escape(url)}</a>"
                + (f" &nbsp;·&nbsp; <a href='{_html.escape(learn_url)}' "
                   f"style='color:var(--accent);font-weight:600'>read the lessons →</a>" if learn_url else "")
                + "</div>")
    return body.replace("<div class='row'>", link_bar + "<div class='row'>", 1)


# ---------------------------------------------------------------- capability browser (openwork's UI idea)
def capability_browser(capabilities, title="Fleet capabilities", theme="light", instructions=""):
    """The one UI worth lifting from different-ai/openwork (https://github.com/different-ai/openwork):
    a searchable CAPABILITY list rather than a menu of tools.

    Their desktop app makes the same argument its MCP does — with hundreds of capabilities, the interface is
    a search box, not a list you scroll. We have 320 agents, so the same applies to us. Rendered from the
    same design tokens as every other card and, like everything here, self-contained: the filter is ~15
    lines of inline JS, so the page works offline with no framework.

    `capabilities`: [{"name","summary","domain","modalities","spec_schema","schema_digest"}] — exactly what
    `agent_routing.capability_index()` returns.
    """
    rows = []
    for c in capabilities:
        mods = ", ".join(c.get("modalities", [])[:4])
        schema = ", ".join(f"{k}:{v}" for k, v in list((c.get("spec_schema") or {}).items())[:4]) or "—"
        hay = _html.escape(" ".join([c.get("name", ""), c.get("summary", ""), c.get("domain", ""), mods]).lower())
        rows.append(
            f'<tr data-h="{hay}"><td><code>{_html.escape(c.get("name",""))}</code></td>'
            f'<td>{_html.escape(c.get("summary","")[:150])}</td>'
            f'<td>{_html.escape(c.get("domain",""))}</td>'
            f'<td>{_html.escape(mods)}</td><td><code>{_html.escape(schema)}</code></td>'
            f'<td><code>{_html.escape(c.get("schema_digest",""))}</code></td></tr>')
    head = "".join(f"<th>{h}</th>" for h in
                   ("capability", "what it does", "domain", "modalities", "spec schema", "digest"))
    note = (f"<p style='color:var(--muted);font-size:13px;white-space:pre-line'>"
            f"{_html.escape(instructions)}</p>" if instructions else "")
    return (f"<!doctype html><html><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<style>{tokens(theme)}{_CSS}"
            "#q{width:100%;padding:11px 14px;font-size:15px;border:1px solid var(--border);"
            "border-radius:10px;background:var(--card);color:var(--fg);margin:6px 0 14px}"
            "#n{color:var(--muted);font-size:13px;margin-bottom:10px}"
            "td code{font-size:12px}</style></head><body>"
            f"<h1>{_html.escape(title)}</h1>{note}"
            "<input id=q placeholder='Search capabilities — try: quantize, cross-validation, tabular, "
            "detection…' autofocus>"
            f"<div id=n>{len(rows)} capabilities</div>"
            f"<table><thead><tr>{head}</tr></thead><tbody id=b>{''.join(rows)}</tbody></table>"
            "<script>(function(){var q=document.getElementById('q'),b=document.getElementById('b'),"
            "n=document.getElementById('n'),rows=[].slice.call(b.rows);"
            "function f(){var t=q.value.trim().toLowerCase().split(/\\s+/).filter(Boolean),c=0;"
            "rows.forEach(function(r){var h=r.getAttribute('data-h'),"
            "ok=t.every(function(w){return h.indexOf(w)>=0});"
            "r.style.display=ok?'':'none';if(ok)c++;});"
            "n.textContent=c+' of '+rows.length+' capabilities';}"
            "q.addEventListener('input',f);})();</script></body></html>")


# ---------------------------------------------------------------- agent
class UIComponent(BaseAgent):
    name = "ui-component"
    thread = "S"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        from .base import AUTO
        AUTO.mkdir(parents=True, exist_ok=True)
        title = s.get("title", "Fleet Dashboard")
        cards = s.get("cards") or [("Best CV", "0.884", 0.003), ("LB", "0.887", None), ("Experiments", "42", None)]
        tables = s.get("tables") or [("Leaderboard", ["rank", "team", "score"],
                                      [[1, "kevin", 0.968], [2, "us", 0.887]])]
        theme = s.get("theme", "light")
        html = dashboard(title, cards, tables, theme)
        out = s.get("out") or str(AUTO / "ui_component_demo.html")
        open(out, "w").write(html)
        msg = (f"ui-component: wrote {len(html)}-byte self-contained HTML dashboard ({len(cards)} KPI cards + "
               f"{len(tables)} table(s), {theme} theme) → {out}. Design-token/CSS-vars components (astryx "
               f"theme-as-vars, no Node) — render leaderboards/CV dashboards/metric reports")
        self.log(msg, kind="finding",
                 recommendation="build dashboards from tokens()+stat_card()+table(); swap theme= for dark; "
                                "publish via the Artifact tool for a shareable page")
        return self.done({"path": out, "bytes": len(html), "n_cards": len(cards)}, msg)


_AGENT = UIComponent()


def run_ui(q, worker):
    return _AGENT.run(q, worker)
