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
