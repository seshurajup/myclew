"""vizkit — the shared visual + explainability layer for paper lessons.

WHY this exists rather than more matplotlib per pack: a lesson about *memory* should let you look inside
the memory, and a lesson claiming "level 2 did the work" should be able to attribute the output to level 2.
Those are different jobs and each has a right tool. Surveyed and MEASURED in this env (2026-07-30):

  treescope      (DeepMind)  interactive, foldable, value-coloured view of a real tensor → 16 kB of
                            SELF-CONTAINED html. The design idea worth stealing: don't summarise an
                            array, render it. Used for "watch the delta rule erase the key direction".
  great_tables   (posit)     a table as a designed object (heat-shaded columns, spanners) → 9 kB
                            self-contained html. Replaces `DataFrame.to_html` for result tables.
  altair+vl_convert          Vega-Lite: declarative charts with real design defaults; `vl_convert`
                            renders them to PNG **offline** (no browser), and `to_html()` gives the
                            interactive version for the hub.
  captum         (Meta)      Integrated Gradients / layer attribution — the actual XAI step: which
                            level, memory or expert caused this output?
  torchview + graphviz       the architecture graph GENERATED from the real `nn.Module`, not drawn by
                            hand (falls back to a networkx layout when `dot` is absent).
  matplotlib                 kept for schematics and for anything that must be a plain PNG.

REJECTED, with reasons: circuitsvis (its html imports JS from unpkg at *view* time — a teaching visual
must not need the network); d3/ECharts/Superset (dashboard-shaped, not lesson-shaped); plotly (fine, but
Vega-Lite + vl_convert already gives interactive + offline PNG with one spec).

Everything returns either a path (PNG, for `--- image`) or an object with `_repr_html_` (inline in
`--- output`, thanks to lessonkit's rich-return support).
"""
from __future__ import annotations

from pathlib import Path

COMP = Path(__file__).resolve().parent.parent

# One palette for every lesson: colourblind-safe, dark-on-light, with a single accent for "ours".
INK, MUTE, GRID = "#1f2430", "#8a8f98", "#e7eaef"
ACCENT, GOOD, WARN, COOL = "#0b6cff", "#00a37a", "#d64545", "#7a5af8"
SERIES = [ACCENT, GOOD, WARN, COOL, "#e08b00", "#00a5c4"]
DIVERGING = "RdBu_r"          # for signed matrices (a memory), centred at 0
SEQUENTIAL = "viridis"        # for magnitudes


class Html:
    """Wrap an html string so a lesson cell can `return` it and lessonkit inlines it."""

    def __init__(self, html: str, note: str = ""):
        self.html, self.note = html, note

    def _repr_html_(self) -> str:
        head = (f'<div style="font:600 12px system-ui;color:{MUTE};margin:2px 0 6px">{self.note}</div>'
                if self.note else "")
        return head + self.html


def asset(rel: str) -> Path:
    """`assets/x.png` → an absolute path under the competition root, parents created."""
    p = COMP / rel if not str(rel).startswith("/") else Path(rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ------------------------------------------------------------------ look INSIDE a tensor
def tensor_view(t, note: str = "", max_side: int = 32) -> Html:
    """Interactive, value-coloured view of a real tensor (treescope). Self-contained html.

    This is the "explainable" primitive for this paper series: a memory `M` is a matrix, so show the
    matrix. Large tensors are sliced to `max_side` so the page stays readable.
    """
    import treescope
    small = t
    try:
        if hasattr(t, "shape") and len(t.shape) == 2:
            small = t[:max_side, :max_side]
        elif hasattr(t, "shape") and len(t.shape) == 1:
            small = t[:max_side * 4]
    except Exception:  # noqa: BLE001
        pass
    try:
        small = small.detach().float().cpu()
    except Exception:  # noqa: BLE001
        pass
    with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
        html = treescope.render_to_html(small, compressed=True)
    return Html(html, note or f"tensor {tuple(getattr(t, 'shape', ()))}")


def heat(t, path: str, note: str = "", signed: bool = True, vmax=None):
    """A matrix as a heatmap PNG — the static counterpart of `tensor_view`, for `--- image` blocks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    a = t.detach().float().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)
    v = float(np.abs(a).max()) if vmax is None else vmax
    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
    im = ax.imshow(a, cmap=DIVERGING if signed else SEQUENTIAL,
                   vmin=-v if signed else None, vmax=v)
    ax.set_title(note, fontsize=9, color=INK)
    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, shrink=0.85)
    p = asset(path); fig.savefig(p, dpi=150); plt.close(fig)
    return str(p.relative_to(COMP))


# ------------------------------------------------------------------ designed charts (Vega-Lite)
def vl_theme(chart):
    """Our house style applied to any Altair chart: no chartjunk, direct labels, one accent."""
    return (chart
            .configure_view(strokeWidth=0)
            .configure_axis(grid=True, gridColor=GRID, domain=False, tickColor=GRID,
                            labelColor=INK, titleColor=MUTE, labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor=INK, titleColor=MUTE, orient="top", direction="horizontal")
            .configure_title(color=INK, fontSize=13, anchor="start"))


def chart_png(chart, path: str, scale: float = 2.0):
    """Render a Vega-Lite chart to PNG **offline** (vl_convert; no browser, no network)."""
    import vl_convert as vlc
    p = asset(path)
    p.write_bytes(vlc.vegalite_to_png(chart.to_json(), scale=scale))
    return str(p.relative_to(COMP))


def chart_html(chart, note: str = "") -> Html:
    """The same chart, interactive (tooltips/zoom) for the hub."""
    return Html(chart.to_html(), note)


def table(df, title: str = "", subtitle: str = "", heat_cols=(), lower_better=()) -> Html:
    """A results table as a designed object (great_tables): heat-shaded numeric columns, real caption."""
    from great_tables import GT, loc, style
    gt = GT(df)
    if title:
        gt = gt.tab_header(title=title, subtitle=subtitle or None)
    for c in heat_cols:
        if c in df.columns:
            rev = c in lower_better
            gt = gt.data_color(columns=[c], palette=["#e8f1ff", ACCENT] if not rev else [ACCENT, "#e8f1ff"],
                               na_color="white")
    return Html(gt.as_raw_html(), "")


# ------------------------------------------------------------------ architecture, generated not drawn
def arch_graph(module, input_size, path: str, depth: int = 3):
    """The real `nn.Module` → a graph PNG via torchview+graphviz; networkx fallback if `dot` is absent."""
    p = asset(path)
    try:
        from torchview import draw_graph
        # torchview defaults to device="cuda" and MOVES the module — that would silently relocate the
        # caller's model and break the next cell. Trace on the module's own device and restore it.
        try:
            dev = next(module.parameters()).device
        except StopIteration:
            dev = "cpu"
        g = draw_graph(module, input_size=input_size, expand_nested=True, depth=depth,
                       graph_dir="TB", hide_inner_tensors=True, save_graph=False, device=dev)
        module.to(dev)
        g.visual_graph.attr(bgcolor="white", fontname="Helvetica")
        g.visual_graph.render(p.with_suffix(""), format="png", cleanup=True)
        return str(p.relative_to(COMP))
    except Exception:                                     # no `dot` on PATH, or an unsupported op
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [f"{i}: {m.__class__.__name__}" for i, m in enumerate(module.modules()) if i]
        fig, ax = plt.subplots(figsize=(3.4, 0.42 * max(len(names), 3)), constrained_layout=True)
        ax.set_axis_off()
        for i, n in enumerate(names):
            ax.add_patch(plt.Rectangle((0.02, len(names) - i - 0.8), 0.96, 0.6, fill=False, ec=ACCENT))
            ax.text(0.5, len(names) - i - 0.5, n, ha="center", va="center", fontsize=8, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(0, len(names) + 0.2)
        fig.savefig(p, dpi=150); plt.close(fig)
        return str(p.relative_to(COMP))


def level_dag(levels: list, path: str, note: str = "Nested levels, fastest at the top"):
    """The NL picture proper: components as a DAG ordered by UPDATE FREQUENCY.

    `levels` = [{"name": str, "freq": float, "needs": [names]}]. Node size ∝ frequency, edges = the
    dependency that breaks a frequency tie (Definition 2's `A ≻ B`).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    G = nx.DiGraph()
    for lv in levels:
        G.add_node(lv["name"], freq=lv["freq"])
    for lv in levels:
        for n in lv.get("needs", []):
            G.add_edge(n, lv["name"])
    order = sorted({lv["freq"] for lv in levels}, reverse=True)
    pos, per = {}, {}
    for lv in levels:
        row = order.index(lv["freq"])
        per[row] = per.get(row, 0) + 1
        pos[lv["name"]] = (per[row] * 1.6, -row)
    fig, ax = plt.subplots(figsize=(7.2, 1.5 + 1.15 * len(order)), constrained_layout=True)
    ax.set_axis_off()
    sizes = [900 + 2600 * (G.nodes[n]["freq"] / max(o for o in order)) for n in G.nodes]
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=MUTE, arrows=True, arrowsize=12,
                           connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=sizes, node_color=[SERIES[order.index(
        G.nodes[n]["freq"]) % len(SERIES)] for n in G.nodes], alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_color="white")
    for row, f in enumerate(order):
        ax.text(-0.4, -row, f"level {row + 1}\nfreq {f:g}", fontsize=8, color=MUTE,
                ha="right", va="center")
    ax.set_title(note, fontsize=10, color=INK, loc="left")
    p = asset(path); fig.savefig(p, dpi=150); plt.close(fig)
    return str(p.relative_to(COMP))


# ------------------------------------------------------------------ the XAI step
def attribute(forward, x, names=None, n_steps: int = 32, target=None):
    """Integrated Gradients (captum) over a forward function → a tidy DataFrame of attributions.

    Used to answer the paper's own question empirically: *which level actually produced this output?*
    `forward` maps a stacked input to a scalar-per-row output; `names` labels the input slots.
    """
    import pandas as pd
    import torch
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(forward)
    x = x if x.requires_grad else x.clone().requires_grad_(True)
    kw = {"target": target} if target is not None else {}
    a = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=n_steps, **kw)
    a = a.detach().float().cpu()
    flat = a.abs().sum(0) if a.dim() > 1 else a.abs()
    labels = names or [f"input {i}" for i in range(flat.numel())]
    df = pd.DataFrame({"component": labels, "attribution": [float(v) for v in flat]})
    df["share_%"] = (100 * df.attribution / df.attribution.sum().clip(min=1e-12)).round(1)
    return df.sort_values("attribution", ascending=False, ignore_index=True)
