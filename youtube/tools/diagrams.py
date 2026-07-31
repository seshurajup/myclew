"""House-style diagram toolkit — concept visuals drawn by really running Python.

Design intent
-------------
A Short that shows the same `>>> 4.0` text box as the other eighty reads as templated, and worse,
a text box teaches nothing the code didn't already say. These diagrams show the *mechanic*: where
a slice actually cuts, what the call stack looks like mid-recursion, which items a set operation
keeps. That is the thing a viewer can't get from reading the code, and it makes every video look
different because every concept looks different.

Every figure is produced by matplotlib from real values, never mocked, and is styled to sit next
to the editor pane: same dark surface, same syntax accents, same monospace.

Output modes (see FLOW in tools/gen_visuals.py):
    terminal  — real captured shell session          (tools/gen_terminal_shot.py)
    cells     — index/slice strip, sequence mechanics
    stack     — call frames for recursion/scope
    venn      — set algebra
    bars      — tallies, timings, before/after ordering
    map       — key -> value arrows (dict, zip, comprehension)
    tree      — class hierarchy / MRO
    traceback — a real error, in red, as a terminal would print it
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle  # noqa: E402

# palette mirrors remotion_shorts/src/astryxTheme.ts so the figure belongs to the same video
BG      = "#0a0a0a"
SURFACE = "#161b22"
BORDER  = "#404040"
TEXT    = "#e5e5e5"
MUTED   = "#a3a3a3"
KEYWORD = "#efa8ff"   # purple
STRING  = "#a6d2a2"   # green
NUMBER  = "#ffb37f"   # orange
FUNC    = "#a0caff"   # blue
YELLOW  = "#eec12f"
RED     = "#ffaeaa"
TEAL    = "#83dac9"
MONO    = "DejaVu Sans Mono"

W, H, DPI = 11.8, 5.0, 100


def _fig(w=W, h=H):
    fig, ax = plt.subplots(figsize=(w, h), dpi=DPI)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return fig, ax


# The renderer gives an image event a fixed zone (Short.tsx OUT_ZONE=470) and scales the image to
# maxHeight 340 within a ~988px-wide card. A tall figure therefore shrinks until its text is
# unreadable on a phone. Anything past this aspect gets flagged at generation time, not discovered
# later in a rendered frame.
MIN_ASPECT = 2.6          # width / height
_WARNED = []


def _save(fig, path):
    fig.savefig(path, facecolor=BG, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    try:
        from PIL import Image
        w, h = Image.open(path).size
        if w / h < MIN_ASPECT:
            msg = (f"{Path(path).name}: {w}x{h} is {w / h:.1f}:1 — taller than {MIN_ASPECT}:1, "
                   f"so it will scale down and read small on a phone")
            _WARNED.append(msg)
            print(f"  ! {msg}")
    except Exception:                                              # noqa: BLE001
        pass
    return path


def _title(ax, text):
    ax.text(0.5, 1.06, text, transform=ax.transAxes, ha="center", va="bottom",
            color=MUTED, fontsize=15, family=MONO)


def cells(path, values, title="", highlight=(), labels_neg=False, note=""):
    """A sequence as boxes with index labels; `highlight` is a set/range of indices to light up.

    Use for: indexing, slicing, negative indices, enumerate, reversing.
    """
    n = len(values)
    fig, ax = _fig(W, 2.9)
    ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-1.3, 1.4)
    hs = set(highlight)
    for i, v in enumerate(values):
        on = i in hs
        ax.add_patch(Rectangle((i - 0.42, -0.45), 0.84, 0.9, facecolor=SURFACE if not on else "#1d3b57",
                               edgecolor=FUNC if on else BORDER, lw=3 if on else 2, zorder=2))
        ax.text(i, 0, str(v), ha="center", va="center", color=TEXT if on else MUTED,
                fontsize=22, family=MONO, zorder=3, fontweight="bold" if on else "normal")
        ax.text(i, 0.75, str(i), ha="center", va="center", color=FUNC if on else MUTED,
                fontsize=15, family=MONO)
        if labels_neg:
            ax.text(i, -0.85, str(i - n), ha="center", va="center", color=NUMBER,
                    fontsize=15, family=MONO)
    if note:
        ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center", color=STRING,
                fontsize=17, family=MONO)
    _title(ax, title)
    return _save(fig, path)


def stack(path, frames, title="", note=""):
    """Call frames stacked bottom-up; `frames` = ["fact(4)", "fact(3)", ...]. Recursion/scope."""
    fig, ax = _fig(W, 3.3)
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, max(4, len(frames)) + 0.3)
    for i, f in enumerate(frames):
        top = i == len(frames) - 1
        ax.add_patch(Rectangle((2.2, i), 5.6, 0.82, facecolor="#1d3b57" if top else SURFACE,
                               edgecolor=FUNC if top else BORDER, lw=3 if top else 2))
        ax.text(5.0, i + 0.41, f, ha="center", va="center", color=TEXT, fontsize=20, family=MONO,
                fontweight="bold" if top else "normal")
        if top:
            ax.text(8.1, i + 0.41, "← now", ha="left", va="center", color=YELLOW,
                    fontsize=17, family=MONO)
    ax.text(1.9, len(frames) / 2, "call stack", ha="right", va="center", color=MUTED,
            fontsize=16, family=MONO, rotation=90)
    if note:
        ax.text(0.5, -0.09, note, transform=ax.transAxes, ha="center", color=STRING,
                fontsize=17, family=MONO)
    _title(ax, title)
    return _save(fig, path)


def venn(path, a, b, title="", op="", result=()):
    """Two-set Venn with real membership; `op` labels the operation, `result` is highlighted."""
    fig, ax = _fig(W, 3.5)
    ax.set_xlim(0, 10); ax.set_ylim(0.3, 4.7)
    ax.set_aspect("equal", adjustable="datalim")      # else Circle() draws as an ellipse
    ax.add_patch(Circle((4.15, 2.5), 1.55, facecolor=FUNC, alpha=0.20, edgecolor=FUNC, lw=3))
    ax.add_patch(Circle((5.85, 2.5), 1.55, facecolor=KEYWORD, alpha=0.20, edgecolor=KEYWORD, lw=3))
    ax.text(2.9, 4.05, "a", color=FUNC, fontsize=22, family=MONO, fontweight="bold")
    ax.text(7.0, 4.05, "b", color=KEYWORD, fontsize=22, family=MONO, fontweight="bold")
    res = set(result)
    only_a, both, only_b = sorted(set(a) - set(b)), sorted(set(a) & set(b)), sorted(set(b) - set(a))
    for xs, x in ((only_a, 3.35), (both, 5.0), (only_b, 6.65)):
        for k, v in enumerate(xs):
            hot = v in res
            ax.text(x, 3.0 - k * 0.55, str(v), ha="center", color=YELLOW if hot else MUTED,
                    fontsize=21 if hot else 18, family=MONO,
                    fontweight="bold" if hot else "normal")
    if op:
        ax.text(5.0, 0.35, op, ha="center", color=STRING, fontsize=19, family=MONO)
    _title(ax, title)
    return _save(fig, path)


def bars(path, labels, values, title="", note="", highlight=None, xlabel=""):
    """Horizontal bars from real numbers — tallies, timings, before/after ordering."""
    fig, ax = _fig(W, 0.55 * len(labels) + 2.0)
    cols = [YELLOW if (highlight is not None and i == highlight) else FUNC for i in range(len(labels))]
    y = range(len(labels))
    ax.barh(list(y), values, color=cols, height=0.62)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, color=TEXT, fontsize=18, family=MONO)
    ax.invert_yaxis()
    ax.tick_params(axis="x", colors=MUTED, labelsize=13)
    for sp in ("bottom",):
        ax.spines[sp].set_visible(True); ax.spines[sp].set_color(BORDER)
    ax.set_xlim(0, max(values) * 1.18)                # headroom so the value label never clips
    for i, v in enumerate(values):
        ax.text(v, i, f" {v}", va="center", color=MUTED, fontsize=16, family=MONO)
    if xlabel:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=14, family=MONO)
    if note:
        ax.text(0.5, -0.22, note, transform=ax.transAxes, ha="center", color=STRING,
                fontsize=16, family=MONO)
    _title(ax, title)
    return _save(fig, path)


def mapping(path, pairs, title="", note="", key_label="key", val_label="value"):
    """key -> value arrows. Use for dicts, zip, comprehensions, defaultdict grouping."""
    n = len(pairs)
    fig, ax = _fig(W, 0.58 * n + 1.4)
    ax.set_xlim(0, 10); ax.set_ylim(-0.7, n)
    ax.text(2.7, n - 0.28, key_label, ha="center", color=MUTED, fontsize=15, family=MONO)
    ax.text(7.3, n - 0.28, val_label, ha="center", color=MUTED, fontsize=15, family=MONO)
    for i, (k, v) in enumerate(pairs):
        y = n - 1.35 - i * 0.75
        ax.add_patch(Rectangle((1.0, y - 0.26), 3.4, 0.56, facecolor=SURFACE, edgecolor=BORDER, lw=2))
        ax.text(2.7, y, str(k), ha="center", va="center", color=TEAL, fontsize=19, family=MONO)
        ax.add_patch(Rectangle((5.6, y - 0.26), 3.4, 0.56, facecolor=SURFACE, edgecolor=BORDER, lw=2))
        ax.text(7.3, y, str(v), ha="center", va="center", color=STRING, fontsize=19, family=MONO)
        ax.add_patch(FancyArrowPatch((4.5, y), (5.5, y), arrowstyle="-|>", mutation_scale=26,
                                     color=FUNC, lw=3.0))
    if note:
        ax.text(0.5, -0.06, note, transform=ax.transAxes, ha="center", color=STRING,
                fontsize=16, family=MONO)
    _title(ax, title)
    return _save(fig, path)


def tree(path, edges, root, title="", note=""):
    """Class hierarchy / MRO. `edges` = [(parent, child), ...] laid out by depth."""
    depth, children = {root: 0}, {}
    for p, c in edges:
        children.setdefault(p, []).append(c)
        depth[c] = depth.get(p, 0) + 1
    levels = {}
    for node, d in depth.items():
        levels.setdefault(d, []).append(node)
    maxd = max(levels)
    fig, ax = _fig(W, 0.92 * (maxd + 1) + 0.6)
    ax.set_xlim(0, 10); ax.set_ylim(-0.4, maxd + 0.6)
    pos = {}
    for d, nodes in levels.items():
        for i, nd in enumerate(sorted(nodes)):
            x = 10 * (i + 1) / (len(nodes) + 1)
            pos[nd] = (x, maxd - d)
    for p, c in edges:
        if p in pos and c in pos:
            ax.add_patch(FancyArrowPatch(pos[p], pos[c], arrowstyle="-|>", mutation_scale=20,
                                         color=MUTED, lw=2.6, shrinkA=34, shrinkB=34))
    for nd, (x, y) in pos.items():
        is_root = nd == root
        ax.add_patch(Rectangle((x - 1.25, y - 0.28), 2.5, 0.58,
                               facecolor="#1d3b57" if is_root else SURFACE,
                               edgecolor=KEYWORD if is_root else BORDER, lw=3 if is_root else 2))
        ax.text(x, y, nd, ha="center", va="center", color=TEXT, fontsize=19, family=MONO)
    if note:
        ax.text(0.5, -0.05, note, transform=ax.transAxes, ha="center", color=STRING,
                fontsize=16, family=MONO)
    _title(ax, title)
    return _save(fig, path)


def traceback_shot(path, tb_text, title="a real traceback"):
    """A genuine captured traceback, painted like a terminal. Error-handling topics."""
    lines = [l for l in tb_text.rstrip("\n").split("\n") if l.strip()][-9:]
    fig, ax = _fig(W, 0.45 * len(lines) + 1.5)
    ax.set_xlim(0, 10); ax.set_ylim(-0.5, len(lines) + 0.3)
    ax.add_patch(Rectangle((0.1, -0.35), 9.8, len(lines) + 0.45, facecolor="#0d1117",
                           edgecolor="#30363d", lw=2))
    for i, l in enumerate(lines):
        y = len(lines) - 1 - i
        col = RED if ("Error" in l or "Traceback" in l) else MUTED
        ax.text(0.35, y, l[:92], ha="left", va="center", color=col, fontsize=15, family=MONO)
    _title(ax, title)
    return _save(fig, path)


def variable_box(path, name, value, title="", note=""):
    """A variable drawn as the labeled box the narration describes. Assignment topics."""
    fig, ax = _fig(W, 3.0)
    ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    ax.add_patch(Rectangle((3.3, 0.85), 3.4, 1.35, facecolor=SURFACE, edgecolor=FUNC, lw=3))
    ax.text(5.0, 1.52, repr(value), ha="center", va="center", color=STRING, fontsize=30, family=MONO)
    # the label sits ON the box, like a name tag
    ax.add_patch(Rectangle((3.9, 2.05), 2.2, 0.52, facecolor=FUNC, edgecolor=FUNC, lw=2))
    ax.text(5.0, 2.31, name, ha="center", va="center", color="#0a0a0a", fontsize=21,
            family=MONO, fontweight="bold")
    ax.text(5.0, 0.45, f"type: {type(value).__name__}", ha="center", color=MUTED,
            fontsize=16, family=MONO)
    if note:
        ax.text(0.5, -0.02, note, transform=ax.transAxes, ha="center", color=STRING,
                fontsize=17, family=MONO)
    _title(ax, title)
    return _save(fig, path)


def substitution(path, template, slot, value, result, title="", note=""):
    """f-string interpolation shown as a swap: the {slot} lights up, the value drops in.

    `result` must be the REALLY evaluated string, never a retyped guess.

    Layout note: character positions are MEASURED from the rendered text (get_window_extent),
    never estimated from a per-character width — an estimate put the highlight box mid-word.
    """
    fig, ax = _fig(W, 3.1)
    ax.set_xlim(0, 10); ax.set_ylim(0.2, 3.5)
    fig.canvas.draw()                                  # renderer needed for measurement
    r = fig.canvas.get_renderer()
    fs, Y = 26, 2.75
    inv = ax.transData.inverted()

    def width(txt):
        t = ax.text(0, -5, txt, fontsize=fs, family=MONO)   # off-screen probe
        bb = t.get_window_extent(renderer=r)
        t.remove()
        (x0, _), (x1, _) = inv.transform([(0, 0), (bb.width, 0)])
        return x1 - x0

    pre, _, post = template.partition(slot)
    wp, ws, wo = width(pre), width(slot), width(post)
    x = 5.0 - (wp + ws + wo) / 2
    ax.text(x, Y, pre, ha="left", va="center", color=TEXT, fontsize=fs, family=MONO)
    ax.add_patch(Rectangle((x + wp - 0.04, Y - 0.26), ws + 0.08, 0.52,
                           facecolor="#1d3b57", edgecolor=YELLOW, lw=2.5, zorder=2))
    ax.text(x + wp, Y, slot, ha="left", va="center", color=YELLOW, fontsize=fs,
            family=MONO, zorder=3)
    ax.text(x + wp + ws, Y, post, ha="left", va="center", color=TEXT, fontsize=fs, family=MONO)
    # the value dropping into the slot
    ax.add_patch(FancyArrowPatch((x + wp + ws / 2, Y - 0.32), (5.0, 1.78),
                                 arrowstyle="-|>", mutation_scale=24, color=YELLOW, lw=2.6))
    ax.text(5.0, 1.55, repr(value), ha="center", va="center", color=STRING, fontsize=23, family=MONO)
    # the real result
    ax.add_patch(Rectangle((1.2, 0.35), 7.6, 0.8, facecolor="#0d1117", edgecolor=BORDER, lw=2))
    ax.text(5.0, 0.75, result, ha="center", va="center", color=STRING, fontsize=25, family=MONO)
    if note:
        ax.text(0.5, -0.04, note, transform=ax.transAxes, ha="center", color=MUTED,
                fontsize=16, family=MONO)
    _title(ax, title)
    return _save(fig, path)
