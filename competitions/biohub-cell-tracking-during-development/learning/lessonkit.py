"""
lessonkit — the reusable engine every lesson generator shares (DRY).

A generator file (e.g. da01_cells_per_frame.py) just defines META + CELLS and calls
build_lesson(); this runs each code cell on real data, captures its REAL output
(DataFrame/Series -> HTML table, tensor -> shape, stdout, ndarray), and writes the
`.learning` file with each output attached right after its code. No output is typed by
hand — it comes from executing the code.
"""
from __future__ import annotations
import ast, io, contextlib
from pathlib import Path
import numpy as np
import pandas as pd
try:
    import torch
except Exception:
    torch = None


def run_capture(code: str, glb: dict):
    """Exec one cell in shared namespace `glb`; return (stdout, last-expression value)."""
    tree = ast.parse(code)
    buf = io.StringIO()
    val = None
    with contextlib.redirect_stdout(buf):
        if tree.body and isinstance(tree.body[-1], ast.Expr):          # Jupyter-style last expr
            exec(compile(ast.Module(tree.body[:-1], type_ignores=[]), "<cell>", "exec"), glb)
            val = eval(compile(ast.Expression(tree.body[-1].value), "<cell>", "eval"), glb)
        else:
            exec(compile(tree, "<cell>", "exec"), glb)
    return buf.getvalue(), val


def render_output(stdout: str, val) -> str | None:
    """Turn a cell's real output into a `--- output` block body.

    A cell may also RETURN a rich object and it lands inline in the lesson:
      • anything with `_repr_html_()`  — treescope tensor views, great_tables tables, Altair/Plotly
        charts, IPython.display.HTML — i.e. the whole "interactive, look-inside" family;
      • a matplotlib Figure          — saved next to nothing and inlined as a data-URI PNG;
      • a raw `<svg …>`/`<div …>` string.
    The hub renders lesson output through markdown, which passes raw HTML through (that is how the
    existing DataFrame tables already work), so this needs no hub change.
    """
    parts = []
    if stdout.strip():
        parts.append("```\n" + stdout.rstrip() + "\n```")
    # --- rich returns, checked before the array/tensor branches so a wrapper wins over its payload
    if hasattr(val, "_repr_html_") and not isinstance(val, (pd.DataFrame, pd.Series)):
        try:
            html = val._repr_html_()
            if html:
                return "\n".join(parts + [html])
        except Exception:  # noqa: BLE001 — fall through to the plain renderers
            pass
    try:
        from matplotlib.figure import Figure as _MplFigure
        if isinstance(val, _MplFigure):
            import base64
            import io as _io
            buf = _io.BytesIO(); val.savefig(buf, format="png", dpi=140, bbox_inches="tight")
            b64 = base64.b64encode(buf.getvalue()).decode()
            return "\n".join(parts + [f'<img src="data:image/png;base64,{b64}" '
                                      f'style="max-width:100%;border-radius:6px">'])
    except Exception:  # noqa: BLE001
        pass
    if isinstance(val, str) and val.lstrip().startswith(("<svg", "<div", "<table", "<figure")):
        return "\n".join(parts + [val])
    if isinstance(val, pd.DataFrame):
        parts.append(val.to_html(index=False, border=0, classes="df"))
    elif isinstance(val, pd.Series):
        parts.append(val.to_frame().to_html(border=0, classes="df"))
    elif torch is not None and isinstance(val, torch.Tensor):
        parts.append(f"```\ntorch.Size({list(val.shape)})  dtype={val.dtype}\n```")
    elif isinstance(val, np.ndarray):
        parts.append(f"```\nndarray shape={val.shape} dtype={val.dtype}\n```")
    elif val is not None:
        parts.append("```\n" + repr(val) + "\n```")
    return "\n".join(parts) or None


def build_lesson(meta: dict, cells: list[dict], out_path, glb: dict | None = None) -> str:
    """meta -> `@ key: value` header; each cell = {note, code?, shape?, image?}. Runs code,
    attaches the REAL output after it, writes `out_path`. Returns the path written."""
    glb = dict(glb or {})
    import zarr
    glb.setdefault("np", np); glb.setdefault("pd", pd); glb.setdefault("zarr", zarr)
    if torch is not None:
        glb.setdefault("torch", torch)
    lines = [f"@ {k}: {v}" for k, v in meta.items()] + [""]
    for c in cells:
        lines += ["--- note", c["note"], ""]
        if c.get("code"):
            lines += ["--- code", c["code"], ""]
            stdout, val = run_capture(c["code"], glb)
            out = render_output(stdout, val)          # the REAL output of running the cell
            if out:
                lines += ["--- output", out, ""]
            if c.get("image"):                        # + an attached figure (e.g. a concept diagram)
                lines += ["--- image", c["image"], ""]
        if c.get("shape"):
            lines += ["--- shape", c["shape"], ""]
    Path(out_path).write_text("\n".join(lines) + "\n")
    print(f"built {out_path}  ({sum(1 for c in cells if c.get('code'))} code cells run)")
    return str(out_path)


def refresh_learning(path, glb: dict | None = None) -> str:
    """PATTERN B: a `.learning` file lives next to its real code and IS the source (no generator).
    Parse it, RUN each `--- code` block in a shared namespace, and rewrite the `--- output` block
    right after it with the REAL captured output. Notes/images/shapes are preserved verbatim."""
    import re, zarr
    g = dict(glb or {})
    g.setdefault("np", np); g.setdefault("pd", pd); g.setdefault("zarr", zarr)
    if torch is not None:
        g.setdefault("torch", torch)
    raw = Path(path).read_text()
    lines = raw.split("\n")
    meta, i = [], 0
    while i < len(lines) and (lines[i].startswith("@ ") or lines[i].strip() == ""):
        if lines[i].startswith("@ "):
            meta.append(lines[i])
        i += 1
    body = "\n".join(lines[i:])
    parts = re.split(r"(?m)^--- (note|code|output|image|shape)\s*$", body)
    blocks = [(parts[k], parts[k + 1].strip("\n")) for k in range(1, len(parts), 2)]
    out, ran = [], 0
    for typ, content in blocks:
        if typ == "output":
            continue                                  # drop stale output; regenerated below
        out.append((typ, content))
        if typ == "code":
            stdout, val = run_capture(content, g)      # RUN the block on real data
            rendered = render_output(stdout, val)
            ran += 1
            if rendered:
                out.append(("output", rendered))       # attach the REAL output after the code
    res = "\n".join(meta) + "\n\n" + "".join(f"--- {t}\n{c}\n\n" for t, c in out)
    Path(path).write_text(res)
    print(f"refreshed {path}  ({ran} code blocks run in place)")
    return str(path)


if __name__ == "__main__":
    import sys
    for _p in sys.argv[1:]:
        refresh_learning(_p)
