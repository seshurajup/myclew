"""
autorun — run a lesson's code cells like a Jupyter notebook and capture their REAL
outputs for the learning UI: stdout, the last-expression value, and specifically
**pandas DataFrames rendered as HTML tables** and tensors rendered as their shape.

This is the "no hallucination" engine: outputs come from EXECUTING the code, never typed.
"""
from __future__ import annotations
import ast, io, contextlib
import pandas as pd
import numpy as np
import torch


def run_cell(code: str, glb: dict) -> str:
    """Execute one cell in the shared namespace `glb`; return its output as HTML
    (stdout + Jupyter-style last-expression value; DataFrame -> table)."""
    tree = ast.parse(code)
    stdout, last = io.StringIO(), None
    with contextlib.redirect_stdout(stdout):
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            exec(compile(ast.Module(tree.body[:-1], type_ignores=[]), "<cell>", "exec"), glb)
            last = eval(compile(ast.Expression(tree.body[-1].value), "<cell>", "eval"), glb)
        else:
            exec(compile(tree, "<cell>", "exec"), glb)
    parts = []
    s = stdout.getvalue()
    if s.strip():
        parts.append(f"<pre>{s.rstrip()}</pre>")
    if isinstance(last, pd.DataFrame):
        parts.append(last.to_html(index=False, border=0, classes="df"))
    elif isinstance(last, pd.Series):
        parts.append(last.to_frame().to_html(border=0, classes="df"))
    elif isinstance(last, torch.Tensor):
        parts.append(f"<pre>torch.Size({tuple(last.shape)})  dtype={last.dtype}</pre>")
    elif isinstance(last, np.ndarray):
        parts.append(f"<pre>ndarray shape={last.shape} dtype={last.dtype}</pre>")
    elif last is not None:
        parts.append(f"<pre>{repr(last)}</pre>")
    return "\n".join(parts)


if __name__ == "__main__":
    # DEMO — a real DataFrame from the real data auto-renders as a table.
    from pathlib import Path
    ROOT = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development")
    glb = {"pd": pd, "np": np, "ROOT": ROOT}
    cell = """
df = pd.read_csv(f"{ROOT}/learning/01_cells_per_frame_per_dataset.csv")   # real EDA output
summary = df.groupby("group").agg(
    n=("dataset", "size"),
    cells_per_frame_max=("cpf_max", "max"),
    cells_per_frame_mean=("cpf_mean", "mean"),
).round(1).reset_index()
summary
"""
    html = run_cell(cell, glb)
    print("=== captured output HTML (auto-rendered from RUNNING the code) ===")
    print(html[:600])
    print("...\nDataFrame -> HTML table:", "<table" in html)
