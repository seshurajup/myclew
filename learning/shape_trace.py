"""
shape_trace — run a snippet of real pipeline code and capture the shape of every
tensor after each line. This is what makes the learning UI's SHAPES panel automatic:
a lesson's code is EXECUTED on real data and the shapes are recorded, not hand-typed.

Usage:
    from shape_trace import trace_shapes
    shapes = trace_shapes(code_str, setup_globals)
    # shapes: {source_line_index: {"var": (1,32,64,32,32), ...}}
"""
from __future__ import annotations
import sys
import torch


def _snap(local_vars):
    """Shapes of every torch.Tensor currently in scope."""
    out = {}
    for k, v in local_vars.items():
        if isinstance(v, torch.Tensor):
            out[k] = tuple(v.shape)
    return out


def trace_shapes(code: str, glb: dict | None = None) -> dict[int, dict]:
    """Execute `code` line by line and return {1-based line -> {var: shape}} captured
    AFTER that line ran. `glb` provides real inputs (a loaded frame, a built model, ...)."""
    glb = dict(glb or {})
    lines = code.strip("\n").split("\n")
    captured: dict[int, dict] = {}
    state = {"prev": None}

    def tracer(frame, event, arg):
        # Only trace the lesson's OWN top-level lines — never descend into torch/nn
        # internals (that would flood us with BatchNorm's private tensors).
        if frame.f_code.co_filename != "<lesson>":
            return None
        if event == "line":
            if state["prev"] is not None:
                captured[state["prev"]] = _snap(frame.f_locals)
            state["prev"] = frame.f_lineno
        elif event == "return":
            if state["prev"] is not None:
                captured[state["prev"]] = _snap(frame.f_locals)
        return tracer

    compiled = compile("\n".join(lines), "<lesson>", "exec")
    sys.settrace(tracer)
    try:
        exec(compiled, glb)
    finally:
        sys.settrace(None)
    # remap absolute line numbers -> 1-based positions within the snippet
    base = min(captured) - 1 if captured else 0
    return {ln - base: sh for ln, sh in captured.items()}


if __name__ == "__main__":
    # DEMO on the REAL conv block + a REAL frame crop (no toy data).
    from pathlib import Path
    import numpy as np, zarr, torch.nn as nn
    TRAIN = Path("/home/seshu/kaggle/2026/biohub-cell-tracking-during-development/"
                 "input/biohub-cell-tracking-during-development/train")
    z = zarr.open(str(TRAIN / "6bba_062c8d37.zarr" / "0"), mode="r")
    crop = np.asarray(z[0, :, 96:128, 96:128]).astype(np.float32)
    lo, hi = np.quantile(crop, 0.01), np.quantile(crop, 0.99)
    crop = np.clip((crop - lo) / (hi - lo + 1e-6), 0, 1)

    def conv_block(ci, co):
        return nn.Sequential(
            nn.Conv3d(ci, co, 3, padding=1, bias=False), nn.BatchNorm3d(co), nn.ReLU(inplace=True),
            nn.Conv3d(co, co, 3, padding=1, bias=False), nn.BatchNorm3d(co), nn.ReLU(inplace=True))

    code = """
x = torch.from_numpy(crop)[None, None]
block = conv_block(1, 32).eval()
y = block(x)
pooled = torch.nn.functional.max_pool3d(y, 2)
"""
    shapes = trace_shapes(code, {"torch": torch, "crop": crop, "conv_block": conv_block})
    print("AUTO-CAPTURED shapes (line -> tensors), from RUNNING real code on a real frame:")
    for ln in sorted(shapes):
        if shapes[ln]:
            print(f"  after line {ln}: " + ", ".join(f"{k}={v}" for k, v in shapes[ln].items()))
