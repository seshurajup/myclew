"""arc-onnx-golf — DETERMINISTIC ONNX emit + verify + OFFICIAL-COST tool for network-golf (ARC-AGI-ONNX)
competitions (neurogolf-2026). PURE PYTHON TOOL — it contains NO LLM call and no "solve the ARC rule"
reasoning. It EMITS a minimal ONNX graph for an already-identified transform, VERIFIES functional
correctness on the task's train/test/arc-gen pairs under the EXACT official semantics, and returns the
OFFICIAL cost (memory_bytes + params) + score = max(1, 25 - ln(max(1, cost))).

The live reasoning layer (the researchpapers `researcher`/`leader` agents) CALLS this tool through
fleet_dispatch; the ARC-solving + architecture-rewrite thinking lives THERE, not here.

Grounded truth (data/neurogolf_utils/neurogolf_utils.py, the official scorer):
  • Network I/O is ONE-HOT [1,10,30,30] float32. A grid cell of colour k → channel k = 1.0 at (r,c);
    grids <30 are top-left anchored, the rest is zero-hot. Read-back: colour = channel with value > 0.
  • Correctness = exact (output > 0.0) one-hot equality on EVERY train+test+arc-gen example ≤30×30.
  • cost = memory(charged intermediate tensors; input & output are FREE) + params(initializer elements +
    Constant node values). score = max(1, 25 - ln(max(1, cost))). Zero-cost graph → 25.
  • opset 10, ir_version 10, single input `input`, single output `output`. Banned ops:
    Loop/Scan/NonZero/Unique/Script/Function/Compress/*Sequence*; no subgraphs/functions/custom domains;
    no initializer↔IO name collision; every tensor must strict-shape-infer to concrete positive dims.

The emitter set below is the VERIFIED, faithfully-costed core distilled from experiments/poc_triage_onnx.py
+ patterns.md (9th place). Each is a single terminal node → zero charged intermediate → cost is exactly its
initializer params. detect_transform() proposes a candidate from the pairs; golf() emits+verifies+costs it.
The tool is HONEST: a candidate that fails the (output>0) gate on any subset is reported unsolved, not forced.

Spec (all optional): {task:{train,test,arc_gen:[{input,output}...]}} OR {tasks_dir,task_id} OR
{transform,meta}; {dtype}; {emit_only:bool}. Empty spec → returns the emitter catalogue (no onnx needed).
Data-wise test: test_fleet_agents/arc_onnx_golf_test.py (synthetic grids, onnxruntime-verified).
"""
from __future__ import annotations
import glob
import json
import math
import os
import tempfile
from pathlib import Path
from .base import BaseAgent
from . import onnx_tools as OT           # the GENERIC cross-comp onnx tool (validity/cost/verify engine)

COMP = Path(__file__).resolve().parent.parent
GRID = [1, 10, 30, 30]
# neurogolf-specific banned ops (passed IN to the generic onnx validity check; not a generic-onnx rule)
NEUROGOLF_BANNED = ["Loop", "Scan", "NonZero", "Unique", "Script", "Function", "Compress", "*Sequence*"]

# emitter name -> (one-line description, ONNX ops, typical cost, patterns.md band)
EMITTER_CATALOGUE = {
    "identity":     ("same grid out (no-op)",                    ["Identity"],   0,  "25"),
    "transpose":    ("main-diagonal reflection (square grids)",  ["Transpose"],  0,  "25"),
    "recolor":      ("palette bijection: output ch = input ch",  ["Gather"],     10, "22-23"),
    "recolor_conv": ("general colour map (incl. many-to-one)",   ["Conv"],       100,"20-21"),
    "flip_w":       ("fixed-width horizontal mirror",            ["Gather"],     30, "21-22"),
    "flip_h":       ("fixed-height vertical mirror",             ["Gather"],     30, "21-22"),
    "constant":     ("fixed output grid (input-independent)",    ["Identity"],   None,"varies"),
}


# ------------------------------------------------------------------ one-hot codec (official semantics)
def onehot(grid):
    import numpy as np
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim != 2 or max(g.shape) > 30:
        return None
    a = np.zeros((1, 10, 30, 30), np.float32)
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            a[0, int(g[r, c]), r, c] = 1.0
    return a


# ------------------------------------------------------------------ transform detection (real semantics)
def detect_transform(pairs):
    """Propose (name, meta) for a single-node emitter that reproduces EVERY pair, else None.
    pairs: list of (input_grid, output_grid) python lists. Detection is by numpy equality; golf() then
    emits the ONNX and RE-VERIFIES under onnxruntime (the authority)."""
    import numpy as np
    ins = [np.asarray(i, np.int64) for i, _ in pairs]
    outs = [np.asarray(o, np.int64) for _, o in pairs]
    if not ins:
        return None
    same_shape = all(i.shape == o.shape for i, o in zip(ins, outs))

    if same_shape and all(np.array_equal(i, o) for i, o in zip(ins, outs)):
        return ("identity", {})

    # diagonal reflection (square grids): output == input.T
    if all(i.shape[0] == i.shape[1] for i in ins) and all(np.array_equal(i.T, o) for i, o in zip(ins, outs)):
        return ("transpose", {})

    # palette map: consistent colour->colour function across all pairs
    if same_shape:
        mp, good = {}, True
        for i, o in zip(ins, outs):
            for a, b in zip(i.ravel(), o.ravel()):
                a, b = int(a), int(b)
                if a in mp and mp[a] != b:
                    good = False
                    break
                mp[a] = b
            if not good:
                break
        if good and any(k != v for k, v in mp.items()):
            bijective = len(set(mp.values())) == len(mp)
            return ("recolor" if bijective else "recolor_conv", {"map": mp})

    # fixed-size horizontal / vertical mirror (single static spatial gather)
    if len({i.shape for i in ins}) == 1:
        H, W = ins[0].shape
        if all(np.array_equal(np.fliplr(i), o) for i, o in zip(ins, outs)):
            return ("flip_w", {"W": int(W)})
        if all(np.array_equal(np.flipud(i), o) for i, o in zip(ins, outs)):
            return ("flip_h", {"H": int(H)})

    # constant output (input-independent) — only when inputs actually differ
    if all(np.array_equal(outs[0], o) for o in outs) and len({i.tobytes() for i in ins}) > 1:
        return ("constant", {"grid": outs[0].tolist()})
    return None


# ------------------------------------------------------------------ ONNX builders (opset 10, ir 10)
def _model(nodes, inits):
    import onnx
    from onnx import helper, TensorProto
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, GRID)
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, GRID)
    g = helper.make_graph(nodes, "arc", [x], [y], inits)
    m = helper.make_model(g, ir_version=10, opset_imports=[helper.make_opsetid("", 10)])
    onnx.checker.check_model(m, full_check=True)
    return m


def build_model(name, meta):
    import numpy as np
    from onnx import helper, numpy_helper
    if name == "identity":
        return _model([helper.make_node("Identity", ["input"], ["output"])], [])
    if name == "transpose":
        return _model([helper.make_node("Transpose", ["input"], ["output"], perm=[0, 1, 3, 2])], [])
    if name == "recolor":
        mp = meta["map"]
        inv = {v: k for k, v in mp.items()}                       # idx[out_ch] = in_ch
        idx = np.array([inv.get(c, c) for c in range(10)], np.int64)
        t = numpy_helper.from_array(idx, "idx")
        return _model([helper.make_node("Gather", ["input", "idx"], ["output"], axis=1)], [t])
    if name == "recolor_conv":
        mp = meta["map"]
        W = np.zeros((10, 10, 1, 1), np.float32)
        for a in range(10):
            W[mp.get(a, a), a, 0, 0] = 1.0
        w = numpy_helper.from_array(W, "W")
        return _model([helper.make_node("Conv", ["input", "W"], ["output"], kernel_shape=[1, 1])], [w])
    if name in ("flip_w", "flip_h"):
        axis = 3 if name == "flip_w" else 2
        n = meta.get("W") if name == "flip_w" else meta.get("H")
        idx = np.array([(n - 1 - j if j < n else j) for j in range(30)], np.int64)
        t = numpy_helper.from_array(idx, "idx")
        return _model([helper.make_node("Gather", ["input", "idx"], ["output"], axis=axis)], [t])
    if name == "constant":
        oh = onehot(meta["grid"])
        c = numpy_helper.from_array(oh.astype(np.float32), "const")
        return _model([helper.make_node("Identity", ["const"], ["output"])], [c])
    raise ValueError(f"unknown emitter: {name}")


# ------------------------------------------------------------------ OFFICIAL cost = the GENERIC onnx tool
# arc-onnx-golf is a THIN neurogolf wrapper: the validity + params + charged-memory engine lives in the
# generic `onnx` tool (onnx_tools). Here we only pass the neurogolf specifics (banned ops, the [1,10,30,30]
# zero sample input) and layer the neurogolf SCORE = max(1, 25 - ln(cost)) on top of the generic cost.
def official_cost(model):
    """Neurogolf cost/score via the generic onnx tool. Returns {valid, reason, memory, params, cost, score}."""
    import numpy as np
    cost = OT.measure_cost(model, sample_inputs={"input": np.zeros((1, 10, 30, 30), np.float32)},
                           banned_ops=NEUROGOLF_BANNED, free_io=("input", "output"))
    if not cost["valid"] or cost.get("cost") is None:
        return {"valid": False, "reason": cost.get("reason"), "memory": cost.get("memory"),
                "params": cost.get("params"), "cost": None, "score": None}
    score = max(1.0, 25.0 - math.log(max(1.0, cost["cost"])))
    return {"valid": True, "reason": "ok", "memory": cost["memory"], "params": cost["params"],
            "cost": cost["cost"], "score": round(score, 6)}


def verify(model, pairs):
    """Exact (output > 0.0) one-hot equality on every pair ≤30×30 (the official correctness gate)."""
    import numpy as np
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    n = 0
    for gi, go in pairs:
        xi = onehot(gi)
        yo = onehot(go)
        if xi is None or yo is None:
            continue                                   # >30 grids are ignored by the host too
        out = (sess.run(["output"], {"input": xi})[0] > 0.0).astype(np.float32)
        if not np.array_equal(out, yo):
            return False, n
        n += 1
    return True, n


# ------------------------------------------------------------------ end-to-end golf
def golf(train, test=None, arc_gen=None, transform=None, meta=None, emit_only=False):
    """Emit + verify + cost a single-node ONNX for a task. Returns a result dict. If (transform, meta) are
    given, uses them; else detect_transform(train). VERIFIES on train+test+arc-gen (the generalization gate)."""
    test = test or []
    arc_gen = arc_gen or []
    if transform is None:
        det = detect_transform(train)
        if det is None:
            return {"solved": False, "transform": None, "reason": "no covered transform detected",
                    "n_covered_emitters": len(EMITTER_CATALOGUE)}
        transform, meta = det
    model = build_model(transform, meta or {})
    cost = official_cost(model)
    if emit_only:
        return {"solved": None, "transform": transform, **cost, "emit_only": True}
    ok_tr, ntr = verify(model, train)
    ok_te, nte = verify(model, test)
    ok_ag, nag = verify(model, arc_gen)
    # HONEST fallback: a cheap palette Gather (cost 10) that fails means the map is not channel-bijective →
    # fall back to the general 1x1-Conv colour map (cost 100). The harness reports the higher cost so the
    # researcher knows a cheaper rewrite (patterns band 22-23) may still exist.
    if not (ok_tr and ok_te and ok_ag) and transform == "recolor" and (meta or {}).get("map"):
        transform = "recolor_conv"
        model = build_model(transform, meta)
        cost = official_cost(model)
        ok_tr, ntr = verify(model, train)
        ok_te, nte = verify(model, test)
        ok_ag, nag = verify(model, arc_gen)
    solved = bool(ok_tr and ok_te and ok_ag and cost["valid"])
    return {"solved": solved, "transform": transform, "ops": EMITTER_CATALOGUE.get(transform, ("", [], 0, ""))[1],
            "verified": {"train": [ok_tr, ntr], "test": [ok_te, nte], "arc_gen": [ok_ag, nag]},
            "memory": cost["memory"], "params": cost["params"], "cost": cost["cost"],
            "score": cost["score"], "valid": cost["valid"], "reason": cost["reason"]}


def _load_task(tasks_dir, task_id):
    p = os.path.join(tasks_dir, f"{task_id}.json")
    if not os.path.exists(p):
        cand = glob.glob(os.path.join(tasks_dir, f"*{task_id}*.json"))
        if cand:
            p = cand[0]
    d = json.load(open(p))
    def gp(k):
        return [(e["input"], e["output"]) for e in d.get(k, [])]
    return gp("train"), gp("test"), gp("arc-gen") or gp("arc_gen")


class ArcOnnxGolf(BaseAgent):
    name = "arc-onnx-golf"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        # empty-spec (smoke) path → return the emitter catalogue, no onnx needed
        if not (spec.get("task") or spec.get("tasks_dir") or spec.get("transform")):
            cat = {k: {"desc": v[0], "ops": v[1], "cost": v[2], "band": v[3]} for k, v in EMITTER_CATALOGUE.items()}
            msg = (f"[{worker}] arc-onnx-golf: deterministic ONNX emit/verify/official-cost tool. "
                   f"{len(cat)} verified emitters (one-hot [1,10,30,30], opset10). Give spec.task or "
                   f"spec.tasks_dir+task_id to golf a task.")
            return self.done({"catalogue": cat, "n_emitters": len(cat)}, msg)
        try:
            if spec.get("task"):
                t = spec["task"]
                train = [(e["input"], e["output"]) for e in t.get("train", [])]
                test = [(e["input"], e["output"]) for e in t.get("test", [])]
                arc_gen = [(e["input"], e["output"]) for e in (t.get("arc_gen") or t.get("arc-gen") or [])]
            elif spec.get("tasks_dir"):
                train, test, arc_gen = _load_task(spec["tasks_dir"], spec.get("task_id", ""))
            else:
                train = test = arc_gen = []
            res = golf(train, test, arc_gen, transform=spec.get("transform"), meta=spec.get("meta"),
                       emit_only=bool(spec.get("emit_only")))
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] arc-onnx-golf failed: {type(e).__name__}: {str(e)[:80]}")
        emoji = "✅" if res.get("solved") else ("🔧" if res.get("emit_only") else "⛔")
        msg = (f"[{worker}] {emoji} arc-onnx-golf `{res.get('transform')}` solved={res.get('solved')} "
               f"cost={res.get('cost')} score={res.get('score')} (mem={res.get('memory')} params={res.get('params')})")
        self.log(summary=msg, kind="verdict")
        return self.done(res, msg, to="researcher")


_AGENT = ArcOnnxGolf()


def run(q, worker):
    return _AGENT.run(q, worker)
