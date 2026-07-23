"""onnx — GENERIC, cross-competition ONNX tool (deterministic python; no LLM). Useful for ANY comp that
ships a model: export a torch/sklearn model to ONNX, VERIFY functional correctness via onnxruntime, MEASURE
cost (params + charged activation memory bytes + optional latency), and QUANTIZE (fp16 / dynamic int8 —
composes with the `quantize` + `compress-select` agents, does not duplicate them). Typical uses:
  • Kaggle offline code-comps: shrink/verify a model for the 2×T4/12h budget (e.g. biohub inference).
  • Network-golf / grid-reasoning (neurogolf): the `arc-onnx-golf` wrapper calls THIS tool for the
    emit→verify→official-cost loop (cost = memory + params). The scoring rule is layered on top there.

The cost engine faithfully mirrors the neurogolf official scorer (data/neurogolf_utils.py): params =
initializer elements + Constant-node values; memory = charged intermediate-tensor bytes (graph input &
output are FREE) using the MAX of static shape-inference and the ORT runtime-profiler trace. `banned_ops`
and the [1,10,30,30] sample input are neurogolf specifics passed IN by the caller, not hardcoded here.

Spec: {action: cost|verify|export|quantize, model_path/onnx_bytes, sample_inputs, cases, banned_ops,
latency, framework(torch|sklearn), ...}. Empty spec → capability summary (no onnx needed).
Data-wise test: test_fleet_agents/onnx_tools_test.py.
"""
from __future__ import annotations
import json
import math
import os
import tempfile
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ load helpers
def _load(model_or_path):
    """Accept an onnx.ModelProto, a path, or serialized bytes → ModelProto."""
    import onnx
    if hasattr(model_or_path, "graph"):
        return model_or_path
    if isinstance(model_or_path, (bytes, bytearray)):
        return onnx.load_from_string(bytes(model_or_path))
    return onnx.load(str(model_or_path))


def _input_shapes(model):
    """Static input {name: shape} from the graph (dim_value; unknown dims → 1)."""
    import onnx  # noqa
    shapes = {}
    init = {i.name for i in model.graph.initializer}
    for inp in model.graph.input:
        if inp.name in init:
            continue
        dims = []
        for d in inp.type.tensor_type.shape.dim:
            dims.append(d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else 1)
        shapes[inp.name] = dims
    return shapes


def _np_dtype_of(model, name):
    import onnx
    for inp in model.graph.input:
        if inp.name == name:
            return onnx.helper.tensor_dtype_to_np_dtype(inp.type.tensor_type.elem_type)
    return None


# ------------------------------------------------------------------ validity + cost (generic engine)
def validity(model, banned_ops=None):
    """Generic ONNX validity: single I/O, no init↔IO name collision, no subgraphs/functions/custom domain,
    every tensor strict-shape-infers to concrete positive dims. `banned_ops` (optional, e.g. neurogolf's
    Loop/Scan/NonZero/…) are additionally rejected. Returns (ok, reason)."""
    import onnx
    from onnx import AttributeProto
    banned = {b.upper() for b in (banned_ops or [])}
    for node in model.graph.node:
        if banned and (node.op_type.upper() in banned or ("SEQUENCE" in node.op_type.upper() and "Sequence" in banned_ops)):
            return False, f"banned op {node.op_type}"
        if banned and "Sequence" in node.op_type and "*Sequence*" in (banned_ops or []):
            return False, f"banned op {node.op_type}"
        for attr in node.attribute:
            if attr.type in (AttributeProto.GRAPH, AttributeProto.GRAPHS):
                return False, "subgraph attr"
    if model.functions:
        return False, "model.functions"
    for op in model.opset_import:
        if op.domain not in ("", "ai.onnx"):
            return False, f"custom domain {op.domain}"
    init_names = {i.name for i in model.graph.initializer}
    io_names = {t.name for t in list(model.graph.input) + list(model.graph.output)}
    if io_names & init_names:
        return False, "init/IO name collision"
    try:
        g = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    except Exception as e:  # noqa: BLE001
        return False, f"shape-infer failed: {str(e)[:60]}"
    for t in list(g.input) + list(g.value_info) + list(g.output):
        tt = t.type.tensor_type
        if not tt.HasField("shape"):
            return False, f"no shape for {t.name}"
        for d in tt.shape.dim:
            if d.HasField("dim_param") or not d.HasField("dim_value") or d.dim_value <= 0:
                return False, f"non-concrete dim on {t.name}"
    return True, "ok"


def calc_params(model):
    """Param count: initializer elements + Constant-node value elements (neurogolf-faithful, generic)."""
    p = 0
    for init in model.graph.initializer:
        if any(d <= 0 for d in init.dims):
            return None
        p += math.prod(init.dims)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for a in node.attribute:
            if a.name == "value":
                p += math.prod(a.t.dims)
            elif a.name == "value_floats":
                p += len(a.floats)
            elif a.name == "value_ints":
                p += len(a.ints)
    return p


def calc_memory(model, sample_inputs=None, free_io=("input", "output")):
    """Charged activation memory: sum of intermediate-tensor bytes (names in `free_io` are FREE), tensor
    shapes = MAX of static shape-inference and the ORT runtime-profiler trace. `sample_inputs` overrides the
    default zero inputs used to drive the profiler; if None they are inferred from the graph input shapes."""
    import numpy as np
    import onnx
    g = onnx.shape_inference.infer_shapes(model, strict_mode=True).graph
    tmap = {t.name: t for t in list(g.input) + list(g.value_info) + list(g.output)}
    mem = {}
    for name, t in tmap.items():
        if name in free_io:
            continue
        tt = t.type.tensor_type
        n = 1
        for d in tt.shape.dim:
            n *= d.dim_value
        dt = onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type)
        mem[name] = n * np.dtype(dt).itemsize
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        node_outputs = {node.output[0]: list(node.output) for node in g.node if node.output}
        dtypes = {name: onnx.helper.tensor_dtype_to_np_dtype(tmap[name].type.tensor_type.elem_type) for name in mem}
        if sample_inputs is None:
            sample_inputs = {}
            for name, dims in _input_shapes(model).items():
                dt = _np_dtype_of(model, name) or np.float32
                sample_inputs[name] = np.zeros(dims, dtype=dt)
        with tempfile.TemporaryDirectory() as td:
            opts = ort.SessionOptions()
            opts.enable_profiling = True
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            opts.profile_file_prefix = os.path.join(td, "p")
            sess = ort.InferenceSession(model.SerializeToString(), opts, providers=["CPUExecutionProvider"])
            out_names = [o.name for o in sess.get_outputs()]
            sess.run(out_names, {k: v for k, v in sample_inputs.items()})
            data = json.load(open(sess.end_profiling()))
            for ev in data:
                if ev.get("cat") != "Node" or "args" not in ev or "output_type_shape" not in ev["args"]:
                    continue
                nm = ev.get("name", "").replace("_kernel_time", "")
                outs = node_outputs.get(nm)
                if not outs:
                    continue
                for i, sd in enumerate(ev["args"]["output_type_shape"]):
                    if i >= len(outs):
                        continue
                    on = outs[i]
                    if on not in dtypes:
                        continue
                    b = np.dtype(dtypes[on]).itemsize * sum(math.prod(dims) for dims in sd.values())
                    mem[on] = max(mem.get(on, 0), b)
    except Exception:  # noqa: BLE001
        pass
    return sum(mem.values())


def measure_cost(model_or_path, sample_inputs=None, banned_ops=None, free_io=("input", "output"), latency=False):
    """Generic cost report: {valid, reason, memory, params, cost, latency_ms}. cost = memory + params.
    Reusable for offline-budget triage (params+memory) on ANY comp's ONNX model."""
    model = _load(model_or_path)
    ok, reason = validity(model, banned_ops=banned_ops)
    if not ok:
        return {"valid": False, "reason": reason, "memory": None, "params": None, "cost": None, "latency_ms": None}
    params = calc_params(model)
    memory = calc_memory(model, sample_inputs=sample_inputs, free_io=free_io)
    out = {"valid": True, "reason": "ok", "memory": int(memory), "params": int(params),
           "cost": int(memory + params), "latency_ms": None}
    if latency:
        out["latency_ms"] = _latency_ms(model, sample_inputs)
    return out


def _latency_ms(model, sample_inputs=None, runs=20):
    import time
    import numpy as np
    try:
        import onnxruntime as ort
        ort.set_default_logger_severity(3)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        if sample_inputs is None:
            sample_inputs = {}
            for name, dims in _input_shapes(model).items():
                dt = _np_dtype_of(model, name) or np.float32
                sample_inputs[name] = np.zeros(dims, dtype=dt)
        out_names = [o.name for o in sess.get_outputs()]
        sess.run(out_names, sample_inputs)  # warm
        t0 = time.perf_counter()
        for _ in range(runs):
            sess.run(out_names, sample_inputs)
        return round((time.perf_counter() - t0) / runs * 1000.0, 4)
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ generic functional verify
def verify_functional(model_or_path, cases):
    """Run onnxruntime and check outputs. cases = [{inputs:{name:array}, outputs:{name:array},
    mode:'exact'|'allclose', rtol, atol, transform:callable(out)->out}]. Returns (all_ok, n_pass, n_total)."""
    import numpy as np
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    model = _load(model_or_path)
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    npass = 0
    for c in cases:
        res = sess.run(out_names, {k: np.asarray(v) for k, v in c["inputs"].items()})
        got = dict(zip(out_names, res))
        ok = True
        for oname, exp in c["outputs"].items():
            g = got.get(oname) if oname in got else res[0]
            tf = c.get("transform")
            if tf:
                g = tf(g)
            exp = np.asarray(exp)
            if c.get("mode", "allclose") == "exact":
                ok = ok and np.array_equal(g, exp)
            else:
                ok = ok and np.allclose(g, exp, rtol=c.get("rtol", 1e-4), atol=c.get("atol", 1e-5))
        npass += int(ok)
    return npass == len(cases), npass, len(cases)


# ------------------------------------------------------------------ export (guarded) + quantize (compose)
def export_torch(module, sample_input, path=None, input_names=("input",), output_names=("output",),
                 opset=17, dynamic_axes=None):
    """torch.nn.Module → ONNX (guarded torch import). Returns the path."""
    import torch  # noqa
    path = path or tempfile.mktemp(suffix=".onnx")
    module.eval()
    torch.onnx.export(module, sample_input, path, input_names=list(input_names),
                      output_names=list(output_names), opset_version=opset, dynamic_axes=dynamic_axes)
    return path


def export_sklearn(model, n_features, path=None):
    """sklearn estimator → ONNX via skl2onnx (guarded). Returns the path."""
    from skl2onnx import to_onnx  # noqa
    import numpy as np
    path = path or tempfile.mktemp(suffix=".onnx")
    onx = to_onnx(model, np.zeros((1, n_features), dtype=np.float32))
    with open(path, "wb") as f:
        f.write(onx.SerializeToString())
    return path


def quantize(model_or_path, mode="fp16", path=None):
    """Shrink an ONNX model. mode='fp16' (onnxconverter_common) or 'int8' (onnxruntime dynamic quant).
    This composes with the `quantize`/`compress-select` fleet agents (PTQ/ToMe) — here it is the ONNX-graph
    dtype path. Guarded; returns {path, mode} or {error}."""
    import onnx
    model = _load(model_or_path)
    path = path or tempfile.mktemp(suffix=f".{mode}.onnx")
    try:
        if mode == "fp16":
            from onnxconverter_common import float16
            onnx.save(float16.convert_float_to_float16(model), path)
        elif mode == "int8":
            src = tempfile.mktemp(suffix=".onnx")
            onnx.save(model, src)
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(src, path, weight_type=QuantType.QInt8)
        else:
            return {"error": f"unknown mode {mode}"}
        return {"path": path, "mode": mode}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {str(e)[:80]}", "hint": "compose with the `quantize`/`compress-select` agents"}


_CAPS = {
    "cost": "measure_cost(model) → params + charged activation memory bytes (+ optional latency); offline-budget triage",
    "verify": "verify_functional(model, cases) → onnxruntime exact/allclose output check",
    "export": "export_torch / export_sklearn → ONNX (guarded torch/skl2onnx)",
    "quantize": "quantize(model, fp16|int8) → shrink graph (composes with quantize/compress-select agents)",
}


class OnnxTool(BaseAgent):
    name = "onnx"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        action = spec.get("action")
        if not action or not (spec.get("model_path") or spec.get("onnx_bytes") or spec.get("module")):
            return self.done({"capabilities": _CAPS, "actions": list(_CAPS)},
                             f"[{worker}] onnx: generic cross-comp ONNX tool — {', '.join(_CAPS)}. "
                             f"Give spec.action + spec.model_path.")
        try:
            m = spec.get("model_path") or spec.get("onnx_bytes")
            if action == "cost":
                res = measure_cost(m, sample_inputs=spec.get("sample_inputs"), banned_ops=spec.get("banned_ops"),
                                   free_io=tuple(spec.get("free_io", ("input", "output"))),
                                   latency=bool(spec.get("latency")))
            elif action == "verify":
                ok, npass, ntot = verify_functional(m, spec.get("cases") or [])
                res = {"ok": ok, "pass": npass, "total": ntot}
            elif action == "quantize":
                res = quantize(m, mode=spec.get("mode", "fp16"), path=spec.get("out_path"))
            else:
                return self.escalate(worker, "researcher", f"[{worker}] onnx: unknown action {action}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] onnx {action} failed: {type(e).__name__}: {str(e)[:80]}")
        msg = f"[{worker}] onnx {action}: {res}"
        self.log(summary=msg[:200], kind="verdict")
        return self.done(res, msg, to="researcher")


_AGENT = OnnxTool()


def run(q, worker):
    return _AGENT.run(q, worker)
