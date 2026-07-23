"""onnx_tools_test — data-wise verifier for the GENERIC `onnx` tool. Builds tiny ONNX graphs (no framework
export needed) and checks verify_functional + measure_cost (params + charged activation memory). Skips
gracefully (PASS) if onnx/onnxruntime are unavailable.
"""
import os
import sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

try:
    import onnx  # noqa
    import onnxruntime  # noqa
    import numpy as np
    from onnx import helper, TensorProto, numpy_helper
    _HAVE = True
except Exception:  # noqa: BLE001
    _HAVE = False


def _relu_model():
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    n = helper.make_node("Relu", ["input"], ["output"])
    m = helper.make_model(helper.make_graph([n], "g", [x], [y], []),
                          ir_version=10, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(m)
    return m


def _twohop_model():
    # Relu -> Add(bias) : the intermediate Relu output is a CHARGED tensor (4 floats = 16 bytes)
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 4])
    b = numpy_helper.from_array(np.ones((1, 4), np.float32), "b")   # 4 params
    n1 = helper.make_node("Relu", ["input"], ["h"])
    n2 = helper.make_node("Add", ["h", "b"], ["output"])
    m = helper.make_model(helper.make_graph([n1, n2], "g", [x], [y], [b]),
                          ir_version=10, opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(m)
    return m


def _run():
    print("=== ONNX (generic tool) DATA-WISE VERIFIER ===")
    if not _HAVE:
        print("  ~ onnx/onnxruntime not in this env — SKIP")
        print("=== onnx: PASS (skipped) ===")
        return True
    from fleet_agents import onnx_tools as OT
    checks = {}

    # measure_cost: single-node Relu → 0 params, 0 charged memory (only input/output, both FREE)
    c1 = OT.measure_cost(_relu_model())
    checks["relu_valid"] = c1["valid"] is True
    checks["relu_zero_cost"] = c1["params"] == 0 and c1["memory"] == 0 and c1["cost"] == 0

    # measure_cost: two-hop → 4 params (bias) + 16 bytes charged intermediate 'h'
    c2 = OT.measure_cost(_twohop_model())
    checks["twohop_params4"] = c2["params"] == 4
    checks["twohop_mem16"] = c2["memory"] == 16
    checks["twohop_cost20"] = c2["cost"] == 20

    # latency measurement returns a number
    c3 = OT.measure_cost(_relu_model(), latency=True)
    checks["latency_measured"] = isinstance(c3["latency_ms"], float)

    # verify_functional: Relu on a signed vector, exact
    cases = [{"inputs": {"input": np.array([[-1.0, 2.0, -3.0, 4.0]], np.float32)},
              "outputs": {"output": np.array([[0.0, 2.0, 0.0, 4.0]], np.float32)}, "mode": "exact"}]
    ok, npass, ntot = OT.verify_functional(_relu_model(), cases)
    checks["verify_ok"] = ok and npass == 1 and ntot == 1

    # banned-op gate (neurogolf-style) rejects; generic (no banned list) accepts the same graph's validity
    okv, _ = OT.validity(_relu_model(), banned_ops=["Relu"])
    checks["banned_rejects"] = okv is False
    okv2, _ = OT.validity(_relu_model())
    checks["generic_accepts"] = okv2 is True

    # agent contract (empty spec → capabilities)
    status, res, to, msg = OT.run({"question": "smoke", "spec": {}}, "test")
    checks["agent_contract"] = status == "done" and "capabilities" in res

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    print(f"  -> relu cost={c1['cost']}; twohop params={c2['params']} mem={c2['memory']} cost={c2['cost']}; "
          f"latency={c3['latency_ms']}ms")
    ok = all(checks.values())
    print(f"=== onnx: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except Exception as e:  # noqa: BLE001
        print(f"  X ERROR: {type(e).__name__}: {e}")
        sys.exit(1)
