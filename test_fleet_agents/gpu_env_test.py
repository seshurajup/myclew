"""Verifier for gpu_env — the batched-simulation framework and its CONFORMANCE gate.

The gate is the point. A reimplemented environment that differs subtly from the official one trains a
policy that is optimal for OUR simulator and wrong on the ladder — the same failure mode as a CV that
misranks against the real leaderboard. So this test checks that `conform` actually CATCHES a divergence,
not merely that it passes when things agree; a gate that always says OK is worse than no gate.

Offline: CPU tensors only, no CUDA required, no network.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "researchpapers"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fleet_agents import gpu_env as G  # noqa: E402


def _run():
    import torch
    checks = {}

    # --- the shape functions must match the official scalar definitions, including f(0)=0 and ln(1+x)
    xs = torch.tensor([0.0, 0.5, 1.0, 7.0, 100.0, 12345.0], dtype=torch.float64)
    ref = {"linear": lambda x: x, "sq": lambda x: x * x, "sqrt": math.sqrt,
           "log": lambda x: math.log(1.0 + x), "log10": lambda x: math.log10(1.0 + x)}
    for name, f in ref.items():
        got = G.shape_batch(name, xs)
        want = torch.tensor([f(float(v)) for v in xs], dtype=torch.float64)
        checks[f"shape `{name}` matches the scalar definition"] = bool(
            torch.allclose(got, want, atol=1e-12, rtol=1e-12))
    checks["every shape maps 0 -> 0"] = all(
        abs(float(G.shape_batch(n, torch.tensor([0.0], dtype=torch.float64))[0])) < 1e-12 for n in G.SHAPES)

    # --- market price: reproduce the official formula independently and compare
    params = {"base": 250, "I0": 10000, "T": 300, "below_func": "log", "below_target": 0.20,
              "above_func": "sq", "above_target": 3.60}

    def ref_price(inv):
        d = abs(inv - params["I0"])
        if inv < params["I0"]:
            f, tgt = params["below_func"], params["below_target"]
            amp = tgt * params["base"] / ref[f](params["T"])
            p = params["base"] + amp * ref[f](d)
        else:
            f, tgt = params["above_func"], params["above_target"]
            amp = tgt * params["base"] / ref[f](params["T"])
            p = params["base"] - amp * ref[f](d)
        return max(round(p), 1)

    cases = [0, 1, 9999, 10000, 10001, 10300, 20000]
    r = G.conform(lambda cs: G.market_price_batch(params, torch.tensor([float(c) for c in cs],
                                                                      dtype=torch.float64)),
                  ref_price, cases)
    checks["batched market price conforms to the formula"] = r["equal"]
    checks["price is floored at 1 under a heavy glut"] = float(
        G.market_price_batch(params, torch.tensor([10_000_000.0], dtype=torch.float64))[0]) == 1.0

    # --- THE GATE MUST CATCH A DIVERGENCE. A conformance check that only ever passes is worthless.
    bad = G.conform(lambda cs: torch.tensor([float(ref_price(c)) + 1.0 for c in cs]), ref_price, cases)
    checks["conform DETECTS a wrong implementation"] = (not bad["equal"]) and bad["n_mismatch"] == len(cases)
    checks["and reports the first offending input"] = (
        bad["first_mismatch"] is not None and "case" in bad["first_mismatch"])
    subtle = list(cases)
    checks["conform catches a single subtle off-by-one too"] = not G.conform(
        lambda cs: torch.tensor([float(ref_price(c)) + (1.0 if i == 3 else 0.0)
                                 for i, c in enumerate(cs)]), ref_price, subtle)["equal"]

    # --- shape mismatch must be reported, not raise
    sm = G.conform(lambda cs: torch.tensor([1.0]), ref_price, cases)
    checks["a shape mismatch is reported, not raised"] = (not sm["equal"]) and "shape" in sm.get("why", "")

    checks["device selection falls back to CPU without CUDA"] = str(G.torch_device(prefer_gpu=False)) == "cpu"

    # --- FUSED all-product path must equal the per-product path exactly, and stay allocation-free.
    # Measured: the per-product version recomputed amp = target*base/f(T) every call (CUDA kernels for
    # scalar maths) and made one call per product — 14 kernels/call, only 17.6us of real GPU work, and a
    # per-call cost pinned at ~51us from B=128 to B=4096 (32x work, same time) = launch-bound. Hoisting the
    # constants and fusing the products halved it; replacing Python-list column indexing with precomputed
    # masks made it CUDA-Graph capturable (fancy indexing allocates mid-capture, which is forbidden).
    mp = {"A": params, "B": dict(params, base=25, above_func="log", above_target=0.20)}
    prods = ["A", "B"]
    pre = G.precompute_market(mp, prods)
    invs = torch.tensor([[0.0, 0.0], [9999.0, 9999.0], [10000.0, 10000.0],
                         [10001.0, 10001.0], [50000.0, 50000.0]], dtype=torch.float64)
    fused = G.market_price_all(pre, invs)
    per_prod = torch.stack([G.market_price_batch(mp[p], invs[:, j]) for j, p in enumerate(prods)], dim=1)
    checks["fused all-product path equals the per-product path"] = bool(torch.equal(fused, per_prod))
    checks["precompute hoists amp constants to the host"] = (
        "amp_below" in pre and "amp_above" in pre)
    checks["and builds graph-safe masks (no per-call indexing)"] = (
        "masks_below" in pre and "masks_above" in pre
        and all(hasattr(m, "shape") for m in pre["masks_below"].values()))

    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"  {sum(1 for v in checks.values() if v)}/{len(checks)} passed")
    return all(checks.values())


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
