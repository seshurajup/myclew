"""fp8_cell_detector_verify — MEASURED proof for fp8_cell_detector on the RTX 5090.

Runs the 5 checks the task asks for and prints a machine-readable JSON block:
  1. fp8 end-to-end train step RUNS (fwd+loss+bwd+opt.step) without error.
  2. CONVERGENCE: loss start->end over N steps, fp8 vs bf16, on synthetic 3D Gaussian-blob volumes.
  3. SPEED: s/iter fp8 vs bf16 (real ratio).
  4. fp8 COMPUTE FRACTION: % of forward MACs through fp8.
  5. PEAK VRAM fp8 vs bf16.
Also confirms hardware_tune.select_train_precision(model) returns 'fp8' and reports the param split.

Run:  OMP_NUM_THREADS=1 research/cellmot_venv/bin/python fleet_agents/fp8_cell_detector_verify.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

from fleet_agents import fp8_cell_detector as M      # noqa: E402
from fleet_agents import hardware_tune as HT         # noqa: E402

DEV = "cuda"
torch.manual_seed(0)


def make_batch(bs, vol, n_blobs=6, sigma=2.0):
    """Synthetic 3D volumes with a few Gaussian cell-center blobs + noise. target = Gaussian heatmap."""
    D, H, W = vol
    x = torch.rand(bs, 1, D, H, W, device=DEV) * 0.1
    y = torch.zeros(bs, 1, D, H, W, device=DEV)
    zz, yy, xx = torch.meshgrid(torch.arange(D, device=DEV), torch.arange(H, device=DEV),
                                torch.arange(W, device=DEV), indexing="ij")
    for b in range(bs):
        for _ in range(n_blobs):
            cz = torch.randint(2, D - 2, (1,), device=DEV).item()
            cy = torch.randint(4, H - 4, (1,), device=DEV).item()
            cx = torch.randint(4, W - 4, (1,), device=DEV).item()
            g = torch.exp(-(((zz - cz) ** 2) / (2 * sigma ** 2)
                            + ((yy - cy) ** 2) / (2 * sigma ** 2)
                            + ((xx - cx) ** 2) / (2 * sigma ** 2)))
            y[b, 0] = torch.maximum(y[b, 0], g)
            x[b, 0] += g            # blobs are visible in the input
    return x, y


def train_loop(model, steps, bs, vol, lr=2e-3, seed=0):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    fixed = make_batch(bs, vol)                    # a fixed batch: convergence must be visible
    for s in range(steps):
        x, y = fixed
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(x)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(out.float(), y.float())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def time_iters(model, bs, vol, iters=30, warmup=8):
    x, y = make_batch(bs, vol)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(warmup):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), y.float())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), y.float())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters


def peak_vram(model, bs, vol):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    x, y = make_batch(bs, vol)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(x).float(), y.float())
    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e9


def main():
    assert torch.cuda.is_available(), "needs the 5090"
    vol = (16, 64, 64)
    bs = 4
    steps = int(os.environ.get("STEPS", "300"))
    res = {"device": torch.cuda.get_device_name(0), "cc": torch.cuda.get_device_capability(0),
           "vol": vol, "bs": bs, "steps": steps,
           "fp8_backward": os.environ.get("CELLMOT_FP8_BACKWARD", "1") == "1"}

    # ---- param split + select_train_precision ----
    m0 = M.build_default().to(DEV)
    split = M.param_split(m0)
    sel = HT.select_train_precision(model=m0)
    frac = M.fp8_flop_fraction(m0, vol)
    res["param_split"] = split
    res["select_train_precision"] = sel
    res["fp8_flop_fraction"] = frac

    # ---- (1) fp8 train step runs ----
    step_ran = None
    try:
        _ = train_loop(m0.set_fp8(True), steps=2, bs=bs, vol=vol)
        step_ran = True
    except Exception as e:  # noqa: BLE001
        step_ran = f"ERROR: {type(e).__name__}: {e}"
    res["fp8_train_step_runs"] = step_ran

    # ---- (2) convergence fp8 vs bf16 (fresh models, same seed) ----
    m_fp8 = M.build_default().to(DEV).set_fp8(True)
    m_bf16 = M.build_default().to(DEV).set_fp8(False)
    l_fp8 = train_loop(m_fp8, steps, bs, vol, seed=1)
    l_bf16 = train_loop(m_bf16, steps, bs, vol, seed=1)
    res["convergence"] = {
        "fp8": {"start": l_fp8[0], "end": l_fp8[-1], "min": min(l_fp8),
                "diverged": (not (l_fp8[-1] < l_fp8[0])) or any(x != x for x in l_fp8)},
        "bf16": {"start": l_bf16[0], "end": l_bf16[-1], "min": min(l_bf16),
                 "diverged": (not (l_bf16[-1] < l_bf16[0])) or any(x != x for x in l_bf16)},
    }

    # ---- (3) speed ----
    t_fp8 = time_iters(M.build_default().to(DEV).set_fp8(True), bs, vol)
    t_bf16 = time_iters(M.build_default().to(DEV).set_fp8(False), bs, vol)
    res["speed"] = {"fp8_s_per_iter": t_fp8, "bf16_s_per_iter": t_bf16,
                    "fp8_speedup_vs_bf16": t_bf16 / t_fp8}

    # ---- (5) peak VRAM ----
    v_fp8 = peak_vram(M.build_default().to(DEV).set_fp8(True), bs, vol)
    v_bf16 = peak_vram(M.build_default().to(DEV).set_fp8(False), bs, vol)
    res["peak_vram_gb"] = {"fp8": v_fp8, "bf16": v_bf16, "fp8_minus_bf16": v_fp8 - v_bf16}

    print("=== FP8_CELL_DETECTOR_PROOF_JSON ===")
    print(json.dumps(res, indent=1, default=str))
    print("=== END ===")
    return res


if __name__ == "__main__":
    main()
