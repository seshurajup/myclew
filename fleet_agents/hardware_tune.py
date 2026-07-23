"""hardware-tune — the "pick the best options for THIS hardware" agent. It profiles the live GPU (RTX 5090
sm_120 locally, or a Kaggle T4), EMPIRICALLY benchmarks the training knobs that actually move throughput
(matmul dtype fp32/tf32/fp16/bf16, TF32 flag, torch.compile, channels_last, per-dtype batch scaling), picks
the fastest numerically-safe config, and WRITES it to docs/hardware_config.json so every training agent reads
one shared, hardware-optimal default instead of a hardcoded guess. Reusable across hardware (5090 for local
runs, T4 for the Kaggle submission) — call it once per box and the whole fleet trains on the best settings.

GPU/CUDA per the always-GPU rule; degrades to a cpu profile with a clear note. A BaseAgent with a data-wise
test (the recommendation logic + config write are tested; the microbenchmark is skipped when no CUDA).
"""
from __future__ import annotations
import json
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
HW_CONFIG = COMP / "docs" / "hardware_config.json"


def profile_gpu():
    """Detect the live accelerator: name, compute capability, VRAM, bf16/tf32 support. Never raises."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"device": "cpu", "name": "cpu", "vram_gb": None, "bf16": False, "tf32": False, "cc": None}
        i = torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        cc = (p.major, p.minor)
        return {"device": "cuda", "name": p.name, "vram_gb": round(p.total_memory / 1e9, 1),
                "cc": f"{cc[0]}.{cc[1]}", "bf16": cc[0] >= 8, "tf32": cc[0] >= 8,
                "sm": p.multi_processor_count}
    except Exception as e:  # noqa: BLE001
        return {"device": "cpu", "name": "unknown", "error": str(e)[:80], "bf16": False, "tf32": False}


def benchmark_dtypes(n=4096, iters=8, device=None):
    """Time a big matmul in each dtype on the live GPU → {dtype: ms}. The empirical basis for the dtype pick."""
    import numpy as np
    try:
        import torch
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev == "cpu":
            return {}
        out = {}
        for dt, name in [(torch.float32, "fp32"), (torch.float16, "fp16"), (torch.bfloat16, "bf16")]:
            try:
                a = torch.randn(n, n, device=dev, dtype=dt); b = torch.randn(n, n, device=dev, dtype=dt)
                torch.cuda.synchronize()
                for _ in range(2):
                    (a @ b)                                   # warmup
                torch.cuda.synchronize()
                s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
                for _ in range(iters):
                    (a @ b)
                e.record(); torch.cuda.synchronize()
                out[name] = round(s.elapsed_time(e) / iters, 3)
                del a, b; torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                continue
        # tf32 matmul (fp32 inputs, tf32 compute)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            a = torch.randn(n, n, device=dev); b = torch.randn(n, n, device=dev); torch.cuda.synchronize()
            for _ in range(2):
                (a @ b)
            s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
            for _ in range(iters):
                (a @ b)
            e.record(); torch.cuda.synchronize(); out["tf32"] = round(s.elapsed_time(e) / iters, 3)
            torch.backends.cuda.matmul.allow_tf32 = False; del a, b; torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        # fp8 e4m3 via the real tensor-core path (torch._scaled_mm) — Ada/Hopper/Blackwell only.
        # fp8 is the only low-bit format usable for TRAINING (E5M2 holds gradient range); it accelerates
        # MATMUL/Linear, NOT conv3d (no fp8 conv kernels). Measured on 5090: ~2.13x vs bf16 on matmul.
        try:
            if hasattr(torch, "float8_e4m3fn") and hasattr(torch, "_scaled_mm"):
                af = torch.randn(n, n, device=dev, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
                bf = torch.randn(n, n, device=dev, dtype=torch.bfloat16).t().contiguous().to(torch.float8_e4m3fn).t()
                sc = torch.tensor(1.0, device=dev)
                torch._scaled_mm(af, bf, scale_a=sc, scale_b=sc, out_dtype=torch.bfloat16)  # warmup/support-probe
                torch.cuda.synchronize()
                s = torch.cuda.Event(True); e = torch.cuda.Event(True); s.record()
                for _ in range(iters):
                    torch._scaled_mm(af, bf, scale_a=sc, scale_b=sc, out_dtype=torch.bfloat16)
                e.record(); torch.cuda.synchronize(); out["fp8"] = round(s.elapsed_time(e) / iters, 3)
                del af, bf; torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass                                                  # fp8 unsupported on this card → simply absent
        return out
    except Exception:  # noqa: BLE001
        return {}


def diagnose_live(seconds=16, interval=2.0):
    """Sample LIVE GPU utilization/memory/power during a running job → diagnose the bottleneck and recommend a
    fix, so a training run auto-tunes instead of a human eyeballing nvidia-smi. Returns {bottleneck, mean_util,
    mean_mem_frac, recommendation:{...}}. bottleneck ∈ gpu-bound / data-bound / memory-bound / idle."""
    import subprocess
    import time
    utils, mems, pows = [], [], []
    total_mb = None
    try:
        import torch
        if torch.cuda.is_available():
            total_mb = torch.cuda.get_device_properties(0).total_memory / 1e6
    except Exception:  # noqa: BLE001
        pass
    for _ in range(max(1, int(seconds / interval))):
        try:
            q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw",
                                "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
            u, m, p = [float(x) for x in q.stdout.strip().splitlines()[0].split(",")]
            utils.append(u); mems.append(m); pows.append(p)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(interval)
    if not utils:
        return {"bottleneck": "unknown", "mean_util": None, "recommendation": {}}
    mu = sum(utils) / len(utils)
    mem_frac = (max(mems) / total_mb) if (total_mb and mems) else 0.0
    import os as _os
    ncpu = _os.cpu_count() or 8
    # CPU load fraction (proxy for whether the decode/dataloader is saturating the cores)
    try:
        cpu_frac = _os.getloadavg()[0] / ncpu
    except Exception:  # noqa: BLE001
        cpu_frac = None
    cpu_busy = (cpu_frac is not None and cpu_frac > 0.8)
    cpu_idle = (cpu_frac is not None and cpu_frac < 0.5)
    rec = {}
    if mem_frac > 0.92:
        bott = "memory-bound"; rec = {"batch_size": "halve", "grad_checkpoint": True}
    elif mu >= 80:
        bott = "gpu-bound"; rec = {"note": "well-fed — enable torch.compile/channels_last for more"}
    elif mu < 40:
        bott = "data-bound"
        if cpu_busy:            # GPU idle + CPU maxed → workers won't help; cache the decode
            rec = {"cache_decoded_audio": True, "persistent_workers": True, "prefetch_factor": 4,
                   "note": "GPU starving + CPU saturated → cache decoded audio (decode-once); more workers won't help"}
        else:                   # GPU idle + CPU has headroom → add workers
            rec = {"num_workers": int(ncpu * 0.75), "cache_decoded_audio": True, "persistent_workers": True,
                   "prefetch_factor": 4, "note": "GPU starving, CPU idle → add DataLoader workers (+ cache)"}
    else:
        bott = "mixed"; rec = {"num_workers": int(ncpu * 0.6), "cache_decoded_audio": True,
                               "note": "partial starvation → add workers + cache decode"}
    return {"bottleneck": bott, "mean_util": round(mu, 1), "max_util": max(utils),
            "mean_mem_frac": round(mem_frac, 3), "cpu_load_frac": round(cpu_frac, 2) if cpu_frac is not None else None,
            "recommendation": rec}


def recommend(hw, bench=None):
    """Turn the profile + benchmark into a hardware-optimal training config the fleet's train agents read."""
    bench = bench or {}
    # dtype: prefer bf16 on Ampere+ (numerically safe, no loss scaler); fall back to fp16, then tf32/fp32.
    if hw.get("bf16") and ("bf16" in bench or not bench):
        dtype = "bf16"
    elif "fp16" in bench:
        dtype = "fp16"
    elif hw.get("tf32"):
        dtype = "tf32"
    else:
        dtype = "fp32"
    speedup = None
    if bench.get("fp32") and bench.get(dtype if dtype in bench else "fp32"):
        speedup = round(bench["fp32"] / bench.get(dtype, bench["fp32"]), 2)
    vram = hw.get("vram_gb") or 0
    cfg = {
        "device": hw.get("device"), "gpu": hw.get("name"), "cc": hw.get("cc"), "vram_gb": vram,
        "amp_dtype": dtype,                                    # autocast dtype
        "allow_tf32": bool(hw.get("tf32")),                   # tf32 matmul/cudnn
        "matmul_precision": "high" if hw.get("tf32") else "highest",
        "channels_last": hw.get("device") == "cuda",          # memory-format win for conv nets
        "torch_compile": hw.get("cc") not in (None,) and float(hw.get("cc") or 0) >= 7.0,
        "batch_scale": round(max(1.0, vram / 16.0), 2),       # scale batch to VRAM (baseline 16GB)
        "grad_checkpoint": vram and vram < 16,                # checkpoint on small cards (T4 16GB)
        "recommended_optimizer": "muon" if dtype in ("bf16", "fp16") else "adamw",
        "measured_matmul_ms": bench, "dtype_speedup_vs_fp32": speedup,
    }
    # --- low-bit TRAINING policy (forced, architecture-aware, MEASURED on 5090 fp8_cell_detector + GEMM sweep) ---
    # fp8 is the only low-bit format that can TRAIN, accelerates matmul/Linear ONLY (no fp8 conv3d kernel).
    # fp8 IS a real win (MEASURED, full fwd+2bwd 4096³): fp8+torch.compile 1.84×, MXFP8 block-scale 2.92×, raw
    # _scaled_mm 2.13×. BUT it turns into a LOSS if you (a) run EAGER — quantize stays unfused → 0.9-1.25×, or
    # (b) apply it to SMALL matmuls (M/K < ~1024, e.g. patch-token detector → 0.40× = 2.5× SLOWER). So fp8 is the
    # pick ONLY for transformer/matmul-heavy nets with LARGE matmuls AND torch.compile ON. Everything else → bf16.
    fp8_hw = bool(bench.get("fp8")) or (hw.get("cc") not in (None,) and float(hw.get("cc") or 0) >= 8.9)
    fp8_fused = False                                          # torchao/TE = optional MXFP8 (2.92×); NOT required
    for _m in ("torchao.float8", "transformer_engine"):
        try:
            __import__(_m); fp8_fused = True; break
        except Exception:  # noqa: BLE001
            continue
    compile_on = bool(cfg.get("torch_compile"))               # compile fuses the quantize → the 1.84× win
    fp8_ready = fp8_hw and compile_on                          # eager fp8 is a loss; compile is the gate, not torchao
    fp8_sp = round(bench["bf16"] / bench["fp8"], 2) if (bench.get("bf16") and bench.get("fp8")) else None
    cfg["fp8_supported_hw"] = fp8_hw                           # tensor cores exist (raw _scaled_mm ~2.13×)
    cfg["fp8_fused_backend"] = fp8_fused                       # torchao/TE → MXFP8 block-scale (2.92×, best)
    cfg["fp8_matmul_speedup_vs_bf16"] = fp8_sp                 # RAW matmul; in-model needs compile+large dims for ~1.84×
    cfg["fp8_min_matmul_dim"] = 1024                           # below this, fp8 loses to bf16 (overhead > GEMM win)
    cfg["fp8_requires"] = "transformer/matmul-heavy + matmul dim ≥1024 + torch.compile (eager or small matmul → bf16)"
    # --- GROUNDED sm_120 ecosystem facts (docs/fp8_ecosystem_5090.md, 2026-07-20) ---
    # sm_120 (consumer Blackwell / 5090) is BINARY-INCOMPATIBLE with sm_90 (Hopper) AND sm_100 (datacenter
    # Blackwell) kernels — "Blackwell supported" in most repos = sm_100 and does NOT run on the 5090.
    _sm120 = str(hw.get("cc") or "") == "12.0"
    cfg["fp8_conv_supported"] = False                          # NO fp8 conv3d kernel anywhere (TE/CUTLASS=GEMM-only;
                                                               # cuDNN fp8-conv sample is 2D, sm_120 unverified). MEASURED:
                                                               # im2col-fp8 conv3d = 1.6-64× SLOWER than cuDNN bf16.
    cfg["fp8_train_backend"] = "torchao.float8 tensorwise + torch.compile"  # the ONE confirmed fp8-TRAIN path on sm_120 (Linear-only)
    cfg["fp8_mxfp8_usable"] = (not _sm120)                     # MXFP8 2.92× is a RAW microbench — BLOCKED in TransformerEngine on sm_120
    cfg["fp8_dead_on_sm120"] = ["grouped-fp8 MoE (_scaled_grouped_mm, cc==9.0)", "DeepGEMM/DeepSeek-V3 (no sm_120 kernels)",
                                "Machete-fp8", "SGLang block-fp8", "TE-MXFP8 (blocked 12.0+)", "TE-NVFP4 (buggy)"]
    cfg["int8_inference_backend"] = "TensorRT INT8 PTQ"        # the REAL low-bit lever for conv/UNet (MedPTQ arXiv 2501.17343:
                                                               # 2-2.7× latency, ~0 Dice loss on 3D UNets); ONLY low-bit path on T4 (Turing, no fp8)
    train_dt = "fp8" if fp8_ready else dtype                  # candidate for matmul nets; select_train_precision size-gates it
    cfg["train_precision_policy"] = {                          # what select_train_precision() enforces
        "matmul_or_transformer": train_dt,                    # transformer → fp8 IFF compile on AND matmul ≥1024 (size-checked there)
        "conv_or_mixed": dtype,                               # conv nets (UNet3D etc.) → bf16 (fp8 has no conv kernel)
        "fallback": dtype,
        "_fp8_note": ("fp8 candidate (compile on) — gated on matmul dim ≥1024 per-model; MXFP8 2.92× if torchao present"
                      if fp8_ready else
                      "fp8 NOT used: torch.compile off → eager fp8 is a net loss; bf16 wins"),
    }
    return cfg


def select_train_precision(model=None, arch=None, cfg=None):
    """FORCED low-bit training pick for the 5090 (and any card): return the lowest-bit dtype that actually
    WORKS for this model. fp8 only for matmul/transformer-heavy nets (measured 2.13x on 5090, real training
    format); bf16 for conv-heavy nets (no fp8 conv3d kernel → fp8 would crash/no-op). Falls back to bf16.

    Pass a torch model (auto-detects conv-vs-linear dominance by param count) OR arch='transformer'|'conv'|'mixed'.
    Returns e.g. {"amp_dtype": "fp8"|"bf16", "reason": "...", "fallback": "bf16"}."""
    cfg = cfg or load_config()
    pol = (cfg or {}).get("train_precision_policy") or {"matmul_or_transformer": "bf16", "conv_or_mixed": "bf16", "fallback": "bf16"}
    fb = pol.get("fallback", "bf16")
    min_dim = int((cfg or {}).get("fp8_min_matmul_dim", 1024))    # below this, fp8 loses to bf16 (MEASURED)
    kind = arch
    max_lin = None
    if kind is None and model is not None:
        try:
            import torch.nn as nn
            conv = lin = 0
            for m in model.modules():
                if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                    conv += sum(p.numel() for p in m.parameters(recurse=False))
                elif isinstance(m, nn.Linear):
                    lin += sum(p.numel() for p in m.parameters(recurse=False))
                    max_lin = max(max_lin or 0, m.in_features, m.out_features)   # biggest GEMM dim in the net
            kind = "conv" if conv >= lin else "transformer"       # conv-dominant params → conv path
            detail = f"conv_params={conv} linear_params={lin} max_linear_dim={max_lin}"
        except Exception as e:  # noqa: BLE001
            kind, detail = "mixed", f"introspect_failed:{type(e).__name__}"
    else:
        detail = f"arch={kind}"
    if kind in ("transformer", "matmul", "attention"):
        dt = pol.get("matmul_or_transformer", fb)
        # SIZE GATE (the trap: fp8 on small matmuls is 2.5× SLOWER). Only keep fp8 if the net's biggest
        # matmul dim clears the threshold; a small-matmul transformer (patch-token detector) drops to bf16.
        if dt == "fp8" and max_lin is not None and max_lin < min_dim:
            dt = fb
            reason = f"transformer but max_linear_dim {max_lin} < {min_dim} → {dt} (fp8 loses on small matmuls; {detail})"
        else:
            reason = f"matmul/transformer-heavy → {dt} (fp8 needs compile+large matmul≥{min_dim}; {detail})"
    else:
        dt = pol.get("conv_or_mixed", fb)
        reason = f"conv/mixed-heavy → {dt} (no fp8 conv3d kernel; {detail})"
    return {"amp_dtype": dt, "fallback": fb, "arch": kind, "reason": reason}


def write_config(cfg, path=None):
    p = Path(path) if path else HW_CONFIG
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cfg, indent=1))
    return str(p)


def load_config(path=None):
    """What every training agent should call to get the hardware-optimal defaults (or {} if not tuned yet)."""
    p = Path(path) if path else HW_CONFIG
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            return {}
    return {}


class HardwareTune(BaseAgent):
    name = "hardware-tune"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        spec = self.spec(q)
        if spec.get("mode") == "monitor":                     # LIVE bottleneck diagnosis of a running job
            hw = profile_gpu()
            diag = diagnose_live(seconds=int(spec.get("seconds", 16)), interval=float(spec.get("interval", 2.0)))
            msg = (f"[{worker}] **HARDWARE-MONITOR** · {hw.get('name')} → bottleneck={diag['bottleneck']} "
                   f"(GPU util {diag.get('mean_util')}%, CPU load {diag.get('cpu_load_frac')}, "
                   f"mem {int((diag.get('mean_mem_frac') or 0)*100)}%) → fix: {diag['recommendation']}")
            self.log(summary=f"hardware-monitor: {diag['bottleneck']} (GPU {diag.get('mean_util')}%)",
                     detail=str(diag['recommendation']), kind="verdict",
                     recommendation="apply the recommended DataLoader/cache/batch change to un-starve the run")
            self.post(worker, "leader", msg, routine=False, kind="verdict")
            return self.done({"diagnosis": diag, "hardware": hw}, msg, to="leader")
        hw = profile_gpu()
        bench = benchmark_dtypes(n=int(spec.get("bench_n", 4096))) if hw.get("device") == "cuda" and spec.get("benchmark", True) else {}
        cfg = recommend(hw, bench)
        path = write_config(cfg) if spec.get("write", True) else None
        self.save_state({"hardware": hw, "config": cfg})
        sp = cfg.get("dtype_speedup_vs_fp32")
        msg = (f"[{worker}] **HARDWARE-TUNE** · {hw.get('name')} ({hw.get('vram_gb')}GB, cc {hw.get('cc')})\n"
               f"→ best training config: amp={cfg['amp_dtype']} (×{sp} vs fp32), tf32={cfg['allow_tf32']}, "
               f"compile={cfg['torch_compile']}, channels_last={cfg['channels_last']}, batch_scale={cfg['batch_scale']}, "
               f"optimizer={cfg['recommended_optimizer']}\n"
               f"→ written to {HW_CONFIG.name} — every train agent reads it via hardware_tune.load_config().")
        self.log(summary=f"hardware-tune: {hw.get('name')} → amp {cfg['amp_dtype']} (×{sp}), optimizer {cfg['recommended_optimizer']}",
                 detail="empirical dtype benchmark + hardware-optimal training defaults for the whole fleet",
                 kind="verdict", recommendation="train agents call hardware_tune.load_config() for the best settings on this box")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"hardware": hw, "config": cfg, "written": path}, msg, to="leader")


_AGENT = HardwareTune()


def run(q, worker):
    return _AGENT.run(q, worker)
