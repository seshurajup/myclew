"""lightning_tricks — a PyTorch Lightning speed/efficiency advisor. Lightning wraps the training loop, so the
performance levers are Trainer flags + a few callbacks, and the RIGHT settings depend on the hardware. This
agent encodes the current best-practice Lightning recipe and emits Trainer kwargs tuned to the detected GPU
(honoring the fleet's always-GPU + low-bit rules): bf16 mixed precision on Ampere+/Blackwell, torch.compile,
gradient accumulation for effective batch size, the right strategy for multi-GPU, and the callbacks that
actually matter (EarlyStopping, ModelCheckpoint(save_top_k), LR monitor, SWA). Pure-python config generation —
offline-testable — plus a knowledge base of Lightning gotchas.

Primitives (stdlib; torch optional for detection):
  • gpu_cap()                              — (major,minor) or None.
  • trainer_kwargs(device, ...)            — recommended Trainer(**kwargs) for the hardware.
  • recommended_callbacks(...)             — the callback set (as spec dicts) worth enabling.
  • tricks()                               — the annotated list of Lightning speed/correctness tricks.
"""
from __future__ import annotations
from .base import BaseAgent


def gpu_cap():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_capability(0)
    except Exception:  # noqa: BLE001
        pass
    return None


def trainer_kwargs(cap=None, n_gpus=1, accumulate_eff_batch=None, base_batch=None, compile_model=True):
    """Recommended `Trainer(**kwargs)` for the detected hardware.
      • precision: 'bf16-mixed' on Ampere+ (sm_80) / Blackwell (bf16 tensor cores, no loss scaling); '16-mixed'
        on older; '32-true' on CPU.
      • accumulate_grad_batches: derived from a target effective batch if given.
      • strategy: 'ddp'/'fsdp' when n_gpus>1 (fsdp for big models), else 'auto'.
      • gradient_clip_val, deterministic hints, benchmark=True for fixed shapes (cudnn autotune)."""
    cap = cap if cap is not None else gpu_cap()
    kw = {"devices": max(1, int(n_gpus)), "benchmark": True, "gradient_clip_val": 1.0, "log_every_n_steps": 20}
    if cap is None or int(n_gpus) == 0:        # no GPU detected, or CPU explicitly requested (n_gpus=0)
        kw.update(accelerator="cpu", precision="32-true", devices=1)
    else:
        kw["accelerator"] = "gpu"
        kw["precision"] = "bf16-mixed" if cap[0] >= 8 else "16-mixed"   # bf16 on Ampere/Ada/Hopper/Blackwell
    if n_gpus and n_gpus > 1:
        kw["strategy"] = "ddp"        # 'fsdp' for models that don't fit one GPU
    if accumulate_eff_batch and base_batch:
        kw["accumulate_grad_batches"] = max(1, int(round(accumulate_eff_batch / base_batch)))
    kw["_compile_hint"] = ("wrap model with torch.compile before Trainer.fit (biggest single speedup on "
                           "large matmuls; skip for tiny/conv-heavy)") if compile_model else None
    return kw


def recommended_callbacks(monitor="val_loss", mode="min", patience=10, save_top_k=2, swa=False):
    """The callbacks worth enabling as spec dicts (map to lightning.pytorch.callbacks)."""
    cbs = [
        {"cls": "EarlyStopping", "monitor": monitor, "mode": mode, "patience": patience},
        {"cls": "ModelCheckpoint", "monitor": monitor, "mode": mode, "save_top_k": save_top_k},
        {"cls": "LearningRateMonitor", "logging_interval": "step"},
    ]
    if swa:
        cbs.append({"cls": "StochasticWeightAveraging", "swa_lrs": 1e-2})
    return cbs


def tricks():
    """Annotated Lightning speed/correctness tricks (the knowledge base)."""
    return [
        ("bf16-mixed precision", "Trainer(precision='bf16-mixed') on Ampere+; ~2× throughput, no loss scaling, "
         "stabler than fp16. #1 free win on the 5090."),
        ("torch.compile", "compile the LightningModule's model before fit; biggest speedup on large matmuls; "
         "eager/small-matmul/conv can regress — measure the full step."),
        ("gradient accumulation", "accumulate_grad_batches raises the EFFECTIVE batch without more VRAM."),
        ("set_float32_matmul_precision('high')", "enable TF32 for fp32 matmuls (Ampere+)."),
        ("benchmark=True", "cudnn autotuner picks fast kernels when input shapes are fixed."),
        ("num_workers + persistent_workers", "DataLoader(num_workers=os.cpu_count(), persistent_workers=True, "
         "pin_memory=True) — dataloading is the usual hidden bottleneck."),
        ("save_top_k, not save-every", "ModelCheckpoint(save_top_k=k) avoids disk thrash."),
        ("EarlyStopping", "stop when the monitored metric plateaus — the successive-halving analog."),
        ("FSDP for big models", "strategy='fsdp' shards params/grads/optim across GPUs when a model won't fit."),
        ("SWA / EMA", "StochasticWeightAveraging or an EMA callback for a cheap generalization bump late in run."),
        ("QAT under Lightning", "wire lowbit_qat.QuantLinear (nvfp4 on 5090 / int8 on T4) into the module for "
         "low-bit training; honors the fleet's train-low-bit rule."),
    ]


# ---------------------------------------------------------------- agent
class LightningTricks(BaseAgent):
    name = "lightning-tricks"
    thread = "M"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        cap = tuple(s["cap"]) if s.get("cap") else gpu_cap()
        n_gpus = int(s.get("n_gpus", 1))
        kw = trainer_kwargs(cap, n_gpus=n_gpus,
                            accumulate_eff_batch=s.get("eff_batch"), base_batch=s.get("base_batch"))
        cbs = recommended_callbacks(swa=bool(s.get("swa")))
        prec = kw.get("precision"); nt = len(tricks())
        msg = (f"lightning-tricks [{'gpu '+str(cap) if cap else 'cpu'}, {n_gpus} dev]: Trainer precision="
               f"{prec}, benchmark=True, clip=1.0"
               f"{', accum='+str(kw['accumulate_grad_batches']) if 'accumulate_grad_batches' in kw else ''}"
               f"{', strategy='+kw['strategy'] if 'strategy' in kw else ''}; {len(cbs)} callbacks "
               f"(EarlyStopping/Checkpoint/LRMonitor{'/SWA' if s.get('swa') else ''}); {nt} tricks in KB "
               f"(bf16-mixed + torch.compile + grad-accum are the top-3). QAT via lowbit_qat for low-bit training")
        self.log(msg, kind="finding",
                 recommendation="apply trainer_kwargs() to Lightning Trainer; compile the model; bf16-mixed on "
                                "the 5090; add EarlyStopping+ModelCheckpoint(save_top_k)")
        return self.done({"precision": prec, "n_callbacks": len(cbs), "n_tricks": nt,
                          "kwargs": {k: v for k, v in kw.items() if not k.startswith("_")}}, msg)


_AGENT = LightningTricks()


def run_lightning(q, worker):
    return _AGENT.run(q, worker)
