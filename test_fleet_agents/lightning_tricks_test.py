"""lightning_tricks_test — data-wise verifier for the PyTorch Lightning advisor.

Core properties:
  1. trainer_kwargs picks bf16-mixed on Ampere+/Blackwell, 16-mixed on older, 32-true on CPU.
  2. gradient accumulation derived from effective/base batch.
  3. multi-GPU → a strategy is set.
  4. callbacks + tricks knowledge base are non-empty and well-formed.
  5. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import lightning_tricks as L


def _run():
    print("=== LIGHTNING-TRICKS VERIFIER ===")
    checks = {}

    # 1. precision by capability
    checks["blackwell_bf16"] = L.trainer_kwargs((12, 0))["precision"] == "bf16-mixed"
    checks["ampere_bf16"] = L.trainer_kwargs((8, 0))["precision"] == "bf16-mixed"
    checks["turing_fp16"] = L.trainer_kwargs((7, 5))["precision"] == "16-mixed"   # Kaggle T4
    cpu = L.trainer_kwargs(None, n_gpus=0)     # n_gpus=0 forces CPU regardless of detected hardware
    checks["cpu_32"] = cpu["precision"] == "32-true" and cpu["accelerator"] == "cpu"
    print(f"  -> precision: 5090→{L.trainer_kwargs((12,0))['precision']} T4→{L.trainer_kwargs((7,5))['precision']} cpu→{cpu['precision']}")

    # 2. grad accumulation
    kw = L.trainer_kwargs((12, 0), accumulate_eff_batch=256, base_batch=32)
    checks["accum"] = kw["accumulate_grad_batches"] == 8

    # 3. multi-gpu strategy
    checks["multi_gpu_strategy"] = "strategy" in L.trainer_kwargs((8, 0), n_gpus=2)
    checks["single_gpu_no_strategy"] = "strategy" not in L.trainer_kwargs((8, 0), n_gpus=1)

    # 4. callbacks + tricks
    cbs = L.recommended_callbacks(swa=True)
    names = {c["cls"] for c in cbs}
    checks["callbacks"] = {"EarlyStopping", "ModelCheckpoint", "LearningRateMonitor", "StochasticWeightAveraging"} <= names
    tr = L.tricks()
    checks["tricks_nonempty"] = len(tr) >= 8 and all(len(t) == 2 for t in tr)
    checks["tricks_mention_bf16_compile"] = any("bf16" in t[0] for t in tr) and any("compile" in t[0] for t in tr)
    print(f"  -> {len(cbs)} callbacks, {len(tr)} tricks")

    # 5. agent
    st, dta, to, msg = L.run_lightning({"spec": {"cap": [12, 0], "eff_batch": 128, "base_batch": 16, "swa": True}}, "t")
    checks["agent_done"] = st == "done" and dta["precision"] == "bf16-mixed" and dta["kwargs"]["accumulate_grad_batches"] == 8

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== lightning-tricks: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
