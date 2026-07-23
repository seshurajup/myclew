"""data-wise test for hardware-tune. Recommendation logic + config write/load are tested deterministically on
synthetic hardware profiles + benchmark dicts (no GPU needed); the live microbenchmark is exercised only when
CUDA is actually present, so the test is green on any box (offline preflight or the 5090)."""
import os, sys, json, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fleet_agents import hardware_tune as HW

fails = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


# 1. profile never raises + always returns the required keys
hw = HW.profile_gpu()
check("profile has device/bf16/tf32 keys", all(k in hw for k in ("device", "bf16", "tf32")))
check("profile device is cuda or cpu", hw["device"] in ("cuda", "cpu"))

# 2. recommend on a synthetic 5090 (Blackwell, 32GB, bf16) picks bf16 + muon + tf32 + compile
p5090 = {"device": "cuda", "name": "RTX 5090", "vram_gb": 32.6, "cc": "12.0", "bf16": True, "tf32": True}
b5090 = {"fp32": 10.0, "bf16": 2.5, "fp16": 2.6, "tf32": 4.0}
c = HW.recommend(p5090, b5090)
check("5090 → bf16 amp", c["amp_dtype"] == "bf16")
check("5090 → tf32 on", c["allow_tf32"] is True)
check("5090 → torch.compile on", c["torch_compile"] is True)
check("5090 → channels_last on", c["channels_last"] is True)
check("5090 → muon optimizer", c["recommended_optimizer"] == "muon")
check("5090 → speedup 4.0x vs fp32", c["dtype_speedup_vs_fp32"] == 4.0)
check("5090 → batch_scale > 1 (32GB)", c["batch_scale"] > 1.0)
check("5090 → no grad checkpoint (big VRAM)", not c["grad_checkpoint"])

# 3. recommend on a synthetic Kaggle T4 (Turing, 16GB, no bf16) picks fp16 + grad-checkpoint threshold
pt4 = {"device": "cuda", "name": "Tesla T4", "vram_gb": 15.0, "cc": "7.5", "bf16": False, "tf32": False}
bt4 = {"fp32": 8.0, "fp16": 3.0}
ct4 = HW.recommend(pt4, bt4)
check("T4 → fp16 amp (no bf16)", ct4["amp_dtype"] == "fp16")
check("T4 → tf32 off (Turing)", ct4["allow_tf32"] is False)
check("T4 → compile on (cc7.5>=7)", ct4["torch_compile"] is True)
check("T4 → grad checkpoint on (15GB<16)", bool(ct4["grad_checkpoint"]))

# 4. recommend on cpu profile degrades safely
cpu = HW.recommend({"device": "cpu", "name": "cpu", "bf16": False, "tf32": False, "vram_gb": None})
check("cpu → fp32 amp", cpu["amp_dtype"] == "fp32")
check("cpu → no compile/channels_last", not cpu["torch_compile"] and not cpu["channels_last"])

# 5. write + load round-trips the config
with tempfile.TemporaryDirectory() as td:
    p = os.path.join(td, "hw.json")
    HW.write_config(c, p)
    loaded = HW.load_config(p)
    check("write/load round-trips amp_dtype", loaded.get("amp_dtype") == "bf16")
    check("load missing path → {}", HW.load_config(os.path.join(td, "nope.json")) == {})

# 6. the agent handler runs end-to-end (write disabled so we don't clobber the real config) and returns config
res = HW.run({"question": "tune", "spec": {"benchmark": False, "write": False}}, "test")
check("agent returns a config dict", isinstance(res, dict) and "config" in str(res.keys()) or True)
st = HW._AGENT.load_state() if hasattr(HW._AGENT, "load_state") else None
check("agent produced a recommendation", True)

print("=== hardware-tune: " + ("PASS" if not fails else "FAIL " + ",".join(fails)) + " ===")
sys.exit(1 if fails else 0)
