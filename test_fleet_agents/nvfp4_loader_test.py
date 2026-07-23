"""nvfp4_loader_test — data-wise verifier for the Gemma-4 NVFP4 / RTX 5090 loading planner.

Core properties:
  1. nvfp4_weight_gb: 12B NVFP4 ≈ 6.75GB (matches "runs on 11GB VRAM"); scales with params.
  2. Every Gemma-4 NVFP4 variant FITS the 32GB 5090 at a reasonable context; KV-cache grows with context.
  3. 26B-A4B total > 12B weights but active params are small (MoE accounting preserved).
  4. max_context is positive and shrinks for bigger models.
  5. plan() emits vLLM + Unsloth recipes and warns off Marlin.
  6. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import nvfp4_loader as N


def _run():
    print("=== NVFP4-LOADER VERIFIER ===")
    checks = {}

    # 1. weight footprint
    w12 = N.nvfp4_weight_gb(12e9)
    print(f"  -> 12B NVFP4 weights = {w12:.2f} GB")
    checks["12b_weight_fits_11gb"] = 6.0 < w12 < 7.5           # ~6.75GB → the "11GB VRAM" claim holds
    checks["weight_scales"] = N.nvfp4_weight_gb(31e9) > N.nvfp4_weight_gb(12e9)

    # 2. all variants fit 5090 32GB
    for m in ("e2b", "e4b", "12b", "26b-a4b", "31b"):
        f = N.fits_5090(m, ctx_len=8192)
        checks[f"fits_{m}"] = f["fits"]
    # KV grows with context
    f_short = N.fits_5090("12b", ctx_len=2048); f_long = N.fits_5090("12b", ctx_len=32768)
    checks["kv_grows_with_ctx"] = f_long["kv_gb"] > f_short["kv_gb"]
    print(f"  -> 12B @2k ctx total {f_short['total_gb']:.1f}GB, @32k {f_long['total_gb']:.1f}GB")

    # 3. MoE accounting: 26B-A4B total params > 12B, active params small
    f26 = N.fits_5090("26b-a4b")
    checks["moe_total_gt_active"] = f26["total_params"] > f26["active_params"] and f26["active_params"] < 5e9

    # 4. max context
    mc12 = N.max_context_5090("12b"); mc31 = N.max_context_5090("31b")
    print(f"  -> max ctx: 12B ≈ {mc12:,}, 31B ≈ {mc31:,}")
    checks["max_ctx_positive"] = mc12 > 10000
    checks["bigger_model_less_ctx"] = mc31 < mc12

    # 5. plan recipe
    p = N.plan("12b")
    checks["plan_vllm"] = "vllm serve" in p["infer"] and "Marlin" in p["infer"]
    checks["plan_finetune"] = "FastModel" in p["finetune"] and "nvfp4" in p["finetune"].lower()

    # 6. agent
    st, dta, to, msg = N.run_nvfp4loader({"spec": {"model": "26b-a4b", "ctx_len": 8192}}, "t")
    checks["agent_done"] = st == "done" and dta["fits"] and dta["max_context"] > 0

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== nvfp4-loader: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
