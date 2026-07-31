"""llama2_infer_test — pure logic: llama2.c param-count, KV-cache, int8 group-wise export sizing, VRAM fit."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import llama2_infer as L


def _run():
    print("=== LLAMA2-INFER LOGIC VERIFIER ===")
    # stories15M config from llama2.c (dim=288,hidden=768,L=6,heads=6,kv=6,V=32000,seq=256) ~ 15M params
    cfg = {"dim": 288, "hidden_dim": 768, "n_layers": 6, "n_heads": 6,
           "n_kv_heads": 6, "vocab_size": 32000, "seq_len": 256}
    p = L.param_count(cfg)
    kv = L.kv_cache_bytes(cfg)                     # 2*6*256*288*4
    exp = L.int8_export_bytes(cfg, group_size=64)
    fit = L.fits(cfg, vram_gb=15.0)
    fit_i8 = L.fits(cfg, vram_gb=15.0, group_size=64)
    # GQA must shrink params: 2 kv-heads < 6 heads
    gqa = dict(cfg); gqa["n_kv_heads"] = 2
    checks = {
        "params_near_15M": 12e6 < p < 20e6,
        "kv_matches_formula": kv == 2 * 6 * 256 * 288 * 4,
        "gqa_shrinks_params": L.param_count(gqa) < p,
        "int8_ratio_gt_3": exp["ratio"] > 3.0,
        "int8_smaller_than_fp32": exp["int8_bytes"] < exp["fp32_bytes"],
        "quantized_div_by_group": exp["quantized_params"] % 64 == 0,
        "tiny_model_fits_15gb": fit["fits"] is True and fit_i8["fits"] is True,
        "int8_needs_less_than_fp32": fit_i8["weight_bytes"] < fit["weight_bytes"],
        "bad_group_raises": _raises(lambda: L.int8_export_bytes(cfg, 0)),
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"    params={p:,} kv={kv:,}B int8_ratio={exp['ratio']} need_fp32={fit['need_gb']}GB int8={fit_i8['need_gb']}GB")
    ok = all(checks.values()); print("RESULT:", "PASS" if ok else "FAIL"); return ok


def _raises(fn):
    try:
        fn(); return False
    except Exception:
        return True


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
