"""llmc_train_test — pure logic: llm.c GPT-2 param-count, vocab-pad, FLOPs/token, MFU wall-clock estimate."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import llmc_train as T


def _run():
    print("=== LLMC-TRAIN LOGIC VERIFIER ===")
    n_layer, n_head, n_embd = T.GPT2["gpt2"]
    p = T.param_count(n_layer, n_embd)             # ~124M (weight-tied)
    pad = T.pad_vocab(50257)
    e_bf16 = T.estimate("gpt2", "5090", "bf16")
    e_fp32 = T.estimate("gpt2", "5090", "fp32")
    e_t4 = T.estimate("gpt2", "T4", "bf16")
    checks = {
        "gpt2_params_near_124M": 120e6 < p < 135e6,
        "vocab_padded_to_50304": pad == 50304 and pad % 128 == 0,
        "medium_bigger_than_base": T.param_count(*(lambda t: (t[0], t[2]))(T.GPT2["gpt2-medium"])) > p,
        "flops_per_token_positive": e_bf16["flops_per_token"] > 0,
        "bf16_faster_than_fp32": e_bf16["tokens_per_sec"] > e_fp32["tokens_per_sec"],
        "5090_faster_than_t4": e_bf16["tokens_per_sec"] > e_t4["tokens_per_sec"],
        "wall_hours_positive": e_bf16["wall_hours"] > 0,
        "steps_from_batch": e_bf16["steps"] == 10 ** 10 // 2 ** 19,
        "unknown_model_raises": _raises(lambda: T.estimate("gpt3")),
        "unknown_device_raises": _raises(lambda: T.estimate("gpt2", "TPU")),
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"    params={p:,} pad={pad} bf16={e_bf16['tokens_per_sec']:,}tok/s {e_bf16['wall_hours']}h | t4={e_t4['tokens_per_sec']:,}tok/s")
    ok = all(checks.values()); print("RESULT:", "PASS" if ok else "FAIL"); return ok


def _raises(fn):
    try:
        fn(); return False
    except Exception:
        return True


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
