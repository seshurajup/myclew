"""component_graft_test — pure logic: graft_plan (reuse/drop + adapter) + accept_graft (no regression)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import component_graft as G


def _run():
    print("=== COMPONENT-GRAFT LOGIC VERIFIER ===")
    blocks = [("blk0", 64), ("blk1", 128), ("blk2", 128), ("blk3", 256)]
    p_adapt = G.graft_plan(blocks, keep_upto=3, our_head={"name": "unet-head", "in_ch": 32})
    p_noadapt = G.graft_plan(blocks, keep_upto=2, our_head={"name": "unet-head", "in_ch": 128})
    acc_ok, d1 = G.accept_graft({"44b6": 0.86, "6bba": 0.80}, {"44b6": 0.85, "6bba": 0.78})
    acc_reg, d2 = G.accept_graft({"44b6": 0.86, "6bba": 0.70}, {"44b6": 0.85, "6bba": 0.78})
    checks = {
        "reuse_count": len(p_adapt["reuse"]) == 3 and len(p_adapt["drop"]) == 1,
        "adapter_when_mismatch": p_adapt["adapter"] is not None and "128" in p_adapt["adapter"],
        "no_adapter_when_match": p_noadapt["adapter"] is None,
        "accept_no_regression": acc_ok is True and d1 >= 0,
        "reject_regression": acc_reg is False and d2 < 0,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("    plan=", p_adapt)
    ok = all(checks.values()); print("RESULT:", "PASS" if ok else "FAIL"); return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
