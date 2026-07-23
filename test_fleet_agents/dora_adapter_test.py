"""dora_adapter_test — data-wise verifier for DoRA.

Core properties:
  1. IDENTITY AT INIT: with B=0 and m=||W0||_row, the DoRA layer reproduces the frozen base linear map.
  2. PARAMETER EFFICIENCY: trainable params = out + rank*(in+out) << out*in.
  3. DoRA BEATS LoRA at equal rank on a per-row-RESCALE target (a full-rank magnitude change that plain
     low-rank LoRA cannot represent but DoRA's magnitude vector captures directly).
  4. The magnitude vector actually MOVES during training. + agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import dora_adapter as D


def _run():
    print("=== DoRA VERIFIER ===")
    checks = {}
    try:
        import torch
        import torch.nn.functional as F
    except Exception as e:  # noqa: BLE001
        print("torch unavailable:", e); print("=== dora-adapt: SKIP ==="); return True
    torch.manual_seed(0)
    out, inp, rank = 16, 16, 2
    W0 = torch.randn(out, inp)

    # 1. identity at init
    dora = D.DoRALinear(W0, rank=rank)
    X = torch.randn(128, inp)
    with torch.no_grad():
        err = float(F.mse_loss(dora(X), X @ W0.T))
    print(f"  -> init identity err={err:.3e}")
    checks["identity_at_init"] = err < 1e-8

    # 2. parameter efficiency
    checks["param_efficient"] = dora.n_trainable() == out + rank * (inp + out) and dora.n_trainable() < out * inp

    # 3. DoRA vs LoRA on a per-row rescale target
    scale = (0.4 + 2.0 * torch.rand(out)).unsqueeze(1); Wt = scale * W0
    Ttarget = X @ Wt.T
    m0 = dora.m.detach().clone()
    ld = D._fit(dora, X, Ttarget, steps=600, lr=0.05)
    lora = D.LoRALinear(W0, rank=rank); ll = D._fit(lora, X, Ttarget, steps=600, lr=0.05)
    print(f"  -> DoRA MSE={ld:.4e}  LoRA MSE={ll:.4e}")
    checks["dora_beats_lora"] = ld < 0.5 * ll
    checks["dora_fits"] = ld < 1e-3 * float((Ttarget ** 2).mean())

    # 4. magnitude moved
    m_moved = float((dora.m - m0).abs().mean())
    print(f"  -> magnitude moved {m_moved:.3f}")
    checks["magnitude_moves"] = m_moved > 0.05

    st, dta, to, msg = D.run({"spec": {"out": 16, "in": 16, "rank": 2, "steps": 500}}, "t")
    checks["agent_done"] = st == "done" and dta["dora_mse"] < dta["lora_mse"]

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== dora-adapt: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
