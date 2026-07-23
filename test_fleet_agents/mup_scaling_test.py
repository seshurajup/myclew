"""mup_scaling_test — data-wise verifier for μP width-scaling (ArchScale).

Core properties:
  1. mup_init_std: hidden = 1/sqrt(fan_in), readout = 1/fan_in.
  2. mup_lr_scale: hidden multiplier = 1; readout = base/width (shrinks as width grows).
  3. Coordinate check: μP keeps |Δlogit| roughly flat across widths; standard param grows ~∝ width.
  4. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import mup_scaling as MP


def _run():
    print("=== μP-SCALING VERIFIER ===")
    checks = {}

    # 1. init std
    checks["init_hidden"] = abs(MP.mup_init_std(256) - 1 / np.sqrt(256)) < 1e-9
    checks["init_readout"] = abs(MP.mup_init_std(256, readout=True) - 1 / 256) < 1e-12

    # 2. lr scale
    checks["lr_hidden_one"] = MP.mup_lr_scale(1024, 256, "hidden") == 1.0
    checks["lr_readout_shrinks"] = abs(MP.mup_lr_scale(1024, 256, "readout") - 256 / 1024) < 1e-12
    checks["lr_readout_monotone"] = MP.mup_lr_scale(2048, 256, "output") < MP.mup_lr_scale(512, 256, "output")

    # 3. coordinate check
    widths = [64, 128, 256, 512, 1024]
    mu = MP.coordinate_check(widths, mup=True, lr=0.1, base_width=64)
    sp = MP.coordinate_check(widths, mup=False, lr=0.1, base_width=64)
    mu_spread = max(mu.values()) / min(mu.values())
    sp_spread = max(sp.values()) / min(sp.values())
    print(f"  -> μP |Δlogit| by width: {[round(v,2) for v in mu.values()]}  spread={mu_spread:.2f}x")
    print(f"  -> SP |Δlogit| by width: {[round(v,1) for v in sp.values()]}  spread={sp_spread:.1f}x")
    checks["mup_flat"] = mu_spread < 2.0                        # width-independent (bounded by rng noise)
    checks["sp_grows"] = sp_spread > 8.0                        # ~16x width range → ~16x growth
    checks["mup_much_flatter"] = mu_spread < 0.25 * sp_spread

    # 4. agent contract
    st, dta, to, msg = MP.run_mup({"spec": {"widths": widths, "base_width": 64}}, "t")
    checks["agent_done"] = st == "done" and dta["mup_spread"] < dta["sp_spread"]

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== mup-scaling: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
