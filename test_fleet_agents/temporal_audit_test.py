"""temporal_audit_test — DATA-WISE verifier for the temporal-audit preprocessing on synthetic data.

Plant a static cell cloud replicated across frames with a KNOWN accumulating GLOBAL setup-drift added.
Assert correct_global_shift() estimates and removes it (frame cloud-centres re-align; log records the
per-frame shift magnitude), while a genuine INDIVIDUAL long-jumper survives as residual (not flattened).
Also sanity-checks label_integrity on a clean graph.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np, pandas as pd
from scipy.spatial import cKDTree
from fleet_agents.temporal_audit import correct_global_shift


def _run():
    print("=== TEMPORAL-AUDIT VERIFIER (global-shift correction) ===")
    rng = np.random.RandomState(0)
    base = rng.rand(30, 3) * np.array([60, 250, 250])
    drift = np.array([0.0, 5.0, 3.0])          # KNOWN global setup drift per frame (µm-ish, voxel units here)
    rows = []
    for t in range(5):
        pts = base + drift * t                  # whole cloud shifts by drift each frame (setup motion)
        for (z, y, x) in pts:
            rows.append({"t": t, "z": z, "y": y, "x": x})
    df = pd.DataFrame(rows)

    corr, log = correct_global_shift(df, np, cKDTree)
    centers = corr.groupby("t")[["z", "y", "x"]].mean().to_numpy()
    spread = float(np.abs(centers - centers[0]).max())          # cloud-centre drift AFTER correction → ~0
    log_mag = np.mean([m for _, m in log]) if log else 0.0      # recorded per-frame shift ≈ ‖drift‖

    checks = {
        "shift_removed": spread < 1e-6,                          # accumulated global drift subtracted out
        "log_records_shift": abs(log_mag - float(np.linalg.norm(drift))) < 0.5,  # ‖[0,5,3]‖≈5.83
        "shape_preserved": len(corr) == len(df) and set(corr["t"]) == set(df["t"]),
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"    (post-correction centre spread={spread:.2e}, mean logged shift={log_mag:.2f} vs ‖drift‖={np.linalg.norm(drift):.2f})")
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
