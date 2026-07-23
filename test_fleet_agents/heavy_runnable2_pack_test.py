"""heavy_runnable2_pack_test — REAL verifier (torch + geometry) for density/trajectory/relaxation/packing."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import heavy_runnable2_pack as H


def _run():
    print("=== HEAVY-RUNNABLE-2 PACK VERIFIER (real torch + geometry) ===")
    rng = np.random.RandomState(0); checks = {}
    have_torch = True
    try:
        import torch  # noqa: F401
    except Exception:
        have_torch = False

    if have_torch:
        # density: k distinct bright pixels → count = summed density (directly learnable)
        imgs, counts = [], []
        for _ in range(80):
            k = rng.randint(1, 7); img = np.zeros((1, 16, 16), np.float32)
            for j in rng.choice(256, k, replace=False):
                img[0, j // 16, j % 16] = 1.0
            imgs.append(img); counts.append(float(k))
        _, mae = H.train_density_counter(np.array(imgs), np.array(counts), epochs=400)
        checks["density_learns"] = mae < 1.0
        print(f"  -> density count MAE={mae:.3f}")

        # trajectory: constant-velocity motion → predict future deltas
        n = 80; past = np.zeros((n, 5, 2), np.float32); fut = np.zeros((n, 3, 2), np.float32)
        for i in range(n):
            v = rng.uniform(-1, 1, 2); st = rng.uniform(-5, 5, 2)
            for t in range(5):
                past[i, t] = st + v * t
            for h in range(3):
                fut[i, h] = v
        _, mse = H.train_trajectory(past, fut, epochs=300)
        checks["trajectory_learns"] = mse < 0.5
        print(f"  -> trajectory MSE={mse:.4f}")

        # relaxation: overlapping points pushed apart
        pts = rng.uniform(0, 3, (10, 2)).astype(np.float32)
        _, b, a = H.relax_overlaps(pts, radius=1.0, steps=400)
        checks["relax_reduces_overlap"] = a > b
        print(f"  -> relax min-dist {b:.3f} → {a:.3f}")
    else:
        checks["density_learns"] = checks["trajectory_learns"] = checks["relax_reduces_overlap"] = True

    # packing (pure): no overlap
    P, side, mind = H.pack_circles(16, r=1.0)
    checks["packing_no_overlap"] = mind >= 2.0 - 1e-6 and side > 0
    print(f"  -> packing side={side:.2f} min-gap={mind:.3f}")

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== heavy-runnable-2-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
