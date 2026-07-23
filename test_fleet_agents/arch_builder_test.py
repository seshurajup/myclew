"""arch_builder_test — plant a GT with KNOWN division rate; assert derived div_pos_weight = 1/rate and
kernel is anisotropic (z-kernel < xy-kernel, from the voxel scale). Verifies the data→architecture map."""
import os, sys, tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
import pandas as pd
from fleet_agents import arch_builder


def _make_gt(path, div_rate=0.01):
    rng = np.random.RandomState(0); rows = []
    for t in range(40):
        n = 100
        div = (rng.rand(n) < div_rate).astype("int8")
        for i in range(n):
            rows.append({"embryo": "E", "t": t, "z": rng.rand() * 10, "y": rng.rand() * 40, "x": rng.rand() * 40,
                         "dz": rng.randn() * 0.5, "dy": rng.randn() * 0.5, "dx": rng.randn() * 0.5,
                         "is_division": int(div[i])})
    df = pd.DataFrame(rows); df.to_parquet(path, index=False)
    return float(df["is_division"].mean())


def test_arch_builder_derives_from_data():
    with tempfile.TemporaryDirectory() as d:
        gt = os.path.join(d, "gt.parquet"); rate = _make_gt(gt, 0.01)
        s, res, to, msg = arch_builder.build(
            {"question": "t", "spec": {"gt_path": gt, "name": "test_arch", "sample_frames": 8}}, "test")
        assert s == "done", msg
        exp_w = round(1.0 / rate, 1)
        got_w = res["measured"]["div_pos_weight"]
        assert abs(got_w - exp_w) / exp_w < 0.15, f"div_pos_weight {got_w} != ~1/rate {exp_w}"
        kz, ky, kx = res["measured"]["kernel"]
        assert kz < kx and kz < ky, f"kernel not anisotropic (z should be smallest): {res['measured']['kernel']}"
        return {"div_weight_from_rate": True, "anisotropic_kernel": True}


def _run():
    print("=== ARCH-BUILDER DATA-WISE VERIFIER ===")
    try:
        r = test_arch_builder_derives_from_data()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== arch-builder: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
