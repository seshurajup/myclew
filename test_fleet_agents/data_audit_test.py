"""data_audit_test — DATA-WISE verifier for the data-audit agent.

Ground truth: build a flow GT with TWO embryos on deliberately different scales (A ~1.0, B ~2.5×) plus
planted outlier jumps. A correct data-audit must (1) normalise both embryos to a common scale and
(2) drop the outliers. We assert the post-audit per-embryo medians converge and outliers are removed.
"""
import os
import sys
import tempfile
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
import pandas as pd
from fleet_agents import data_audit


def _make_gt(path):
    rng = np.random.RandomState(0)
    rows = []
    for emb, scale in [("EMB_A", 1.0), ("EMB_B", 2.5)]:          # B is 2.5x A's scale (the bug)
        for t in range(40):
            n = 60
            z = rng.rand(n) * 10; y = rng.rand(n) * 40; x = rng.rand(n) * 40
            dz = rng.randn(n) * 0.5 * scale; dy = rng.randn(n) * 0.5 * scale; dx = rng.randn(n) * 0.5 * scale
            div = (rng.rand(n) < 0.005).astype("int8")
            for i in range(n):
                rows.append({"embryo": emb, "t": t, "z": z[i], "y": y[i], "x": x[i],
                             "dz": dz[i], "dy": dy[i], "dx": dx[i], "is_division": int(div[i])})
    df = pd.DataFrame(rows)
    # plant 50 outlier track-switching jumps
    idx = rng.choice(len(df), 50, replace=False)
    df.loc[idx, ["dz", "dy", "dx"]] = 30.0
    df.to_parquet(path, index=False)
    return df


def test_data_audit_normalises_scale_and_drops_outliers():
    with tempfile.TemporaryDirectory() as d:
        gt = os.path.join(d, "gt.parquet"); out = os.path.join(d, "clean.parquet")
        _make_gt(gt)
        status, res, to, msg = data_audit.report({"question": "t", "spec": {"gt_path": gt, "out_path": out}}, "test")
        assert status == "done", f"agent errored: {msg}"
        assert res["dropped"] >= 40, f"should drop the ~50 planted outliers, dropped {res['dropped']}"
        clean = pd.read_parquet(out)
        meds = {}
        for e in clean["embryo"].unique():
            m = np.sqrt((clean[clean.embryo == e][["dz", "dy", "dx"]].fillna(0) ** 2).sum(1))
            meds[e] = float(m[m > 0].median())
        ratio = max(meds.values()) / min(meds.values())
        assert ratio < 1.4, f"embryos not normalised to a common scale: medians {meds} (ratio {ratio:.2f})"
        return {"scale_normalised": ratio < 1.4, "outliers_dropped": res["dropped"] >= 40}


def _run():
    print("=== DATA-AUDIT DATA-WISE VERIFIER ===")
    try:
        r = test_data_audit_normalises_scale_and_drops_outliers()
        for k, v in r.items():
            print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== data-audit: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
