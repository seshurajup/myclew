"""cv_lb_calibrate_test — plant (cv,lb) anchors on a KNOWN line lb = 0.8·cv + 0.18 and assert the agent
recovers slope/intercept and predicts a held-out CV correctly. Also checks the 1-anchor offset fallback."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import cv_lb_calibrate as C


def _run():
    print("=== CV-LB-CALIBRATE DATA-WISE VERIFIER ===")
    # known line: lb = 0.8*cv + 0.18  (so cv .90→lb .90, cv .8837→.8870, cv .9257→.9206)
    anchors = [{"cv": 0.8803, "lb": 0.8*0.8803+0.18, "exp": "A"},
               {"cv": 0.9257, "lb": 0.8*0.9257+0.18, "exp": "B"},
               {"cv": 0.8500, "lb": 0.8*0.8500+0.18, "exp": "C"},
               {"cv": 0.9000, "lb": 0.8*0.9000+0.18, "exp": "D"}]
    s, d, to, msg = C.CvLbCalibrate().run({"question": "cal", "spec": {"anchors": anchors, "predict": [0.9161]}}, "test")
    slope, inter = d["slope"], d["intercept"]
    pred = d["preds"][0]["predicted_lb"]
    checks = {
        "slope_~0.8": abs(slope - 0.8) < 1e-3,
        "intercept_~0.18": abs(inter - 0.18) < 1e-3,
        "predicts_0.9161": abs(pred - (0.8*0.9161+0.18)) < 1e-3,
        "kind_linear": d["kind"] == "linear",
        "low_resid": (d["mean_resid"] if d["mean_resid"] is not None else 1) < 1e-3,
    }
    # 1-anchor offset fallback
    s2, d2, _, _ = C.CvLbCalibrate().run({"question": "cal1", "spec": {"anchors": [{"cv": 0.9257, "lb": 0.900}], "predict": [0.8837]}}, "test")
    checks["offset_fallback"] = d2["kind"] == "offset" and abs(d2["preds"][0]["predicted_lb"] - (0.8837 + (0.900-0.9257))) < 1e-6
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== cv-lb-calibrate: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
