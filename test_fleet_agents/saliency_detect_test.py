"""saliency_detect_test — plant a saliency volume with TWO bright blobs: one already covered by an
existing detection, one NOT. Assert saliency-detect recovers ONLY the uncovered blob as a NEW add-only
candidate (add-only recall repair), and counts peaks correctly. No GPU."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import saliency_detect as SD


def _run():
    print("=== SALIENCY-DETECT DATA-WISE VERIFIER ===")
    # 8x16x16 saliency; blob A at (4,4,4) already detected, blob B at (4,12,12) MISSED by the primary head
    sal = np.zeros((8, 16, 16), "float32")
    sal[4, 4, 4] = 1.0
    sal[4, 12, 12] = 0.9
    existing = [[4, 4, 4]]                              # only blob A is already a detection

    new, n_total = SD.peaks_from_saliency(sal, thresh_frac=0.5, merge_vox=3.0, existing=existing)
    # also via the agent
    s, d, to, msg = SD.SaliencyDetect().run(
        {"question": "detect", "spec": {"saliency": sal, "existing_nodes": existing, "thresh_frac": 0.5}}, "test")

    checks = {
        "found_two_peaks": n_total == 2,
        "one_new_candidate": len(new) == 1,
        "new_is_blobB": len(new) == 1 and abs(new[0][1] - 12) < 1 and abs(new[0][2] - 12) < 1,
        "agent_reports_1_new": d.get("new_candidates") == 1,
        "add_only_skips_covered": d.get("peaks_total") == 2 and d.get("new_candidates") == 1,
    }
    # with NO existing nodes → both blobs are candidates
    new2, _ = SD.peaks_from_saliency(sal, thresh_frac=0.5, merge_vox=3.0, existing=None)
    checks["no_existing_returns_both"] = len(new2) == 2
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== saliency-detect: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
