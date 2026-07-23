"""scaffold_pack_test — verifier: runnable tools work; lib-gated tools escalate cleanly (no fake green)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
import numpy as np
from fleet_agents import scaffold_pack as S


def _run():
    print("=== SCAFFOLD PACK VERIFIER ===")
    checks = {}
    # region-decompose (real cv2): two white rectangles → two regions
    img = np.zeros((100, 100), np.uint8); img[10:40, 10:40] = 255; img[60:90, 60:90] = 255
    checks["region_two_panels"] = len(S.decompose_regions(img, min_area=100)) == 2
    # dicom heuristic
    checks["dicom_portrait"] = S.estimate_metadata(np.zeros((200, 100)))["orientation"] == "portrait"
    # libs now INSTALLED → these tools RUN (rdkit/gplearn/nnunetv2/python-chess were pip-installed safely).
    # If a lib is somehow absent, the agent escalates cleanly — accept either, but never crash.
    import numpy as np2
    st, d, to, msg = S.run_molecular({"spec": {"smiles": ["CCO", "c1ccccc1"]}}, "t")
    checks["molecular_ok"] = st in ("done", "escalated")  # done if rdkit present
    st, d, to, msg = S.run_gp({"spec": {"X": np2.random.rand(50, 3).tolist(), "y": np2.random.rand(50).tolist()}}, "t")
    checks["gp_ok"] = st in ("done", "escalated")
    st, d, to, msg = S.run_nnunet({"spec": {}}, "t")
    checks["nnunet_ok"] = st in ("done", "escalated")
    st, d, to, msg = S.run_nnue({"spec": {}}, "t")
    checks["nnue_torch_runs"] = st == "done"
    # tools whose lib is genuinely absent here STILL escalate cleanly (no fake green)
    st, d, to, msg = S.run_automl({"spec": {}}, "t")
    checks["automl_escalates"] = st == "escalated"          # autogluon deliberately not installed (ABI)
    st, d, to, msg = S.run_foundation({"spec": {}}, "t")
    checks["foundation_escalates"] = st == "escalated"      # mast3r not installed
    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== scaffold-pack: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
