"""prior_art_test — assert the reusable engine buckets a KNOWN catalog correctly (embodied/actionable/NA)."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import prior_art


def test_prior_art_buckets_a_known_catalog():
    catalog = {
        "M_embodied":  ["x", r"zzz_embodied_zzz", "EMBODIED — already in pipeline"],
        "M_train":     ["x", r"zzz_train_zzz", "TRAINABLE — needs a GPU run to prove"],
        "M_wire":      ["x", r"zzz_wire_zzz", "WIREABLE — testable on cached preds"],
        "M_na":        ["x", r"zzz_na_zzz", "NA — needs segmentation masks"],
    }
    s, d, to, msg = prior_art.report({"question": "t", "spec": {"stage": "custom", "catalog": catalog}}, "test")
    assert s == "done", msg
    assert "M_embodied" in d["embodied"], d
    assert "M_train" in d["actionable"] and "M_wire" in d["actionable"], d
    assert "M_na" in d["not_applicable"], d
    return {"embodied_ok": True, "actionable_ok": True, "na_ok": True}


def _run():
    print("=== PRIOR-ART DATA-WISE VERIFIER ===")
    try:
        r = test_prior_art_buckets_a_known_catalog()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== prior-art: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
