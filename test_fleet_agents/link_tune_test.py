"""link_tune_test — mock canonical scorer with a KNOWN best linking config (motion-relink OFF + gap 4.0
win; the rest neutral/worse). Assert link-tune coordinate-ascends to it, keeps only improving knobs, and
reports the gain. No GPU — scorer injected."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import link_tune


def _run():
    print("=== LINK-TUNE DATA-WISE VERIFIER ===")
    base = {"BIOHUB_MOTION_RELINK": "1", "BIOHUB_GAP_CLOSE_UM": "6.0", "BIOHUB_OUTPUT_MIN_TRACK_LEN": "4"}
    grid = {"BIOHUB_MOTION_RELINK": ["0", "1"], "BIOHUB_GAP_CLOSE_UM": ["4.0", "6.0"],
            "BIOHUB_OUTPUT_MIN_TRACK_LEN": ["4", "10"]}
    BASE = 0.8837

    def mock_score(cfg):
        cv = BASE
        if cfg.get("BIOHUB_MOTION_RELINK") == "0":         # motion-relink OFF → +0.017 (the ablation win)
            cv += 0.017
        if cfg.get("BIOHUB_GAP_CLOSE_UM") == "4.0":         # gap 4.0 → +0.003
            cv += 0.003
        if cfg.get("BIOHUB_OUTPUT_MIN_TRACK_LEN") == "10":  # mtl 10 → -0.005 (hurts here)
            cv -= 0.005
        return cv, cv - 0.001                                # (score, edge_jaccard)

    s, d, to, msg = link_tune.LinkTune().run(
        {"question": "tune linking", "spec": {"base": base, "grid": grid, "score_fn": mock_score}}, "test")
    cfg = d["best_config"]; kept = d["kept"]
    checks = {
        "turned_motion_relink_off": cfg["BIOHUB_MOTION_RELINK"] == "0" and kept.get("BIOHUB_MOTION_RELINK") == "0",
        "set_gap_4.0": cfg["BIOHUB_GAP_CLOSE_UM"] == "4.0",
        "kept_mtl_at_base_4": cfg["BIOHUB_OUTPUT_MIN_TRACK_LEN"] == "4",   # mtl 10 hurts → not adopted
        "gain_is_0.020": abs(d["gain"] - 0.020) < 1e-6,                    # +0.017 + 0.003
        "best_cv": abs(d["best_cv"] - (BASE + 0.020)) < 1e-6,
    }
    for k, v in checks.items():
        print(f"  {'✅' if v else '❌'} {k}")
    ok = all(checks.values())
    print(f"\n=== link-tune: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    try:
        sys.exit(0 if _run() else 1)
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); sys.exit(1)
