"""heal_test — assert the self-healing agent maps KNOWN error signatures to the correct fix diagnosis."""
import os, re, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import heal


def _diagnose(err):
    for pat, (where, d) in heal.DIAGNOSES:
        if re.search(pat, err, re.I):
            return where, d
    return None, None


def test_heal_maps_errors_to_fixes():
    cases = {
        "Traceback ... KeyError: 'paths'": "train_from_config",
        "torch.cuda.OutOfMemoryError: CUDA out of memory": "batch_size",
        "DataLoader worker (pid 123) is killed": "num_workers",
    }
    results = {}
    for err, expect in cases.items():
        where, d = _diagnose(err)
        hit = where is not None and (expect in where or expect in (d or ""))
        results[err[:30]] = hit
        assert hit, f"error '{err[:40]}' → wrong/missing diagnosis (where={where})"
    return results


def _run():
    print("=== HEAL DATA-WISE VERIFIER ===")
    try:
        r = test_heal_maps_errors_to_fixes()
        for k, v in r.items(): print(f"  {'✅' if v else '❌'} {k}")
        ok = all(r.values())
    except AssertionError as e:
        print(f"  ❌ FAILED: {e}"); ok = False
    print(f"\n=== heal: {'PASS' if ok else 'FAIL'} ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
