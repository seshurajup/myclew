"""nb_preflight_test — DATA-WISE verifier of the preflight decision logic (_verdict) + import parsing,
no venv/network. Asserts: an unresolved import → RED; hard-coded mount path → RED; all-resolved +
content-discovery → GREEN; the zarr→typing_extensions trap is representable.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import nb_preflight as N


def _run():
    print("=== NB-PREFLIGHT VERDICT-LOGIC VERIFIER ===")
    mods = N._imports_of_source("import os, zarr\nfrom cellpose import models\nimport numpy as np\n")
    ok_green, rep_green = N._verdict({"zarr": True, "cellpose": True}, True, 0)
    ok_red, rep_red = N._verdict({"zarr": False, "cellpose": True}, True, 0)   # the real zarr trap
    ok_path, rep_path = N._verdict({"zarr": True}, False, 3)                    # hard-coded path → RED
    checks = {
        "parses_imports": {"zarr", "cellpose", "numpy"} <= mods and "os" in mods,
        "green_when_all_resolve": ok_green is True,
        "red_on_unresolved_import": ok_red is False and "zarr" in rep_red["unresolved_imports"],
        "red_on_bad_path_discovery": ok_path is False,
        "comment_lines_reported": rep_path["comment_lines"] == 3,
    }
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print("    green=", rep_green["verdict"], "| red=", rep_red["verdict"])
    ok = all(checks.values())
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
