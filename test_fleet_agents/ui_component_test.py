"""ui_component_test — data-wise verifier for the astryx-style UI/dashboard generator.

Core properties:
  1. tokens() emits CSS custom properties for light and dark; themes differ.
  2. stat_card renders label+value; delta colored pos/neg.
  3. table escapes content and has the right row/col count.
  4. dashboard is a complete self-contained HTML doc (doctype, inline style, no external refs).
  5. agent writes a valid HTML file.
"""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import ui_component as U


def _run():
    print("=== UI-COMPONENT VERIFIER ===")
    checks = {}

    # 1. tokens
    lt, dk = U.tokens("light"), U.tokens("dark")
    checks["tokens_css_vars"] = "--bg:" in lt and "--accent:" in lt and lt.startswith(":root{")
    checks["themes_differ"] = lt != dk

    # 2. stat card
    c = U.stat_card("Best CV", "0.884", 0.003)
    checks["card_has_value"] = "0.884" in c and "Best CV" in c
    checks["delta_pos"] = 'class="delta pos"' in c and "+0.003" in c
    checks["delta_neg"] = 'class="delta neg"' in U.stat_card("x", "1", -0.5)

    # 3. table escaping + shape
    t = U.table(["a", "b"], [[1, "<script>"], [3, 4]])
    checks["table_escaped"] = "&lt;script&gt;" in t and "<script>" not in t.replace("&lt;script&gt;", "")
    checks["table_rows"] = t.count("<tr>") == 3     # 1 header + 2 body

    # 4. full dashboard self-contained
    h = U.dashboard("T", [("CV", "0.88", 0.01)], [("LB", ["r", "team"], [[1, "us"]])], theme="dark")
    checks["is_html_doc"] = h.startswith("<!doctype html>") and "</html>" in h
    checks["inline_style_no_external"] = "<style>" in h and "http://" not in h and "https://" not in h
    checks["dark_theme_applied"] = "--bg:#0f1117" in h
    print(f"  -> dashboard {len(h)} bytes, self-contained, dark theme")

    # 5. agent
    st, dta, to, msg = U.run_ui({"spec": {"theme": "light"}}, "t")
    checks["agent_done"] = st == "done" and os.path.exists(dta["path"]) and dta["bytes"] > 500

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== ui-component: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
