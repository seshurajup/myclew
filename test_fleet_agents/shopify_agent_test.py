"""shopify_agent_test — data-wise verifier for the Shopify commerce agent (offline core).

Core properties:
  1. product_payload has the Shopify REST product shape (variants with price/sku/inventory).
  2. order_analytics: correct revenue, AOV, top products by quantity, refund rate.
  3. rfm_segments assigns labels; a high-R/F/M customer is a 'champion'.
  4. admin_request builds the right URL/headers; send is guarded without creds.
  5. agent contract."""
import os, sys
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP); sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))
from fleet_agents import shopify_agent as S


def _run():
    print("=== SHOPIFY-AGENT VERIFIER ===")
    checks = {}

    # 1. product payload
    p = S.product_payload("Tee", 19.95, sku="T1", inventory=50, tags=["new", "sale"])
    v = p["product"]["variants"][0]
    checks["payload_shape"] = v["price"] == "19.95" and v["sku"] == "T1" and v["inventory_quantity"] == 50
    checks["payload_tags"] = p["product"]["tags"] == "new, sale"

    # 2. order analytics
    orders = [
        {"total_price": "50.00", "financial_status": "paid",
         "line_items": [{"title": "Tee", "quantity": 3, "price": "10"}, {"title": "Cap", "quantity": 1, "price": "20"}]},
        {"total_price": "30.00", "financial_status": "paid", "line_items": [{"title": "Tee", "quantity": 2, "price": "15"}]},
        {"total_price": "0.00", "financial_status": "refunded", "line_items": [{"title": "Cap", "quantity": 1, "price": "20"}]},
    ]
    a = S.order_analytics(orders)
    checks["revenue"] = a["revenue"] == 80.0
    checks["aov"] = a["aov"] == round(80.0 / 3, 2)
    checks["top_product"] = a["top_products"][0] == ("Tee", 5)     # 3+2 tees
    checks["refund_rate"] = a["refund_rate"] == round(1 / 3, 4)
    print(f"  -> revenue ${a['revenue']} AOV ${a['aov']} top {a['top_products'][:2]} refund {a['refund_rate']}")

    # 3. RFM
    custs = [
        {"id": 1, "last_order_day": 9, "n_orders": 20, "total_spent": 2000},   # recent, frequent, big → champion
        {"id": 2, "last_order_day": 1, "n_orders": 15, "total_spent": 1500},
        {"id": 3, "last_order_day": 2, "n_orders": 1, "total_spent": 50},
        {"id": 4, "last_order_day": 0, "n_orders": 2, "total_spent": 80},
        {"id": 5, "last_order_day": 8, "n_orders": 18, "total_spent": 1800},
    ]
    segs = {c["id"]: c["segment"] for c in S.rfm_segments(custs, now_day=10)}
    checks["rfm_labels_exist"] = all(v in ("champion", "loyal", "at_risk", "new", "other") for v in segs.values())
    checks["rfm_champion"] = segs[1] == "champion"
    print(f"  -> RFM segments: {segs}")

    # 4. admin request build + guard
    req = S.admin_request("shop.myshopify.com", "tok123", "POST", "products.json", {"product": {}})
    checks["req_url"] = req["url"].endswith("/products.json") and "2025-01" in req["url"]
    checks["req_auth"] = req["headers"]["X-Shopify-Access-Token"] == "tok123" and req["method"] == "POST"
    try:
        S.send_admin_request(S.admin_request("s", "REPLACE", "GET", "x.json"))
        checks["send_guarded"] = False
    except RuntimeError:
        checks["send_guarded"] = True

    # 4b. app scaffold uses the 2026 recommended SDKs + auth
    files = S.app_scaffold("My App", with_function=True)
    checks["scaffold_toml"] = "shopify.app.toml" in files and "embedded = true" in files["shopify.app.toml"]
    checks["scaffold_token_exchange"] = "token-exchange" in files["shopify.app.toml"]  # 2026 managed-install auth
    pkg = files["package.json"]
    checks["scaffold_react_router"] = "shopify-app-react-router" in pkg and "react-router" in pkg
    checks["scaffold_polaris_appbridge"] = "@shopify/polaris" in pkg and "app-bridge-react" in pkg
    checks["scaffold_typescript"] = "app/shopify.server.ts" in files and "typescript" in pkg
    checks["scaffold_function"] = any("function" in k for k in files) and any(k.endswith(".rs") for k in files)
    import tempfile
    written = S.write_app_scaffold("My App", tempfile.mkdtemp(), with_function=True)
    checks["scaffold_written"] = len(written) >= 5 and all(os.path.exists(p) for p in written)
    print(f"  -> app scaffold: {len(files)} files (React Router + TS + Polaris + token-exchange + Function)")

    # 5. agent
    st, dta, to, msg = S.run_shopify({"spec": {}}, "t")
    checks["agent_done"] = st == "done" and dta["revenue"] > 0

    for k, v in checks.items():
        print(f"  {'OK' if v else 'X'} {k}")
    ok = all(checks.values())
    print(f"=== shopify-agent: {'PASS' if ok else 'FAIL'} ({sum(checks.values())}/{len(checks)}) ===")
    return ok


if __name__ == "__main__":
    sys.exit(0 if _run() else 1)
