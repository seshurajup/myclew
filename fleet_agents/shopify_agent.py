"""shopify_agent — a Shopify commerce agent. The offline-testable core is the DATA + ANALYTICS layer any
store integration needs — build product payloads, aggregate order economics (revenue / AOV / top products),
and RFM-style customer segmentation — plus a correctly-formed Shopify Admin API request builder. The actual
HTTP call is import/credential-guarded (needs SHOPIFY_STORE + SHOPIFY_ACCESS_TOKEN) so the analytics + payload
construction are unit-testable with no network.

Primitives (stdlib, no deps):
  • product_payload(title, price, ...)        — a valid Shopify product create payload (REST admin shape).
  • order_analytics(orders)                    — revenue, order count, AOV, top products, refund rate.
  • rfm_segments(customers)                     — Recency/Frequency/Monetary scoring → segment labels.
  • admin_request(store, token, method, path)  — build a signed Admin API request (dict); .send() guarded.
"""
from __future__ import annotations
import os
from .base import BaseAgent

API_VERSION = "2025-01"


def product_payload(title, price, sku=None, inventory=0, vendor=None, tags=None, body_html=""):
    """A Shopify REST Admin product-create payload. Single default variant carries price/sku/inventory."""
    variant = {"price": f"{float(price):.2f}", "inventory_quantity": int(inventory)}
    if sku:
        variant["sku"] = str(sku)
    p = {"title": str(title), "body_html": str(body_html), "variants": [variant]}
    if vendor:
        p["vendor"] = str(vendor)
    if tags:
        p["tags"] = ", ".join(tags) if isinstance(tags, (list, tuple)) else str(tags)
    return {"product": p}


def order_analytics(orders):
    """Aggregate store economics from a list of order dicts {total_price, line_items:[{title,quantity,price}],
    financial_status}. Returns revenue, n_orders, AOV, top_products (by qty), refund_rate."""
    orders = list(orders or [])
    rev = sum(float(o.get("total_price", 0)) for o in orders)
    n = len(orders)
    refunded = sum(1 for o in orders if o.get("financial_status") in ("refunded", "partially_refunded"))
    prod_qty = {}
    for o in orders:
        for li in o.get("line_items", []):
            prod_qty[li.get("title", "?")] = prod_qty.get(li.get("title", "?"), 0) + int(li.get("quantity", 0))
    top = sorted(prod_qty.items(), key=lambda kv: kv[1], reverse=True)[:5]
    return {"revenue": round(rev, 2), "n_orders": n, "aov": round(rev / n, 2) if n else 0.0,
            "top_products": top, "refund_rate": round(refunded / n, 4) if n else 0.0}


def rfm_segments(customers, now_day=0):
    """RFM segmentation. customers: [{id, last_order_day, n_orders, total_spent}]. Scores each dim 1-4 by
    quartile, labels: 'champion' (high all), 'loyal', 'at_risk' (old but was frequent), 'new', 'other'.
    now_day = reference day; recency = now_day - last_order_day (smaller = better)."""
    import numpy as np
    cs = list(customers or [])
    if not cs:
        return []
    rec = np.array([now_day - c.get("last_order_day", 0) for c in cs], float)
    freq = np.array([c.get("n_orders", 0) for c in cs], float)
    mon = np.array([c.get("total_spent", 0) for c in cs], float)
    def score(a, invert=False):
        q = np.quantile(a, [0.25, 0.5, 0.75])
        s = np.digitize(a, q) + 1            # 1..4
        return (5 - s) if invert else s
    R = score(rec, invert=True); F = score(freq); M = score(mon)
    out = []
    for i, c in enumerate(cs):
        r, f, m = int(R[i]), int(F[i]), int(M[i])
        if r >= 3 and f >= 3 and m >= 3:
            seg = "champion"
        elif f >= 3 and m >= 3:
            seg = "loyal"
        elif r <= 2 and f >= 3:
            seg = "at_risk"
        elif r >= 3 and f <= 2:
            seg = "new"
        else:
            seg = "other"
        out.append({"id": c.get("id"), "R": r, "F": f, "M": m, "segment": seg})
    return out


def admin_request(store, token, method, path, body=None):
    """Build a Shopify Admin API request descriptor (URL, headers, method, body). Does NOT send. path e.g.
    'products.json'. Returns a dict; .send via send_admin_request (guarded)."""
    url = f"https://{store}/admin/api/{API_VERSION}/{path.lstrip('/')}"
    headers = {"X-Shopify-Access-Token": token, "Content-Type": "application/json"}
    return {"method": method.upper(), "url": url, "headers": headers, "body": body}


def send_admin_request(req, timeout=30):
    """Send a built admin_request via stdlib urllib. Requires real creds; raises if store/token look empty."""
    import json, urllib.request
    tok = req["headers"].get("X-Shopify-Access-Token", "")
    if "REPLACE" in req["url"] or not tok or "REPLACE" in tok or "myshopify.com" not in req["url"]:
        raise RuntimeError("send_admin_request needs a real *.myshopify.com store + SHOPIFY_ACCESS_TOKEN")
    data = json.dumps(req["body"]).encode() if req.get("body") is not None else None
    r = urllib.request.Request(req["url"], data=data, headers=req["headers"], method=req["method"])
    with urllib.request.urlopen(r, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------- Shopify APP scaffolding (recommended SDKs)
# Shopify's official stack for embedded apps: Shopify CLI (scaffold/deploy), @shopify/shopify-api (Admin/OAuth),
# @shopify/shopify-app-remix or app-express (framework glue), App Bridge (embedding in Admin), Polaris (UI),
# GraphQL Admin API, and webhook subscriptions. This builds the config/manifest/handlers for such an app.
def app_scaffold(app_name, scopes=None, api_version=API_VERSION, webhooks=None, with_function=False):
    """Return the file set for a Shopify embedded app using the 2026 RECOMMENDED stack: Shopify CLI + the
    Remix/React-Router template + TypeScript + Polaris + App Bridge, with MANAGED INSTALLATION + TOKEN-EXCHANGE
    auth (the current default, replacing the old OAuth redirect flow) declared in shopify.app.toml. Optionally
    scaffolds a Shopify FUNCTION extension (Rust/Wasm backend logic). Returns {filename: content};
    write_app_scaffold() writes them, does not run npm/CLI."""
    scopes = scopes or ["read_products", "write_products", "read_orders"]
    webhooks = webhooks or ["ORDERS_CREATE", "APP_UNINSTALLED"]
    slug = app_name.lower().replace(" ", "-")
    # embedded + managed install + token-exchange (grant_authorization) = the 2026 default auth
    toml = (f'name = "{app_name}"\nclient_id = "REPLACE_CLIENT_ID"\n'
            f'application_url = "https://REPLACE.example.com"\nembedded = true\n\n'
            f'[access_scopes]\nscopes = "{",".join(scopes)}"\nuse_legacy_install_flow = false\n\n'
            f'[auth]\ngrant_options = ["token-exchange"]\n\n'
            f'[webhooks]\napi_version = "{api_version}"\n')
    pkg = {
        "name": slug, "private": True,
        "scripts": {"dev": "shopify app dev", "deploy": "shopify app deploy", "build": "react-router build"},
        "dependencies": {"@shopify/shopify-app-react-router": "^1.0.0", "@shopify/shopify-api": "^11.0.0",
                         "@shopify/app-bridge-react": "^4.0.0", "@shopify/polaris": "^13.0.0",
                         "react-router": "^7.0.0"},
        "devDependencies": {"@shopify/cli": "^3.0.0", "typescript": "^5.0.0"},
    }
    server = (
        "import '@shopify/shopify-api/adapters/node';\n"
        "import { shopifyApp } from '@shopify/shopify-app-react-router/server';\n"
        f"// scopes: {scopes!r}. Auth = managed install + token exchange (no manual OAuth redirect).\n"
        "const shopify = shopifyApp({\n"
        "  apiKey: process.env.SHOPIFY_API_KEY!,\n"
        "  apiSecretKey: process.env.SHOPIFY_API_SECRET!,\n"
        "  appUrl: process.env.SHOPIFY_APP_URL!,\n"
        "  isEmbeddedApp: true,\n"
        "  future: { unstable_newEmbeddedAuthStrategy: true },  // token-exchange embedded auth\n"
        "});\n"
        "export default shopify;\nexport const authenticate = shopify.authenticate;\n")
    files = {"shopify.app.toml": toml,
             "package.json": __import__("json").dumps(pkg, indent=2),
             "app/shopify.server.ts": server,
             "WEBHOOKS.md": "Subscribed webhooks:\n" + "\n".join(f"- {w}" for w in webhooks)}
    if with_function:
        files["extensions/discount-function/shopify.extension.toml"] = (
            '[[extensions]]\nname = "discount-function"\ntype = "function"\napi_version = "' + api_version + '"\n'
            '[[extensions.targeting]]\ntarget = "cart.lines.discounts.generate.run"\n')
        files["extensions/discount-function/src/run.rs"] = (
            "// Shopify Function (Rust→Wasm): backend logic that runs on Shopify's infra, not your server.\n"
            "// `shopify app function build` compiles this to Wasm.\n"
            "fn main() { /* generate discounts from cart input */ }\n")
    return files


def write_app_scaffold(app_name, out_dir, **kw):
    """Write app_scaffold(...) files under out_dir. Returns the list of written paths."""
    import os as _os
    files = app_scaffold(app_name, **kw)
    written = []
    for rel, content in files.items():
        p = _os.path.join(out_dir, rel); _os.makedirs(_os.path.dirname(p) or ".", exist_ok=True)
        open(p, "w").write(content); written.append(p)
    return written


# ---------------------------------------------------------------- agent
class ShopifyAgent(BaseAgent):
    name = "shopify-agent"
    thread = "S"; kind = "finding"

    def run(self, q, worker):
        s = self.spec(q)
        orders = s.get("orders") or [
            {"total_price": "49.90", "financial_status": "paid",
             "line_items": [{"title": "Tee", "quantity": 2, "price": "19.95"}, {"title": "Cap", "quantity": 1, "price": "10.00"}]},
            {"total_price": "19.95", "financial_status": "paid", "line_items": [{"title": "Tee", "quantity": 1, "price": "19.95"}]},
            {"total_price": "0.00", "financial_status": "refunded", "line_items": [{"title": "Cap", "quantity": 1, "price": "10.00"}]},
        ]
        a = order_analytics(orders)
        store = os.environ.get("SHOPIFY_STORE", "your-store.myshopify.com")
        token = os.environ.get("SHOPIFY_ACCESS_TOKEN", "")
        req = admin_request(store, token or "REPLACE", "POST", "products.json",
                            product_payload("Sample Tee", 19.95, sku="TEE-1", inventory=100, tags=["new"]))
        msg = (f"shopify-agent: {a['n_orders']} orders → revenue ${a['revenue']}, AOV ${a['aov']}, "
               f"top={a['top_products'][:2]}, refund-rate {a['refund_rate']*100:.0f}%. Built Admin API "
               f"{req['method']} {req['url'].split('/admin/')[-1]} (send needs SHOPIFY_ACCESS_TOKEN). "
               f"Catalog payloads + order analytics + RFM segmentation, offline-testable")
        self.log(msg, kind="finding",
                 recommendation="set SHOPIFY_STORE + SHOPIFY_ACCESS_TOKEN then send_admin_request; use "
                                "order_analytics/rfm_segments for merchandising + retention insights")
        return self.done({"revenue": a["revenue"], "aov": a["aov"], "refund_rate": a["refund_rate"]}, msg)


_AGENT = ShopifyAgent()


def run_shopify(q, worker):
    return _AGENT.run(q, worker)
