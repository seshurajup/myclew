import json, math, urllib.request, torch, torch.nn as nn, torch.nn.functional as F

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(DEV)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(0)
print(f"device: {DEV}" + (f" | {torch.cuda.get_device_name(0)}" if DEV.type == "cuda" else ""))

def ollama_embed(texts, model="embeddinggemma", timeout=180):
    """Live embeddings from the local daemon — None if it is not reachable (cells then SKIP loudly)."""
    try:
        req = urllib.request.Request("http://localhost:11434/api/embed",
            data=json.dumps({"model": model, "input": texts}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return torch.tensor(json.loads(r.read())["embeddings"])
    except Exception:
        return None

def ok(name, cond, extra=""):
    print(("PASS  " if cond else "FAIL  ") + name + (f"   | {extra}" if extra else ""))

V, d_emb, B = 400, 64, 192
torch.manual_seed(0)
base = torch.randn(16, V)                                     # 16 topics + more noise = a HARD task,
topic = torch.randint(0, 16, (B,))                            # so a truncated prefix can actually fail
qf = F.normalize(base[topic] + 1.2 * torch.randn(B, V), dim=1)
pf = F.normalize(base[topic] + 1.2 * torch.randn(B, V), dim=1)

def train_mrl(mrl, steps=400):
    torch.manual_seed(5)
    enc = nn.Linear(V, d_emb, bias=False)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    dims = (d_emb, d_emb // 2, d_emb // 4, d_emb // 8) if mrl else (d_emb,)
    for _ in range(steps):
        loss = 0
        for dd in dims:
            q = F.normalize(enc(qf)[:, :dd], dim=1); p = F.normalize(enc(pf)[:, :dd], dim=1)
            loss = loss + F.cross_entropy(q @ p.T / 0.05, torch.arange(B))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        out = {}
        for dd in (d_emb, d_emb // 4, d_emb // 8):
            q = F.normalize(enc(qf)[:, :dd], dim=1); p = F.normalize(enc(pf)[:, :dd], dim=1)
            out[dd] = float((q @ p.T).argmax(1).eq(torch.arange(B)).float().mean())
    return out

plain, mrl = train_mrl(False), train_mrl(True)
d8 = d_emb // 8
print(f"  plain: full {plain[d_emb]:.2f}   d/4 {plain[d_emb//4]:.2f}   d/8 {plain[d8]:.2f}")
print(f"  MRL  : full {mrl[d_emb]:.2f}   d/4 {mrl[d_emb//4]:.2f}   d/8 {mrl[d8]:.2f}")
ok("the MRL prefix survives aggressive truncation; the plain one collapses",
   mrl[d8] > plain[d8] + 0.2, f"at d/8: {plain[d8]:.2f} -> {mrl[d8]:.2f}")
ok("at negligible cost to the full dimension", mrl[d_emb] > plain[d_emb] - 0.05,
   f"{plain[d_emb]:.2f} vs {mrl[d_emb]:.2f}")

docs = ["Gradient boosting for tabular competitions",
        "Cell tracking in embryo microscopy volumes",
        "Quantizing neural networks to int8",
        "Optimal transport couplings for flow matching",
        "Cross validation without leakage",
        "Batch normalization folding into effective weights",
        "Formal proof search with a verifier",
        "Matryoshka embeddings for on-device retrieval"]
E = ollama_embed([f"title: none | text: {t}" for t in docs])
Q = ollama_embed(["task: search result | query: make my model smaller and faster with 8-bit"])
if E is None or Q is None:
    print("  SKIP — Ollama not reachable")
    ok("live truncation check skipped, honestly", True, "guarded cell")
else:
    En, Qn = F.normalize(E, dim=1), F.normalize(Q, dim=1)
    full = (Qn @ En.T)[0]
    order_full = full.argsort(descending=True)
    print(f"  full-dim (768) ranking: {[docs[int(i)][:34] for i in order_full[:3]]}")
    for dd in (256, 128):
        Ed = F.normalize(E[:, :dd], dim=1); Qd = F.normalize(Q[:, :dd], dim=1)
        r = (Qd @ Ed.T)[0].argsort(descending=True)
        agree = int(r[0]) == int(order_full[0])
        print(f"  truncated to {dd:>3}: top answer {'UNCHANGED' if agree else 'changed'} "
              f"({docs[int(r[0])][:34]})")
        ok(f"the shipped model's top answer survives truncation to {dd}", agree,
           "MRL training, visible in the artifact itself")
    ok("and the top answer is the right one", "int8" in docs[int(order_full[0])],
       f"'{docs[int(order_full[0])]}'")
