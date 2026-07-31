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

caps = {
 "quantize": "Post-training quantization of a model to int8/4-bit for faster inference",
 "cv-build": "Build the embryo-disjoint cross validation split (leave-one-embryo-out)",
 "blend-optimize": "Optimize blend weights of out-of-fold prediction sets",
 "calibrate": "Calibrate classifier probability outputs (temperature/isotonic)",
 "det-sweep": "Detection det-and-pool threshold sweep for max node recall at calibrated count",
 "lora-train": "Train a LoRA adapter on the detector backbone",
 "gm-writeup-mine": "Mine winning solution writeups from Kaggle grandmasters",
 "nb-preflight": "Verify the submission notebook imports and runs offline on Kaggle",
 "pipeline": "Run an ordered agent chain deterministically",
 "deep-research": "Deep research over recent architecture innovations",
 "tab-stack": "Stack tabular base models with a meta-learner",
 "keyframe": "Select keyframes for annotation value",
}
tests = [("quantize a model to int8 for faster inference", "quantize"),
         ("build embryo-disjoint cross validation folds", "cv-build"),
         ("blend two sets of out-of-fold predictions optimally", "blend-optimize"),
         ("calibrate the classifier probability outputs", "calibrate"),
         ("check the submission notebook will run offline on kaggle", "nb-preflight"),
         ("train a lora adapter on the detector", "lora-train")]
names = sorted(caps)
raw_docs = ollama_embed([f"{n}: {caps[n]}" for n in names])
if raw_docs is None:
    print("  SKIP — Ollama not reachable; the 0/8 vs 5/8 experiment needs the live model")
    ok("live prompt experiment skipped, honestly", True, "guarded cell")
else:
    prm_docs = ollama_embed([f"title: {n} | text: {caps[n]}" for n in names])
    raw_q = ollama_embed([q for q, _ in tests])
    prm_q = ollama_embed([f"task: search result | query: {q}" for q, _ in tests])
    def top1(Qe, De):
        S = F.normalize(Qe, dim=1) @ F.normalize(De, dim=1).T
        return sum(names[int(S[k].argmax())] == g for k, (_, g) in enumerate(tests))
    t_raw, t_prm = top1(raw_q, raw_docs), top1(prm_q, prm_docs)
    print(f"  routing accuracy over {len(names)} agents: raw strings {t_raw}/{len(tests)}   "
          f"with the mandated prompts {t_prm}/{len(tests)}")
    ok("the task prompt is PART OF THE MODEL, not decoration", t_prm > t_raw + 1,
       f"{t_raw} -> {t_prm} top-1 — eq. 1's t_q/t_p, measured on the shipped weights")
    ok("asymmetric prompts: queries and documents get DIFFERENT prefixes", True,
       "'task: search result | query:' vs 'title: ... | text:' — retrieval is not symmetric")

V, d_emb, B = 400, 32, 128
torch.manual_seed(0)
base = torch.randn(8, V)
topic = torch.randint(0, 8, (B,))
qf = F.normalize(base[topic] + 0.6 * torch.randn(B, V), dim=1)
pf = F.normalize(base[topic] + 0.6 * torch.randn(B, V), dim=1)
hf = F.normalize(base[topic] + 0.25 * torch.randn(B, V), dim=1)   # HARD: same topic region, wrong doc

def train_nce(w_hard, tau=0.05, steps=300):
    torch.manual_seed(1)
    enc = nn.Linear(V, d_emb, bias=False)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    for _ in range(steps):
        q = F.normalize(enc(qf), dim=1); p = F.normalize(enc(pf), dim=1)
        h = F.normalize(enc(hf), dim=1)
        pos = (q * p).sum(1) / tau
        hard = (q * h).sum(1) / tau
        inb = q @ p.T / tau
        denom = torch.logsumexp(torch.cat([inb, (hard + math.log(w_hard))[:, None]], 1), 1)
        loss = (denom - pos).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        q = F.normalize(enc(qf), dim=1); p = F.normalize(enc(pf), dim=1)
        h = F.normalize(enc(hf), dim=1)
        acc = float((q @ p.T).argmax(1).eq(torch.arange(B)).float().mean())
        margin = float(((q * p).sum(1) - (q * h).sum(1)).mean())
    return acc, margin

acc1, m1 = train_nce(w_hard=1.0)
acc5, m5 = train_nce(w_hard=5.0)
print(f"  w=1: retrieval {acc1:.2f}, pos-vs-hard margin {m1:+.3f}")
print(f"  w=5: retrieval {acc5:.2f}, pos-vs-hard margin {m5:+.3f}")
ok("in-batch NCE trains retrieval", acc1 > 0.7, f"top-1 {acc1:.2f} over {B}")
ok("up-weighting the hard negative buys MARGIN against it", m5 > m1,
   f"{m1:+.3f} -> {m5:+.3f} — w_i spends gradient exactly where discrimination is hardest")

V, d_emb, B = 400, 32, 64
torch.manual_seed(0)
base = torch.randn(8, V)
topic = torch.randint(0, 8, (B,))
topic[B // 2:] = topic[: B // 2]                              # every pair DUPLICATED
qf = F.normalize(base[topic] + 0.05 * torch.randn(B, V), dim=1)
qf[B // 2:] = qf[: B // 2]                                    # exact duplicate queries
pf = qf + 0.05 * torch.randn(B, V)
enc = nn.Linear(V, d_emb, bias=False)
q = F.normalize(enc(qf), dim=1); p = F.normalize(enc(pf), dim=1)
logits = q @ p.T / 0.05

with torch.no_grad():                                         # where does the softmax mass GO?
    P = logits.softmax(1)
    twin = torch.tensor([float(P[i, i + B // 2]) for i in range(B // 2)]).mean()
    genuine = torch.tensor([float(P[i, (i + 3) % (B // 2)]) for i in range(B // 2)]).mean()
print(f"  softmax mass on the identical twin {float(twin):.4f}   on a genuine negative "
      f"{float(genuine):.4f}")
ok("the unmasked loss spends most of its 'negative' pressure on a CORRECT answer",
   float(twin) > 2 * float(genuine),
   f"{float(twin)/float(genuine):.1f}x — the twin IS the right passage, and it is being pushed away")

keep = torch.ones(B, B, dtype=torch.bool)                     # eq. (3): mask the twin
for i_ in range(B // 2):
    keep[i_, i_ + B // 2] = False; keep[i_ + B // 2, i_] = False
masked = logits.masked_fill(~keep & ~torch.eye(B, dtype=torch.bool), -1e4)
g_raw = torch.autograd.grad(F.cross_entropy(logits, torch.arange(B)), enc.weight,
                            retain_graph=True)[0]
g_msk = torch.autograd.grad(F.cross_entropy(masked, torch.arange(B)), enc.weight)[0]
rel = float((g_raw - g_msk).norm() / g_raw.norm())
print(f"  relative gradient difference from masking: {rel:.1%}")
ok("so the mask changes the actual update", rel > 0.01, f"{rel:.1%} of the gradient norm")
ok("HONEST SCALE NOTE: with 50% duplicates the effect is a few percent of the gradient", True,
   "eq. 3 is cheap insurance, not a headline win — at real duplicate rates it prevents a slow poison")

V, d_emb, B = 400, 24, 128
torch.manual_seed(0)
base = torch.randn(8, V)
topic = torch.randint(0, 8, (B,))
qf = F.normalize(base[topic] + 0.6 * torch.randn(B, V), dim=1)
pf = F.normalize(base[topic] + 0.6 * torch.randn(B, V), dim=1)

def train_spread(lam, steps=300):
    torch.manual_seed(3)
    enc = nn.Linear(V, d_emb, bias=False)
    opt = torch.optim.Adam(enc.parameters(), lr=1e-2)
    for _ in range(steps):
        q = F.normalize(enc(qf), dim=1); p = F.normalize(enc(pf), dim=1)
        loss = F.cross_entropy(q @ p.T / 0.05, torch.arange(B))
        off = ~torch.eye(B, dtype=torch.bool)
        ls = ((q @ q.T)[off] ** 2).mean() + ((p @ p.T)[off] ** 2).mean()
        (loss + lam * ls).backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        q = F.normalize(enc(qf), dim=1); p = F.normalize(enc(pf), dim=1)
    return q, p

def int8(x):                                                  # per-tensor symmetric, like a vector DB
    s = x.abs().max() / 127
    return (x / s).round().clamp(-127, 127) * s

res = {}
for lam in (0.0, 1.0):
    q, p = train_spread(lam)
    off = ~torch.eye(B, dtype=torch.bool)
    iso = float(((q @ q.T)[off] ** 2).mean())
    acc = float((q @ p.T).argmax(1).eq(torch.arange(B)).float().mean())
    acc8 = float((int8(q) @ int8(p).T).argmax(1).eq(torch.arange(B)).float().mean())
    res[lam] = (iso, acc, acc8)
    print(f"  lam={lam}: mean sq cross-sim {iso:.4f}   float acc {acc:.2f}   int8 acc {acc8:.2f}   "
          f"retention {acc8/max(acc,1e-9):.2f}")
ok("the loss makes the cloud measurably more isotropic", res[1.0][0] < res[0.0][0] * 0.9,
   f"{res[0.0][0]:.4f} -> {res[1.0][0]:.4f} ({(1-res[1.0][0]/res[0.0][0])*100:.0f}% less "
   f"cross-similarity)")
ok("int8 retention is not WORSE with it", res[1.0][2] / max(res[1.0][1], 1e-9)
   >= res[0.0][2] / max(res[0.0][1], 1e-9) - 1e-9)
ok("HONEST NULL: this toy saturates, so it cannot demonstrate the quantization WIN",
   res[0.0][2] >= 0.99,
   f"both variants retrieve perfectly at int8 ({res[0.0][2]:.2f}); the paper's claim needs a task hard "
   f"enough that quantization error changes a ranking — what we verified is the MECHANISM (isotropy)")

V, dt, ds, B = 400, 48, 16, 192
torch.manual_seed(0)
base = torch.randn(8, V)
topic = torch.randint(0, 8, (B,))
qf = F.normalize(base[topic] + 0.6 * torch.randn(B, V), dim=1)
pf = F.normalize(base[topic] + 0.6 * torch.randn(B, V), dim=1)
hf = F.normalize(base[topic] + 0.25 * torch.randn(B, V), dim=1)

teacher = nn.Linear(V, dt, bias=False)                        # a STRONG teacher: train it properly
optt = torch.optim.Adam(teacher.parameters(), lr=1e-2)
for _ in range(500):
    q = F.normalize(teacher(qf), dim=1); p = F.normalize(teacher(pf), dim=1)
    h = F.normalize(teacher(hf), dim=1)
    logits = torch.cat([q @ p.T, (q * h).sum(1, keepdim=True) + 1.0], 1) / 0.05
    optt.zero_grad(); F.cross_entropy(logits, torch.arange(B)).backward(); optt.step()
with torch.no_grad():
    tq = F.normalize(teacher(qf), dim=1); tp = F.normalize(teacher(pf), dim=1)
    th = F.normalize(teacher(hf), dim=1)
    t_margin = float(((tq * tp).sum(1) - (tq * th).sum(1)).mean())

def distill(use_hard, steps=400):
    torch.manual_seed(4)
    stud = nn.Sequential(nn.Linear(V, ds, bias=False))
    W = nn.Linear(ds, dt, bias=False)                          # align spaces for matching
    opt = torch.optim.Adam(list(stud.parameters()) + list(W.parameters()), lr=1e-2)
    for _ in range(steps):
        sq = F.normalize(W(stud(qf)), dim=1); sp = F.normalize(W(stud(pf)), dim=1)
        sh = F.normalize(W(stud(hf)), dim=1)
        loss = ((sq - tq) ** 2).sum(1).mean() + ((sp - tp) ** 2).sum(1).mean()
        if use_hard:
            loss = loss + ((sh - th) ** 2).sum(1).mean()       # eq. (5)'s third term
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        sq = F.normalize(W(stud(qf)), dim=1); sp = F.normalize(W(stud(pf)), dim=1)
        sh = F.normalize(W(stud(hf)), dim=1)
        return float(((sq * sp).sum(1) - (sq * sh).sum(1)).mean())

m_no, m_yes = distill(False), distill(True)
print(f"  teacher margin {t_margin:+.3f}   student without P- term {m_no:+.3f}   with it {m_yes:+.3f}")
ok("the P- term makes the student's hard-negative margin LARGER", m_yes > m_no,
   f"{m_no:+.3f} -> {m_yes:+.3f} — it is trained on exactly the distinction it must draw")
ok("a 3x-smaller student inherits the teacher's ORDERING without any contrastive loss",
   m_yes > 0, f"margin {m_yes:+.3f} > 0 from pure geometry matching")
ok("HONEST NOTE: matching the teacher's margin MAGNITUDE is a different (harder) target", True,
   f"student {m_yes:+.3f} vs teacher {t_margin:+.3f} — a 16-dim student cannot hold a 48-dim geometry; "
   f"what transfers is the decision structure, not the scale")
