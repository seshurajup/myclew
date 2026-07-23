"""multimodal_pack_test — DATA-WISE, offline, deterministic (BLAS-pinned) verifier for the MULTIMODAL
FEATURE-LEVEL FUSION pack.

Builds tiny synthetic multi-modality feature tensors (no files, no network) and asserts ground-truth
behaviour of each agent's underlying function:
  • fusion returns the right shape + finite for EVERY strategy (concat/sum/mean/gated/film/bilinear/
    cross_attention), for N modalities of variable input dims;
  • an INFORMATIVE modality + a NOISE modality: a small supervised probe on the GATED-fused / cross-
    attention-fused representation separates the two classes BETTER than a probe on the noise modality
    alone (fusion tracks the informative signal, not the noise);
  • modality-dropout preserves shape + stays finite when a modality is masked at train time, and the
    learned NULL-token path fires when a modality is fully ABSENT at eval time;
  • encoder-adapter aligns different-dim inputs to the shared dim and L2-normalizes (unit-norm rows);
plus that each raw handler returns a valid (status,data,to,msg) contract on an EMPTY spec (fleet smoke
contract). Exit 0 iff all checks pass.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
COMP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, COMP)
sys.path.insert(0, os.path.join(COMP, "tools", "researchpapers"))

import torch

from fleet_agents import multimodal_pack as M

torch.manual_seed(0)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
_fails = []


def check(name, cond):
    print(("  ok  " if cond else "  FAIL") + "  " + name)
    if not cond:
        _fails.append(name)


# ── 0. encoder-adapter aligns heterogeneous dims -> shared dim + L2-norm ──────────────────────────────
B = 6
in_dims = {"image": 40, "text": 24, "tabular": 8}
adapter = M.build_encoder_adapter(in_dims, shared_dim=32, device=DEV).eval()
feats = {m: torch.randn(B, d, device=DEV) for m, d in in_dims.items()}
with torch.no_grad():
    aligned = adapter(feats)
check("adapter aligns every modality to shared dim (B, 32)",
      all(tuple(aligned[m].shape) == (B, 32) for m in in_dims))
check("adapter output finite", all(bool(torch.isfinite(aligned[m]).all()) for m in in_dims))
check("adapter L2-normalizes rows to unit norm",
      all(bool((aligned[m].norm(dim=-1) - 1.0).abs().max() < 1e-3) for m in in_dims))
# missing modality in input dict is simply skipped
with torch.no_grad():
    part = adapter({k: v for k, v in feats.items() if k != "text"})
check("adapter skips a missing modality", set(part) == {"image", "tabular"})

# ── 1. fusion: every strategy returns (B, shared_dim) + finite, for N variable-dim modalities ─────────
shared = 24
for s in M._ALL_STRATEGIES:
    model = M.build_multimodal_fusion(in_dims, shared_dim=shared, strategy=s, device=DEV).eval()
    with torch.no_grad():
        fused = model(feats)
    check(f"fusion[{s}] -> (B, shared) finite",
          tuple(fused.shape) == (B, shared) and bool(torch.isfinite(fused).all()))

# optional regression head path
hmodel = M.build_multimodal_fusion(in_dims, shared_dim=shared, strategy="concat", head="regression",
                                   out_dim=1, device=DEV).eval()
with torch.no_grad():
    y = hmodel(feats)
check("fusion regression head -> (B, 1)", tuple(y.shape) == (B, 1) and bool(torch.isfinite(y).all()))

# missing-modality-safe forward (drop 'text' from the dict)
mmodel = M.build_multimodal_fusion(in_dims, shared_dim=shared, strategy="concat", device=DEV).eval()
with torch.no_grad():
    fmiss = mmodel({k: v for k, v in feats.items() if k != "text"})
check("fusion is missing-modality-safe (still (B, shared))",
      tuple(fmiss.shape) == (B, shared) and bool(torch.isfinite(fmiss).all()))


# ── 2. informative vs noise: fusion tracks the INFORMATIVE modality, beats noise-alone probe ──────────
# Build a 2-class dataset: modality 'sig' carries the label (class-separated means); modality 'noise' is
# pure noise. Train a tiny linear probe on (a) the fused rep vs (b) the noise modality alone; fused must
# separate the classes better (lower training loss / higher train accuracy).
def make_data(n_per=64, d_sig=12, d_noise=12, sep=2.5, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    y = torch.cat([torch.zeros(n_per), torch.ones(n_per)]).long()
    mu = torch.zeros(2, d_sig)
    mu[0, :] = -sep
    mu[1, :] = sep
    sig = torch.randn(2 * n_per, d_sig, generator=g) + mu[y]
    noise = torch.randn(2 * n_per, d_noise, generator=g)          # label-independent
    return sig.to(DEV), noise.to(DEV), y.to(DEV)


def probe_acc(rep, y, epochs=250, lr=0.05):
    """Train a tiny logistic-regression probe on a FIXED representation; return train accuracy."""
    import torch.nn as nn
    rep = rep.detach()
    clf = nn.Linear(rep.shape[1], 2).to(rep.device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(clf(rep), y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (clf(rep).argmax(1) == y).float().mean().item()
    return acc


torch.manual_seed(1)
sig, noise, y = make_data()
dims = {"sig": sig.shape[1], "noise": noise.shape[1]}
data = {"sig": sig, "noise": noise}
# gated fusion: the learned gate can down-weight the noise modality
for strat in ("gated", "cross_attention"):
    fmodel = M.build_multimodal_fusion(dims, shared_dim=32, strategy=strat, device=DEV)
    # brief supervised training of the fusion so its gate/attention can learn to favour the signal
    import torch.nn as nn
    head = nn.Linear(32, 2).to(DEV)
    opt = torch.optim.Adam(list(fmodel.parameters()) + list(head.parameters()), lr=0.02)
    lossf = nn.CrossEntropyLoss()
    fmodel.train()
    for _ in range(300):
        opt.zero_grad()
        logits = head(fmodel(data))
        loss = lossf(logits, y)
        loss.backward()
        opt.step()
    fmodel.eval()
    with torch.no_grad():
        fused = fmodel(data)
    acc_fused = probe_acc(fused, y)
    acc_noise = probe_acc(noise, y)
    check(f"fusion[{strat}] rep separates classes ({acc_fused:.2f}) better than noise-alone ({acc_noise:.2f})",
          acc_fused > acc_noise + 0.10 and acc_fused > 0.80)

# ── 3. modality-dropout: shape-safe + finite when masked; learned null-token on a fully-absent modality ─
names = ["image", "text", "tabular"]
dim = 16
md = M.build_modality_dropout(names, dim, p=0.5, min_keep=1, device=DEV)
fd = {m: torch.randn(B, dim, device=DEV) for m in names}
md.train()
out_tr = md(fd)
check("modality-dropout TRAIN preserves shape for all modalities",
      all(tuple(out_tr[m].shape) == (B, dim) for m in names) and set(out_tr) == set(names))
check("modality-dropout TRAIN stays finite", all(bool(torch.isfinite(out_tr[m]).all()) for m in names))
# at least min_keep real (non-null) modalities kept per row: check no row is fully the null tokens
nulls = torch.stack([md.null[m].detach() for m in names], 0)      # (M, dim)
stk = torch.stack([out_tr[m] for m in names], 1)                  # (B, M, dim)
is_null = torch.stack([torch.isclose(stk[:, j], nulls[j].unsqueeze(0), atol=1e-5).all(-1)
                       for j in range(len(names))], 1)            # (B, M) True where that modality==null
kept = (~is_null).sum(1)                                          # real modalities kept per row
check("modality-dropout keeps >= min_keep real modalities per row", bool((kept >= 1).all()))

# EVAL: a modality fully ABSENT from the input is imputed with its learned null token
md.eval()
miss = {k: v for k, v in fd.items() if k != "text"}
out_ev = md(miss)
check("modality-dropout EVAL preserves shape + fills absent modality",
      all(tuple(out_ev[m].shape) == (B, dim) for m in names))
check("modality-dropout EVAL imputes absent modality with its learned null token",
      torch.allclose(out_ev["text"], md.null["text"].detach().unsqueeze(0).expand(B, dim), atol=1e-5))
check("modality-dropout EVAL leaves present modalities untouched",
      torch.allclose(out_ev["image"], fd["image"], atol=1e-6))

# ── 4. every raw handler returns a valid contract on EMPTY spec (fleet smoke contract) ────────────────
VALID = {"done", "escalated", "holding", "error", "failed", "skipped"}
for h in (M.run_fusion, M.run_encoder_adapter, M.run_modality_dropout):
    r = h({"question": "test", "spec": {}}, "unit")
    check(f"handler {h.__name__} valid contract", isinstance(r, tuple) and len(r) == 4 and r[0] in VALID)
    check(f"handler {h.__name__} returns done on healthy default", r[0] == "done")

print()
if _fails:
    print("FAILURES:", _fails)
    sys.exit(1)
print("ALL MULTIMODAL PACK CHECKS PASSED")
sys.exit(0)
