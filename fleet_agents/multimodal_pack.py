"""multimodal_pack — the MULTIMODAL modality pack: FEATURE/MODEL-level FUSION, GROUNDED in the top
solutions of real multimodal competitions (petfinder-pawpularity-score = image+tabular, shopee-product-
matching = image+text, ariel-data-challenge-2024/2025 = spectroscopic signal+metadata). Mined via the
fleet's `gm-writeup-mine` path (nvidia-kaggle bearer); distilled recurring techniques + full provenance
in docs/multimodal_pack_grounded.md.

A "multimodal" comp = a SINGLE model consuming >=2 modalities (image+text+tabular / signal+metadata),
fused INSIDE the network. The fleet already does LATE / prediction-level fusion (averaging model outputs)
— `ensemble`, `blend-optimize`, `infer-cascade`, `tab-stack`, `checkpoint-merger` — so that is only
REFERENCED here, NOT rebuilt. The genuine gap is FEATURE/MODEL-level fusion.

What this pack adds (each recurring across the mined winners):
  • multimodal-fusion         — dict of per-modality feature tensors -> project each to a shared dim ->
        fuse via a configurable strategy (concat / sum / mean / gated / FiLM / cross-attention /
        bilinear) -> fused representation (+ optional head). N modalities, variable input dims.
        Grounded: shopee "NFNet-F0 + Indonesian-BERT concatenated at final feature layers"; petfinder
        Swin-embedding (+) 12 metadata features -> SVR/MLP head; shopee 2nd GAT-over-similarity (attn).
  • modality-encoder-adapter  — wraps heterogeneous per-modality inputs (image/text/tabular) into
        aligned, L2-normed, shared-dim embeddings with per-modality LayerNorm + projection + learnable
        modality-type embeddings. Grounded: shopee `F.normalize(torch.cat([F.normalize(e1),
        F.normalize(e2)],1))`; ariel per-planet normalization.
  • modality-dropout          — training-time random modality masking + inference missing-modality
        imputation via a learned per-modality NULL token, so the model is robust when a modality is
        absent. Grounded: petfinder's KEY negative finding — image-only ~= image+metadata, so the model
        must never DEPEND on one modality.

Pure torch/numpy, GPU-FIRST (every tensor op runs on CUDA when available; CPU fallback only if no CUDA).
No numpy/torch version is touched, no external deps. Data-wise tests:
test_fleet_agents/multimodal_pack_test.py.
"""
from __future__ import annotations
from .base import BaseAgent

# fusion strategies that COMBINE two already-aligned (same-dim) representations pairwise
_PAIR_STRATEGIES = ("sum", "mean", "gated", "film", "bilinear", "cross_attention")
# fusion strategies + the one that changes dimensionality (concat)
_ALL_STRATEGIES = ("concat",) + _PAIR_STRATEGIES


def _device(spec):
    import torch
    d = (spec or {}).get("device")
    if d:
        return d
    return "cuda" if torch.cuda.is_available() else "cpu"


def l2norm(x, eps=1e-8):
    """Row-wise L2 normalization to the unit sphere (the shopee `F.normalize` before-concat trick)."""
    import torch
    return x / (x.norm(dim=-1, keepdim=True) + eps)


# ════════════════════════════════════════════════════════════ 1. modality-encoder-adapter
def build_encoder_adapter(in_dims, shared_dim=128, l2=True, use_type_emb=True, norm=True, device=None):
    """Align heterogeneous per-modality feature vectors into a common `shared_dim` space (pure torch).

    `in_dims` = dict {modality_name: raw_feature_dim} (image embedding 1280, text embedding 768, tabular
    12, ...). Each modality gets its OWN LayerNorm + Linear projection to `shared_dim`, a learnable
    modality-TYPE embedding added (so the fuser knows which modality a row came from), and optional final
    L2-normalization (the shopee `F.normalize(cat(F.normalize(...)))` pattern). Returns an nn.Module whose
    forward(dict of (B, d_m) tensors) -> dict of (B, shared_dim) aligned, unit-norm embeddings.
    Missing modalities in the input dict are simply skipped (composes with modality-dropout).
    """
    import torch
    from torch import nn
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(in_dims.keys())

    class EncoderAdapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.names = names
            self.shared_dim = shared_dim
            self.l2 = l2
            self.norms = nn.ModuleDict(
                {m: (nn.LayerNorm(int(in_dims[m])) if norm else nn.Identity()) for m in names})
            self.proj = nn.ModuleDict({m: nn.Linear(int(in_dims[m]), shared_dim) for m in names})
            # learnable per-modality TYPE embedding (like a token-type / segment embedding)
            self.type_emb = nn.Parameter(torch.randn(len(names), shared_dim) * 0.02) if use_type_emb \
                else None
            self._idx = {m: i for i, m in enumerate(names)}

        def forward(self, feats: dict):
            out = {}
            for m, x in feats.items():
                if m not in self.proj:
                    continue
                h = self.proj[m](self.norms[m](x))
                if self.type_emb is not None:
                    h = h + self.type_emb[self._idx[m]]
                if self.l2:
                    h = l2norm(h)
                out[m] = h
            return out

    return EncoderAdapter().to(dev)


class ModalityEncoderAdapter(BaseAgent):
    name = "modality-encoder-adapter"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            torch.manual_seed(int(spec.get("seed", 0)))
            B = int(spec.get("batch", 6))
            shared = int(spec.get("shared_dim", 64))
            # heterogeneous raw dims (image embedding / text embedding / tabular vector) — the winners' mix
            in_dims = spec.get("in_dims") or {"image": 40, "text": 24, "tabular": 8}
            adapter = build_encoder_adapter(in_dims, shared_dim=shared, device=dev).eval()
            feats = {m: torch.randn(B, int(d), device=dev) for m, d in in_dims.items()}
            with torch.no_grad():
                aligned = adapter(feats)
            shapes_ok = all(tuple(v.shape) == (B, shared) for v in aligned.values())
            finite = all(bool(torch.isfinite(v).all()) for v in aligned.values())
            # L2-normed => every row has unit norm (the shared-sphere alignment)
            unit = all(bool((v.norm(dim=-1) - 1.0).abs().max() < 1e-3) for v in aligned.values())
            ok = shapes_ok and finite and unit and set(aligned) == set(in_dims)
            msg = (f"modality-encoder-adapter: aligned {len(in_dims)} modalities {dict(in_dims)} -> shared "
                   f"dim {shared} (per-modality LayerNorm+proj + learnable type-emb, L2-normed). "
                   f"shapes_ok={shapes_ok} finite={finite} unit_norm={unit} device={dev}. "
                   f"Grounded: shopee F.normalize(cat(F.normalize(...))); ariel per-planet norm.")
            self.log(msg, kind="finding",
                     recommendation="build_encoder_adapter(in_dims_dict, shared_dim) BEFORE multimodal-fusion; "
                                    "aligns image/text/tabular to one unit-sphere space; missing modalities skipped")
            return self.done({"n_modalities": len(in_dims), "shared_dim": shared, "shapes_ok": shapes_ok,
                              "finite": finite, "unit_norm": unit, "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher",
                                         f"[{worker}] modality-encoder-adapter bad output (shapes_ok={shapes_ok} "
                                         f"finite={finite} unit={unit})")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] modality-encoder-adapter FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ 2. multimodal-fusion
def _fusion_module(torch, nn, strategy, shared_dim, n_modalities, heads=4):
    """One fusion block over a LIST of already-aligned (B, shared_dim) modality tensors -> (B, out_dim).
    Strategy grounds:
      concat          — stack all modalities -> Linear (shopee 'concatenated at final feature layers').
      sum / mean      — parameter-light additive fusion of aligned embeddings.
      gated           — per-modality learned gate (sigmoid) weights each modality then sums (soft select).
      film            — modality[0] is FiLM-modulated (scale,shift) by the others (feature-wise cond).
      bilinear        — low-rank bilinear interaction between the first two modalities (+ residual sum).
      cross_attention — multi-head attention with each modality as a token; fused = mean of attended tokens
                        (shopee-2nd GAT-over-similarity / VQA co-attention family).
    """
    class Fusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.strategy = strategy
            self.shared_dim = shared_dim
            self.n = n_modalities
            if strategy == "concat":
                self.out = nn.Linear(shared_dim * n_modalities, shared_dim)
            elif strategy == "gated":
                self.gate = nn.Linear(shared_dim, 1)
                self.out = nn.Linear(shared_dim, shared_dim)
            elif strategy == "film":
                self.film = nn.Linear(shared_dim, 2 * shared_dim)   # from the "other" modalities -> (scale,shift)
                self.out = nn.Linear(shared_dim, shared_dim)
            elif strategy == "bilinear":
                rank = max(8, shared_dim // 2)
                self.u = nn.Linear(shared_dim, rank, bias=False)
                self.v = nn.Linear(shared_dim, rank, bias=False)
                self.out = nn.Linear(rank, shared_dim)
            elif strategy == "cross_attention":
                self.attn = nn.MultiheadAttention(shared_dim, num_heads=heads, batch_first=True)
                self.out = nn.Linear(shared_dim, shared_dim)
            else:  # sum / mean
                self.out = nn.Linear(shared_dim, shared_dim)

        def forward(self, mods):
            # mods: list of (B, shared_dim)
            if self.strategy == "concat":
                return self.out(torch.cat(mods, dim=-1))
            if self.strategy == "sum":
                return self.out(torch.stack(mods, 0).sum(0))
            if self.strategy == "mean":
                return self.out(torch.stack(mods, 0).mean(0))
            if self.strategy == "gated":
                stk = torch.stack(mods, 1)                       # (B, M, D)
                w = torch.softmax(self.gate(stk), dim=1)          # (B, M, 1) soft modality selection
                return self.out((w * stk).sum(1))
            if self.strategy == "film":
                base = mods[0]
                ctx = torch.stack(mods[1:], 0).mean(0) if len(mods) > 1 else base
                scale, shift = self.film(ctx).chunk(2, dim=-1)
                return self.out(base * (1.0 + scale) + shift)     # feature-wise modulation
            if self.strategy == "bilinear":
                a, b = mods[0], mods[1] if len(mods) > 1 else mods[0]
                inter = self.out(self.u(a) * self.v(b))           # low-rank bilinear
                return inter + torch.stack(mods, 0).sum(0)        # + residual so nothing is dropped
            if self.strategy == "cross_attention":
                tok = torch.stack(mods, 1)                        # (B, M, D) each modality a token
                att, _ = self.attn(tok, tok, tok)
                return self.out(att.mean(1))
            return self.out(torch.stack(mods, 0).mean(0))

    return Fusion()


def build_multimodal_fusion(in_dims, shared_dim=128, strategy="concat", out_dim=None, head=None,
                            adapter=True, heads=4, device=None):
    """Build a FEATURE-LEVEL multimodal fusion model (pure torch, GPU-first).

    `in_dims` = dict {modality: raw_dim}. If `adapter=True` each modality is first aligned to `shared_dim`
    (per-modality LayerNorm + projection + type-emb + L2, via build_encoder_adapter); then fused by
    `strategy` in {concat, sum, mean, gated, film, bilinear, cross_attention}; then an OPTIONAL head
    (`head='regression'|'classification'` with `out_dim`) maps the fused (B, shared_dim) vector to the
    target. Forward(dict of (B, d_m) tensors) -> fused (B, shared_dim) OR head output (B, out_dim).
    Robust to a MISSING modality: absent modalities are skipped by the adapter and (for concat/bilinear
    which need a fixed arity) filled with a zero vector so the shape stays valid — combine with
    modality-dropout's learned null token for the trained-in version.
    """
    import torch
    from torch import nn
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    names = list(in_dims.keys())
    n = len(names)

    class MultimodalFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.names = names
            self.strategy = strategy
            self.shared_dim = shared_dim
            self.adapter = build_encoder_adapter(in_dims, shared_dim=shared_dim, device=dev) if adapter else None
            # if no adapter, modalities are assumed already at shared_dim
            self.fuse = _fusion_module(torch, nn, strategy, shared_dim, n, heads=heads)
            self.head = None
            if head in ("regression", "classification"):
                od = int(out_dim or (1 if head == "regression" else 2))
                self.head = nn.Sequential(nn.LayerNorm(shared_dim), nn.GELU(), nn.Linear(shared_dim, od))

        def encode(self, feats: dict):
            if self.adapter is not None:
                aligned = self.adapter(feats)
            else:
                aligned = {m: feats[m] for m in feats}
            # keep a fixed modality ORDER; fill any missing modality with zeros (shape-safe fusion)
            mods = []
            for m in self.names:
                if m in aligned:
                    mods.append(aligned[m])
                else:
                    ref = next(iter(aligned.values()))
                    mods.append(torch.zeros(ref.shape[0], self.shared_dim, device=ref.device, dtype=ref.dtype))
            return self.fuse(mods)

        def forward(self, feats: dict):
            fused = self.encode(feats)
            return self.head(fused) if self.head is not None else fused

    return MultimodalFusion().to(dev)


class MultimodalFusion(BaseAgent):
    name = "multimodal-fusion"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            torch.manual_seed(int(spec.get("seed", 0)))
            B = int(spec.get("batch", 6))
            shared = int(spec.get("shared_dim", 48))
            in_dims = spec.get("in_dims") or {"image": 40, "text": 24, "tabular": 8}
            feats = {m: torch.randn(B, int(d), device=dev) for m, d in in_dims.items()}
            strategies = spec.get("strategies") or list(_ALL_STRATEGIES)
            checks = {}
            for s in strategies:
                model = build_multimodal_fusion(in_dims, shared_dim=shared, strategy=s, device=dev).eval()
                with torch.no_grad():
                    fused = model(feats)
                checks[s] = tuple(fused.shape) == (B, shared) and bool(torch.isfinite(fused).all())
            # optional head path (regression)
            hmodel = build_multimodal_fusion(in_dims, shared_dim=shared, strategy="gated",
                                             head="regression", out_dim=1, device=dev).eval()
            with torch.no_grad():
                y = hmodel(feats)
            head_ok = tuple(y.shape) == (B, 1) and bool(torch.isfinite(y).all())
            # missing-modality robustness: drop 'text' from the input, fusion must still return valid shape
            miss = {k: v for k, v in feats.items() if k != "text"}
            mmodel = build_multimodal_fusion(in_dims, shared_dim=shared, strategy="concat", device=dev).eval()
            with torch.no_grad():
                fmiss = mmodel(miss)
            miss_ok = tuple(fmiss.shape) == (B, shared) and bool(torch.isfinite(fmiss).all())
            ok = all(checks.values()) and head_ok and miss_ok
            msg = (f"multimodal-fusion: {sum(checks.values())}/{len(checks)} strategies ok ({checks}); "
                   f"head={head_ok} missing-modality-safe={miss_ok}; fused {len(in_dims)} modalities "
                   f"{dict(in_dims)} -> (B,{shared}) via concat/sum/gated/film/bilinear/cross_attention. device={dev}. "
                   f"Grounded: shopee concat-at-final-layers; petfinder embed+metadata->head; shopee-2nd attn.")
            self.log(msg, kind="finding",
                     recommendation="build_multimodal_fusion(in_dims, strategy='concat'|'gated'|'cross_attention', "
                                    "head='regression'); FEATURE-level fusion (late/decision fusion stays in "
                                    "ensemble/blend-optimize/infer-cascade)")
            return self.done({"strategies": {k: bool(v) for k, v in checks.items()}, "head_ok": head_ok,
                              "missing_modality_safe": miss_ok, "shared_dim": shared,
                              "n_modalities": len(in_dims), "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher",
                                         f"[{worker}] multimodal-fusion checks failed: {checks} head={head_ok} miss={miss_ok}")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] multimodal-fusion FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ 3. modality-dropout
def build_modality_dropout(names, dim, p=0.5, min_keep=1, device=None):
    """Training-time modality masking + inference missing-modality imputation (pure torch).

    Holds a learnable NULL token per modality (shape (dim,)). forward(dict {modality: (B, dim)}, training):
      • TRAIN: each modality independently dropped with prob `p` (replaced by its null token), but at least
        `min_keep` modalities are always kept per sample (so a row is never fully blanked). This is the
        petfinder robustness trick — the model must not DEPEND on any single modality.
      • EVAL: any modality ABSENT from the input dict is imputed with its learned null token (the model
        works when a real modality is missing at test time).
    Shapes are always preserved: returns dict {modality: (B, dim)} for ALL registered `names`.
    """
    import torch
    from torch import nn
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")

    class ModalityDropout(nn.Module):
        def __init__(self):
            super().__init__()
            self.names = list(names)
            self.dim = dim
            self.p = p
            self.min_keep = max(1, int(min_keep))
            self.null = nn.ParameterDict(
                {m: nn.Parameter(torch.randn(dim) * 0.02) for m in self.names})

        def forward(self, feats: dict, training=None):
            training = self.training if training is None else training
            # reference batch size / device from any present modality
            present = [feats[m] for m in self.names if m in feats]
            if not present:
                raise ValueError("modality-dropout: at least one modality must be present")
            B = present[0].shape[0]
            dev_ = present[0].device
            out = {}
            if not training:
                # EVAL: impute missing modalities with their null token; keep present ones as-is
                for m in self.names:
                    if m in feats:
                        out[m] = feats[m]
                    else:
                        out[m] = self.null[m].to(dev_).unsqueeze(0).expand(B, self.dim)
                return out
            # TRAIN: random per-sample per-modality mask, guaranteeing >= min_keep kept
            avail = [m for m in self.names if m in feats]
            M = len(avail)
            keep = (torch.rand(B, M, device=dev_) >= self.p)          # (B, M) True=keep
            # enforce min_keep: if a row keeps < min_keep, force the top-scoring modalities on
            deficit = (self.min_keep - keep.sum(1)).clamp(min=0)
            if bool((deficit > 0).any()):
                score = torch.rand(B, M, device=dev_)
                order = score.argsort(dim=1, descending=True)         # random priority per row
                for i in range(B):
                    need = int(deficit[i].item())
                    if need > 0:
                        keep[i, order[i, :self.min_keep]] = True
            for j, m in enumerate(avail):
                mask = keep[:, j].unsqueeze(-1).to(feats[m].dtype)     # (B,1)
                null = self.null[m].to(dev_).unsqueeze(0)
                out[m] = mask * feats[m] + (1.0 - mask) * null
            # modalities not in the input at train time also imputed with null (fully-absent path)
            for m in self.names:
                if m not in out:
                    out[m] = self.null[m].to(dev_).unsqueeze(0).expand(B, self.dim)
            return out

    return ModalityDropout().to(dev)


class ModalityDropout(BaseAgent):
    name = "modality-dropout"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch
            spec = self.spec(q)
            dev = _device(spec)
            torch.manual_seed(int(spec.get("seed", 0)))
            B = int(spec.get("batch", 8))
            dim = int(spec.get("dim", 16))
            names = spec.get("names") or ["image", "text", "tabular"]
            p = float(spec.get("p", 0.5))
            md = build_modality_dropout(names, dim, p=p, min_keep=1, device=dev)
            feats = {m: torch.randn(B, dim, device=dev) for m in names}
            # TRAIN path: shapes preserved, finite, >= min_keep modalities kept per row
            md.train()
            out_tr = md(feats)
            shapes_ok = all(tuple(out_tr[m].shape) == (B, dim) for m in names) and set(out_tr) == set(names)
            finite_tr = all(bool(torch.isfinite(out_tr[m]).all()) for m in names)
            # EVAL missing-modality: drop 'text' entirely -> imputed by its null token
            md.eval()
            miss = {k: v for k, v in feats.items() if k != "text"}
            out_ev = md(miss)
            null_used = torch.allclose(out_ev["text"], md.null["text"].detach().unsqueeze(0).expand(B, dim),
                                       atol=1e-5)
            eval_ok = all(tuple(out_ev[m].shape) == (B, dim) for m in names) and null_used
            ok = shapes_ok and finite_tr and eval_ok
            msg = (f"modality-dropout: train mask (p={p}, min_keep=1) shapes_ok={shapes_ok} finite={finite_tr}; "
                   f"eval missing-modality imputed by learned null token null_used={null_used}. "
                   f"{len(names)} modalities {names} dim={dim} device={dev}. "
                   f"Grounded: petfinder image-only ~= image+metadata -> never depend on one modality.")
            self.log(msg, kind="finding",
                     recommendation="build_modality_dropout(names, dim, p) BEFORE fusion at train; at inference the "
                                    "learned null token imputes any absent modality (missing-modality robustness)")
            return self.done({"shapes_ok": shapes_ok, "finite_train": finite_tr, "null_used": bool(null_used),
                              "n_modalities": len(names), "dim": dim, "device": str(dev)}, msg) \
                if ok else self.escalate(worker, "researcher",
                                         f"[{worker}] modality-dropout bad (shapes={shapes_ok} finite={finite_tr} "
                                         f"eval={eval_ok})")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] modality-dropout FAILED ({str(e)[:180]})")


# ════════════════════════════════════════════════════════════ handlers
_FUSE = MultimodalFusion()
_ADAPT = ModalityEncoderAdapter()
_DROP = ModalityDropout()


def run_fusion(q, worker):
    return _FUSE.run(q, worker)


def run_encoder_adapter(q, worker):
    return _ADAPT.run(q, worker)


def run_modality_dropout(q, worker):
    return _DROP.run(q, worker)
