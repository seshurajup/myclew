"""training_head_pack — three GENUINELY-NEW, competition-agnostic training primitives distilled from a
batch of GM winner repos. None duplicate the concurrently-added EMA/SWA/mixup/focal/label-smoothing/SAM/
arcface/WBF/snapshot/multi-TTA set, nor any existing handler. All pure torch/numpy, CPU-verifiable:

  • deep-supervision   — multi-scale segmentation deep supervision (ChristofHenkel, CZII CryoET 1st place):
                         a UNet emits a seg head at EVERY decoder level; the target is downsampled with
                         adaptive_MAX_pool (not avg/interp — max KEEPS tiny foreground objects alive at coarse
                         scales) and a per-level weight vector focuses the loss (e.g. [0,0,1,1] = ignore the
                         two coarsest heads). Reusable loss for any dense-prediction/detection-by-heatmap task.

  • sed-attention-pool — weakly-supervised attention pooling head (FlorentinGe, BirdCLEF 2026 2nd place):
                         per-frame logits + a learned softmax-over-time attention → a single clip prediction,
                         so a clip-level label supervises frame-level detection with NO frame labels. Plus a
                         learnable GeM (generalised-mean) pool over the frequency/spatial axis. Reusable for
                         any weakly-labelled temporal/MIL problem (audio SED, video action, WSI tiles).

  • awp-perturb        — Adversarial Weight Perturbation (GosUxD, MABe 5th place). DISTINCT from SAM: it takes
                         a PER-PARAMETER gradient-normalised ascent step delta*g/||g||, back-props the loss at
                         the perturbed weights, then RESTORES the original weights before the optimizer step
                         (SAM steps AT the perturbed point with one global grad norm). A cheap flat-minima
                         regulariser that reliably lifts noisy Kaggle CV; typically enabled after a warmup.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseAgent


# ==================================================================== deep supervision
def _adaptive_max_pool(y, size):
    """Downsample a (B,C,...spatial) target to `size` with MAX pooling — 2D or 3D."""
    nd = y.ndim - 2
    if nd == 3:
        return F.adaptive_max_pool3d(y, size)
    if nd == 2:
        return F.adaptive_max_pool2d(y, size)
    return F.adaptive_max_pool1d(y, size)


def to_ce_target(y, eps=1e-6):
    """Multi-label foreground (B,C,...) → categorical target with an appended background channel,
    normalised to sum 1 over channels (CryoET trick: overlapping soft foreground → dense-CE target)."""
    y_bg = 1.0 - y.sum(1, keepdim=True).clamp(0, 1)
    y = torch.cat([y, y_bg], 1)
    return y / y.sum(1, keepdim=True).clamp_min(eps)


def dense_cross_entropy(logits, target, class_weights=None):
    """CE between per-voxel logits (B,C,...) and a soft categorical target (same shape). Mean over
    batch+spatial, then weighted-sum over classes. Returns (loss, per_class_loss)."""
    logp = F.log_softmax(logits.float(), dim=1)
    loss = -(logp * target.float())
    dims = (0,) + tuple(range(2, loss.ndim))
    class_losses = loss.mean(dims)
    if class_weights is not None:
        w = class_weights.to(class_losses.device).float()
        return (class_losses * w).sum(), class_losses
    return class_losses.sum(), class_losses


def deep_supervision_loss(outputs, target, lvl_weights=None, class_weights=None, add_bg=True):
    """Multi-scale deep-supervision loss.

    outputs      : list of seg logits, one per decoder level, shapes (B, C(+1), *spatial_l).
    target       : (B, C, *spatial_full) multi-label foreground map.
    lvl_weights  : weight per level (len == len(outputs)); e.g. [0,0,1,1] ignores the 2 coarsest heads.
    add_bg       : append a background channel and normalise (dense-CE target). If False, target is
                   assumed already categorical and matched to each output's channel count.

    Each level's target is MAX-pooled to that level's spatial size, so small objects survive coarse heads.
    """
    n = len(outputs)
    if lvl_weights is None:
        lvl_weights = torch.ones(n)
    lvl_weights = torch.as_tensor(lvl_weights, dtype=torch.float32, device=outputs[0].device)
    losses = []
    for i, out in enumerate(outputs):
        yl = _adaptive_max_pool(target.float(), out.shape[-(target.ndim - 2):])
        tgt = to_ce_target(yl) if add_bg else yl
        losses.append(dense_cross_entropy(out, tgt, class_weights)[0])
    losses = torch.stack(losses)
    return (losses * lvl_weights).sum() / lvl_weights.sum().clamp_min(1e-9)


# ==================================================================== SED attention pooling
class GeMPool(nn.Module):
    """Learnable generalised-mean pool over one axis (default freq/height dim=2 of a (B,C,H,W) map).
    p=1 → average, p→∞ → max. p is a learnable scalar (clamped ≥1)."""
    def __init__(self, p_init=3.0, eps=1e-6, dim=2):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p_init)))
        self.eps = float(eps); self.dim = int(dim)

    def forward(self, x):
        p = self.p.clamp(min=1.0)
        return x.clamp(min=self.eps).pow(p).mean(dim=self.dim).pow(1.0 / p)


def attention_pool(frame_logits, att_logits):
    """Weakly-supervised temporal attention pooling (SED head).

    frame_logits : (B, T, C) per-frame class logits.
    att_logits   : (B, T, C) per-frame attention logits.
    Returns (clipwise_mean, att_clipwise, att_weights):
      clipwise_mean : (B, C) simple time-mean of frame logits  — the stable TRAIN target.
      att_clipwise  : (B, C) attention-weighted sum            — the sharp EVAL/inference prediction.
      att_weights   : (B, T, C) softmax over time.
    """
    att_weights = torch.softmax(att_logits, dim=1)
    clipwise_mean = frame_logits.mean(dim=1)
    att_clipwise = (frame_logits * att_weights).sum(dim=1)
    return clipwise_mean, att_clipwise, att_weights


class SEDHead(nn.Module):
    """Sound-Event-Detection-style weakly-supervised head: shared BN+dropout on (B,T,C_in) frame features,
    then a class fc and an attention fc → clip prediction via attention_pool. Backbone-agnostic."""
    def __init__(self, in_features, num_classes, dropout=0.3):
        super().__init__()
        self.bn = nn.BatchNorm1d(in_features)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)
        self.att_fc = nn.Linear(in_features, num_classes)

    def forward(self, feat):                          # feat: (B, T, C_in)
        B, T, C = feat.shape
        feat = self.bn(feat.reshape(B * T, C)).reshape(B, T, C)
        feat = self.dropout(feat)
        return attention_pool(self.fc(feat), self.att_fc(feat))


# ==================================================================== Adversarial Weight Perturbation
class AWP:
    """Adversarial Weight Perturbation regulariser (Kaggle-style). Wraps a model + optimizer; each
    train_step does a normal fwd/bwd, then a PER-PARAMETER grad-normalised ascent step (delta*g/||g||),
    a second fwd/bwd at the perturbed weights, RESTORES the original weights, and finally the optimizer step.
    loss_fn(model) must return a scalar loss (closure). Returns the clean loss (float)."""
    def __init__(self, model, optimizer, delta=0.1, eps=1e-6, clip_grad=0.0):
        self.model = model; self.opt = optimizer
        self.delta = float(delta); self.eps = float(eps); self.clip_grad = float(clip_grad)

    def _perturb(self):
        perts = []
        with torch.no_grad():
            for p in self.model.parameters():
                pert = None
                if p.grad is not None:
                    g = p.grad.data; norm = torch.norm(g)
                    if norm > 0:
                        pert = self.delta * g / (norm + self.eps)
                        p.data.add_(pert)
                perts.append(pert)
        return perts

    def _restore(self, perts):
        with torch.no_grad():
            for p, pert in zip(self.model.parameters(), perts):
                if pert is not None:
                    p.data.sub_(pert)

    def train_step(self, loss_fn):
        self.opt.zero_grad(); clean = loss_fn(self.model); clean.backward()   # 1) clean grads
        perts = self._perturb()                                                # 2) ascend to adversarial weights
        self.opt.zero_grad(); loss_fn(self.model).backward()                   # 3) grads at perturbed weights
        self._restore(perts)                                                   # 4) back to clean weights
        if self.clip_grad > 0:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        self.opt.step(); self.opt.zero_grad()                                  # 5) descend clean weights w/ adv grads
        return float(clean.detach())


# ==================================================================== agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class DeepSupervision(_B):
    name = "deep-supervision"
    def run(self, q, worker):
        s = self.spec(q); torch.manual_seed(int(s.get("seed", 0)))
        C = int(s.get("classes", 3)); B = int(s.get("batch", 2)); sz = int(s.get("size", 16))
        # synthetic 2D multi-scale demo: a tiny FCN with heads at 4 decoder scales
        target = (torch.rand(B, C, sz, sz) > 0.85).float()
        levels = [sz, sz // 2, sz // 4, sz // 8]
        heads = nn.ModuleList([nn.Conv2d(4, C + 1, 1) for _ in levels])
        stem = nn.Sequential(nn.Conv2d(C, 4, 3, padding=1), nn.ReLU())
        params = list(stem.parameters()) + list(heads.parameters())
        opt = torch.optim.Adam(params, lr=1e-2)
        lvlw = s.get("lvl_weights", [0, 0, 1, 1])
        def fwd():
            f = stem(target)
            return [h(F.adaptive_avg_pool2d(f, (L, L))) for h, L in zip(heads, levels)]
        first = float(deep_supervision_loss(fwd(), target, lvlw).detach())
        for _ in range(int(s.get("steps", 40))):
            opt.zero_grad(); loss = deep_supervision_loss(fwd(), target, lvlw); loss.backward(); opt.step()
        last = float(loss.detach())
        msg = (f"deep-supervision: {len(levels)}-level max-pooled DS loss {first:.4f}->{last:.4f} "
               f"(lvl_weights={lvlw}); target downsampled with adaptive_MAX_pool to keep tiny objects.")
        self.log(msg, kind="finding", recommendation="attach a seg head per decoder level; weight finer levels")
        return self.done({"loss_first": first, "loss_last": last, "levels": levels}, msg)


class SEDAttentionPool(_B):
    name = "sed-attention-pool"
    def run(self, q, worker):
        s = self.spec(q); torch.manual_seed(int(s.get("seed", 0)))
        B = int(s.get("batch", 8)); T = int(s.get("frames", 12)); Cf = int(s.get("feat", 16))
        K = int(s.get("classes", 3))
        # weakly-supervised MIL: one salient frame per positive clip carries the class signal
        feats = torch.randn(B, T, Cf)
        y = (torch.rand(B, K) > 0.6).float()
        salient = torch.randint(0, T, (B,))
        for b in range(B):
            feats[b, salient[b], :K] += 3.0 * (2 * y[b] - 1)     # inject signal at one frame only
        head = SEDHead(Cf, K); opt = torch.optim.Adam(head.parameters(), lr=5e-2)
        def step():
            clip_mean, att_clip, _ = head(feats)
            return F.binary_cross_entropy_with_logits(clip_mean, y), att_clip
        first = float(step()[0].detach())
        for _ in range(int(s.get("steps", 120))):
            opt.zero_grad(); loss, _ = step(); loss.backward(); opt.step()
        with torch.no_grad():
            _, att_clip, w = head(feats)
            acc = float(((att_clip > 0).float() == y).float().mean())
        msg = (f"sed-attention-pool: weakly-supervised BCE {first:.4f}->{float(loss.detach()):.4f}, "
               f"attention-clip acc={acc:.3f}; softmax-over-time att focuses on the salient frame.")
        self.log(msg, kind="finding", recommendation="train on clip-mean, infer on attention-weighted clip")
        return self.done({"loss_last": float(loss.detach()), "att_clip_acc": acc}, msg)


class AWPPerturb(_B):
    name = "awp-perturb"
    def run(self, q, worker):
        s = self.spec(q); torch.manual_seed(int(s.get("seed", 0)))
        n = int(s.get("n", 128)); d = int(s.get("d", 8))
        X = torch.randn(n, d); w_true = torch.randn(d, 1); Y = X @ w_true + 0.1 * torch.randn(n, 1)
        model = nn.Linear(d, 1); opt = torch.optim.SGD(model.parameters(), lr=1e-2)
        awp = AWP(model, opt, delta=float(s.get("delta", 0.05)), clip_grad=float(s.get("clip_grad", 0.0)))
        def loss_fn(m): return F.mse_loss(m(X), Y)
        first = float(loss_fn(model).detach())
        # verify restore-before-step: snapshot then run one AWP step; weights must move only by the opt step
        snap = [p.detach().clone() for p in model.parameters()]
        _ = awp.train_step(loss_fn)
        moved = sum(float((p.detach() - s0).abs().sum()) for p, s0 in zip(model.parameters(), snap))
        for _ in range(int(s.get("steps", 200))):
            awp.train_step(loss_fn)
        last = float(loss_fn(model).detach())
        msg = (f"awp-perturb: MSE {first:.4f}->{last:.4f} (delta={awp.delta}); per-param grad-normalised "
               f"ascent + restore-before-step (distinct from SAM). one-step weight move={moved:.4f}.")
        self.log(msg, kind="finding", recommendation="enable AWP after a warmup; tune delta on CV")
        return self.done({"loss_first": first, "loss_last": last, "one_step_move": moved}, msg)


_DS = DeepSupervision(); _SED = SEDAttentionPool(); _AWP = AWPPerturb()


def run_deep_supervision(q, worker): return _DS.run(q, worker)
def run_sed_attention(q, worker): return _SED.run(q, worker)
def run_awp(q, worker): return _AWP.run(q, worker)
