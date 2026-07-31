"""train-tricks — REUSABLE (any competition) pack of the most common Kaggle-grandmaster TRAINING techniques
the fleet was MISSING, distilled by scanning 179 winner repos (docs/gm_distill_manifest.json). The fleet had
augmentation-search, arch-search and optimizers, but none of the winner-standard training-loop primitives.
This adds them as clean, small, correct, GPU-capable (torch/CUDA) utilities usable from ANY trainer:

  • ModelEMA                     — exponential moving average of weights (register/update/copy_to). ~37 repos.
  • SWA (averaged model + BN)    — stochastic weight averaging: averaged_model + update_bn helper. ~12 repos.
  • mixup / cutmix               — batch augmentation → (mixed_x, y_a, y_b, lam) + mixup_criterion. ~25 repos.
  • label_smoothing_cross_entropy — smoothed-target CE / smooth_one_hot targets (sum to 1). ~24 repos.
  • focal loss                   — binary_focal_loss + FocalLoss (multiclass), down-weights easy examples. ~23 repos.
  • SAM                          — sharpness-aware minimization optimizer wrapper (first_step / second_step).
  • ArcFace / sub-center ArcFace — additive-angular-margin metric-learning head + loss.

Every utility is pure PyTorch, tiny, documented, and runs on CUDA when available (the always-GPU rule) while
degrading gracefully to CPU. The `TrainTricksPack` BaseAgent self-verifies the pack on a synthetic batch and
emits a finding; its data-wise test lives in test_fleet_agents/train_tricks_pack_test.py.
"""
from __future__ import annotations
import copy
import math
from .base import BaseAgent


# ══════════════════════════════════════════════════════════════════ 1. EMA (Exponential Moving Average)
class ModelEMA:
    """Exponential moving average of model weights — the single most common winner trick (~37 repos).

    Keeps a shadow copy of the model's parameters+buffers updated as `ema = decay*ema + (1-decay)*param`
    after every optimizer step. The EMA weights are typically MORE robust than the raw weights at eval time.

    Usage:
        ema = ModelEMA(model, decay=0.999)
        for batch in loader:
            ... loss.backward(); optimizer.step()
            ema.update(model)            # after each step
        ema.copy_to(model)               # swap EMA weights in for evaluation / checkpointing

    A warmup schedule ramps the decay so early (noisy) steps don't dominate the average.
    """
    def __init__(self, model, decay: float = 0.9999, warmup: int = 0, device=None):
        import torch
        self.decay = float(decay)
        self.warmup = int(warmup)
        self.num_updates = 0
        # a frozen deep copy is the shadow; never receives gradients
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.device = device
        if device is not None:
            self.ema.to(device)
        self._torch = torch

    def _cur_decay(self) -> float:
        if self.warmup <= 0:
            return self.decay
        # ramp: min(decay, (1+t)/(10+t)) so early steps average faster (timm-style)
        d = (1 + self.num_updates) / (10 + self.num_updates)
        return min(self.decay, d)

    @property
    def module(self):
        """The EMA (shadow) model — use this for evaluation."""
        return self.ema

    def update(self, model):
        """Update the shadow weights toward the live model. Call after every optimizer.step()."""
        torch = self._torch
        self.num_updates += 1
        d = self._cur_decay()
        with torch.no_grad():
            ema_params = dict(self.ema.named_parameters())
            for name, p in model.named_parameters():
                e = ema_params[name]
                if self.device is not None:
                    p = p.to(self.device)
                if e.dtype.is_floating_point:
                    e.mul_(d).add_(p.detach(), alpha=1.0 - d)
                else:
                    e.copy_(p.detach())
            # buffers (e.g. BatchNorm running stats) tracked by direct copy
            ema_bufs = dict(self.ema.named_buffers())
            for name, b in model.named_buffers():
                if name in ema_bufs:
                    src = b.to(self.device) if self.device is not None else b
                    ema_bufs[name].copy_(src)

    def copy_to(self, model):
        """Copy the EMA weights INTO `model` (in-place) — swap them in for eval/export."""
        torch = self._torch
        with torch.no_grad():
            ema_params = dict(self.ema.named_parameters())
            for name, p in model.named_parameters():
                p.copy_(ema_params[name].to(p.device))
            ema_bufs = dict(self.ema.named_buffers())
            for name, b in model.named_buffers():
                if name in ema_bufs:
                    b.copy_(ema_bufs[name].to(b.device))

    def state_dict(self):
        return self.ema.state_dict()


# ══════════════════════════════════════════════════════════════════ 1b. CMS — per-block update FREQUENCIES
# Behrouz, Razaviyayn, Zhong & Mirrokni, "Nested Learning: The Illusion of Deep Learning Architecture",
# NeurIPS 2025 — paper: https://alibehrouz.com/files/NL.pdf  (§7.1, eqs. 70-71)
# local: docs/papers/nested-learning/nested-learning.md · lessons: learning/annotated/nl07.learning
#
# The idea we can use today without changing an architecture: a Transformer is already a two-frequency
# machine (attention updates per token, the MLP is frozen after pre-training) and the whole proposal is to
# fill in the spectrum between those extremes. A "Continuum Memory System" is a chain of blocks whose
# parameters are updated every C^(l) steps, so fast blocks adapt while slow blocks keep persistent
# knowledge — and when a fast block forgets something, the slower blocks still hold it.
#
# MEASURED on the 5090 (lesson nl07): the average parameters written per step is exactly sum(n_l / C_l)
# (matched the prediction to <2%), and the inference cost is UNCHANGED because the gate is on the
# optimiser step, not on the forward pass.
def cms_param_groups(named_modules, periods):
    """Assign update PERIODS to parameter groups → [{"params": [...], "period": C, "name": …}].

    `named_modules` = [(name, module), …] in depth order; `periods` = the period per group (e.g.
    (1, 4, 16, 64) → the first group updates every step, the last every 64). Groups are cut evenly over
    the modules given, so this works for any backbone without per-model wiring.
    """
    mods = [(n, m) for n, m in named_modules if any(p.requires_grad for p in m.parameters(recurse=False))
            or list(m.parameters(recurse=False))]
    if not mods or not periods:
        return []
    per = max(1, len(mods) // len(periods))
    groups = []
    for gi, C in enumerate(periods):
        chunk = mods[gi * per: (gi + 1) * per] if gi < len(periods) - 1 else mods[gi * per:]
        params = [p for _, m in chunk for p in m.parameters(recurse=False) if p.requires_grad]
        if params:
            groups.append({"params": params, "period": int(C), "name": f"f{gi + 1}",
                           "modules": [n for n, _ in chunk],
                           "n_params": int(sum(p.numel() for p in params))})
    return groups


def cms_step_gate(step, groups):
    """Zero the grads of every group whose turn it is not (NL eq. 71). Call AFTER backward, BEFORE step.

    Returns the number of parameters actually written this step, so a trainer can log the real update cost
    (`sum(n_l / C_l)` on average) instead of assuming it.
    """
    written = 0
    for g in groups:
        if int(step) % int(g["period"]):
            for p in g["params"]:
                p.grad = None
        else:
            written += int(g.get("n_params", sum(p.numel() for p in g["params"])))
    return written


def cms_expected_cost(groups):
    """The predicted average parameters-written-per-step, `sum(n_l / C_l)` — compare against the measured
    value from `cms_step_gate` to catch a mis-wired schedule."""
    return float(sum(g.get("n_params", 0) / g["period"] for g in groups))


# ══════════════════════════════════════════════════════════════════ 2. SWA (Stochastic Weight Averaging)
def swa_average_model(model):
    """Create an AveragedModel wrapper for SWA (~12 repos). After warmup, call `.update_parameters(model)`
    at the end of each epoch; the wrapper stores the running average of the weights. Uses torch's official
    `AveragedModel` (equal-weight running mean). Returns the wrapped averaged model.

        swa_model = swa_average_model(model)
        for epoch in range(swa_start, n_epochs):
            train_one_epoch(...)
            swa_model.update_parameters(model)
        update_bn(loader, swa_model)     # recompute BN running stats on the averaged weights
    """
    from torch.optim.swa_utils import AveragedModel
    return AveragedModel(model)


def swa_update_bn(loader, swa_model, device=None, forward_fn=None):
    """Recompute BatchNorm running statistics for an SWA-averaged model by a single forward pass over the
    data (the averaged weights have never "seen" data through BN, so their running stats are stale).

    `forward_fn(swa_model, batch)` lets callers adapt to arbitrary batch structures; by default it assumes
    `batch` is the input tensor (or `(input, ...)`) and calls `swa_model(input)`. No-op if the model has no BN.
    """
    import torch
    # detect BN layers; if none, nothing to do
    momenta = {}
    for m in swa_model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.reset_running_stats()
            momenta[m] = m.momentum
    if not momenta:
        return swa_model
    was_training = swa_model.training
    swa_model.train()
    for m in momenta:
        m.momentum = None   # cumulative moving average
    with torch.no_grad():
        for batch in loader:
            if forward_fn is not None:
                forward_fn(swa_model, batch)
            else:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if device is not None:
                    x = x.to(device)
                swa_model(x)
    for m, mom in momenta.items():
        m.momentum = mom
    swa_model.train(was_training)
    return swa_model


# ══════════════════════════════════════════════════════════════════ 3. Mixup / CutMix
def mixup_data(x, y, alpha: float = 0.2, seed=None):
    """Mixup batch augmentation (~25 repos). Blends samples: `x' = lam*x + (1-lam)*x[perm]` with
    `lam ~ Beta(alpha, alpha)`. Returns `(mixed_x, y_a, y_b, lam)` for use with `mixup_criterion`.

    alpha<=0 disables mixing (lam=1). lam is always in [0, 1].
    """
    import torch, numpy as np
    if alpha and alpha > 0:
        rng = np.random.RandomState(seed) if seed is not None else np.random
        lam = float(rng.beta(alpha, alpha))
    else:
        lam = 1.0
    lam = max(0.0, min(1.0, lam))
    B = x.size(0)
    perm = torch.randperm(B, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[perm]
    return mixed_x, y, y[perm], lam


def cutmix_data(x, y, alpha: float = 1.0, seed=None):
    """CutMix batch augmentation (~25 repos). Pastes a random rectangular patch from `x[perm]` onto `x`;
    `lam` is corrected to the true pixel-area ratio so the label mix matches the image mix. Expects an
    image tensor `x` of shape (B, C, H, W). Returns `(mixed_x, y_a, y_b, lam)`.
    """
    import torch, numpy as np
    rng = np.random.RandomState(seed) if seed is not None else np.random
    lam = float(rng.beta(alpha, alpha)) if (alpha and alpha > 0) else 1.0
    B, C, H, W = x.shape
    perm = torch.randperm(B, device=x.device)
    # random box whose area ≈ (1-lam) of the image
    r = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * r), int(W * r)
    cy, cx = int(rng.randint(H)), int(rng.randint(W))
    y1, y2 = max(0, cy - cut_h // 2), min(H, cy + cut_h // 2)
    x1, x2 = max(0, cx - cut_w // 2), min(W, cx + cut_w // 2)
    mixed_x = x.clone()
    mixed_x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    # correct lam to actual pasted area
    lam = 1.0 - ((y2 - y1) * (x2 - x1) / float(H * W))
    lam = max(0.0, min(1.0, lam))
    return mixed_x, y, y[perm], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """The mixed-loss helper for mixup/cutmix: `lam*loss(pred, y_a) + (1-lam)*loss(pred, y_b)`.
    Works with any per-sample-reducing criterion (CrossEntropyLoss, BCEWithLogits, focal, ...).
    """
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


# ══════════════════════════════════════════════════════════════════ 4. Label smoothing
def smooth_one_hot(targets, num_classes: int, smoothing: float = 0.1):
    """Build smoothed one-hot targets: true class gets `1-smoothing`, the remaining mass `smoothing` is
    spread uniformly over the other `num_classes-1` classes. Each row sums to 1. (~24 repos.)
    """
    import torch
    smoothing = float(smoothing)
    with torch.no_grad():
        t = torch.full((targets.size(0), num_classes), smoothing / max(1, num_classes - 1),
                       device=targets.device, dtype=torch.float32)
        t.scatter_(1, targets.long().unsqueeze(1), 1.0 - smoothing)
    return t


def label_smoothing_cross_entropy(logits, targets, smoothing: float = 0.1, weight=None):
    """Label-smoothed cross-entropy (~24 repos). Softens hard targets to regularize confidence and improve
    calibration/generalization. `targets` are integer class indices; `smoothing`=0 recovers plain CE.
    """
    import torch
    import torch.nn.functional as F
    n = logits.size(-1)
    logp = F.log_softmax(logits, dim=-1)
    if weight is not None:
        logp = logp * weight.to(logp.device).unsqueeze(0)
    nll = -logp.gather(1, targets.long().unsqueeze(1)).squeeze(1)     # true-class term
    smooth = -logp.mean(dim=-1)                                       # uniform term
    loss = (1.0 - smoothing) * nll + smoothing * smooth
    return loss.mean()


# ══════════════════════════════════════════════════════════════════ 5. Focal loss
def binary_focal_loss(logits, targets, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
    """Binary focal loss (~23 repos). Down-weights easy examples via `(1-p_t)^gamma`, focusing training on
    hard/rare positives — the standard fix for heavy class imbalance in detection/segmentation.
    `logits` and `targets` are the same shape; targets are float in {0,1} (or soft).
    """
    import torch
    import torch.nn.functional as F
    targets = targets.float()
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)          # prob of the true class
    focal = ((1.0 - p_t) ** gamma) * bce
    if alpha is not None and alpha >= 0:
        a_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        focal = a_t * focal
    if reduction == "mean":
        return focal.mean()
    if reduction == "sum":
        return focal.sum()
    return focal


class FocalLoss:
    """Multiclass focal loss (~23 repos) as a callable module. `FL = -alpha*(1-p_t)^gamma * log(p_t)` over
    softmax probabilities. `logits` shape (B, C), integer `targets` shape (B,). Down-weights easy classes.
    """
    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = "mean"):
        self.alpha = alpha        # optional per-class weight tensor / scalar
        self.gamma = float(gamma)
        self.reduction = reduction

    def __call__(self, logits, targets):
        import torch
        import torch.nn.functional as F
        logp = F.log_softmax(logits, dim=-1)
        logp_t = logp.gather(1, targets.long().unsqueeze(1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            if torch.is_tensor(self.alpha):
                a_t = self.alpha.to(logits.device)[targets.long()]
            else:
                a_t = float(self.alpha)
            loss = a_t * loss
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# ══════════════════════════════════════════════════════════════════ 6. SAM (Sharpness-Aware Minimization)
def _make_sam(base_optimizer_cls, *, rho=0.05, adaptive=False, **base_kwargs):
    # local import so torch is only required when SAM is actually used
    import torch

    class SAM(torch.optim.Optimizer):
        """Sharpness-Aware Minimization optimizer wrapper (davda54/sam style). Seeks flat minima by
        perturbing weights to the local worst case (`first_step`) then taking the real step from there
        (`second_step`). Wraps any base optimizer.

        Usage (two forward/backward passes per batch):
            optimizer = make_sam(torch.optim.SGD, lr=0.01, momentum=0.9, rho=0.05)
            loss_fn(model(x), y).backward()
            optimizer.first_step(zero_grad=True)     # climb to the sharp point
            loss_fn(model(x), y).backward()
            optimizer.second_step(zero_grad=True)    # descend from there
        """
        def __init__(self, params, base_optimizer_cls, rho=0.05, adaptive=False, **kwargs):
            assert rho >= 0, "rho (neighborhood size) must be non-negative"
            defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
            super().__init__(params, defaults)
            self.base_optimizer = base_optimizer_cls(self.param_groups, **kwargs)
            self.param_groups = self.base_optimizer.param_groups
            self.defaults.update(self.base_optimizer.defaults)

        @torch.no_grad()
        def first_step(self, zero_grad=False):
            grad_norm = self._grad_norm()
            for group in self.param_groups:
                scale = group["rho"] / (grad_norm + 1e-12)
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    self.state[p]["e_w"] = e_w = (
                        (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                    )
                    p.add_(e_w)          # climb to the local maximum "w + e(w)"
            if zero_grad:
                self.zero_grad()

        @torch.no_grad()
        def second_step(self, zero_grad=False):
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None or "e_w" not in self.state[p]:
                        continue
                    p.sub_(self.state[p]["e_w"])     # back to the original weights
            self.base_optimizer.step()               # real update from the sharp gradient
            if zero_grad:
                self.zero_grad()

        @torch.no_grad()
        def step(self, closure=None):
            assert closure is not None, "SAM requires a closure that re-evaluates the loss"
            closure = torch.enable_grad()(closure)
            self.first_step(zero_grad=True)
            closure()
            self.second_step()

        def _grad_norm(self):
            shared_device = self.param_groups[0]["params"][0].device
            norms = []
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    w = (torch.abs(p) if group["adaptive"] else 1.0)
                    norms.append((w * p.grad).norm(p=2).to(shared_device))
            if not norms:
                return torch.tensor(0.0, device=shared_device)
            return torch.norm(torch.stack(norms), p=2)

    return SAM, torch


def make_sam(base_optimizer_cls, params, rho=0.05, adaptive=False, **base_kwargs):
    """Factory returning a ready SAM optimizer wrapping `base_optimizer_cls` (e.g. torch.optim.SGD/Adam).
    See the SAM docstring for the first_step/second_step training loop.
    """
    SAM, _ = _make_sam(base_optimizer_cls, rho=rho, adaptive=adaptive, **base_kwargs)
    return SAM(params, base_optimizer_cls, rho=rho, adaptive=adaptive, **base_kwargs)


# ══════════════════════════════════════════════════════════════════ 7. ArcFace / sub-center ArcFace
def build_arcface(in_features: int, out_features: int, s: float = 30.0, m: float = 0.50, k: int = 1,
                  easy_margin: bool = False):
    """Additive-Angular-Margin (ArcFace) head + loss for metric learning (embeddings → identities/species).
    Set `k>1` for SUB-CENTER ArcFace (k centers per class → robust to noisy labels). Returns an nn.Module
    whose forward(embeddings, labels)->logits are fed to a plain cross-entropy.

        head = build_arcface(emb_dim, n_classes, s=30, m=0.5, k=3)
        logits = head(F.normalize(embeddings), labels)
        loss = F.cross_entropy(logits, labels)
    """
    import torch
    from torch import nn
    import torch.nn.functional as F

    class ArcFace(nn.Module):
        def __init__(s_, in_features, out_features, scale, margin, k, easy_margin):
            super().__init__()
            s_.out_features = out_features
            s_.k = int(k)
            s_.scale = float(scale)
            s_.margin = float(margin)
            s_.easy_margin = easy_margin
            # weight = class centers (k sub-centers per class), L2-normalized at use time
            s_.weight = nn.Parameter(torch.empty(out_features * s_.k, in_features))
            nn.init.xavier_uniform_(s_.weight)
            s_.cos_m = math.cos(margin)
            s_.sin_m = math.sin(margin)
            s_.th = math.cos(math.pi - margin)          # threshold for monotonicity
            s_.mm = math.sin(math.pi - margin) * margin

        def forward(s_, embeddings, labels=None):
            # cosine between L2-normalized embeddings and L2-normalized centers
            cos = F.linear(F.normalize(embeddings), F.normalize(s_.weight))
            if s_.k > 1:                                # sub-center: max over the k centers of each class
                cos = cos.view(-1, s_.out_features, s_.k).max(dim=2).values
            if labels is None:
                return cos * s_.scale                  # inference: plain scaled cosine logits
            sine = torch.sqrt((1.0 - cos.pow(2)).clamp(1e-9, 1.0))
            phi = cos * s_.cos_m - sine * s_.sin_m      # cos(theta + m)
            if s_.easy_margin:
                phi = torch.where(cos > 0, phi, cos)
            else:
                phi = torch.where(cos > s_.th, phi, cos - s_.mm)
            onehot = torch.zeros_like(cos)
            onehot.scatter_(1, labels.long().view(-1, 1), 1.0)
            logits = (onehot * phi + (1.0 - onehot) * cos) * s_.scale
            return logits

    return ArcFace(in_features, out_features, s, m, k, easy_margin)


# ══════════════════════════════════════════════════════════════════ agent
class TrainTricksPack(BaseAgent):
    """Fleet agent that self-verifies the training-tricks pack on a tiny synthetic batch and reports which
    winner-standard techniques are now available to the fleet (EMA/SWA/mixup/cutmix/label-smoothing/focal/
    SAM/ArcFace). The utilities above are the reusable product; this run() is the health-check + advertisement.
    """
    name = "train-tricks"
    thread = "S"
    kind = "finding"

    def run(self, q, worker):
        checks = {}
        try:
            import torch
            from torch import nn
            spec = self.spec(q)
            dev = spec.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
            torch.manual_seed(0)
            model = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(dev)

            # EMA tracks weights
            ema = ModelEMA(model, decay=0.9, device=dev)
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.ones_like(p))
            ema.update(model)
            checks["ema"] = True

            # mixup / cutmix
            x = torch.randn(6, 8, device=dev); y = torch.randint(0, 4, (6,), device=dev)
            _, ya, yb, lam = mixup_data(x, y, alpha=0.4, seed=0)
            checks["mixup"] = 0.0 <= lam <= 1.0 and ya.shape == yb.shape == y.shape
            img = torch.randn(6, 3, 16, 16, device=dev)
            _, _, _, lam2 = cutmix_data(img, y, alpha=1.0, seed=0)
            checks["cutmix"] = 0.0 <= lam2 <= 1.0

            # losses
            logits = torch.randn(6, 4, device=dev)
            checks["label_smoothing"] = torch.isfinite(label_smoothing_cross_entropy(logits, y, 0.1))
            checks["focal"] = torch.isfinite(FocalLoss(gamma=2.0)(logits, y))
            checks["binary_focal"] = torch.isfinite(
                binary_focal_loss(torch.randn(6, device=dev), (torch.rand(6, device=dev) > 0.5).float()))

            # SAM two-step
            m2 = nn.Linear(8, 4).to(dev)
            opt = make_sam(torch.optim.SGD, m2.parameters(), lr=0.01, momentum=0.9, rho=0.05)
            m2(x).sum().backward(); opt.first_step(zero_grad=True)
            m2(x).sum().backward(); opt.second_step(zero_grad=True)
            checks["sam"] = True

            # ArcFace
            head = build_arcface(8, 4, s=30.0, m=0.5, k=3).to(dev)
            al = head(x, y)
            checks["arcface"] = tuple(al.shape) == (6, 4)

            checks = {k: bool(v) for k, v in checks.items()}
        except Exception as e:  # noqa: BLE001 — never crash the fleet; report the failure
            msg = f"[{worker}] train-tricks: pack self-check FAILED ({str(e)[:180]})"
            return self.escalate(worker, "researcher", msg)

        n_ok = sum(checks.values()); n = len(checks)
        tricks = "EMA, SWA, mixup, cutmix, label-smoothing, focal(bin+multi), SAM, sub-center ArcFace"
        msg = (f"train-tricks: {n_ok}/{n} winner training-tricks verified on-device "
               f"({checks}); reusable utils: {tricks}")
        self.log(msg, kind="finding",
                 recommendation="import from fleet_agents.train_tricks_pack in any trainer — "
                                "EMA/SWA for eval-robust weights, mixup/cutmix+focal+label-smoothing for the loss, "
                                "SAM for flat minima, ArcFace for metric learning")
        return self.done({"checks": checks, "device": str(dev), "tricks": tricks}, msg)


_AGENT = TrainTricksPack()


def run(q, worker):
    return _AGENT.run(q, worker)
