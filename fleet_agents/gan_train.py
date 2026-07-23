"""gan-train — REUSABLE (any competition) adversarial image trainer on the GPU (torch/CUDA). One general
Generator/Discriminator + training loop that covers the recurring GAN use-cases in Kaggle vision comps:

  • mode "translate"  — map a SOURCE image domain onto a TARGET appearance (domain adaptation / style transfer),
                        with an optional STRUCTURE-PRESERVATION loss so the content (cells/objects) survives.
  • mode "augment"    — learn the target distribution and SAMPLE new variants for data augmentation / synthetic
                        training data (generator conditioned on the source + noise).

Comp-agnostic: takes plain numpy volumes/stacks via spec (or .npy paths); no biohub paths baked in. The
`domain-match` agent delegates its learned mapper here (single GAN implementation, reused). A BaseAgent with
its own data-wise test. Everything runs on CUDA when available (per the always-GPU rule); numpy only for I/O.
"""
from __future__ import annotations
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent


def _stack(vol, eps=1e-6):
    import numpy as np
    v = np.nan_to_num(np.asarray(vol, float), nan=0.0, posinf=0.0, neginf=0.0)
    if v.ndim == 2:
        v = v[None]
    return (v - v.mean()) / (v.std() + max(float(eps), 1e-12))


def _build(device, gen_ch=32, noise_dim=0):
    """Residual conv Generator (starts near identity → preserves layout) + patch Discriminator. `noise_dim`>0
    concatenates a per-pixel noise channel (augment mode) so the generator can SAMPLE variants."""
    import torch
    from torch import nn

    class Gen(nn.Module):
        def __init__(s):
            super().__init__()
            cin = 1 + (1 if noise_dim else 0)
            s.net = nn.Sequential(nn.Conv2d(cin, gen_ch, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(gen_ch, gen_ch, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(gen_ch, 1, 3, padding=1))

        def forward(s, x, z=None):
            inp = x if z is None else torch.cat([x, z], 1)
            return x + s.net(inp)

    class Disc(nn.Module):
        def __init__(s):
            super().__init__()
            s.net = nn.Sequential(nn.Conv2d(1, gen_ch, 4, 2, 1), nn.LeakyReLU(0.2),
                                  nn.Conv2d(gen_ch, gen_ch * 2, 4, 2, 1), nn.LeakyReLU(0.2),
                                  nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(gen_ch * 2, 1))

        def forward(s, x):
            return s.net(x)

    return Gen().to(device), Disc().to(device)


def train_gan(src_vol, tgt_vol, mode="translate", iters=400, lambda_struct=3.0, patch=32, batch=32,
              lr=2e-4, gen_ch=32, device=None, seed=0, adv_auc_fn=None, prewarp_fn=None,
              early_stop=False, eval_every=50, patience=3, struct_min=0.5, adv_patch=400):
    """Train the GAN on GPU. Returns (apply_fn, metrics): apply_fn(volume)->mapped volume (slice-wise), and
    metrics {adv_auc_before, adv_auc_after, structure_corr, honest_match, device, mode}. `prewarp_fn(src,tgt)`
    (e.g. domain_match's fixed transform) warm-starts translate so the GAN only closes the residual. The
    adv-AUC metric is INDEPENDENT (passed in; defaults to math_master) — never the discriminator itself.
    `early_stop` monitors adv-AUC every `eval_every` iters and stops after `patience` non-improving checks.
    `struct_min` is the structure-corr floor honest_match requires (default 0.5 = legacy). `adv_patch` sizes
    the patch-feature sample for scoring. `device` respects the caller; falls back to cpu when CUDA is absent.
    NaN/Inf inputs are sanitized; `patch`/`batch` are clamped to safe minimums so tiny volumes never crash."""
    import numpy as np, torch
    from torch import nn
    torch.manual_seed(seed); np.random.seed(seed)
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(dev, str) and dev.startswith("cuda") and not torch.cuda.is_available():
        dev = "cpu"                                            # graceful device fallback
    E0 = _stack(src_vol); C = _stack(tgt_vol)
    E = _stack(prewarp_fn(src_vol, tgt_vol)) if prewarp_fn is not None else E0
    # clamp patch/batch so tiny volumes or single-sample requests never crash the sampler/discriminator
    patch = int(max(8, min(patch, E.shape[1], E.shape[2], C.shape[1], C.shape[2])))
    batch = int(max(1, batch))
    noise_dim = 1 if mode == "augment" else 0
    G, D = _build(dev, gen_ch, noise_dim)
    og = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    od = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()

    def sample(stack, n):
        Z, Y, X = stack.shape; out = []
        for _ in range(n):
            z = np.random.randint(0, Z); y = np.random.randint(0, max(1, Y - patch)); x = np.random.randint(0, max(1, X - patch))
            out.append(stack[z, y:y + patch, x:x + patch])
        return torch.tensor(np.stack(out)[:, None], dtype=torch.float32, device=dev)

    def corr(a, b):
        a = a.flatten(1); b = b.flatten(1)
        a = a - a.mean(1, keepdim=True); b = b - b.mean(1, keepdim=True)
        return ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-6)).mean()

    def _current_map():                                       # slice-wise apply of the current G (for early-stop)
        with torch.no_grad():
            return np.stack([G(torch.tensor(E[k][None, None], dtype=torch.float32, device=dev),
                               torch.zeros(1, 1, *E[k].shape, device=dev) if noise_dim else None).cpu().numpy()[0, 0]
                             for k in range(E.shape[0])])

    _es_fn = None
    if early_stop:
        try:
            _es_fn = adv_auc_fn if adv_auc_fn is not None else __import__("importlib").import_module(
                "fleet_agents.math_master").adversarial_auc
            from . import domain_match as _DM
        except Exception:  # noqa: BLE001
            _es_fn = None
    best_auc, bad = 2.0, 0
    for it in range(iters):
        xe = sample(E, batch); xc = sample(C, batch)
        z = torch.randn_like(xe) if noise_dim else None
        with torch.no_grad():
            fake = G(xe, z)
        od.zero_grad()
        ld = bce(D(xc), torch.ones(batch, 1, device=dev)) + bce(D(fake), torch.zeros(batch, 1, device=dev))
        ld.backward(); od.step()
        og.zero_grad()
        gen = G(xe, z)
        lg = bce(D(gen), torch.ones(batch, 1, device=dev)) + lambda_struct * (1 - corr(gen, xe))
        lg.backward(); og.step()
        if _es_fn is not None and (it + 1) % max(1, int(eval_every)) == 0:
            try:
                cur = round(float(_es_fn(_DM.patch_feats(_current_map(), adv_patch), _DM.patch_feats(C, adv_patch))), 3)
            except Exception:  # noqa: BLE001
                cur = None
            if cur is not None:
                if cur < best_auc - 1e-3:
                    best_auc, bad = cur, 0
                else:
                    bad += 1
                    if bad >= max(1, int(patience)):
                        break

    G.eval()

    def apply_fn(vol):
        st = _stack(prewarp_fn(vol, tgt_vol)) if prewarp_fn is not None else _stack(vol)
        with torch.no_grad():
            zc = None
            return np.stack([G(torch.tensor(st[k][None, None], dtype=torch.float32, device=dev),
                               torch.zeros(1, 1, *st[k].shape, device=dev) if noise_dim else None).cpu().numpy()[0, 0]
                             for k in range(st.shape[0])])

    if adv_auc_fn is None:
        from . import math_master as MM
        adv_auc_fn = MM.adversarial_auc
    from . import domain_match as DM
    mapped = apply_fn(src_vol)
    try:
        before = round(float(adv_auc_fn(DM.patch_feats(E0, adv_patch), DM.patch_feats(C, adv_patch))), 3)
        after = round(float(adv_auc_fn(DM.patch_feats(mapped, adv_patch), DM.patch_feats(C, adv_patch))), 3)
    except Exception:  # noqa: BLE001 — scoring failed → report None instead of crashing the fleet
        before = after = None
    mapped_s = np.nan_to_num(mapped); E0_s = np.nan_to_num(E0)
    fe = (mapped_s - mapped_s.mean()).ravel(); oe = (E0_s - E0_s.mean()).ravel()
    struct = round(float((fe @ oe) / (np.linalg.norm(fe) * np.linalg.norm(oe) + 1e-9)), 3)
    matched = bool(after is not None and after <= 0.6)
    metrics = {"adv_auc_before": before, "adv_auc_after": after, "structure_corr": struct,
               "matched": matched, "honest_match": bool(matched and struct >= float(struct_min)),
               "device": dev if isinstance(dev, str) else str(dev), "mode": mode, "iters": iters}
    return apply_fn, metrics


class GanTrain(BaseAgent):
    name = "gan-train"
    thread = "S"
    kind = "verdict"

    def run(self, q, worker):
        import numpy as np
        spec = self.spec(q)
        src = spec.get("src"); tgt = spec.get("target")
        if src is None or tgt is None:
            return self.escalate(worker, "researcher", f"[{worker}] gan-train: need spec.src and spec.target (arrays or .npy paths).")
        src = np.load(src) if isinstance(src, str) else np.asarray(src, float)
        tgt = np.load(tgt) if isinstance(tgt, str) else np.asarray(tgt, float)
        prewarp = None
        if spec.get("prewarp", True):
            from . import domain_match as DM
            prewarp = lambda s, t: DM.appearance_match_search(s, t, sigmas=(1.5, 3, 5), n_patch=300, auto=False)[1] or DM.zscore_norm(s)
        try:
            _, m = train_gan(src, tgt, mode=spec.get("mode", "translate"), iters=int(spec.get("iters", 400)),
                             lambda_struct=float(spec.get("lambda_struct", 3.0)), patch=int(spec.get("patch", 32)),
                             batch=int(spec.get("batch", 32)), gen_ch=int(spec.get("gen_ch", 32)),
                             lr=float(spec.get("lr", 2e-4)), seed=int(spec.get("seed", 0)),
                             early_stop=bool(spec.get("early_stop", False)), eval_every=int(spec.get("eval_every", 50)),
                             patience=int(spec.get("patience", 3)), struct_min=float(spec.get("struct_min", 0.5)),
                             device=spec.get("device"), prewarp_fn=prewarp)
        except Exception as e:  # noqa: BLE001 — torch/GAN failure must not crash the fleet
            return self.escalate(worker, "researcher", f"[{worker}] gan-train: training failed ({str(e)[:200]}).")
        self.save_state({"gan_metrics": m})
        tag = ("✅ honest match (signal preserved)" if m["honest_match"]
               else "⚠️ reached target but destroys signal" if m["matched"] else "partial")
        msg = (f"[{worker}] **GAN-TRAIN** ({m['mode']}, {m['device']}) · adversarial image translation\n"
               f"adv-AUC {m['adv_auc_before']}→**{m['adv_auc_after']}**, structure-corr {m['structure_corr']} → {tag}")
        self.log(summary=f"gan-train ({m['mode']}): adv-AUC {m['adv_auc_before']}→{m['adv_auc_after']}, struct {m['structure_corr']}, honest={m['honest_match']}",
                 detail="reusable GPU adversarial trainer (residual Gen vs patch Disc + structure guard)",
                 kind="verdict", recommendation="honest_match ⇒ translated source usable for training; else the gap is structural (content, not style)")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"gan_metrics": m}, msg, to="leader")


_AGENT = GanTrain()


def run(q, worker):
    return _AGENT.run(q, worker)
