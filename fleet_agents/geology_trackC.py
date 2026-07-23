"""geology_trackC — REUSABLE neural sequence track (Track C) for wellbore geosteering, FP8-attempted.

A GRU over the horizontal well's (GR + geometry + typewell-signature) sequence predicts the residual
dtvt = TVT - tvt_ps per MD step.

FP8 backend: Hugging Face Hub `kernels` (kernels-community/finegrained-fp8, Triton, per user request —
NOT torchao). MEASURED on sm_120 (RTX 5090, 2026-07-23): the kernel loads and its forward is numerically
correct (~3.8% rel error, expected fp8 quant noise), but (a) has NO registered autograd backward (raw
Triton custom op, forward-only) and (b) is ~10x SLOWER than a plain bf16 matmul even at 512x512x512
after warmup (worse at our GRU's 256-dim). A `HFFP8Linear` wrapper below supplies a straight-through-ish
bf16 backward for correctness, but a cheap timing PREFLIGHT (`_hf_fp8_worth_it`, one warmed-up matmul at
the model's actual hidden size) rejects fp8 before spending a full epoch, because it's already known to
be slower — this is the "measure the full step, don't assume" rule applied at the cheapest possible
point. Net effect: HF-kernels fp8 is NOT usable for training here either (same conclusion as torchao,
via an independent kernel) → falls back to `fp8_fallback` (bf16). See memory `rogii_trackC_fp8_seq.md`
and the updated `fp8_sm120_ecosystem_verdict.md`.

Contract mirrors Track A/B for blend integration:
  trackC_oof(train_dir, test_dir, out_oof, out_test, cfg_params) -> writes standardized ledgers,
  returns (cv_rmse, precision_used).
"""
from __future__ import annotations
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- feature engineering (numpy)
TW_OFFSETS = (-20.0, -10.0, 0.0, 10.0, 20.0)  # typewell GR sampled at tvt_ps + these offsets


def _well_seq(hw, tw, training):
    """Return (feats [n,F], target dtvt [n], eval_mask [n], sup_mask [n], ids[list], tvt_ps, well_len)."""
    if "TVT_input" not in hw.columns:
        return None
    pred = hw["TVT_input"].isna().to_numpy()
    if training:
        pred = pred & hw["TVT"].notna().to_numpy()
    if pred.sum() == 0:
        return None
    ps = int(np.argmax(hw["TVT_input"].isna().to_numpy()))
    j = max(ps - 1, 0)
    tvt_ps = float(hw["TVT"].iloc[j]) if training else float(hw["TVT_input"].iloc[j])
    md = hw["MD"].to_numpy(float); X = hw["X"].to_numpy(float); Y = hw["Y"].to_numpy(float); Z = hw["Z"].to_numpy(float)
    md_ps, x_ps, y_ps, z_ps = md[j], X[j], Y[j], Z[j]
    gr = hw["GR"].to_numpy(float)
    gr_ff = pd.Series(gr).ffill().bfill().to_numpy()
    grad = np.gradient(gr_ff, md)
    dmd = md - md_ps; dz = Z - z_ps
    horiz = np.sqrt((X - x_ps) ** 2 + (Y - y_ps) ** 2)
    incl = np.gradient(Z, md)
    # typewell GR signature sampled around tvt_ps
    tw_s = tw.dropna(subset=["TVT", "GR"]).sort_values("TVT")
    tvt_grid = tw_s["TVT"].to_numpy(float); gr_grid = tw_s["GR"].to_numpy(float)
    sig = [np.interp(tvt_ps + o, tvt_grid, gr_grid) * np.ones(len(md)) for o in TW_OFFSETS]
    feats = np.column_stack([
        gr_ff, np.isnan(gr).astype(float), grad,
        dmd / 1000.0, dz / 100.0, horiz / 1000.0, incl, (Z + 9000.0) / 1000.0,
        *sig,
    ]).astype(np.float32)
    # known dtvt as teacher-forcing input channel (0 on eval)
    known = hw["TVT_input"].to_numpy(float)
    tf_dtvt = np.where(np.isnan(known), 0.0, known - tvt_ps).astype(np.float32)
    feats = np.column_stack([feats, tf_dtvt, (~np.isnan(known)).astype(np.float32)])
    if training:
        target = (hw["TVT"].to_numpy(float) - tvt_ps).astype(np.float32)
        sup = np.isfinite(target)
    else:
        target = np.zeros(len(md), np.float32); sup = np.zeros(len(md), bool)
    ids = [f"{{}}_{i}" for i in range(len(md))]  # filled by caller with well id
    return feats, target, pred, sup, ids, tvt_ps, len(md)


# --------------------------------------------------------------------------- torch model + training
def _build(params):
    """GRU over the full well (data-derived: 773 effective samples -> small+regularized RNN beats a
    Transformer here). Wide Linears (in/out %16) get fp8; the out=1 projection stays bf16."""
    import torch.nn as nn

    class TrackCNet(nn.Module):
        def __init__(self, F, d=256, L=2, dropout=0.25):
            super().__init__()
            d16 = (d + 15) // 16 * 16  # keep proj dims %16 for fp8 eligibility
            self.inp = nn.Linear(F, d16)
            self.gru = nn.GRU(d16, d, L, batch_first=True, dropout=(dropout if L > 1 else 0.0),
                              bidirectional=False)
            self.mix = nn.Sequential(nn.Linear(d, d16), nn.GELU(), nn.Dropout(dropout))
            self.out = nn.Linear(d16, 1)  # linear head (target |dtvt| p99≈75, no saturating activation)

        def forward(self, x, lengths=None):
            import torch
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            h = self.inp(x)
            if lengths is not None:
                packed = pack_padded_sequence(h, lengths.cpu(), batch_first=True, enforce_sorted=False)
                y, _ = self.gru(packed)
                y, _ = pad_packed_sequence(y, batch_first=True, total_length=x.shape[1])
            else:
                y, _ = self.gru(h)
            return self.out(self.mix(y)).squeeze(-1)

    return TrackCNet


def _make_optimizer(model, name, wd, lr=3e-4):
    """Reusable optimizer factory: "adamw" (default) or "muon" (Newton-Schulz orthogonalized updates on
    2D matrices, per the fleet's muon_optimizer.py + hardware_tune's recommendation on bf16 boxes). Muon
    convention (kept consistent with muon_optimizer.py): only >=2D weight matrices get orthogonalized
    updates; biases, norm params, and the small output head stay on AdamW (K3's "per-head" split doesn't
    literally apply to a GRU — there are no attention heads — but keeping the head/embedding-analog params
    off Muon is the same spirit). gru.weight_hh_l* is a fused 3-gate matrix (reset/update/candidate); a
    truer per-gate block-split Muon is a possible follow-up, not implemented here — this is the honest
    first step (whole-matrix Muon on eligible 2D weights), CV-gated against plain AdamW."""
    import torch
    if name != "muon":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=True)
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from muon_optimizer import Muon
    except Exception as e:  # noqa: BLE001
        print(f"  ! muon_optimizer unavailable ({e}) -> falling back to AdamW")
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=True)
    muon_params, adamw_params = [], []
    head_names = ("out.",)  # keep the small output head on AdamW (K3 "per-head" analog: don't fuse it in)
    for n, p in model.named_parameters():
        if p.ndim >= 2 and not n.startswith(head_names):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    opt_muon = Muon(muon_params, lr=0.02, weight_decay=wd)
    opt_adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=wd)

    class _Combined:
        """Thin wrapper so the training loop's opt.zero_grad()/opt.step() calls hit both optimizers."""
        def zero_grad(self, *a, **kw):
            opt_muon.zero_grad(*a, **kw); opt_adamw.zero_grad(*a, **kw)

        def step(self, *a, **kw):
            opt_muon.step(*a, **kw); opt_adamw.step(*a, **kw)

    return _Combined()


# --------------------------------------------------------------------------- HF-kernels FP8 (not torchao)
_HF_FP8_KERNEL = None  # lazy-loaded singleton: kernels.get_kernel("kernels-community/finegrained-fp8")


def _hf_fp8_kernel():
    global _HF_FP8_KERNEL
    if _HF_FP8_KERNEL is None:
        from kernels import get_kernel
        _HF_FP8_KERNEL = get_kernel("kernels-community/finegrained-fp8", version=4)
    return _HF_FP8_KERNEL


class _HFFP8MatmulFn:
    """torch.autograd.Function: fp8 forward via HF kernels-community/finegrained-fp8 (tensor-wide
    dynamic quant of the weight, matmul_2d), bf16 backward (the kernel has no registered autograd
    formula — straight-through: gradients computed as if the matmul were plain bf16)."""

    @staticmethod
    def build():
        import torch

        class Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, weight):
                k = _hf_fp8_kernel()
                wq, ws = k.fp8_act_quant(weight.reshape(1, -1), block_size=weight.numel())
                wq = wq.reshape(weight.shape); ws = ws.reshape(1)
                ctx.save_for_backward(x, weight)
                return k.matmul_2d(x, wq, ws, None, output_dtype=x.dtype)

            @staticmethod
            def backward(ctx, grad_out):
                x, weight = ctx.saved_tensors
                grad_x = grad_out @ weight if ctx.needs_input_grad[0] else None
                grad_w = None
                if ctx.needs_input_grad[1]:
                    flat_go = grad_out.reshape(-1, grad_out.shape[-1])
                    flat_x = x.reshape(-1, x.shape[-1])
                    grad_w = flat_go.t() @ flat_x
                return grad_x, grad_w

        return Fn


def _hf_fp8_worth_it(dim, dev, reps=50):
    """Cheap PREFLIGHT: time one warmed-up HF-fp8 matmul_2d vs plain bf16 matmul at the model's actual
    hidden size. MEASURED on sm_120: fp8 is ~10x SLOWER even at 512-dim (worse at 256) — this check
    rejects fp8 in ~0.1s instead of wasting a full CV-gate epoch on a backend already known to lose."""
    import torch
    import time
    k = _hf_fp8_kernel()
    x = torch.randn(64, dim, device=dev, dtype=torch.bfloat16)
    w = torch.randn(dim, dim, device=dev, dtype=torch.bfloat16)
    wq, ws = k.fp8_act_quant(w.reshape(1, -1), block_size=w.numel())
    wq = wq.reshape(dim, dim); ws = ws.reshape(1)
    for _ in range(10):
        k.matmul_2d(x, wq, ws, None, output_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        k.matmul_2d(x, wq, ws, None, output_dtype=torch.bfloat16)
    torch.cuda.synchronize(); t_fp8 = (time.time() - t0) / reps
    for _ in range(10):
        x @ w.T
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        x @ w.T
    torch.cuda.synchronize(); t_bf16 = (time.time() - t0) / reps
    return t_bf16 < t_fp8, t_fp8, t_bf16


def _fp8_convert(model):
    """Replace eligible nn.Linear layers with an HF-kernels fp8-forward/bf16-backward wrapper.
    Kept for completeness/testability; the preflight above should already have rejected fp8 by the
    time this would be called on real hidden sizes."""
    import torch.nn as nn

    Fn = _HFFP8MatmulFn.build()

    class HFFP8Linear(nn.Module):
        def __init__(self, lin):
            super().__init__()
            self.weight = lin.weight
            self.bias = lin.bias

        def forward(self, x):
            shp = x.shape
            out = Fn.apply(x.reshape(-1, shp[-1]), self.weight).reshape(*shp[:-1], self.weight.shape[0])
            return out + self.bias if self.bias is not None else out

    def eligible(mod):
        return isinstance(mod, nn.Linear) and mod.in_features % 16 == 0 and mod.out_features % 16 == 0

    for name, child in list(model.named_children()):
        if eligible(child):
            setattr(model, name, HFFP8Linear(child))
        else:
            _fp8_convert(child)
    return model


def _list_pairs(data_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(data_dir, "*__horizontal_well.csv"))):
        wid = os.path.basename(f).split("__")[0]
        tp = os.path.join(data_dir, f"{wid}__typewell.csv")
        if os.path.exists(tp):
            out.append((wid, f, tp))
    return out


def _load_all(data_dir, training, limit=None):
    pairs = _list_pairs(data_dir)
    if limit:
        pairs = pairs[:limit]
    wells = []
    for wid, hp, tp in pairs:
        r = _well_seq(pd.read_csv(hp), pd.read_csv(tp), training)
        if r is None:
            continue
        feats, target, pred, sup, _ids, tvt_ps, n = r
        wells.append(dict(wid=wid, feats=feats, target=target, pred=pred, sup=sup, tvt_ps=tvt_ps, n=n))
    return wells


def trackC_oof(train_dir, test_dir, out_oof, out_test, params, log=print):
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = int(params.get("epochs", 25)); bs = int(params.get("batch_size", 8))
    folds = int(params.get("folds", 5)); limit = params.get("limit")
    precision = params.get("precision", "fp8"); fallback = params.get("fp8_fallback", "bf16")
    d = int(params.get("hidden", 256)); L = int(params.get("layers", 2))
    dropout = float(params.get("dropout", 0.25)); wd = float(params.get("weight_decay", 0.01))
    tv_lambda = float(params.get("tv_lambda", 0.05)); do_tstd = bool(params.get("target_std", True))
    optimizer_name = params.get("optimizer", "adamw")  # "adamw" | "muon" (hardware_tune recommends muon on this box)

    train = _load_all(train_dir, True, limit)
    if not train:
        raise RuntimeError("no training wells")
    F = train[0]["feats"].shape[1]
    allf = np.concatenate([w["feats"] for w in train], 0)
    mu = allf.mean(0); sd = allf.std(0) + 1e-6
    for w in train:
        w["feats"] = (w["feats"] - mu) / sd
    Net = _build(params)

    def make_batches(wells, shuffle):
        order = list(range(len(wells)))
        if shuffle:
            np.random.default_rng(0).shuffle(order)
        for i in range(0, len(order), bs):
            yield order[i:i + bs]

    def collate(wells, batch):
        """Full-well packed batch: pad to max len; keep sup/eval masks and lengths. Teacher-forced known
        section is already a feature channel; loss is masked to the supervised region (eval weighted)."""
        lens = [wells[wi]["n"] for wi in batch]
        T = (max(lens) + 15) // 16 * 16   # pad time to multiple of 16 so B*T is fp8 scaled_mm-eligible
        Xb = np.zeros((len(batch), T, F), np.float32)
        Yb = np.zeros((len(batch), T), np.float32)
        Mb = np.zeros((len(batch), T), np.float32)   # loss mask (eval region weighted 3x)
        Eb = np.zeros((len(batch), T), np.float32)   # eval-only mask for TV penalty
        for k, wi in enumerate(batch):
            w = wells[wi]; ln = w["n"]
            Xb[k, :ln] = w["feats"]; Yb[k, :ln] = w["target"]
            m = w["sup"].astype(np.float32) * np.where(w["pred"], 3.0, 1.0)
            Mb[k, :ln] = m; Eb[k, :ln] = w["pred"].astype(np.float32)
        return Xb, Yb, Mb, Eb, np.array(lens)

    def train_eval(tr_wells, va_wells, use_fp8, tstd):
        torch.manual_seed(0)
        model = Net(F, d=d, L=L, dropout=dropout).to(dev)
        if use_fp8:
            model = _fp8_convert(model)
        model = torch.compile(model)
        opt = _make_optimizer(model, optimizer_name, wd)
        for ep in range(epochs):
            model.train()
            for batch in make_batches(tr_wells, True):
                Xb, Yb, Mb, Eb, lens = collate(tr_wells, batch)
                Xt = torch.from_numpy(Xb).to(dev); Yt = torch.from_numpy(Yb / tstd).to(dev)
                Mt = torch.from_numpy(Mb).to(dev); Et = torch.from_numpy(Eb).to(dev)
                lt = torch.from_numpy(lens).to(dev)
                opt.zero_grad()
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    pred = model(Xt, lengths=lt)
                    mse = (((pred - Yt) ** 2) * Mt).sum() / (Mt.sum() + 1e-6)
                    # total-variation smoothness prior on the eval region (target is near-piecewise-const)
                    dpred = (pred[:, 1:] - pred[:, :-1]).abs() * Et[:, 1:]
                    tv = dpred.sum() / (Et[:, 1:].sum() + 1e-6)
                    loss = mse + tv_lambda * tv
                loss.backward(); opt.step()
        model.eval()
        preds = {}
        with torch.no_grad():
            for batch in make_batches(va_wells, False):
                Xb, Yb, Mb, Eb, lens = collate(va_wells, batch)
                Xt = torch.from_numpy(Xb).to(dev); lt = torch.from_numpy(lens).to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                    out = (model(Xt, lengths=lt).float().cpu().numpy()) * tstd
                for k, wi in enumerate(batch):
                    w = va_wells[wi]
                    preds[wi] = {i: [out[k, i]] for i in range(w["n"]) if w["pred"][i]}
        return preds, model

    groups = np.array([w["wid"] for w in train])
    gkf = GroupKFold(min(folds, len(set(groups))))
    idx = np.arange(len(train))

    # ---- CV-gate: fp8 vs bf16 on fold-0 small subset (measured, not assumed) ----
    use_fp8 = precision in ("fp8", "hf_fp8")
    gate_note = "fp8 requested"
    if use_fp8 and dev == "cuda":
        try:
            bf16_faster, t_fp8, t_bf16 = _hf_fp8_worth_it(d, dev)
        except Exception as e:  # noqa: BLE001
            bf16_faster, t_fp8, t_bf16 = True, float("nan"), float("nan")
            log(f"[C] fp8 preflight load/run failed ({type(e).__name__}: {str(e)[:150]}) -> {fallback}")
        if bf16_faster:
            use_fp8 = False
            gate_note = (f"fp8 preflight: HF-kernels matmul_2d {t_fp8*1e6:.0f}us vs bf16 {t_bf16*1e6:.0f}us "
                         f"at hidden={d} -> bf16 faster, skipping fp8 CV-gate to save an epoch; using {fallback}")
            log(f"[C] {gate_note}")
    if use_fp8:
        tr0, va0 = next(iter(gkf.split(idx, groups=groups)))
        sub_tr = [train[i] for i in tr0[:40]]; sub_va = [train[i] for i in va0[:15]]
        def _tstd(wells):
            v = np.concatenate([w["target"][w["pred"] & w["sup"]] for w in wells if (w["pred"] & w["sup"]).any()])
            return float(v.std()) + 1e-6 if do_tstd and len(v) else 1.0
        ts0 = _tstd(sub_tr)
        def quick_cv(fp8):
            pr, _ = train_eval(sub_tr, sub_va, fp8, ts0)
            e = []
            for wi, w in enumerate(sub_va):
                for i, vs in pr.get(wi, {}).items():
                    if w["pred"][i] and w["sup"][i]:
                        e.append(np.mean(vs) - w["target"][i])
            return float(np.sqrt(np.mean(np.square(e)))) if e else float("inf")
        try:
            cv_fp8 = quick_cv(True); cv_bf16 = quick_cv(False)
            log(f"[C] fp8-gate: fp8 CV {cv_fp8:.3f} vs {fallback} CV {cv_bf16:.3f}")
            if cv_fp8 > cv_bf16 * 1.02:   # >2% worse -> fall back
                use_fp8 = False; gate_note = f"fp8 degraded ({cv_fp8:.2f}>{cv_bf16:.2f}); fell back to {fallback}"
            else:
                gate_note = f"fp8 kept (CV {cv_fp8:.2f} ~ bf16 {cv_bf16:.2f})"
        except Exception as e:  # noqa: BLE001
            # sm_120 fp8 scaled_mm needs token-dim (B*T) %16 — dynamic for variable-length GRU seqs.
            # GRU has no measured fp8 speedup anyway (matmuls too small) -> honest fallback to bf16.
            use_fp8 = False
            gate_note = f"fp8 unavailable for variable-len GRU on sm_120 ({type(e).__name__}); using {fallback} (no fp8 speedup for GRU regardless)"
            log(f"[C] fp8-gate: {gate_note}")
    log(f"[C] precision decision: {'fp8' if use_fp8 else fallback} — {gate_note}")

    # ---- full GroupKFold OOF ----
    def tstd_of(wells):
        v = [w["target"][w["pred"] & w["sup"]] for w in wells if (w["pred"] & w["sup"]).any()]
        return (float(np.concatenate(v).std()) + 1e-6) if (do_tstd and v) else 1.0

    oof_rows = []
    for tr_i, va_i in gkf.split(idx, groups=groups):
        tr_wells = [train[i] for i in tr_i]; va_wells = [train[i] for i in va_i]
        pr, _ = train_eval(tr_wells, va_wells, use_fp8, tstd_of(tr_wells))
        for wi, w in enumerate(va_wells):
            for i, vs in pr.get(wi, {}).items():
                if w["pred"][i]:
                    oof_rows.append((f"{w['wid']}_{i}", w["wid"], float(np.mean(vs)),
                                     float(w["target"][i]) if w["sup"][i] else np.nan))
    oof = pd.DataFrame(oof_rows, columns=["id", "well", "dtvt_pred", "dtvt_true"])
    Path(out_oof).parent.mkdir(parents=True, exist_ok=True)
    oof.to_csv(out_oof, index=False)
    valid = oof.dropna(subset=["dtvt_true"])
    cv = float(np.sqrt(np.mean((valid.dtvt_pred - valid.dtvt_true) ** 2)))
    log(f"[C] golden CV RMSE {cv:.4f} ({'fp8' if use_fp8 else fallback})")

    # ---- retrain on ALL train, predict test ----
    test = _load_all(test_dir, False, limit)
    for w in test:
        w["feats"] = (w["feats"] - mu) / sd
    pr, _ = train_eval(train, test, use_fp8, tstd_of(train))
    trows = []
    for wi, w in enumerate(test):
        for i, vs in pr.get(wi, {}).items():
            if w["pred"][i]:
                trows.append((f"{w['wid']}_{i}", float(np.mean(vs)), w["tvt_ps"]))
    tdf = pd.DataFrame(trows, columns=["id", "dtvt_pred", "tvt_ps"])
    tdf.to_csv(out_test, index=False)
    return cv, ("fp8" if use_fp8 else fallback)
