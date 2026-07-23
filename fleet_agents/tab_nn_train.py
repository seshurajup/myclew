"""tab-nn-train — REAL neural-tabular trainer (torch), the diversity backbone flagged by 4+ Playground
winners (RealMLP/TabM/FT-Transformer families beat or matched GBDTs and drove ensemble decorrelation). This
is a genuine model, not a stub: a residual MLP with BatchNorm + dropout, standardized inputs, OOF CV, GPU-
auto, producing OOF + test predictions scored through the CompConfig metric — exactly like tab-train but the
NN member the GBDT-only tab-train couldn't provide.

Small nets train in seconds on CPU/GPU, so it is verified on synthetic data (not stub-tested). Reads the
same CompConfig contract + reuses tab_common (CV/FE). Torch is required; if absent the agent escalates.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent
from . import comp_config as CC
from . import tab_common as TC


def _device(gpu=None):
    import torch
    if gpu is not None:
        return "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_mlp(d_in, d_out, width=256, blocks=3, dropout=0.2):
    import torch.nn as nn

    class ResBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout))

        def forward(self, x):
            return x + self.net(x)

    layers = [nn.Linear(d_in, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
    layers += [ResBlock(width) for _ in range(blocks)]
    layers += [nn.Linear(width, d_out)]
    return nn.Sequential(*layers)


def _task_kind(cfg, y):
    if cfg.task in ("regression",) or cfg.metric in ("rmse", "rmsle", "mae", "r2", "smape"):
        return "regression"
    return "classification"


def train_nn(cfg, epochs=60, width=256, blocks=3, lr=1e-3, seed=42, fe=False,
             dropout=0.2, weight_decay=1e-4, n_folds=None, gpu=None, patience=None):
    """OOF neural-tabular training. Returns ({'nn': {'oof','test','cv'}}, meta) — same shape as tab-train.
    dropout: residual-block dropout probability (regularization).
    weight_decay: AdamW L2 weight decay.
    n_folds: override CV fold count (else cfg.n_folds).
    gpu: force GPU on/off (None = auto-detect).
    patience: if set, early-stop a fold when its training loss fails to improve for this many epochs."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    df_train, df_test, df_sample = TC.load_frames(cfg)
    tgt = TC._target_name(cfg, df_train); y = df_train[tgt].to_numpy()
    kind = _task_kind(cfg, y)
    if fe:
        from . import tab_fe as FE
        folds = TC.make_cv(cfg, y, seed=seed, n_folds=n_folds)
        X, Xte, _ = FE.engineer(df_train, df_test, cfg, y.astype(float), folds)
    else:
        Xtr, _y, Xte_df, feats, _t, _ = TC.basic_features(df_train, df_test, cfg)
        X = Xtr.to_numpy(); Xte = Xte_df.to_numpy() if Xte_df is not None else None
        folds = TC.make_cv(cfg, y, seed=seed, n_folds=n_folds)
    # sanitize + standardize (epsilon guards constant columns)
    X = np.nan_to_num(np.asarray(X, float), nan=0.0, posinf=0.0, neginf=0.0)
    if Xte is not None:
        Xte = np.nan_to_num(np.asarray(Xte, float), nan=0.0, posinf=0.0, neginf=0.0)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    Xtes = ((Xte - mu) / sd).astype(np.float32) if Xte is not None else None
    n_classes = int(len(np.unique(y))) if kind == "classification" else 1
    d_out = 1 if (kind == "regression" or n_classes == 2) else n_classes
    dev = _device(gpu)
    oof = np.zeros(len(y)) if d_out == 1 else np.zeros((len(y), d_out))
    test_acc = None
    for tr, va in folds:
        net = _make_mlp(Xs.shape[1], d_out, width, blocks, dropout=dropout).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        xt = torch.tensor(Xs[tr]).to(dev)
        if kind == "regression":
            yt = torch.tensor(y[tr].astype(np.float32)).view(-1, 1).to(dev); lossf = nn.MSELoss()
        elif d_out == 1:
            yt = torch.tensor(y[tr].astype(np.float32)).view(-1, 1).to(dev); lossf = nn.BCEWithLogitsLoss()
        else:
            yt = torch.tensor(y[tr].astype(np.int64)).to(dev); lossf = nn.CrossEntropyLoss()
        best_loss = float("inf"); bad = 0
        for ep in range(epochs):
            net.train(); opt.zero_grad()
            out = net(xt); loss = lossf(out, yt); loss.backward(); opt.step()
            if patience:
                lv = float(loss.detach().cpu())
                if lv < best_loss - 1e-5:
                    best_loss = lv; bad = 0
                else:
                    bad += 1
                    if bad >= int(patience):
                        break
        net.eval()
        with torch.no_grad():
            def predict(Xa):
                o = net(torch.tensor(Xa).to(dev))
                if kind == "regression":
                    return o.view(-1).cpu().numpy()
                if d_out == 1:
                    return torch.sigmoid(o).view(-1).cpu().numpy()
                return torch.softmax(o, 1).cpu().numpy()
            oof[va] = predict(Xs[va])
            if Xtes is not None:
                tp = predict(Xtes); test_acc = tp if test_acc is None else test_acc + tp
    test_pred = (test_acc / len(folds)) if test_acc is not None else None
    cv = CC.score(cfg.metric, y, oof) if CC.metric_spec(cfg.metric)["fn"] else float("nan")
    ids = df_test[cfg.id_col].to_numpy() if (df_test is not None and cfg.id_col in df_test.columns) else None
    return {"nn": {"oof": oof, "test": test_pred, "cv": cv}}, {"y": y, "test_ids": ids, "kind": kind, "device": dev}


class TabNnTrain(BaseAgent):
    name = "tab-nn-train"
    thread = "M"
    kind = "finding"

    def run(self, q, worker):
        try:
            import torch  # noqa: F401
        except Exception:
            return self.escalate(worker, "researcher", "tab-nn-train needs torch (missing in this env).")
        spec = self.spec(q)
        if "config" not in spec and "config_file" not in spec:
            return self.escalate(worker, "leader", "tab-nn-train needs spec keys ['config' or 'config_file'] — none provided")
        cfg = CC.CompConfig.from_dict(spec["config"]) if "config" in spec else CC.CompConfig.load(spec["config_file"])
        res, meta = train_nn(cfg, epochs=int(spec.get("epochs", 60)), width=int(spec.get("width", 256)),
                             blocks=int(spec.get("blocks", 3)), seed=int(spec.get("seed", 42)),
                             fe=bool(spec.get("fe", False)), dropout=float(spec.get("dropout", 0.2)),
                             weight_decay=float(spec.get("weight_decay", 1e-4)),
                             n_folds=spec.get("n_folds"), gpu=spec.get("gpu"), patience=spec.get("patience"))
        cv = res["nn"]["cv"]
        msg = f"tab-nn-train: neural-tabular OOF CV({cfg.metric})={cv:.5f} on {meta['device']} — ensemble diversity member"
        self.log(msg, kind="finding", recommendation="blend the NN OOF with GBDTs via blend-optimize (decorrelated)")
        return self.done({"cv": cv, "device": meta["device"], "_preds": res}, msg)


_AGENT = TabNnTrain()


def run(q, worker):
    return _AGENT.run(q, worker)
