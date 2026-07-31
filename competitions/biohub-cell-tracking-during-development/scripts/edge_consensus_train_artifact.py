"""Train the SHIPPABLE edge-error consensus artifact (one HGB model + frozen threshold).

For the submission we train on ALL available labelled golden datasets (both embryos) so the model
sees maximum data; the hidden test is the same distribution (train-size). The LOEO CV
(edge_consensus_run.py) already validated that this train-procedure generalises both-embryo>=0.

Threshold: net-max (FP-TP removed) with precision>=MINPREC on the training edges, capped at TAUMAX
(=only remove edges the model is >~92% confident are FP). Saves model.joblib + ec_meta.json
(feature order, tau, taumax) to --out. The apply logic lives in edge_consensus_apply.py (no GT).
"""
import argparse, sys, glob, os, json
from pathlib import Path
import numpy as np
COMP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(COMP / "scripts"))
import importlib.util
_spec = importlib.util.spec_from_file_location("ec", str(COMP / "scripts/edge_consensus.py"))
ec = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ec)
from sklearn.ensemble import HistGradientBoostingClassifier
import joblib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratchpad/ec_artifact")
    ap.add_argument("--cache", default="scratchpad/ec_cache")
    ap.add_argument("--taumax", type=float, default=0.08)
    ap.add_argument("--minprec", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache)
    dss = sorted(os.path.basename(g)[:-5] for g in glob.glob("scratchpad/base_stacked/*.geff"))
    dss = [d for d in dss if d[:4] in ("44b6", "6bba")]
    X, y, P_all = [], [], []
    Xv, yv, DIVv = [], [], []
    for ds in dss:
        z = np.load(cache / f"{ds}.npz", allow_pickle=True)
        feat, lab, valid, s_out = z["feat"], z["lab"], z["valid"], z["s_out"]
        X.append(feat[valid]); y.append(lab[valid].astype(int))
        Xv.append(feat[valid]); yv.append(lab[valid].astype(int)); DIVv.append(s_out[valid] >= 2)
    X = np.vstack(X); y = np.concatenate(y)
    model = HistGradientBoostingClassifier(max_leaf_nodes=31, learning_rate=0.06, max_iter=300,
                                           l2_regularization=1.0, random_state=args.seed,
                                           class_weight="balanced")
    model.fit(X, y)
    Xv = np.vstack(Xv); yv = np.concatenate(yv); DIVv = np.concatenate(DIVv).astype(bool)
    proba = model.predict_proba(Xv)[:, 1]
    best = (0.02, -1, 0, 0)
    for tau in np.linspace(0.02, args.taumax, 40):
        rem = (proba < tau) & (~DIVv)
        fpr = int((rem & (yv == 0)).sum()); tpr = int((rem & (yv == 1)).sum())
        tot = fpr + tpr; prec = fpr / tot if tot else 0.0
        if tot >= 1 and prec >= args.minprec and (fpr - tpr) > best[1]:
            best = (float(tau), fpr - tpr, fpr, tpr)
    tau = best[0]
    joblib.dump(model, out / "ec_model.joblib")
    meta = dict(features=ec.FEATNAMES, tau=tau, taumax=args.taumax, minprec=args.minprec,
                density_r_um=ec.DENSITY_R_UM, train_datasets=dss,
                train_net=best[1], train_fp_rem=best[2], train_tp_rem=best[3],
                n_train_valid_edges=int(len(yv)))
    (out / "ec_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
