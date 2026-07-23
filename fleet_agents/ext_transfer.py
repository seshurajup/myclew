"""ext-transfer — the external-data training CV the user's design needs: TRAIN the division/flow model on
the box-sampled EXTERNAL embryos (augmentation = the many boxes), then measure TRANSFER to the real
COMPETITION train embryos (embryo-disjoint — external ≠ competition, exactly how Kaggle splits). Reports
the transfer division-AP PER competition embryo (the "2 CV datasets, one per embryo") + the mean.

Composes existing agents (no new training code):
  1. gnn-link-train  → train on external box-sampled subset (include_embryos, many epochs).
  2. competition eval → load the trained gnn_link.pt, extract sister-geometry features on each competition
     embryo (reusing gnn_link_train._features), compute division AP → the honest transfer score.

This is what "train external, see the score on 2 embryos first, then scale to 4" measures on the REAL
target. Start with 2 external embryos; scale by adding to include_embryos. Every score is a measured AP,
no assumption ([[feedback_public_notebook_golden_cv_rule]] discipline applied to training too).

Reusable / spec-driven: {ext_gt, include_embryos, epochs, sample_frames, comp_parquet, hidden, n_layers}.
A BaseAgent subclass with its own data-wise test.
"""
from __future__ import annotations
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent

# Image/feature appearance matching is the REUSABLE `domain-match` agent — biohub composes it (no dup).
from .domain_match import (appearance_match_search, learned_domain_map,  # noqa: E402,F401
                           histogram_match, local_contrast_norm, spectrum_match, zscore_norm, patch_feats as _patch_feats)


class ExtTransfer(BaseAgent):
    name = "ext-transfer"
    thread = "B"
    kind = "verdict"

    def _agents(self):
        from . import _RAW_HANDLERS
        return _RAW_HANDLERS

    def _eval_competition(self, ckpt_path, comp_parquet, spec):
        """Load the trained model, score division-AP on EACH competition embryo (transfer). Returns
        {embryo: ap}. Reuses gnn_link_train._features so the feature contract is identical to training."""
        import numpy as np, pandas as pd, torch
        from torch import nn
        from scipy.spatial import cKDTree
        from sklearn.metrics import average_precision_score
        from . import gnn_link_train as G
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        hidden, nlayers = ck["hidden"], ck["n_layers"]
        mu, sd = ck["mu"], ck["sd"]

        def mlp(out):
            layers, d = [], len(G.FEATURE_NAMES)
            for _ in range(nlayers):
                layers += [nn.Linear(d, hidden), nn.GELU()]; d = hidden
            layers += [nn.Linear(d, out)]
            return nn.Sequential(*layers)
        div = mlp(1); div.load_state_dict(ck["div"]); div.eval()

        df = pd.read_parquet(comp_parquet)
        frames = int(spec.get("eval_frames", 10 ** 9))
        # POOL by base competition EMBRYO (44b6_12dfb391 → 44b6) → ONE AP per embryo = the 2 CV datasets.
        # Per-dataset AP is noise (each dataset has 1-2 divisions); pooling all a embryo's nodes is the
        # honest embryo-level CV the competition split uses.
        df["__base"] = df["embryo"].map(lambda e: str(e).split("_")[0])
        out = {}
        for base in sorted(df["__base"].unique()):
            sub = df[df["__base"] == base]
            Xs, Ds = [], []
            for ds in sub["embryo"].unique():                 # features are per-dataset (per-frame cKDTree)
                X, D, _ = G._features(sub[sub["embryo"] == ds], pd, np, cKDTree, 6.0, frames)
                if len(X):
                    Xs.append(X); Ds.append(D)
            if not Xs:
                continue
            X = np.concatenate(Xs); D = np.concatenate(Ds)
            if D.sum() == 0:                                  # no divisions in this embryo → AP undefined
                out[base] = None; continue
            Xn = (X - mu) / sd
            with torch.no_grad():
                p = torch.sigmoid(div(torch.tensor(Xn))).numpy().ravel()
            out[base] = {"ap": round(float(average_precision_score(D, p)), 4),
                         "n_nodes": int(len(D)), "n_div": int(D.sum())}
        return out

    def _load_mid_slab(self, zarr_path, ds=4, zmax=16):
        """Load a zarr movie's mid-time volume (down-sampled Y,X by `ds`, ≤`zmax` central Z slices) for the
        appearance search. Coarse is fine — the search compares transforms RELATIVELY, not absolute recall."""
        import zarr, numpy as np
        img = zarr.open(zarr_path)
        k = "0" if hasattr(img, "keys") and "0" in list(img.keys()) else None
        a = (img[k][:] if k else img[:]).astype(np.float32)
        a = a[a.shape[0] // 2]                                      # mid time-point → 3D (Z,Y,X)
        z0 = max(0, a.shape[0] // 2 - zmax // 2)
        return a[z0:z0 + zmax, ::ds, ::ds]

    def _appearance_match(self, spec, worker):
        """Drive the external→competition IMAGE appearance gap toward adv-AUC 0.5 with the spatial/frequency
        transforms (spectrum/LCN/histmatch), reporting the full search. Spec: {ext_zarr, comp_zarr, sigmas}."""
        import glob
        ext_zarr = spec.get("ext_zarr")
        comp_zarr = spec.get("comp_zarr")
        if not ext_zarr:
            g = sorted(glob.glob(str(COMP / "research/zebrahub/geff_trainset/ZSNS005*.zarr")))
            ext_zarr = g[0] if g else None
        if not comp_zarr:
            g = [d for d in glob.glob(str(COMP / "research/**/*6bba*.zarr"), recursive=True) if "venv" not in d]
            comp_zarr = g[0] if g else None
        if not ext_zarr or not comp_zarr:
            return self.escalate(worker, "researcher", f"[{worker}] appearance-match: need ext_zarr & comp_zarr (external/competition movies).")
        try:
            ext = self._load_mid_slab(ext_zarr, ds=int(spec.get("ds", 4)), zmax=int(spec.get("zmax", 16)))
            comp = self._load_mid_slab(comp_zarr, ds=int(spec.get("ds", 4)), zmax=int(spec.get("zmax", 16)))
        except Exception as e:  # noqa: BLE001 — unreadable zarr → escalate cleanly
            return self.escalate(worker, "researcher", f"[{worker}] appearance-match: could not load movie volumes ({str(e)[:80]}).")
        # delegate to the REUSABLE domain-match agent — fixed transforms, and (if they fall short) the LEARNED
        # adversarial mapper automatically. `learned` defaults ON here since biohub's gap is texture/PSF-level.
        from . import domain_match as DM
        report, _ = appearance_match_search(ext, comp, sigmas=tuple(spec.get("sigmas", (1.5, 3, 5))),
                                            n_patch=int(spec.get("n_patch", 400)))
        learned = None
        if bool(spec.get("learned", True)) and report["verdict"] != "matched":
            learned, _ = DM.learned_domain_map(ext, comp, iters=int(spec.get("iters", 400)),
                                               lambda_struct=float(spec.get("lambda_struct", 3.0)))
        self.save_state({"appearance_match": report, "learned_match": learned, "ext_zarr": ext_zarr, "comp_zarr": comp_zarr})
        rows = "\n".join(f"| {t['recipe']} | {t['adv_auc']} |" for t in report["trials"])
        tag = {"matched": "✅ MATCHED (indistinguishable)", "partial": "🟡 PARTIAL",
               "structural-gap": "❌ STRUCTURAL GAP (fixed transforms)"}[report["verdict"]]
        lrow = ""
        if learned:
            lrow = (f"\n**learned adversarial mapper**: adv-AUC {learned['adv_auc_before']}→**{learned['adv_auc_after']}**, "
                    f"structure-corr {learned['structure_corr']} → "
                    f"{'✅ honest match (signal preserved)' if learned['honest_match'] else ('⚠️ reached 0.5 but destroys signal' if learned['matched'] else '❌ still separable')}")
        best_overall = min(report["best_adv_auc"], learned["adv_auc_after"]) if learned else report["best_adv_auc"]
        msg = (f"[{worker}] **APPEARANCE-MATCH** · external→competition image gap → drive adv-AUC to 0.5\n"
               f"| recipe | adv-AUC |\n|:-|--:|\n{rows}\n"
               f"→ fixed best **{report['best_adv_auc']}** via `{report['best_recipe']}` — {tag}{lrow}\n"
               f"→ **overall best adv-AUC {best_overall}**")
        self.log(summary=f"appearance-match: fixed {report['best_adv_auc']} ({report['best_recipe']})"
                         + (f", learned {learned['adv_auc_after']} (struct {learned['structure_corr']}, honest={learned['honest_match']})" if learned else ""),
                 detail=f"reusable domain-match on {Path(ext_zarr).name}→{Path(comp_zarr).name}",
                 kind="verdict", recommendation=("appearance matched & signal preserved — external CAN transfer" if (learned and learned["honest_match"]) or report["verdict"] == "matched"
                                                 else "gap survives learned mapping — pivot to competition-domain self-training"))
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"appearance_match": report, "learned_match": learned, "best_adv_auc": best_overall}, msg, to="leader")

    def run(self, q, worker):
        spec = self.spec(q)
        if spec.get("appearance_match"):
            return self._appearance_match(spec, worker)
        A = self._agents()
        ext_gt = spec.get("ext_gt", "results/flow_gt/flow_node_gt_boxed.parquet")
        inc = spec.get("include_embryos") or ["ZSNS001", "ZSNS003"]
        comp_parquet = spec.get("comp_parquet", "results/flow_gt/competition_flow.parquet")

        # 1) train on external via the gnn-link-train agent (box-sampling = augmentation; many epochs)
        tspec = {"gt_path": ext_gt, "include_embryos": inc, "test_embryo": inc[-1],
                 "epochs": int(spec.get("epochs", 400)), "sample_frames": int(spec.get("sample_frames", 80)),
                 "hidden": int(spec.get("hidden", 128)), "n_layers": int(spec.get("n_layers", 3))}
        if spec.get("lr") is not None:                       # forward optional Adam lr to the trainer
            tspec["lr"] = spec["lr"]
        if spec.get("device"):                               # forward optional device (cpu fallback handled downstream)
            tspec["device"] = spec["device"]
        if "gnn-link-train" not in A:
            return self.escalate(worker, "researcher", f"[{worker}] ext-transfer: gnn-link-train agent missing.")
        st, tr = A["gnn-link-train"]({"question": f"train ext {inc}", "spec": tspec}, worker)[:2]
        tr = tr if isinstance(tr, dict) else {}
        ext_ap = tr.get("div_ap")

        # 2) transfer eval on the REAL competition embryos (embryo-disjoint)
        ckpt = COMP / "results" / "gnn_link" / "gnn_link.pt"
        comp = COMP / comp_parquet
        transfer = {}
        if ckpt.exists() and comp.exists():
            try:
                transfer = self._eval_competition(str(ckpt), str(comp), spec)
            except Exception as e:  # noqa: BLE001
                transfer = {"error": str(e)[:120]}
        def _ap(v):
            return v["ap"] if isinstance(v, dict) else (v if isinstance(v, (int, float)) else None)
        vals = [_ap(v) for v in transfer.values() if _ap(v) is not None]
        mean_transfer = round(sum(vals) / len(vals), 4) if vals else None

        self.save_state({"trained_on": inc, "epochs": tspec["epochs"], "ext_heldout_ap": ext_ap,
                         "competition_transfer_ap": transfer, "mean_transfer_ap": mean_transfer})
        self.log(summary=f"ext-transfer: trained on {inc} ({tspec['epochs']}ep) → ext-heldout AP {ext_ap}, "
                         f"competition per-embryo transfer AP (mean {mean_transfer}): "
                         + ", ".join(f"{e}={_ap(v)}" for e, v in transfer.items()),
                 detail="external box-sampled (aug) → embryo-disjoint transfer to the 2 competition embryos (2 CV folds)",
                 kind="verdict", recommendation="compare 2-embryo vs 4-embryo; if transfer stays ~0.5 (random), external division geometry doesn't transfer")
        rows = "\n".join(f"| {e} | {_ap(v)} | {v.get('n_div') if isinstance(v, dict) else '—'} |" for e, v in transfer.items())
        msg = (f"[{worker}] **EXT-TRANSFER** · trained on {len(inc)} external embryos {inc} ({tspec['epochs']} ep, box-sampled aug)\n"
               f"• external held-out AP **{ext_ap}**\n"
               f"• competition transfer AP — **2 CV folds (per embryo)**, mean **{mean_transfer}**\n"
               f"| competition embryo (CV) | div AP | #div |\n|---|--:|--:|\n{rows}\n"
               f"→ scale include_embryos to all 4 to see if more external data lifts the transfer.")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"trained_on": inc, "ext_heldout_ap": ext_ap, "competition_transfer_ap": transfer,
                          "mean_transfer_ap": mean_transfer}, msg, to="leader")


_AGENT = ExtTransfer()


def run(q, worker):
    return _AGENT.run(q, worker)
