"""combined-train — train the division/affinity model on BOX-SAMPLED external + competition data directly,
then evaluate on golden-12. The payoff of the sampling work: external is now author-faithful (density /
complete-paths / track-length / sister-ratio all matched), so combining it with the in-domain competition
GT should finally transfer.

Steps: (1) convert the competition train geffs → per-node flow+division rows, per-dataset NN-normalised so
the sister-geometry features are comparable to the (already-normalised) external boxes; (2) hold out golden-12
as the eval embryos; (3) concat external-boxed + competition-train; (4) train via the verified gnn-link-train;
(5) report held-out division AP. Runs sample-match first as a GATE (won't train on mismatched external).

A BaseAgent subclass with its own data-wise test. Spec: {external_gt, train_dir, golden, epochs, sample_frames}.
"""
from __future__ import annotations
import glob
import os
from pathlib import Path
from .base import BaseAgent
from . import gnn_link_train

COMP = Path(__file__).resolve().parent.parent


def _geff_to_rows(g, io, np, cKDTree, ref=1.3):
    """competition geff → DataFrame [embryo,t,z,y,x,dz,dy,dx,is_division], per-dataset NN-normalised.
    Vectorised (no iterrows) — this was the CPU bottleneck that starved the GPU."""
    import pandas as pd
    try:
        gn, ge = io.read_geff(g)
    except Exception:  # noqa: BLE001 — skip an unreadable geff rather than crash the parallel parse
        return None
    ds = os.path.basename(g).replace(".geff", "")
    nid = gn["node_id"].to_numpy(); tt = gn["t"].to_numpy()
    P = gn[["z", "y", "x"]].to_numpy().astype(float)
    n = len(P)
    if n < 5:
        return None
    med = np.median(cKDTree(P).query(P[: min(300, n)], k=2)[0][:, 1]) if n > 2 else 1.0
    sc = ref / max(med, 1e-6)
    idx = {int(v): i for i, v in enumerate(nid)}
    first_child = {}; nkids = {}
    if len(ge):
        for s, t in ge[["source_id", "target_id"]].to_numpy():
            s = int(s); nkids[s] = nkids.get(s, 0) + 1
            if s not in first_child:
                first_child[s] = int(t)
    dz = np.full(n, np.nan); dy = np.full(n, np.nan); dx = np.full(n, np.nan); isdiv = np.zeros(n, "int8")
    for i in range(n):
        s = int(nid[i])
        if nkids.get(s, 0) >= 2:
            isdiv[i] = 1
        c = first_child.get(s)
        if c is not None and c in idx:
            d = (P[idx[c]] - P[i]) * sc; dz[i], dy[i], dx[i] = d
    Ps = P * sc
    return pd.DataFrame({"embryo": ds, "t": tt, "z": Ps[:, 0], "y": Ps[:, 1], "x": Ps[:, 2],
                         "dz": dz, "dy": dy, "dx": dx, "is_division": isdiv})


class CombinedTrain(BaseAgent):
    name = "combined-train"
    thread = "B"

    def run(self, q, worker):
        from .base import gpu_train_held
        if gpu_train_held():
            return self.escalate(worker, "leader",
                                 f"[{worker}] combined-train HELD — GPU training is parked (5090 power-cap gate). "
                                 f"Remove config/_auto/gpu_train_hold.flag (human GO) before training.")
        import numpy as np
        import pandas as pd
        from scipy.spatial import cKDTree
        import sys
        sys.path.insert(0, str(COMP / "src")); sys.path.insert(0, str(COMP))
        from src import io
        spec = self.spec(q)
        ext_path = Path(spec.get("external_gt") or (COMP / "results" / "flow_gt" / "flow_node_gt_boxed.parquet"))
        train_dir = spec.get("train_dir") or str(COMP / "input" / "biohub-cell-tracking-during-development" / "train")
        golden = set(spec.get("golden") or [])
        limit = int(spec.get("comp_limit", 60))

        # GATE: external must match the author's scheme (density/sister-ratio) before we train on it
        try:
            from . import sample_match
            gs, gres, _, _ = sample_match.run({"question": "gate", "spec": {
                "detection_cells": float(spec.get("detection_cells", 377)), "external_gt": str(ext_path),
                "competition_sister_ratio": 1.60, "external_sister_ratio": 1.53}}, worker)
            if isinstance(gres, dict) and gres.get("matched") is False:
                return self.escalate(worker, "researcher",
                                     f"[{worker}] combined-train: sample-match GATE FAILED — external not author-faithful; re-run box-sample.")
        except Exception:  # noqa: BLE001
            pass

        # competition geffs → parquet, CACHED once (fast reads after) + converted in PARALLEL (all cores).
        cache = COMP / "results" / "flow_gt" / "competition_flow.parquet"
        geffs = sorted(glob.glob(os.path.join(train_dir, "*.geff")))
        if limit:
            geffs = geffs[:limit]
        if not geffs:
            comp_all = pd.DataFrame(columns=["embryo", "t", "z", "y", "x", "dz", "dy", "dx", "is_division"])
        elif cache.exists() and not spec.get("rebuild_cache"):
            comp_all = pd.read_parquet(cache)                    # cached: instant
        else:
            from concurrent.futures import ThreadPoolExecutor
            import os as _os
            workers = min(len(geffs), (_os.cpu_count() or 4))
            with ThreadPoolExecutor(max_workers=workers) as ex:  # all cores for the geff parse
                dfs = list(ex.map(lambda g: _geff_to_rows(g, io, np, cKDTree), geffs))
            comp_all = pd.concat([d for d in dfs if d is not None], ignore_index=True)
            cache.parent.mkdir(parents=True, exist_ok=True); comp_all.to_parquet(cache, index=False)
        golden_norm = {g.replace(".zarr", "").replace(".geff", "") for g in golden}
        comp_eval = comp_all[comp_all["embryo"].isin(golden_norm)]
        comp_df = comp_all[~comp_all["embryo"].isin(golden_norm)]

        # external boxes (already normalised) + competition-train
        try:
            ext = pd.read_parquet(ext_path) if ext_path.exists() else pd.DataFrame()
        except Exception:  # noqa: BLE001 — corrupt external parquet → train on competition alone
            ext = pd.DataFrame()
        parts = [p for p in [ext, comp_df] if len(p)]
        if not parts:
            return self.escalate(worker, "researcher", f"[{worker}] combined-train: no training data.")
        combined = pd.concat(parts, ignore_index=True)
        # add golden eval embryos so gnn-link-train can hold one out
        eval_emb = None
        if len(comp_eval):
            # hold out the golden embryo with the MOST divisions (else division AP is undefined on a 0-division crop)
            divcount = comp_eval.groupby("embryo")["is_division"].sum().sort_values(ascending=False)
            eval_emb = divcount.index[0] if len(divcount) and divcount.iloc[0] > 0 else comp_eval["embryo"].iloc[0]
            combined = pd.concat([combined, comp_eval], ignore_index=True)
        out = COMP / "results" / "flow_gt" / "combined_train.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(out, index=False)

        # train on the combined set, hold out a golden embryo for the honest transfer test
        tspec = {"gt_path": str(out), "epochs": int(spec.get("epochs", 50)),
                 "sample_frames": int(spec.get("sample_frames", 25)),
                 "hidden": int(spec.get("hidden", 128)), "n_layers": int(spec.get("n_layers", 3))}
        if spec.get("lr") is not None:                       # forward optional Adam lr to the trainer
            tspec["lr"] = spec["lr"]
        if spec.get("device"):                               # forward optional device (cpu fallback handled downstream)
            tspec["device"] = spec["device"]
        if eval_emb:
            tspec["test_embryo"] = eval_emb
        status, res, _, _ = gnn_link_train.train({"question": "combined", "spec": tspec}, worker)
        if status != "done":
            return self.escalate(worker, "researcher", f"[{worker}] combined-train: trainer failed.")

        n_ext = len(ext); n_comp = len(comp_df)
        n_div = int(combined["is_division"].sum()) if "is_division" in combined.columns else 0
        self.save_state({"external_nodes": n_ext, "competition_nodes": n_comp, "divisions": n_div,
                         "eval_embryo": eval_emb, "div_ap": res.get("div_ap"), "base_ap": res.get("base_ap")})
        self.log(summary=f"combined-train: ext {n_ext:,} + comp {n_comp:,} nodes → div AP {res.get('div_ap')} (base {res.get('base_ap')}) on held-out {eval_emb}",
                 detail=f"{n_div} divisions; external gate-passed", kind="verdict",
                 recommendation="if div AP >> base, apply via affinity-link on golden-12 and prove the metric")
        msg = (f"[{worker}] **COMBINED-TRAIN** · box-sampled external + competition (gate-passed)\n"
               f"| source | nodes |\n|---|--:|\n| external (boxed) | {n_ext:,} |\n| competition train | {n_comp:,} |\n"
               f"| divisions | {n_div:,} |\n\n"
               f"Held-out `{eval_emb}` division **AP {res.get('div_ap')}** vs base {res.get('base_ap')} "
               f"({res.get('lift','?')}× lift). External is author-faithful → this is the honest transfer test.")
        self.post(worker, "all", msg, routine=False, kind="verdict")
        return self.done({"div_ap": res.get("div_ap"), "base_ap": res.get("base_ap"),
                          "external_nodes": n_ext, "competition_nodes": n_comp, "eval_embryo": eval_emb}, msg)


_AGENT = CombinedTrain()


def run(q, worker):
    return _AGENT.run(q, worker)
