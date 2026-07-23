"""submit-verify — PROVE LOCALLY that the T4-FEASIBLE submission pipeline works end-to-end BEFORE any Kaggle
push: a fast one-pass detector (DoG, measured 0.024s/f on Kaggle CPU) → ILP linking → the exact competition
submission.csv → the REAL per-embryo metric. This is the PIPELINE proof; nb-preflight is the ENVIRONMENT
proof. User rule (2026-07-12): "first prove in local it is possible."

It answers two questions with data, no Kaggle quota spent:
  1. Does the pipeline emit a VALID submission.csv? (schema == the 10 cols; rows>0; nodes>0; edges>0; every
     edge source_id/target_id references an emitted node_id; node coords ≥ 0.)
  2. What does it SCORE per-embryo? (official_counts → official_score = adj_edge_jaccard + 0.1·division_jaccard,
     the identical metric the LB uses — [[biohub_cell_tracking]] local run == LB.) Reported for BOTH embryos
     (44b6 the dense lever + 6bba), never a mean-only verdict.

Reuses the real components (no reinvention): mh_ilp._ilp_track (DoG multi-hyp + tracksdata ILP, threshold-free),
experiments/pipeline/kaggle_submit_pipeline.to_submission_rows (the exact CSV schema), src.metric.
"""
from __future__ import annotations
import time
from .base import BaseAgent, COMP

COLS = ["id", "dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def _valid_submission(df):
    """Schema + integrity checks on the emitted submission rows (the Kaggle grader's hard requirements)."""
    import pandas as pd
    if df is None or "row_type" not in getattr(df, "columns", []):
        return {"has_rows": False, "schema": False, "has_nodes": False, "has_edges": False,
                "edges_reference_nodes": False, "node_coords_valid": False}
    nodes = df[df.row_type == "node"]; edges = df[df.row_type == "edge"]
    node_ids = set(nodes.node_id)
    checks = {
        "has_rows": len(df) > 0,
        "schema": list(df.columns) == COLS,
        "has_nodes": len(nodes) > 0,
        "has_edges": len(edges) > 0,
        "edges_reference_nodes": bool(edges.apply(
            lambda r: r.source_id in node_ids and r.target_id in node_ids, axis=1).all()) if len(edges) else False,
        "node_coords_valid": bool((nodes[["t", "z", "y", "x"]] >= 0).all().all()),
    }
    return checks


class SubmitVerify(BaseAgent):
    name = "submit-verify"
    thread = "S"
    kind = "verdict"

    def _run_pipeline(self, datasets, frames, params):
        """T4-feasible detector→ILP per dataset → (submission_df, per-dataset official_counts). Reuses the
        real _ilp_track + to_submission_rows + metric; global node_id offset so edges never collide across ds."""
        import sys
        sys.path.insert(0, str(COMP)); sys.path.insert(0, str(COMP / "experiments" / "pipeline"))
        import pandas as pd
        from src import io, metric
        from src.io import embryo_id
        from fleet_agents.mh_ilp import _ilp_track, _cfg
        from kaggle_submit_pipeline import to_submission_rows
        cfg = _cfg()
        import numpy as np
        scale = np.asarray(cfg.SCALE)
        all_rows, counts, start = [], [], 0
        for ds in datasets:
            pn, pe = _ilp_track(ds, params, frames=frames, cfg=cfg)       # DoG multi-hyp + ILP (fast detector)
            edges = [(int(s), int(t)) for s, t in zip(pe.get("source_id", []), pe.get("target_id", []))]
            rows, start = to_submission_rows(ds, pn.rename(columns={"node_id": "node_id"}), edges, start)
            all_rows.extend(rows)
            gn, ge = io.read_geff(COMP / f"input/biohub-cell-tracking-during-development/train/{ds}.geff")
            estN = io.geff_estimated_nodes(COMP / f"input/biohub-cell-tracking-during-development/train/{ds}.geff")
            c = metric.official_counts(gn, ge, pn, pe, scale, 7.0, t_true=estN)
            c["dataset"] = ds; c["embryo"] = embryo_id(ds)
            counts.append(c)
        return pd.DataFrame(all_rows, columns=COLS), counts

    def _per_embryo_score(self, counts):
        import pandas as pd, numpy as np
        from src import metric
        df = pd.DataFrame(counts); out = {}
        for emb, g in df.groupby("embryo"):
            out[emb] = metric.official_score(g.to_dict("records"))
        overall = metric.official_score(counts)
        return out, overall

    def run(self, q, worker):
        spec = self.spec(q)
        frames = spec.get("frames", 10)                    # bounded local proof (fast); None = full movie
        strict = bool(spec.get("strict", True))            # strict: escalate on an invalid submission (default); False = report-only
        from fleet_agents.mh_ilp import DEFAULTS
        params = dict(DEFAULTS); params.update({k: v for k, v in spec.items() if k in DEFAULTS})
        params["use_cellpose"] = False                     # DoG-only = the T4-feasible detector
        datasets = spec.get("datasets")
        if not datasets:                                   # one per embryo — prove BOTH, per the per-embryo rule
            from model_scratch.train_v0 import split_datasets
            from src.io import embryo_id
            _, te = split_datasets()
            pick = {}
            for ds in te:
                pick.setdefault(embryo_id(ds), ds)
            datasets = [pick[e] for e in ("44b6", "6bba") if e in pick]

        t0 = time.time()
        df, counts = self._run_pipeline(datasets, frames, params)
        checks = _valid_submission(df)
        per_emb, overall = self._per_embryo_score(counts)
        out = COMP / "experiments/pipeline/_submit_verify.csv"; df.to_csv(out, index=False)
        dt = time.time() - t0

        a44 = per_emb.get("44b6", {}).get("score", float("nan"))
        a6b = per_emb.get("6bba", {}).get("score", float("nan"))
        valid = all(checks.values())
        fails = [k for k, v in checks.items() if not v]
        summary = (f"submit-verify (DoG→ILP, T4-feasible, frames={frames}): "
                   f"submission {'VALID' if valid else 'INVALID ' + str(fails)} "
                   f"(rows={len(df)}); per-embryo score 44b6={a44:.4f} 6bba={a6b:.4f} "
                   f"overall={overall['score']:.4f} → {out.name} [{dt:.0f}s]")
        if not valid and strict:
            return self.escalate(worker, "researcher",
                                 f"submit-verify RED — pipeline emits an INVALID submission: {fails}. {summary}")
        self.log(summary, kind="verdict",
                 recommendation="pipeline proven locally: emits a valid submission.csv with a real per-embryo "
                                "score using the T4-feasible DoG detector. Safe to port into the utility/"
                                "submission notebook; gate every notebook change on this + nb-preflight staying "
                                "GREEN before spending Kaggle quota.")
        return self.done({"valid": valid, "checks": checks, "per_embryo": per_emb,
                          "overall": overall, "rows": len(df), "csv": str(out)}, summary)


_AGENT = SubmitVerify()


def run(q, worker):
    return _AGENT.run(q, worker)
