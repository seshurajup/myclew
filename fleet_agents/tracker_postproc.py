"""tracker-postproc — the pilkwang notebook post-processing + submission writer, as a fleet agent.

Wraps ``research/tracker_pipeline/postproc.run_postproc`` which ADAPTS the verbatim pilkwang
post-proc (learning/ensemble_work/pilk_post.py — motion_relink_edges, close_single_frame_gaps,
recover_strict_gap2, add_safe_divisions_postlink, division/short-track filters, linefit smooth)
into one callable that turns raw prediction ``.geff``s into post-processed ``.geff``s +
``submission.csv``. No post-proc logic is reimplemented — pilk_post functions are called as-is.

Optionally scores the result against GT (edge_jaccard + 0.1·div_jaccard) with src.metric when a
``gt_dir`` is given — passing t_true = geff estimated_number_of_nodes so adj_edge_jaccard is faithful.

Spec: {pred_geff_dir, image_dir, out_dir, datasets, min_track_len, gt_dir (optional score),
       write_geffs, gate_um}.
A BaseAgent subclass with its own data-wise test (test_fleet_agents/tracker_postproc_test.py)
that MUST actually run the post-proc on ≤2 datasets and report edge/div.
"""
from __future__ import annotations
import sys
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent
_TRACKER = COMP / "research" / "tracker_pipeline"
_PRED_DEFAULT = COMP / "research" / "pilkwang_support_pack" / "repo" / "predictions" / "seshu" / "unet_transformer" / "split_0"
_IMG_DEFAULT = COMP / "input" / "biohub-cell-tracking-during-development" / "train"

_VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)


def _paths():
    for p in (str(_TRACKER), str(COMP), str(COMP / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)


class TrackerPostproc(BaseAgent):
    name = "tracker-postproc"
    thread = "B"
    kind = "verdict"

    def _score(self, per_dataset, out_dir, gt_dir, gate_um):
        """Score post-processed geffs vs GT: official adj_edge_jaccard + division_jaccard."""
        from src import io as SIO
        from src import metric as M
        gt_dir = Path(gt_dir)
        counts = []
        for row in per_dataset:
            ds = row["dataset"]
            gt = gt_dir / f"{ds}.geff"
            pred = Path(out_dir) / f"{ds}.geff"
            if not gt.exists() or not pred.exists():
                continue
            gn, ge = SIO.read_geff(gt)
            pn, pe = SIO.read_geff(pred)
            est = SIO.geff_estimated_nodes(gt)
            counts.append(M.official_counts(gn, ge, pn, pe, _VOXEL_SCALE_UM, gate_um=gate_um, t_true=est))
        if not counts:
            return None
        s = M.official_score(counts)
        div = s["division_jaccard"]
        return {"adj_edge_jaccard": round(float(s["adj_edge_jaccard"]), 4),
                "division_jaccard": (None if div != div else round(float(div), 4)),  # NaN → None
                "score": round(float(s["score"]), 4) if s["score"] == s["score"] else None,
                "div_tp": sum(c["div_tp"] for c in counts), "div_fp": sum(c["div_fp"] for c in counts),
                "div_fn": sum(c["div_fn"] for c in counts), "n_scored": len(counts)}

    def run(self, q, worker):
        _paths()
        try:
            import postproc as PPMOD
        except Exception as e:  # noqa: BLE001 — post-proc module/deps missing → escalate cleanly
            return self.escalate(worker, "researcher",
                                 f"[{worker}] tracker-postproc: cannot import postproc module ({str(e)[:80]}).")
        spec = self.spec(q)
        pred_geff_dir = spec.get("pred_geff_dir") or str(_PRED_DEFAULT)
        image_dir = spec.get("image_dir") or str(_IMG_DEFAULT)
        out_dir = spec.get("out_dir") or str(COMP / "results" / "tracker_postproc")
        datasets = spec.get("datasets")
        try:
            res = PPMOD.run_postproc(pred_geff_dir, image_dir, out_dir,
                                     datasets=datasets, min_track_len=spec.get("min_track_len"),
                                     write_geffs=spec.get("write_geffs", True))
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher",
                                 f"[{worker}] tracker-postproc FAILED: {type(e).__name__}: {str(e)[:160]}")
        if res.get("status") == "empty" or res.get("n_datasets", 0) == 0:
            return self.escalate(worker, "researcher",
                                 f"[{worker}] tracker-postproc: no geffs found under {pred_geff_dir}.")

        score = None
        if spec.get("gt_dir"):
            try:
                score = self._score(res["per_dataset"], out_dir, spec["gt_dir"],
                                    float(spec.get("gate_um", 7.0)))
            except Exception as e:  # noqa: BLE001
                score = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
        res["score"] = score

        self.save_state({"n_datasets": res["n_datasets"], "total_nodes": res["total_nodes"],
                         "submission_csv": res["submission_csv"], "score": score})
        sc = ""
        if isinstance(score, dict) and "adj_edge_jaccard" in score:
            sc = (f" · adj_edge_jaccard {score['adj_edge_jaccard']} · div_jaccard {score['division_jaccard']} "
                  f"(div TP={score['div_tp']}/FP={score['div_fp']}/FN={score['div_fn']})")
        self.log(summary=f"tracker-postproc: {res['n_datasets']} ds, {res['total_nodes']:,} nodes, "
                         f"{res['total_edges']:,} edges → submission.csv{sc}",
                 detail=str(res["per_dataset"][:4]), kind="verdict",
                 recommendation="submission.csv is competition-schema; score vs GT to tune post-proc knobs")
        msg = (f"[{worker}] **TRACKER-POSTPROC** · pilkwang fuse/relink/gap/safe-div/linefit\n"
               f"{res['n_datasets']} dataset(s) · {res['total_nodes']:,} nodes / {res['total_edges']:,} edges\n"
               f"submission → `{res['submission_csv']}`{sc}")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done(res, msg, to="leader")


_AGENT = TrackerPostproc()


def run(q, worker):
    return _AGENT.run(q, worker)
