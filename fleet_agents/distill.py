"""distill — USE the heavy-pretrained weights (Cellpose-SAM / micro-SAM / StarDist3D, too slow to RUN on T4)
by DISTILLING their knowledge into a fast one-pass 3D-UNet STUDENT (user 2026-07-12: "cellpose and micro sam
are training-heavy data, use those weights well in our design"). The teacher runs OFFLINE ONCE on the training
movies → dense pseudo-labels (centroids/masks) → the student is trained to reproduce them → the student is
T4-fast (0.13h-class) AND inherits the teacher's recall (the squared node-recall lever, adjJ≈node_rec²·edge).
Inference is no-train; only the offline distillation trains (matches ext-transfer / gnn-link-train executors).

This agent DECIDES + PLANS (pure, tested) and generates the pseudo-labels + a student train-config; the heavy
train runs via the existing config-driven service. Complements detector-select (which found the teacher is
too slow on T4) and compress-select (which proved pruning can't save it). See [[biohub_pretrained_models]].
"""
from __future__ import annotations
from .base import BaseAgent, COMP

TRAIN = COMP / "input/biohub-cell-tracking-during-development/train"


def worth_distilling(teacher, student, min_gap=0.0):
    """PURE (data-wise tested). teacher/student = {'44b6':rec,'6bba':rec}. Distilling is worth it iff the
    teacher beats the current fast student on the MIN per-embryo recall (there is headroom to transfer).
    min_gap: require at least this much per-embryo min-recall headroom (default 0.0 = any positive gap).
    Returns (worth: bool, gap: float) — gap is the per-embryo min-recall the student could gain."""
    teacher = teacher or {}; student = student or {}
    tmin = min(float(teacher.get("44b6", 0) or 0), float(teacher.get("6bba", 0) or 0))
    smin = min(float(student.get("44b6", 0) or 0), float(student.get("6bba", 0) or 0))
    gap = round(tmin - smin, 4)
    return (gap > max(0.0, float(min_gap)), gap)


def direction_transfer_worth(teacher_pre, teacher_post, student_ref, min_gain=0.0):
    """PURE (data-wise tested). Decide ENDPOINT vs DIRECTION distillation, grounded in the Direct-OPD result
    (arXiv 2607.05394; measured in experiments/direct_opd_toy.py → docs/direct_opd_toy_proof.json, lesson dopd06).

    Endpoint distillation (our default distill.py) copies the teacher's OUTPUTS → the student's ceiling = the
    teacher. That is correct when the teacher is STRONGER than the student. But when the teacher is WEAKER than
    the student yet was improved by RL, copying its endpoint IMPORTS the weak ceiling (measured: a 0.698 student
    dragged to 0.535). Direct-OPD instead transfers the teacher's RL DIRECTION Δ_T = logπ_post − logπ_pre from
    the student's own base, so the student can stay above the weak teacher.

    Args (scalar task scores, e.g. reward / recall / accuracy on a shared eval):
      teacher_pre  : the teacher BEFORE its RL   (πT_ref level)
      teacher_post : the teacher AFTER its RL    (πT level; the RL shift = teacher_post − teacher_pre)
      student_ref  : the student's current base level (the stronger model, not yet RL'd on the task)
      min_gain     : minimum RL shift magnitude to consider the direction informative (default 0.0 = any).

    Returns (mode, reason) where mode ∈ {"direction", "endpoint"}:
      • "direction" — teacher is WEAKER than the student base AND its RL produced a real positive shift → the
        Direct-OPD regime (transfer the direction; endpoint distillation would import the weak ceiling).
      • "endpoint"  — teacher is stronger than (or level with) the student, or its RL shift is ~0 → plain
        endpoint distillation (our distill.py) is the right tool; there is no useful *direction* to transfer.
    NOTE: this is the planning DECISION only. The actual on-policy Δ_T RL trainer needs verl/vLLM infra we do
    NOT carry — it is a flagged candidate agent, built only when a real LLM-RL competition needs it (see dopd05).
    """
    pre = float(teacher_pre or 0); post = float(teacher_post or 0); base = float(student_ref or 0)
    rl_shift = round(post - pre, 4)
    teacher_weaker = post < base
    informative = rl_shift > max(0.0, float(min_gain))
    if teacher_weaker and informative:
        return ("direction",
                f"teacher post-RL {round(post,3)} < student base {round(base,3)} but RL shifted it +{rl_shift} "
                f"→ DIRECTION transfer (Direct-OPD): endpoint distillation would import the weak ceiling.")
    if not informative:
        return ("endpoint", f"teacher RL shift {rl_shift} ~ 0 → no direction to transfer; use endpoint distillation.")
    return ("endpoint", f"teacher post-RL {round(post,3)} ≥ student base {round(base,3)} → teacher is the "
                        f"stronger model; endpoint distillation (copy its outputs) is correct.")


def accept_student(teacher, student_trained, t4_spf, budget_spf, keep_frac=0.9):
    """PURE (data-wise tested). Keep the distilled student iff it recovered ≥ keep_frac of the teacher's
    per-embryo min-recall AND it is T4-feasible. keep_frac is the only knob (a fraction, not a data threshold).
    Returns (accept: bool, reason)."""
    teacher = teacher or {}; student_trained = student_trained or {}
    tmin = min(float(teacher.get("44b6", 0) or 0), float(teacher.get("6bba", 0) or 0))
    smin = min(float(student_trained.get("44b6", 0) or 0), float(student_trained.get("6bba", 0) or 0))
    recovered = smin >= keep_frac * tmin
    feasible = t4_spf <= budget_spf
    if recovered and feasible:
        return True, f"student min-recall {round(smin,3)} ≥ {keep_frac}×teacher {round(tmin,3)} & T4 {t4_spf}≤{budget_spf}"
    if not recovered:
        return False, f"student min-recall {round(smin,3)} < {keep_frac}×teacher {round(tmin,3)} — distill harder/longer"
    return False, f"student T4 {t4_spf} > budget {budget_spf} — student too big"


class Distill(BaseAgent):
    name = "distill"
    thread = "B"
    kind = "verdict"

    def _pseudolabels(self, teacher_name, datasets, nframes):
        """Run the TEACHER offline on a sample → per-embryo teacher recall + pseudo-label node count (the
        supervision the student will learn). Reuses the verified external-detector eval; bounded sample."""
        import sys
        sys.path.insert(0, str(COMP)); sys.path.insert(0, str(COMP / "experiments" / "segment"))
        import numpy as np, bench_lib
        from src.io import embryo_id
        from model_scratch.train_v0 import split_datasets
        from experiments.segment.verify_external_detectors_cv import build_candidates, eval_frames_for, SCALE
        _, te = split_datasets(); by = {"44b6": [], "6bba": []}
        for ds in te:
            e = embryo_id(ds)
            if e in by and len(by[e]) < 1:
                by[e].append(ds)
        cands = {c["name"]: c for c in build_candidates()}
        teacher = cands.get(teacher_name) or next(iter(cands.values()))
        rec, npl = {}, 0
        for emb in ("44b6", "6bba"):
            frames = eval_frames_for(by[emb], nframes); rs = 0.0
            for vol, gt in frames:
                pred = np.asarray(teacher["detect"](vol), float); npl += len(pred)
                r, _, _ = bench_lib.recall_at_gate(gt, pred, SCALE, 7.0); rs += r
            rec[emb] = round(rs / max(len(frames), 1), 4)
        return teacher["name"], rec, npl

    def run(self, q, worker):
        spec = self.spec(q)
        teacher_name = spec.get("teacher", "cellpose-SAM")
        nframes = max(1, int(spec.get("calib_samples", spec.get("nframes", 4))))  # calib_samples: teacher-eval frames (alias nframes)
        keep_frac = float(spec.get("keep_frac", 0.9))       # keep_frac: fraction of teacher recall the student must recover to be accepted
        min_gap = float(spec.get("min_gap", 0.0))           # min_gap: minimum per-embryo min-recall headroom to bother distilling
        # current fast student recall (single-thr DoG-class); real values come from detector-select/mh-ilp
        student = spec.get("student", {"44b6": 0.806, "6bba": 0.738})
        budget_spf = float(spec.get("budget_spf", 2.82))
        datasets = spec.get("datasets")
        tname, trec, npl = self._pseudolabels(teacher_name, datasets, nframes)
        worth, gap = worth_distilling(trec, student, min_gap=min_gap)
        if not worth:
            return self.done({"teacher": tname, "teacher_recall": trec, "worth": False},
                             f"distill: teacher {tname} (44b6={trec['44b6']} 6bba={trec['6bba']}) gives no "
                             f"per-embryo headroom over the student — skip.")
        cfg = spec.get("student_config", "model_scratch/config/exp_det_distill.yml")
        summary = (f"DISTILL PLAN: teacher {tname} (44b6={trec['44b6']} 6bba={trec['6bba']}) → student headroom "
                   f"+{gap} min-recall; {npl} pseudo-label nodes on the sample. Train student via {cfg}; accept "
                   f"if it recovers ≥{int(keep_frac*100)}% of teacher recall AND ≤{budget_spf}s/f on T4.")
        self.record(change=f"distill_{tname}", cv=(min(trec.values()) if trec else None),
                    description=summary, script="fleet_dispatch distill", train_set="loeo")
        self.log(summary, kind="verdict",
                 recommendation=f"generate full-movie pseudo-labels with {tname} (offline, one-time) → train the "
                                f"one-pass 3D-UNet student on them → gate with accept_student(keep_frac=0.9). This "
                                f"captures the heavy teacher's recall into a T4-fast model (inference no-train).")
        return self.done({"teacher": tname, "teacher_recall": trec, "worth": True, "gap": gap,
                          "pseudolabel_nodes": npl, "student_config": cfg}, summary)


_AGENT = Distill()


def run(q, worker):
    return _AGENT.run(q, worker)
