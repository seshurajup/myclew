"""detector-select — CHOOSE the detector (architecture + weights) by DATA PROOF on the competition
TRAINING CV, per-embryo. The model choice is made by this agent from evidence, NOT by hand.

User rule (2026-07-11): "take that architecture + its weights IF it is really good for our training-CV
datasets — choose with a python agent, with data proof." So the pick is: for each candidate detector,
measure per-embryo recall@7µm on the embryo-disjoint training CV (recall = unbiased node-recall estimate
under the sparse 1% GT; the squared lever adjJ≈node_rec²·edge_prec). Then _choose() picks the winner that
is good on BOTH embryos (max of the MIN per-embryo recall) among the Kaggle-FEASIBLE, NON-LEAKY candidates.

EXCLUDED by design: models trained on the competition data (pilkwang/hongdaekim Kaggle support-packs) — a
high score there is leakage, not skill. Only external/no-train weights are eligible.
"""
from __future__ import annotations
import time
from .base import BaseAgent, COMP


# Kaggle code-comp budget: 2×T4, 12h wall, internet OFF. Hidden test = TRAINING size = 199×100 = 19,900 fr.
# 2×T4 HALVES wall-clock (user 2026-07-12: "18 on T4 = 9 on 2×T4"): the per-frame factor is PER SINGLE T4;
# splitting frames across the 2 GPUs cuts wall-clock in half. So a detector's wall-clock full-test time is
#   t4_spf(per-single-T4) × TEST_FRAMES / N_GPUS(=2) / 3600  hours.
# The budget below bakes the 2 GPUs in via GPU_SECONDS = 12h×3600×2 = 86,400 total-GPU-s → 4.34 GPU-s/frame,
# so comparing a single-stream t4_spf to DETECT_BUDGET_SPF already credits both GPUs. Reserve ~35% for track.
TEST_FRAMES = 19_900
N_GPUS = 2
GPU_SECONDS = 12 * 3600 * N_GPUS      # 2×T4, 12h — total GPU-seconds (both cards)
DETECT_BUDGET_SPF = round(GPU_SECONDS / TEST_FRAMES * 0.65, 2)   # ~2.82 GPU-s/f (single-stream; 2 GPUs credited)

# CRITICAL (biohub 2026-07 lesson): speed measured on our RTX 5090 must be mapped to the actual T4 — the 5090
# is 5–21× FASTER and the factor depends on the workload. MEASURED on Kaggle 2×T4 (biohub-bench v4):
#   Cellpose-SAM ViT+2D-stitch: 2.4s/f(5090) → 51.7s/f(T4) = ~21× (143h full test — infeasible at ANY block K).
#   DoG classical on Kaggle's 4-vCPU: 0.024s/f → 0.13h full test (measured directly on Kaggle, factor N/A).
# cnn3d/cnn factors remain PRIORS until a one-pass UNet is timed on T4. Compare the T4-ESTIMATED spf to budget.
T4_FACTOR = {"vit_2dstitch": 21.0, "vit": 14.0, "cnn3d": 8.0, "cnn": 6.0, "cpu": 1.0, "unknown": 14.0}


def t4_spf(measured_spf_5090, model_kind="unknown"):
    """Map a dev-GPU (5090) per-frame time to the Kaggle T4. Use the model-kind factor; a raw 5090 number
    is meaningless for the T4 budget. Confirm with on-Kaggle calibration (notebook self-times + aborts)."""
    return round(measured_spf_5090 * T4_FACTOR.get(model_kind, T4_FACTOR["unknown"]), 2)


def _feasible(spf, model_kind=None):
    """Feasible only if the T4-ESTIMATED spf clears the detector budget with margin. `spf` is a 5090 measure
    when model_kind is given (mapped to T4); already-T4 when model_kind is None."""
    est = t4_spf(spf, model_kind) if model_kind else spf
    if est <= DETECT_BUDGET_SPF:
        return f"fits(T4~{est}≤{DETECT_BUDGET_SPF})"
    return None                             # timeout on T4 = zero score → exclude


def _choose(results, require_feasible=True):
    """PURE decision logic (data-wise tested). results = {name: {"44b6":r,"6bba":r,"spf":s,"leaky":bool}}.
    Pick the detector maximising the MIN per-embryo recall (good on BOTH — the per-embryo rule), among
    NON-LEAKY and (optionally) Kaggle-FEASIBLE candidates. Returns (winner_name, ranked_list)."""
    ranked = []
    for name, r in results.items():
        if r.get("leaky"):
            continue
        spf = r.get("spf", 0.0)
        kind = r.get("kind")                        # model_kind → maps the 5090 spf to the real T4 budget
        feas = _feasible(spf, kind)
        if require_feasible and feas is None:
            continue
        r44 = r.get("44b6"); r6b = r.get("6bba")
        if r44 is None or r6b is None:
            continue
        if r44 != r44 or r6b != r6b:                # NaN recall → skip (mis-measured candidate)
            continue
        score = min(r44, r6b)                       # good on BOTH embryos, not the mean
        ranked.append({"name": name, "44b6": r44, "6bba": r6b, "spf": spf, "kind": kind,
                       "t4_spf": t4_spf(spf, kind) if kind else spf,
                       "feasible": feas, "min_recall": round(score, 4)})
    ranked.sort(key=lambda d: -d["min_recall"])
    return (ranked[0]["name"] if ranked else None), ranked


class DetectorSelect(BaseAgent):
    name = "detector-select"
    thread = "S"
    kind = "verdict"

    def _measure(self, nds, nframes):
        """Measure per-embryo recall@7µm for each external candidate detector on the training CV."""
        import sys
        sys.path.insert(0, str(COMP)); sys.path.insert(0, str(COMP / "experiments" / "segment"))
        import numpy as np
        import bench_lib
        from src.io import embryo_id
        from model_scratch.train_v0 import split_datasets
        from experiments.segment.verify_external_detectors_cv import build_candidates, eval_frames_for, SCALE
        _, te = split_datasets()
        by = {"44b6": [], "6bba": []}
        for ds in te:
            e = embryo_id(ds)
            if e in by and len(by[e]) < nds:
                by[e].append(ds)
        cands = build_candidates()
        results = {}
        for emb in ("44b6", "6bba"):
            frames = eval_frames_for(by[emb], nframes)
            for c in cands:
                rec_sum = 0.0; secs = 0.0
                for vol, gt in frames:
                    t0 = time.time(); pred = c["detect"](vol); secs += time.time() - t0
                    r, _, _ = bench_lib.recall_at_gate(gt, np.asarray(pred, float), SCALE, 7.0)
                    rec_sum += r
                d = results.setdefault(c["name"], {"leaky": False})
                d[emb] = round(rec_sum / max(len(frames), 1), 4)
                d["spf"] = round(max(d.get("spf", 0.0), secs / max(len(frames), 1)), 2)
        return results

    def run(self, q, worker):
        spec = self.spec(q)
        nds = int(spec.get("nds", 3)); nframes = int(spec.get("nframes", 6))
        results = spec.get("results") or self._measure(nds, nframes)   # allow injected results (test/reuse)
        winner, ranked = _choose(results, require_feasible=spec.get("require_feasible", True))
        proof = "; ".join(f"{d['name']} [44b6={d['44b6']} 6bba={d['6bba']} {d['spf']}s/f {d['feasible']}]"
                          for d in ranked)
        if winner is None:
            return self.escalate(worker, "researcher",
                                 f"detector-select: no feasible non-leaky detector met the bar. Data: {proof}")
        w = next(d for d in ranked if d["name"] == winner)
        # STAGE-AWARE (2026-07): the data spans zebrafish S0..S4; a single operating point won't fit both a
        # sparse early-gastrula (few large well-separated nuclei) and a dense segmentation crop (many small packed).
        stage_note = ""
        try:
            from .sample_match import dataset_stages
            from collections import Counter
            st = dataset_stages()
            if st:
                dist = dict(sorted(Counter(v["stage"] for v in st.values()).items()))
                stage_note = (f" STAGE-AWARE: data spans S0–S4 {dist} (6bba→early S0–S1, 44b6→late S3–S4); "
                              f"evaluate + tune the detector PER STAGE (density-adaptive pool/kernel), not one "
                              f"global point. Also match the label scheme: recall-tilt on ISOLATED cells. "
                              f"(results/label_selection/dataset_zf_stage.parquet)")
        except Exception:  # noqa: BLE001
            pass
        # 100%-CONFIRMED label protocol (paper-verify.label_facts) — the detector choice must MATCH it.
        try:
            from .paper_verify import label_facts
            lf = label_facts(); imp = lf["implications"]
            stage_note += (f" CONFIRMED-GT ({lf['source'].split(';')[0].strip()}): {imp['detect_all_nuclei']} "
                           f"{imp['recall_tilt']} ⚠️ {imp['cv_over_credits_easy']}"
                           + (f" paper-verify={lf['verdicts']}." if lf.get('verdicts') else ""))
        except Exception:  # noqa: BLE001
            pass
        summary = (f"CHOSEN detector (data-proof, training CV per-embryo): {winner} — "
                   f"44b6={w['44b6']} 6bba={w['6bba']} (min={w['min_recall']}) {w['spf']}s/f {w['feasible']}. "
                   f"Ranked: {proof}")
        self.log(summary, kind="verdict",
                 recommendation=f"adopt {winner} as the mh-ilp candidate generator (threshold-free, external, "
                                f"no-leak). Kaggle support-packs excluded (trained on our data). Fine-tune the "
                                f"winner on zebrafish to make it domain-correct + push recall higher." + stage_note)
        return self.done({"winner": winner, "ranked": ranked, "results": results}, summary)


_AGENT = DetectorSelect()


def run(q, worker):
    return _AGENT.run(q, worker)
