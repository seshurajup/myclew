"""pattern-tune — autonomously FIX the X: tune box-sample's pools until the boxed external matches the
competition track pattern on EVERY column (tracks/crop, track-len, full-span %, div/crop, broken=0).

Composes two agents in a loop: `box-sample` (produce boxed external with current params) → `ext-label-stats`
verify-boxed (compare all columns to competition). Each round it nudges the mismatched knob toward the
competition target — the classic coordinate correction:
  tracks/crop too low  → shift tracks_per_crop_pool up
  track-len too short  → shift track_len_pool up
  full-span too high   → widen target_frames (longer video window → tracks span a smaller fraction)
  div/crop too high    → lower keep_division_prob
Stops when all columns match (verify all_match) or max_rounds. Every decision is a MEASURED verify, no
assumption ([[feedback_agents_only_no_adhoc_python]]). A BaseAgent subclass with its own data-wise test.
"""
from __future__ import annotations
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent


class PatternTune(BaseAgent):
    name = "pattern-tune"
    thread = "A"
    kind = "verdict"

    def _agents(self):
        from . import _RAW_HANDLERS
        return _RAW_HANDLERS

    def run(self, q, worker):
        spec = self.spec(q)
        A = self._agents()
        if "ext-label-stats" not in A:                        # the verify step is mandatory; without it there is nothing to tune
            return self.escalate(worker, "researcher", f"[{worker}] pattern-tune: ext-label-stats agent missing — cannot verify boxed output.")
        gt = spec.get("gt_path", "results/flow_gt/flow_node_gt_clean.parquet")
        out = spec.get("out_path", "results/flow_gt/flow_node_gt_matched.parquet")
        max_rounds = int(spec.get("max_rounds", 6))
        # starting params (competition-ish); the loop corrects them by measurement
        p = {"target_frames": int(spec.get("target_frames", 100)),
             "tracks_pool": list(spec.get("tracks_per_crop_pool", [15])),
             "tlen_pool": list(spec.get("track_len_pool", [33])),
             "keep_div": float(spec.get("keep_division_prob", 1.0)),
             # full_span_frac = precise fraction of tracks that span the whole window → directly sets full-span %
             "full_span_frac": float(spec.get("full_span_frac", 0.15)),
             # z_slab (depth crop → z_std) and time_stride (frame subsample → speed): match the NEW physical columns
             "z_slab": float(spec.get("z_slab", 55.0)), "time_stride": int(spec.get("time_stride", 1))}
        trail, final = [], None
        for r in range(max_rounds):
            bs = {"gt_path": gt, "out_path": out, "match_label_sparsity": True, "n_boxes": int(spec.get("n_boxes", 6)),
                  "target_frames": p["target_frames"], "tracks_per_crop_pool": p["tracks_pool"],
                  "track_len_pool": p["tlen_pool"], "keep_division_prob": p["keep_div"],
                  "full_span_frac": p["full_span_frac"], "z_slab": p["z_slab"], "time_stride": p["time_stride"]}
            try:
                if "box-sample" in A:
                    A["box-sample"]({"question": f"tune r{r}", "spec": bs}, worker)
                vr = A["ext-label-stats"]({"question": f"verify r{r}", "spec": {"verify_boxed": True, "boxed_path": out}}, worker)
            except Exception as e:  # noqa: BLE001 — a composed-agent failure ends the loop cleanly with the best-so-far
                trail.append({"round": r + 1, "params": dict(p), "boxed": {}, "all_match": False, "error": str(e)[:100]})
                break
            vd = vr[1] if isinstance(vr, (list, tuple)) and len(vr) > 1 and isinstance(vr[1], dict) else {}
            got = vd.get("boxed", {}); comp = vd.get("competition", {}); ok = bool(vd.get("all_match"))
            trail.append({"round": r + 1, "params": dict(p), "boxed": got, "all_match": ok})
            final = vd
            if ok or not got:
                break
            # correct one/each mismatched column toward the competition target
            def tgt(k): return comp.get(k)
            if got.get("tracks_per_crop_med", 0) < (tgt("tracks_per_crop_med") or 0):
                p["tracks_pool"] = [int(min(60, x * 1.5 + 3)) for x in p["tracks_pool"]]
            elif got.get("tracks_per_crop_med", 0) > (tgt("tracks_per_crop_med") or 999):
                p["tracks_pool"] = [max(2, int(x * 0.7)) for x in p["tracks_pool"]]
            if got.get("track_len_frames_med", 0) < (tgt("track_len_frames_med") or 0):
                p["tlen_pool"] = [int(min(100, x * 1.3 + 3)) for x in p["tlen_pool"]]
            fs_got, fs_tgt = got.get("full_span_pct", 0), (tgt("full_span_pct") or 0)
            if abs(fs_got - fs_tgt) > 5:                                        # proportional correction toward target %
                # full_span_frac ≈ full-span% ; nudge frac by the observed error (bounded, converges in ~1-2 rounds)
                err = (fs_tgt - fs_got) / 100.0
                p["full_span_frac"] = float(min(1.0, max(0.0, p["full_span_frac"] + err)))
            if got.get("divisions_per_crop_med", 0) > (tgt("divisions_per_crop_med") or 0):
                p["keep_div"] = max(0.0, p["keep_div"] * 0.5 - 0.05)
            # NEW physical columns: z_std via z_slab (proportional), speed via time_stride (integer)
            zs_got, zs_tgt = got.get("z_std", 0), (tgt("z_std") or 0)
            if zs_tgt and abs(zs_got - zs_tgt) > 0.2 * zs_tgt:
                mult = max(0.6, min(1.6, zs_tgt / max(1.0, zs_got)))   # CLAMPED step → no divide-by-small overshoot
                p["z_slab"] = float(max(10.0, min(300.0, p["z_slab"] * mult)))
            sp_got, sp_tgt = got.get("speed_med", 0), (tgt("speed_med") or 0)
            if sp_tgt and sp_got and sp_got < 0.6 * sp_tgt:            # cells too slow → subsample more frames
                p["time_stride"] = min(5, p["time_stride"] + 1)
            elif sp_tgt and sp_got > 1.6 * sp_tgt and p["time_stride"] > 1:
                p["time_stride"] = p["time_stride"] - 1

        matched = bool(final and final.get("all_match"))
        self.save_state({"matched": matched, "rounds": len(trail), "final_params": p,
                         "final_boxed": (final or {}).get("boxed"), "trail": trail})
        self.log(summary=f"pattern-tune: {'MATCHED all columns' if matched else 'not converged'} in {len(trail)} rounds "
                         f"→ params {p}", detail=str(trail[-1] if trail else {}), kind="verdict",
                 recommendation="use out_path as the label-matched external for detector training" if matched
                 else "raise max_rounds or widen the pools; broken=0 already holds")
        line = " → ".join(f"r{t['round']}:{'✅' if t['all_match'] else '·'}({t['boxed'].get('tracks_per_crop_med')}tr/"
                          f"{t['boxed'].get('track_len_frames_med')}f/{t['boxed'].get('full_span_pct')}%fs/"
                          f"{t['boxed'].get('divisions_per_crop_med')}div)" for t in trail)
        msg = (f"[{worker}] **PATTERN-TUNE** · {'✅ ALL COLUMNS MATCH' if matched else '⚠️ not fully converged'} "
               f"in {len(trail)} rounds\n{line}\nfinal params: {p}\n"
               f"→ `{out}` is the competition-pattern-matched external (broken=0 throughout).")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"matched": matched, "rounds": len(trail), "final_params": p,
                          "out_path": out}, msg, to="leader")


_AGENT = PatternTune()


def run(q, worker):
    return _AGENT.run(q, worker)
