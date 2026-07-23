"""psf-deconv — feasibility GATE for PSF DECONVOLUTION of the (never-deconvolved) light-sheet images.

Grounded in the host imaging process (Tomer/Keller 2012 SiMView, docs/host_process/imaging_pipeline_notes.md):
the competition volumes are fused, background-corrected light-sheet OPTICAL SECTIONS that were NOT
PSF-deconvolved. Light-sheet's elongated z-PSF blurs/merges nuclei in dense tissue (nearest-neighbour ~5.5µm)
— exactly the merged-peak PRECISION wall. This agent asks, cheaply and honestly, whether a light anisotropic
Richardson-Lucy deconvolution separates merged nuclei enough to matter, WITHOUT a train/test mismatch tanking it.

Three gates (per-embryo, NO golden-12 LB prediction):
  GATE A  merged-peak-resolution DIAGNOSTIC — at GT close-pairs (<=7µm, the merged regime) count local maxima
          RAW vs DECONV; a pair is "resolved" when it goes 1 blob -> >=2 peaks. Detector-agnostic core signal.
  GATE B  RUNTIME — measured s/frame -> ETA for 199*100=19,900 hidden-test frames on 2xT4/12h.
  GATE C  DETECTOR precision — UNLEARNED DoG (mismatch-free) node P/R raw vs deconv; NOTE the learned pilkwang
          detector (trained on RAW) would need a RETRAIN to deploy on deconv (the real cost).

Every knob is XAI-derived from a MEASURED number (voxel scale, 7µm gate, z/xy anisotropy); significance via
math_master.paired_delta_report; precision/tf32 via hardware_tune.load_config. Compute runs under cellmot_venv
in a subprocess (zarr/geff + torch/cuda + scipy/skimage) so the fleet env is untouched.

Spec: {mode:'mini'|'full', n_datasets_per_embryo, frames_per_ds, rl_iters, sigma_xy, sigma_z, cap_sigma_z,
       run_detector, t4_slowdown, timeout}. Data-wise test: test_fleet_agents/psf_deconv_test.py.
"""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from .base import BaseAgent, COMP

PY_ENV = COMP / "research" / "cellmot_venv" / "bin" / "python"
WORKER = COMP / "fleet_agents" / "_psf_deconv_worker.py"


class PsfDeconv(BaseAgent):
    name = "psf-deconv"
    thread = "B"
    kind = "verdict"

    def run(self, q, worker="w"):
        spec = self.spec(q)
        if not PY_ENV.exists() or not WORKER.exists():
            return self.escalate(worker, "researcher", f"[{worker}] psf-deconv: venv/worker missing.")
        payload = json.dumps({
            "mode": spec.get("mode", "mini"),
            "n_datasets_per_embryo": spec.get("n_datasets_per_embryo"),
            "frames_per_ds": spec.get("frames_per_ds"),
            "rl_iters": int(spec.get("rl_iters", 5)),
            "sigma_xy": float(spec.get("sigma_xy", 1.0)),
            "sigma_z": spec.get("sigma_z"),
            "cap_sigma_z": bool(spec.get("cap_sigma_z", True)),
            "run_detector": bool(spec.get("run_detector", True)),
            "t4_slowdown": float(spec.get("t4_slowdown", 5.0)),
            "out": str(COMP / "results" / "psf_deconv_feas.json"),
        })
        timeout = int(spec.get("timeout", 1800))
        try:
            r = subprocess.run([str(PY_ENV), str(WORKER), payload], capture_output=True, text=True,
                               timeout=timeout, cwd=str(COMP))
        except subprocess.TimeoutExpired:
            return self.escalate(worker, "researcher", f"[{worker}] psf-deconv: timed out after {timeout}s.")
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"[{worker}] psf-deconv subprocess failed: {str(e)[:120]}")
        line = [l for l in (r.stdout or "").splitlines() if l.startswith("{")]
        if not line:
            return self.escalate(worker, "researcher",
                                 f"[{worker}] psf-deconv: no result. stderr={(r.stderr or '')[-240:]}")
        res = json.loads(line[-1])
        v = res.get("verdict", {}); A = res.get("gateA_diagnostic", {}); B = res.get("gateB_runtime", {})
        C = res.get("gateC_detector", {}) or {}
        by = A.get("by_embryo", {})
        pe = " · ".join(
            f"{e}: raw2peak={by.get(e, {}).get('raw_2peak_frac')}→deconv2peak={by.get(e, {}).get('deconv_2peak_frac')} "
            f"(n={by.get(e, {}).get('n')}, resolved={by.get(e, {}).get('resolved')})"
            for e in ("44b6", "6bba"))
        detpe = " · ".join(f"{e}: Δprec={C.get(e, {}).get('d_precision')}" for e in ("44b6", "6bba")) if C else "skipped"
        emoji = {"GO": "✅", "WEAK-GO": "🟡", "NO-GO": "⛔"}.get(v.get("decision"), "❓")
        summary = (f"psf-deconv [{res.get('mode')}] {v.get('decision')}: {pe} | detector {detpe} | "
                   f"ETA={B.get('eta_hours_2xT4')}h/2xT4 (fits12h={B.get('fits_12h')})")
        self.log(summary=summary, detail=json.dumps(res)[:1600], kind="verdict",
                 recommendation=v.get("reason", ""))
        # journal an EXP row (feasibility gate; cv=None — this is a diagnostic, not a scored submission).
        # Only for a REAL gate result (guards against journaling on a stubbed/smoke subprocess).
        if v.get("decision") and res.get("gateB_runtime"):
            self.record(change=f"psf_deconv_feas_{res.get('mode')}", script="fleet:psf-deconv", cv=None,
                        train_set="dense-2emb", stage="detection-precision",
                        description="PSF-deconv feasibility gate (SiMView no-deconv): merged-peak resolution + DoG "
                                    "precision + 2xT4 runtime",
                        observation=f"{v.get('decision')} — {v.get('reason', '')[:160]}")
        msg = (f"[{worker}] {emoji} **PSF-DECONV** [{res.get('mode')}] · **{v.get('decision')}**\n"
               f"GATE A (merged-peak resolution): {pe}\n"
               f"GATE C (DoG, mismatch-free): {detpe}\n"
               f"GATE B (runtime): {B.get('sec_per_frame_local')}s/frame local → {B.get('eta_hours_2xT4')}h on 2xT4 "
               f"(fits 12h={B.get('fits_12h')})\n{v.get('reason', '')}")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"verdict": v, "gateA": A, "gateB": B, "gateC": C, "mode": res.get("mode"),
                          "out": str(COMP / "results" / "psf_deconv_feas.json")}, msg, to="leader")


_AGENT = PsfDeconv()


def run(q, worker="w"):
    return _AGENT.run(q, worker)
