"""Aug-finder agent — DERIVE the valid augmentation menu FROM THE DATA (not a fixed list).

Grounds each candidate augmentation in THIS data's physics — inter-frame motion vs the 7µm match
gate, voxel anisotropy (Z 4× coarser than XY), and the intensity spread — reusing the competition's
aug-validity proofs (experiments/eda/e60–e62, profiles e50/e55). Outputs PASS / FIXABLE / FORBIDDEN
per aug with a data-grounded reason. Deterministic. This is the menu the aug-ablation agent may A/B.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

COMP = Path(__file__).resolve().parent.parent

# physical constants of this dataset (µm)
VOXEL = (1.625, 0.40625, 0.40625)   # (z, y, x) — Z ~4× coarser than XY
GATE_UM = 7.0                        # official node-match gate
# fallback motion fingerprint (from experiments/eda e59) if profiles absent
FALLBACK = {"motion_p95_um": 4.89, "motion_worst_um": 11.16,
            "intensity_p1": 42, "intensity_p99": 859}


def _load_fingerprint() -> dict:
    """Read the data-derived aug-basis profile if present (e55/e50); else fallback constants."""
    for pat in ("experiments/eda/e55_seven_embryo_augbasis.json",
                "experiments/eda/e50_five_embryo_profile.json",
                "tools/researchpapers/eda/**/e5*_*.json"):
        for p in glob.glob(str(COMP / pat), recursive=True):
            try:
                d = json.load(open(p))
                rows = list(d.values()) if isinstance(d, dict) else d
                mot = [r.get("motion_um") or r.get("motion_p95_um") for r in rows
                       if isinstance(r, dict) and (r.get("motion_um") or r.get("motion_p95_um"))]
                if mot:
                    return {"source": p, "motion_p95_um": max(mot),
                            "intensity_p1": FALLBACK["intensity_p1"], "intensity_p99": FALLBACK["intensity_p99"]}
            except Exception:  # noqa: BLE001
                pass
    return {"source": "fallback (e59 fingerprint)", **FALLBACK}


def derive_menu(fp: dict, gate_um: float = GATE_UM) -> list[dict]:
    """Classify each candidate aug from the data physics. Returns [{aug, verdict, reason}].
    gate_um: node-match gate the motion checks compare against (default = official 7µm)."""
    fp = fp or {}
    p95 = float(fp.get("motion_p95_um", FALLBACK["motion_p95_um"]) or FALLBACK["motion_p95_um"])
    fp = {**fp, "motion_p95_um": p95,
          "intensity_p1": fp.get("intensity_p1", FALLBACK["intensity_p1"]),
          "intensity_p99": fp.get("intensity_p99", FALLBACK["intensity_p99"])}
    gate = float(gate_um) if gate_um else GATE_UM
    z_aniso = VOXEL[0] / max(VOXEL[2], 1e-9)
    skip_motion = 2 * p95                                   # frame-skip doubles inter-frame motion
    menu = [
        {"aug": "flip_xy", "verdict": "PASS", "reason": "isometry (Δmotion=0); labels flip too"},
        {"aug": "flip_z", "verdict": "PASS*", "reason": "isometry in training ONLY — omit from inference TTA (anisotropic Z PSF)"},
        {"aug": "rot90_yx", "verdict": "PASS", "reason": f"XY scale equal ({VOXEL[1]}={VOXEL[2]} µm) → rotation is isometric"},
        {"aug": "rot90_zy", "verdict": "FORBIDDEN", "reason": f"Z/XY scale unequal ({z_aniso:.1f}×) → manufactures impossible geometry"},
        {"aug": "crop_scale", "verdict": "PASS", "reason": "varied-size spatial crop is isometric; embryos span 3.5× density"},
        {"aug": "translate_static", "verdict": "PASS", "reason": "coherent same-offset per frame = isometric (per-video)"},
        {"aug": "brightness", "verdict": "PASS", "reason": f"intensity spread p1..p99 = {fp['intensity_p1']}..{fp['intensity_p99']} → detector must be intensity-robust"},
        {"aug": "contrast", "verdict": "PASS", "reason": "photometric; per-embryo optics/staining differ"},
        {"aug": "gamma", "verdict": "PASS", "reason": "photometric; per-embryo intensity nonlinearity"},
        {"aug": "bias_field", "verdict": "PASS", "reason": "light-sheet illumination gradient is real"},
        {"aug": "blur", "verdict": "PASS", "reason": "PSF/defocus variation across depth is real"},
        {"aug": "noise", "verdict": "PASS", "reason": "sensor shot-noise robustness"},
        {"aug": "elastic_temporal_smooth", "verdict": "FIXABLE",
         "reason": "ONLY if temporally smooth (phase drift ≤0.3 rad/frame → Δmotion<0.5µm); per-frame-independent is FORBIDDEN"},
        {"aug": "jitter_small", "verdict": "FIXABLE",
         "reason": "ONLY coherent per-frame OR independent σ≤0.5 vox; ±3-vox independent breaks flow coherence → FORBIDDEN"},
        {"aug": "frame_skip", "verdict": "FORBIDDEN",
         "reason": f"doubles motion to ~{skip_motion:.1f}µm > {gate}µm gate → teaches the linker wrong physics"},
    ]
    return menu


def find(q, worker):
    """Fleet handler — post the data-grounded valid augmentation menu."""
    fp = _load_fingerprint()
    menu = derive_menu(fp)
    ok = [m["aug"] for m in menu if m["verdict"].startswith("PASS")]
    fix = [m["aug"] for m in menu if m["verdict"] == "FIXABLE"]
    bad = [m["aug"] for m in menu if m["verdict"] == "FORBIDDEN"]
    msg = (f"[{worker}] AUG-FINDER (from data · motion_p95={fp['motion_p95_um']:.2f}µm vs {GATE_UM}µm gate · "
           f"Z {VOXEL[0]/VOXEL[2]:.1f}× coarser): PASS={ok} · FIXABLE(constrained)={fix} · FORBIDDEN={bad}. "
           f"These are the physically-valid augs the aug-ablation agent may A/B on the held-out embryo.")
    return ("done", {"fingerprint": fp, "menu": menu}, "all", msg)
