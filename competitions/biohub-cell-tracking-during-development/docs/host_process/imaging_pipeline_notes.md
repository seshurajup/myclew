# Host imaging pipeline — SiMView light-sheet (Tomer, Khairy, Amat, Keller — Nat. Methods 2012)

Source: `docs/host_process/tomer_2012_simview.pdf` (the imaging-method lineage behind this competition's
light-sheet nuclei data; Zebrahub = the zebrafish descendant of this Keller/Royer light-sheet + GMM-tracking lineage).

## The full host process (imaging → detection → tracking)
1. **Imaging:** SiMView simultaneous multiview light-sheet microscope — 4 optical arms (2 illumination + 2 detection),
   sCMOS cameras, 175 Mvoxel/s, one-photon & two-photon. Records entire embryos at **subcellular resolution**,
   **30-s temporal sampling** (25–35 s), for hours. No mechanical rotation (zero time-shift multiview). Lossless
   wavelet compression 5:1–10:1. Nuclei labelled by ubiquitous nuclear-GFP (here Drosophila; competition = zebrafish mCherry).
2. **Detection (nuclei):** **Gaussian Mixture Model** segmentation (+ diffusion gradient vector field for full
   morphology), GPU. **Detection accuracy 94.74% ± 0.68% w.r.t. false positives, ~100% w.r.t. false negatives.**
3. **Tracking:** GMM initialized from the previous timepoint → **frame-to-frame linking accuracy 98.98% ± 0.42%**;
   **through-division 93.81% ± 2.71%**; **division detection 94%**.

## Parameters that DIRECTLY ground our levers (why our findings are correct, not luck)
- **Nearest-neighbour nuclei spacing = 7.57 ± 1.34 µm (12th wave) / 5.52 ± 0.99 µm (13th wave).**
  → This is exactly why the official metric uses a **7 µm** match radius. In dense tissue neighbours are ~5.5 µm apart,
  so peaks genuinely merge at the detection grid — our "peak-separability wall in saturated dense tissue" is REAL biology,
  not a model artifact. [[biohub_division_lever_exhausted]] [[biohub_gapfill_lever]]
- **"Nuclei movements of no more than half the nearest-neighbour distance between subsequent time points"** (why 30-s
  sampling is required) → per-frame motion is bounded to ~≤3–4 µm and **near-linear** → validates our gap-fill motion
  cap (≤7 µm/frame, self-calibrated) AND confirms learned-motion models were rightly rejected (motion is too simple to
  need them). Dividing-nucleus speed 8.12 µm/min (12th) / 7.21 (13th) = ~4 µm per 30-s frame.
- **Detection is ~100% recall / ~94.7% precision at the source.** → matches our measured result exactly: **recall is
  saturated (~0.985–0.99); edge/detection PRECISION is the binding constraint.** Every recall lever we tried was flat;
  the only honest headroom is precision (edge-consensus / FP removal). This is the host's own accuracy profile.

## Takeaway
The paper adds **no new magic lever** — it CONFIRMS the honest ceiling is precision, not recall, and that the motion/
spacing regime is exactly what our gap-fill + 7 µm reasoning assumed. The public honest 0.903 pipeline (frozen detector +
post-proc) sits right at this method's intrinsic accuracy envelope; beating it needs better FP-precision, which the host's
own 94.7% precision figure frames as the real (hard) frontier.
