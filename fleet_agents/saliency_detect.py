"""saliency-detect — turn an XAI saliency/CAM map (results/xai/*.npy) into ADD-ONLY candidate nuclei to
RECOVER detections the primary head thresholded out → boosts node_recall, the lever that actually moves
the metric (adjJ ≈ node_rec²·edge_prec, [[biohub_node_recall_lever]]).

The idea (the user's): saliency highlights WHERE the model sees a cell. A CAM can stay bright at a nucleus
that the final detection head scored just under threshold — so local maxima of the saliency map are
candidate cells. We keep ONLY the peaks that are NOT already covered by an existing detection (add-only,
never remove) — the same shape as abhijith's DeepCenter "add-only repair gate" that recovered missed nuclei
(node_recall 0.988) without flooding FP. A gated ADD, not a replacement ([[biohub_detectors_complementary]]).

Pipeline: saliency [D,H,W] → normalize → threshold (frac of max) → 3D non-max-suppression (max-filter) →
peaks → drop peaks within `merge_vox` of an existing node → NEW candidates. Report count + coords so the
linker can absorb them (and det-sweep / recipe-adopt can measure the golden-12 delta).

Reusable / spec-driven: {saliency: np-array or npy path, existing_nodes: [[z,y,x],...], thresh_frac: 0.5,
   merge_vox: 3.0, max_add: 5000}. A BaseAgent subclass with its own data-wise test.
"""
from __future__ import annotations
from pathlib import Path
from .base import BaseAgent

COMP = Path(__file__).resolve().parent.parent


def peaks_from_saliency(sal, thresh_frac=0.5, merge_vox=3.0, existing=None, max_add=5000, nms_size=3):
    """Local maxima of a saliency volume above thresh_frac·max, NMS'd, minus any within merge_vox of an
    existing node. Returns (new_peaks[[z,y,x]], n_peaks_total). Pure/vectorised — testable without GPU.

    nms_size: cube side (voxels) for the local-maximum filter — larger = sparser peaks (default 3)."""
    import numpy as np
    from scipy.ndimage import maximum_filter
    sal = np.nan_to_num(np.asarray(sal, "float32"))       # sanitize NaN/Inf so peaks are well-defined
    if sal.ndim != 3 or sal.size == 0:
        return np.zeros((0, 3)), 0
    s = (sal - sal.min()) / (np.ptp(sal) + 1e-8)
    thr = float(thresh_frac)
    mx = maximum_filter(s, size=max(1, int(nms_size)))
    ispeak = (s == mx) & (s >= thr)                           # local maxima above threshold
    coords = np.argwhere(ispeak).astype(float)                # [n,3] (z,y,x)
    if len(coords) == 0:
        return np.zeros((0, 3)), 0
    n_total = len(coords)
    if existing is not None and len(existing):
        ex = np.asarray(existing, "float32")
        from scipy.spatial import cKDTree
        d, _ = cKDTree(ex).query(coords, k=1)
        coords = coords[d > merge_vox]                        # ADD-ONLY: drop peaks already covered
    return coords[:max_add], n_total


class SaliencyDetect(BaseAgent):
    name = "saliency-detect"
    thread = "B"
    kind = "verdict"

    def run(self, q, worker):
        import numpy as np
        spec = self.spec(q)
        sal = spec.get("saliency")
        try:
            if isinstance(sal, str):
                sal = np.load(sal)
            elif sal is None:                                 # default: newest CNN saliency the xai agent saved
                import glob
                cands = sorted(glob.glob(str(COMP / "results" / "xai" / "cnn_*.npy")))
                if not cands:
                    return self.escalate(worker, "researcher",
                                         f"[{worker}] saliency-detect: no saliency npy found — run the xai (cnn family) agent first.")
                sal = np.load(cands[-1])
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher",
                                 f"[{worker}] saliency-detect: could not load saliency map ({str(e)[:80]}).")
        existing = spec.get("existing_nodes")
        new, n_total = peaks_from_saliency(sal, thresh_frac=float(spec.get("thresh_frac", 0.5)),
                                           merge_vox=float(spec.get("merge_vox", 3.0)),
                                           existing=existing, max_add=int(spec.get("max_add", 5000)),
                                           nms_size=int(spec.get("nms_size", 3)))
        n_new = int(len(new))
        n_ex = int(len(existing)) if existing is not None else 0
        recall_gain = round(n_new / max(n_ex, 1), 3) if n_ex else None
        self.save_state({"peaks_total": n_total, "already_covered": n_total - n_new if existing is not None else 0,
                         "new_candidates": n_new, "existing": n_ex, "recall_gain_frac": recall_gain,
                         "thresh_frac": spec.get("thresh_frac", 0.5)})
        self.log(summary=f"saliency-detect: {n_new} ADD-ONLY candidate nuclei from {n_total} saliency peaks "
                         f"(+{recall_gain} vs {n_ex} existing)" if n_ex else f"saliency-detect: {n_new} candidate nuclei from {n_total} peaks",
                 detail="add-only recall repair — union with primary detector, then measure golden-12 delta (det-sweep)",
                 kind="verdict", recommendation="union these as extra nodes + re-link + score canonical golden-12 (recipe-adopt/det-sweep gate)")
        msg = (f"[{worker}] **SALIENCY-DETECT** · {n_total} saliency peaks → **{n_new} NEW** add-only candidate nuclei"
               + (f" (+{recall_gain} over {n_ex} existing detections)" if n_ex else "") + "\n"
               f"→ union with the primary detector to recover missed nodes (the node-recall lever), then gate on canonical golden-12.")
        self.post(worker, "leader", msg, routine=False, kind="verdict")
        return self.done({"new_candidates": n_new, "peaks_total": n_total, "recall_gain_frac": recall_gain,
                          "coords": new.tolist()[:50]}, msg, to="leader")


_AGENT = SaliencyDetect()


def run(q, worker):
    return _AGENT.run(q, worker)
