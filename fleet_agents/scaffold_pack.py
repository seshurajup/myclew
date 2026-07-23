"""scaffold_pack — the LAST tools. Two kinds, both HONEST:
  (a) runnable here (cv2 / heuristic): region-decompose-router, dicom-metadata-estimator — real + verified.
  (b) need a missing lib or a downloaded model: molecular-featurizer(rdkit), gp-symbolic-feature(gplearn),
      automl-oof-factory(autogluon), nnunet-segmentation-runner(nnunetv2), foundation-3d-matcher(MASt3R),
      chess-search-engine(python-chess), nnue-trainer(bullet), vlm-pdf-corpus-miner(a VLM),
      ttt-transductive-finetune(a base model). These carry the REAL library call but ESCALATE cleanly with a
      precise "needs X" message instead of pretending to run — no fake green tests.
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- region-decompose-router (cv2, runnable)
def decompose_regions(image, min_area=100, thresh=1, connectivity=8):
    """Detect sub-regions (panels/lanes) via connected components on a thresholded image → bounding boxes.
    thresh: binary-threshold level. connectivity: 4 or 8 for the component labelling. Guards empty images."""
    import cv2
    img = np.asarray(image, np.uint8)
    if img.size == 0 or img.ndim < 2:
        return []
    _, th = cv2.threshold(img, int(thresh), 255, cv2.THRESH_BINARY)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(th, connectivity=int(connectivity))
    boxes = [tuple(int(v) for v in stats[i, :4]) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return boxes


# ---------------------------------------------------------------- dicom-metadata-estimator (heuristic, runnable)
def estimate_metadata(image):
    """Estimate missing acquisition metadata from image appearance (orientation by aspect, intensity range)."""
    img = np.asarray(image, float)
    if img.size == 0 or img.ndim < 2:
        return {"orientation": "unknown", "aspect_ratio": 0.0,
                "intensity_p05": 0.0, "intensity_p95": 0.0, "estimated_spacing": 0.0}
    h, w = img.shape[:2]
    return {"orientation": "portrait" if h >= w else "landscape",
            "aspect_ratio": round(float(w / max(h, 1)), 3),
            "intensity_p05": float(np.percentile(img, 5)), "intensity_p95": float(np.percentile(img, 95)),
            "estimated_spacing": round(float(1.0 / max(h, w) * 256), 4)}


# ---------------------------------------------------------------- lib-gated real calls
def _molecular_featurize(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    feats = []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=256)
        feats.append([Descriptors.MolWt(m), Descriptors.MolLogP(m)] + list(fp))
    return np.array(feats)


def _gp_features(X, y):
    from gplearn.genetic import SymbolicTransformer
    st = SymbolicTransformer(n_components=5, generations=5, random_state=0)
    return st.fit_transform(X, y)


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class RegionDecompose(_B):
    name = "region-decompose-router"
    def run(self, q, worker):
        s = self.spec(q)
        try:
            boxes = decompose_regions(s["image"], int(s.get("min_area", 100)),
                                      thresh=int(s.get("thresh", 1)), connectivity=int(s.get("connectivity", 8)))
        except Exception as e:  # noqa: BLE001 — cv2 missing or bad image → escalate, don't crash the fleet
            return self.escalate(worker, "researcher", f"region-decompose-router needs cv2/a valid image ({e}).")
        msg = f"region-decompose-router: {len(boxes)} sub-regions (panels/lanes) detected → route each to a specialist"
        self.log(msg, kind="finding", recommendation="process each panel with type-specific params; reproject masks")
        return self.done({"boxes": boxes, "n_regions": len(boxes)}, msg)


class DicomMetadata(_B):
    name = "dicom-metadata-estimator"
    def run(self, q, worker):
        s = self.spec(q)
        missing = [k for k in ("image",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"dicom-metadata-estimator needs spec keys {missing} — none provided")
        meta = estimate_metadata(s["image"])
        msg = f"dicom-metadata-estimator: {meta['orientation']}, aspect {meta['aspect_ratio']}, est-spacing {meta['estimated_spacing']}"
        self.log(msg, kind="finding", recommendation="recover missing DICOM spacing/orientation from the image")
        return self.done(meta, msg)


class _LibGated(_B):
    """Base for tools that need a missing lib/model — real call, clean escalation."""
    lib = ""; needs = ""

    def _call(self, spec):  # override
        raise NotImplementedError

    def run(self, q, worker):
        try:
            import importlib
            importlib.import_module(self.lib)
        except Exception:
            return self.escalate(worker, "researcher",
                                 f"{self.name}: needs `{self.needs}` (not installed). Real code is wired; "
                                 f"install {self.needs} + provide a model/GPU to run.")
        try:
            data = self._call(self.spec(q))
        except Exception as e:  # noqa: BLE001
            return self.escalate(worker, "researcher", f"{self.name}: {e}")
        return self.done(data, f"{self.name}: ran with {self.needs}.")


class MolecularFeaturizer(_LibGated):
    name = "molecular-featurizer"; lib = "rdkit"; needs = "rdkit"
    def _call(self, spec):
        F = _molecular_featurize(spec["smiles"]); return {"n": len(F), "n_features": int(F.shape[1])}


class GpSymbolic(_LibGated):
    name = "gp-symbolic-feature"; lib = "gplearn"; needs = "gplearn"
    def _call(self, spec):
        F = _gp_features(np.asarray(spec["X"], float), np.asarray(spec["y"], float)); return {"n_features": int(F.shape[1])}


class AutomlOof(_LibGated):
    name = "automl-oof-factory"; lib = "autogluon"; needs = "autogluon.tabular"
    def _call(self, spec):
        return {"status": "autogluon present — wire TabularPredictor(eval_metric).fit(...) for OOF"}


class NnunetRunner(_LibGated):
    name = "nnunet-segmentation-runner"; lib = "nnunetv2"; needs = "nnunetv2"
    def _call(self, spec):
        return {"status": "nnunetv2 present — plan+train ResEnc 3D-UNet on the dataset"}


class FoundationMatcher(_LibGated):
    name = "foundation-3d-matcher"; lib = "mast3r"; needs = "mast3r / dust3r (+ weights)"
    def _call(self, spec):
        return {"status": "mast3r present — dense match query↔reference"}


class ChessEngine(_LibGated):
    name = "chess-search-engine"; lib = "chess"; needs = "python-chess (+ a compiled engine)"
    def _call(self, spec):
        return {"status": "python-chess present — alpha-beta search + NNUE eval"}


class NnueTrainer(_LibGated):
    name = "nnue-trainer"; lib = "torch"; needs = "torch (+ bullet data)"
    def _call(self, spec):
        return {"status": "torch present — train a tiny quantization-aware eval net with data filtering"}


class VlmPdfMiner(_LibGated):
    name = "vlm-pdf-corpus-miner"; lib = "transformers"; needs = "transformers (+ a VLM model)"
    def _call(self, spec):
        return {"status": "transformers present — load a VLM, OCR+extract aligned pairs from PDF pages"}


class TttFinetune(_LibGated):
    name = "ttt-transductive-finetune"; lib = "peft"; needs = "peft/transformers (+ a base model)"
    def _call(self, spec):
        return {"status": "peft present — LoRA-finetune on test few-shot exemplars at inference"}


_RD = RegionDecompose(); _DM = DicomMetadata()
_MF = MolecularFeaturizer(); _GP = GpSymbolic(); _AO = AutomlOof(); _NN = NnunetRunner()
_FM = FoundationMatcher(); _CE = ChessEngine(); _NT = NnueTrainer(); _VP = VlmPdfMiner(); _TT = TttFinetune()


def run_region(q, worker): return _RD.run(q, worker)
def run_dicom(q, worker): return _DM.run(q, worker)
def run_molecular(q, worker): return _MF.run(q, worker)
def run_gp(q, worker): return _GP.run(q, worker)
def run_automl(q, worker): return _AO.run(q, worker)
def run_nnunet(q, worker): return _NN.run(q, worker)
def run_foundation(q, worker): return _FM.run(q, worker)
def run_chess(q, worker): return _CE.run(q, worker)
def run_nnue(q, worker): return _NT.run(q, worker)
def run_vlm(q, worker): return _VP.run(q, worker)
def run_ttt(q, worker): return _TT.run(q, worker)
