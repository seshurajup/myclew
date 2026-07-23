"""heavy_runnable_pack — "heavy" tools that DO have their real dependency in this env (torch, cv2) so they
are genuinely built + verified on small data, not stubbed:

  • sdf-regression-loss     — signed-distance-field regression target + boundary-weighted loss (BYU/vesuvius
                              sharper-than-Gaussian center).
  • topology-aware-loss     — topology-aware segmentation score (Dice + Betti-0 component agreement) for thin/
                              connected structures (vesuvius/RSNA vessels).
  • ae-latent-view          — REAL torch autoencoder → latent features for ensemble diversity (playground NNs).
  • keypoint-match-verifier — cv2 SIFT/ORB matching + RANSAC homography inliers (recodai copy-move, image match).
  • grid-rectification-unwarp — cv2 homography dewarp of a gridded document to a canonical template (ECG digitization).
"""
from __future__ import annotations
import numpy as np
from .base import BaseAgent


# ---------------------------------------------------------------- sdf-regression-loss
def sdf_from_mask(mask):
    """Signed distance field from a binary mask: +dist inside, -dist outside (sharper center than Gaussian)."""
    from scipy.ndimage import distance_transform_edt
    m = np.asarray(mask, float)
    inside = distance_transform_edt(m); outside = distance_transform_edt(1 - m)
    return inside - outside


def sdf_loss(pred, target, boundary_weight=5.0):
    """Boundary-weighted SDF regression loss (weights the zero-level-set region more)."""
    p = np.asarray(pred, float); t = np.asarray(target, float)
    w = 1.0 + boundary_weight * np.exp(-np.abs(t))
    return float(np.mean(w * (p - t) ** 2))


# ---------------------------------------------------------------- topology-aware-loss
def _betti0(mask):
    from scipy.ndimage import label
    return int(label(np.asarray(mask) > 0.5)[1])


def topology_score(pred_mask, gt_mask):
    """Dice + Betti-0 (connected-component count) agreement — rewards correct TOPOLOGY, not just overlap."""
    p = np.asarray(pred_mask) > 0.5; g = np.asarray(gt_mask) > 0.5
    inter = (p & g).sum(); dice = 2 * inter / (p.sum() + g.sum() + 1e-9)
    b_p, b_g = _betti0(p), _betti0(g)
    betti_agree = 1.0 / (1.0 + abs(b_p - b_g))
    return {"dice": float(dice), "betti0_pred": b_p, "betti0_gt": b_g,
            "topology_score": float(0.5 * dice + 0.5 * betti_agree)}


# ---------------------------------------------------------------- ae-latent-view (real torch)
def autoencoder_latents(X, dim=8, epochs=100, seed=0, lr=1e-3, hidden=64):
    """Train a small autoencoder; return the latent features (n, dim) + final reconstruction MSE.
    lr: AdamW learning rate. hidden: width of the encoder/decoder hidden layer."""
    import torch, torch.nn as nn
    torch.manual_seed(int(seed))
    X = np.asarray(X, float); mu, sd = X.mean(0), X.std(0) + 1e-6
    Xs = torch.tensor(((X - mu) / sd).astype(np.float32))
    dev = "cuda" if torch.cuda.is_available() else "cpu"; Xs = Xs.to(dev)
    d = X.shape[1]; h = int(hidden)
    enc = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Linear(h, dim)).to(dev)
    dec = nn.Sequential(nn.Linear(dim, h), nn.ReLU(), nn.Linear(h, d)).to(dev)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=float(lr))
    lossf = nn.MSELoss(); last = None
    for _ in range(epochs):
        opt.zero_grad(); z = enc(Xs); rec = dec(z); loss = lossf(rec, Xs); loss.backward(); opt.step(); last = float(loss)
    with torch.no_grad():
        Z = enc(Xs).cpu().numpy()
    return Z, last


# ---------------------------------------------------------------- keypoint-match-verifier (cv2)
def match_keypoints(imgA, imgB, min_inliers=8, ratio=0.75):
    """SIFT (or ORB) match + RANSAC homography; returns inlier count (copy-move / image-matching verification).
    ratio: Lowe ratio-test threshold (lower = stricter matches)."""
    import cv2
    a = np.asarray(imgA, np.uint8); b = np.asarray(imgB, np.uint8)
    if a.size == 0 or b.size == 0:
        return {"inliers": 0, "matched": False}
    try:
        det = cv2.SIFT_create()
        norm = cv2.NORM_L2
    except Exception:  # noqa: BLE001
        det = cv2.ORB_create(1000); norm = cv2.NORM_HAMMING
    ka, da = det.detectAndCompute(a, None); kb, db = det.detectAndCompute(b, None)
    if da is None or db is None or len(ka) < 4 or len(kb) < 4:
        return {"inliers": 0, "matched": False}
    bf = cv2.BFMatcher(norm)
    matches = bf.knnMatch(da, db, k=2)
    good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < float(ratio) * pair[1].distance]
    if len(good) < 4:
        return {"inliers": len(good), "matched": False}
    src = np.float32([ka[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kb[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    inl = int(mask.sum()) if mask is not None else 0
    return {"inliers": inl, "matched": inl >= min_inliers}


# ---------------------------------------------------------------- grid-rectification-unwarp (cv2)
def rectify(image, src_quad, dst_size):
    """Warp a quadrilateral region to a canonical rectangle (document/grid dewarp)."""
    import cv2
    img = np.asarray(image, np.uint8)
    W, Hh = dst_size
    dst = np.float32([[0, 0], [W - 1, 0], [W - 1, Hh - 1], [0, Hh - 1]])
    M = cv2.getPerspectiveTransform(np.float32(src_quad), dst)
    return cv2.warpPerspective(img, M, (W, Hh))


# ---------------------------------------------------------------- agents
class _B(BaseAgent):
    thread = "M"; kind = "finding"


class SdfLoss(_B):
    name = "sdf-regression-loss"
    def run(self, q, worker):
        s = self.spec(q)
        try:
            tgt = sdf_from_mask(s["mask"]) if "mask" in s else np.asarray(s["target"], float)
        except Exception as e:  # noqa: BLE001 — scipy missing → escalate cleanly
            return self.escalate(worker, "researcher", f"sdf-regression-loss needs scipy ({e}).")
        loss = sdf_loss(s["pred"], tgt, float(s.get("boundary_weight", 5.0))) if "pred" in s else None
        msg = f"sdf-regression-loss: SDF target built" + (f", loss={loss:.4f}" if loss is not None else "")
        self.log(msg, kind="finding", recommendation="regress the SDF instead of a binary mask for sharper centers")
        return self.done({"loss": loss, "_target": np.asarray(tgt).tolist()}, msg)


class TopologyLoss(_B):
    name = "topology-aware-loss"
    def run(self, q, worker):
        s = self.spec(q)
        try:
            res = topology_score(s["pred_mask"], s["gt_mask"])
        except Exception as e:  # noqa: BLE001 — scipy missing → escalate cleanly
            return self.escalate(worker, "researcher", f"topology-aware-loss needs scipy ({e}).")
        msg = f"topology-aware-loss: score={res['topology_score']:.3f} (dice {res['dice']:.3f}, Betti0 {res['betti0_pred']}vs{res['betti0_gt']})"
        self.log(msg, kind="finding", recommendation="optimize connectivity, not just overlap (thin structures)")
        return self.done(res, msg)


class AeLatentView(_B):
    name = "ae-latent-view"
    def run(self, q, worker):
        try:
            import torch  # noqa: F401
        except Exception:
            return self.escalate(worker, "researcher", "ae-latent-view needs torch (missing).")
        s = self.spec(q)
        missing = [k for k in ("X",) if k not in s]
        if missing:
            return self.escalate(worker, "leader", f"ae-latent-view needs spec keys {missing} — none provided")
        Z, rec = autoencoder_latents(s["X"], int(s.get("dim", 8)), int(s.get("epochs", 100)),
                                                       seed=int(s.get("seed", 0)), lr=float(s.get("lr", 1e-3)),
                                                       hidden=int(s.get("hidden", 64)))
        msg = f"ae-latent-view: {Z.shape[1]}-dim latent features (recon MSE {rec:.4f}) for ensemble diversity"
        self.log(msg, kind="finding", recommendation="append latents as features to GBDTs/NNs for decorrelation")
        return self.done({"recon_mse": rec, "_latents": Z.tolist()}, msg)


class KeypointMatch(_B):
    name = "keypoint-match-verifier"
    def run(self, q, worker):
        s = self.spec(q)
        try:
            res = match_keypoints(s["imgA"], s["imgB"], int(s.get("min_inliers", 8)), ratio=float(s.get("ratio", 0.75)))
        except Exception as e:  # noqa: BLE001 — cv2 missing → escalate cleanly
            return self.escalate(worker, "researcher", f"keypoint-match-verifier needs cv2 ({e}).")
        msg = f"keypoint-match-verifier: {res['inliers']} RANSAC inliers → matched={res['matched']}"
        self.log(msg, kind="finding", recommendation="use for copy-move forgery / image-matching verification")
        return self.done(res, msg)


class GridRectify(_B):
    name = "grid-rectification-unwarp"
    def run(self, q, worker):
        s = self.spec(q)
        try:
            out = rectify(s["image"], s["src_quad"], tuple(s["dst_size"]))
        except Exception as e:  # noqa: BLE001 — cv2 missing / bad quad → escalate cleanly
            return self.escalate(worker, "researcher", f"grid-rectification-unwarp needs cv2 + a valid quad ({e}).")
        msg = f"grid-rectification-unwarp: dewarped to {out.shape}"
        self.log(msg, kind="finding", recommendation="rectify the sheet before waveform/grid extraction (ECG)")
        return self.done({"shape": list(out.shape)}, msg)


_SDF = SdfLoss(); _TOP = TopologyLoss(); _AE = AeLatentView(); _KP = KeypointMatch(); _GR = GridRectify()


def run_sdf(q, worker): return _SDF.run(q, worker)
def run_topology(q, worker): return _TOP.run(q, worker)
def run_ae(q, worker): return _AE.run(q, worker)
def run_keypoint(q, worker): return _KP.run(q, worker)
def run_rectify(q, worker): return _GR.run(q, worker)
